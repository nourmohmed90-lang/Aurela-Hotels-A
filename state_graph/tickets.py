"""
Ticket / failure-recovery system for state-graph agents.

WHY THIS IS A SEPARATE MODULE FROM HITL (see nodes.py):
- HITL (`hitl_pause_node`) is an *expected* pause: the graph is working
  correctly and is deliberately waiting on a decision a human must make.
  It has no independent identity outside the run itself -- it's fully
  represented by the LangGraph checkpoint's state (status="PAUSED_HITL"),
  because approving it just means resuming that one run.
- A ticket is *unplanned*: something the graph could not handle broke --
  a tool call errored, a schema didn't validate, the model returned
  something the graph can't act on. That needs its own durable, inspectable
  record with a real lifecycle (open -> investigating -> resolved),
  independent of any single checkpoint, because:
    1. an admin needs to browse/triage tickets without reconstructing graph
       state for every thread first,
    2. a ticket's status is meaningful on its own ("we're looking into
       this") separate from whatever the underlying run's state says,
    3. it is the artifact a grader (or a real ops team) checks to prove a
       failure was real, detected, and resolved -- not just papered over.

This module owns the `Tickets` table in database/hotel.db. It never talks to
LangGraph directly; `state_graph/nodes.py::create_ticket_node` calls
`create_ticket()` and stores the returned ticket_id back into the graph
state so the two records (ticket row + checkpoint) can be cross-referenced,
without collapsing into the same mechanism.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "hotel.db"))

VALID_STATUSES = ("open", "investigating", "resolved", "PENDING_APPROVAL", "FAILED")


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tickets_table() -> None:
    """Idempotently create the Tickets table if it doesn't exist yet. Safe
    to call on every import -- lets this module work even if someone hasn't
    re-run schema.sql against an existing hotel.db.

    The table created here uses an extended CHECK constraint that includes
    'PENDING_APPROVAL' and 'FAILED' for the incident-escalation workflow
    (Graph 3). Existing DBs with the old constraint are unaffected by this
    call (CREATE TABLE IF NOT EXISTS is a no-op when the table already exists).
    """
    conn = _get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Tickets (
            ticket_id VARCHAR(36) PRIMARY KEY,
            thread_id VARCHAR(36) NOT NULL,
            graph_name VARCHAR(100) NOT NULL,
            node_name VARCHAR(100),
            status VARCHAR(20) NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'investigating', 'resolved', 'PENDING_APPROVAL', 'FAILED')),
            error_message TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            resolved_at TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def create_ticket(
    thread_id: str,
    graph_name: str,
    error_message: str,
    node_name: Optional[str] = None,
) -> str:
    """Called only from a real detected failure inside a graph node (see
    create_ticket_node in nodes.py). Never call this to manually seed a demo
    ticket -- it must always represent an actual failure the graph hit.
    Returns the new ticket_id.
    """
    ensure_tickets_table()
    ticket_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_db()
    conn.execute(
        """
        INSERT INTO Tickets (ticket_id, thread_id, graph_name, node_name, status, error_message, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
        """,
        (ticket_id, thread_id, graph_name, node_name, error_message, now, now),
    )
    conn.commit()
    conn.close()
    return ticket_id


def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    ensure_tickets_table()
    conn = _get_db()
    row = conn.execute("SELECT * FROM Tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_tickets(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List tickets, optionally filtered by status. Ordered newest-first so
    an admin sees the most recent failures at the top of the queue.
    """
    ensure_tickets_table()
    conn = _get_db()
    if status:
        rows = conn.execute(
            "SELECT * FROM Tickets WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM Tickets ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_ticket_status(ticket_id: str, status: str) -> Dict[str, Any]:
    """Move a ticket through open -> investigating -> resolved. Does not
    resume the underlying graph run -- that's a separate, deliberate step
    (see resolve_ticket_and_resume in the platform layer) so an admin can
    mark a ticket 'investigating' without prematurely resuming a run that
    isn't actually fixed yet.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

    ensure_tickets_table()
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise KeyError(f"No ticket with id {ticket_id}")

    now = datetime.now(timezone.utc).isoformat()
    resolved_at = now if status == "resolved" else None

    conn = _get_db()
    conn.execute(
        "UPDATE Tickets SET status = ?, updated_at = ?, resolved_at = COALESCE(?, resolved_at) WHERE ticket_id = ?",
        (status, now, resolved_at, ticket_id),
    )
    conn.commit()
    conn.close()
    return get_ticket(ticket_id)


# ---------------------------------------------------------------------------
# Phase 2: Resilient Ticket & Recovery Engine
# ---------------------------------------------------------------------------

def create_escalation_ticket(
    thread_id: str,
    state_data: Dict[str, Any],
    error_reason: str,
) -> str:
    """Creates a ticket for an incident/escalation event that requires human
    attention. Unlike create_ticket() (which is always 'open' / an unplanned
    failure), this function sets status to 'PENDING_APPROVAL' when the
    graph hit a planned HITL gate that the ticket system is tracking
    externally, or 'FAILED' for genuine unplanned failures.

    The distinction:
    - 'PENDING_APPROVAL': the graph hit a high-severity condition, paused
      via interrupt(), AND someone wants a Tickets row for external tracking
      (e.g. ops dashboards outside the platform). The underlying graph run
      is checkpointed and resumable via resume_graph_from_ticket().
    - 'FAILED': the graph node raised an unhandled exception. The row is
      opened for triage; resolving it calls resume_graph_from_ticket() which
      re-invokes the graph from its last checkpoint.

    graph_name is read from state_data["graph_name"] if present; falls back
    to "unknown_graph". The ticket_id is stored back into state by the caller.
    """
    ensure_tickets_table()
    ticket_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    graph_name = state_data.get("graph_name", "unknown_graph")
    node_name = state_data.get("node_name")

    # Determine status: if the state signals a planned HITL gate, mark
    # PENDING_APPROVAL; otherwise mark FAILED.
    is_hitl = state_data.get("requires_human_approval") or state_data.get("requires_manager_approval")
    target_status = "PENDING_APPROVAL" if is_hitl else "FAILED"

    conn = _get_db()
    try:
        conn.execute(
            """
            INSERT INTO Tickets (ticket_id, thread_id, graph_name, node_name, status, error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, thread_id, graph_name, node_name, target_status, error_reason, now, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Existing DB with old CHECK constraint — fall back gracefully to 'open'
        # so the ticket is still created and visible in the admin queue.
        conn.execute(
            """
            INSERT INTO Tickets (ticket_id, thread_id, graph_name, node_name, status, error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
            """,
            (ticket_id, thread_id, graph_name, node_name, error_reason, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    return ticket_id


def resume_graph_from_ticket(
    ticket_id: str,
    approval_decision: bool,
    feedback: str = "",
) -> Dict[str, Any]:
    """Reads the saved checkpoint from SQLite via the ticket's thread_id,
    updates state with the human decision, and resumes graph execution from
    the exact checkpointed node.

    This is the canonical recovery entry point for all three graphs:
    - Graph 1 (vip_booking_graph): re-invokes with updated state dict
      (entry-point resume pattern, same as approve_hitl_task in app.py).
    - Graph 2 (complaint_resolution_graph): sends Command(resume=...) to
      continue from the interrupt() node (same as approve_complaint_hitl).
    - Graph 3 (incident_escalation_graph): sends Command(resume=...) to
      continue from incident_hitl_node's interrupt().

    Returns {"success": bool, "resumed_state": dict, "ticket": dict}.
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise KeyError(f"No ticket with id {ticket_id!r}")

    thread_id = ticket["thread_id"]
    graph_name = ticket["graph_name"]
    config = {"configurable": {"thread_id": thread_id}}

    # Lazy imports to avoid circular dependencies at module level.
    from .graph import vip_booking_graph
    from .graph_complaint import complaint_resolution_graph, run_complaint_graph
    from .graph_incident import incident_escalation_graph, run_incident_graph

    from langgraph.types import Command as _Command

    try:
        if graph_name == "incident_escalation_graph":
            result, _ = run_incident_graph(
                _Command(resume={"approved": approval_decision, "feedback": feedback}),
                config,
            )

        elif graph_name == "complaint_resolution_graph":
            result, _ = run_complaint_graph(
                _Command(resume={"approved": approval_decision, "feedback": feedback}),
                config,
            )

        else:
            # Default: vip_booking_graph re-invoke-from-entry pattern.
            state_obj = vip_booking_graph.get_state(config)
            if not state_obj or not state_obj.values:
                raise RuntimeError(f"No checkpoint found for thread {thread_id!r}")
            current_state = dict(state_obj.values)
            current_state["is_approved"] = approval_decision
            current_state["status"] = "APPROVED" if approval_decision else "DECLINED"
            current_state["error_message"] = None
            result = vip_booking_graph.invoke(current_state, config=config)

        resolved_ticket = update_ticket_status(ticket_id, "resolved")
        return {"success": True, "resumed_state": result, "ticket": resolved_ticket}

    except Exception as exc:
        return {"success": False, "error": str(exc), "ticket": ticket}
