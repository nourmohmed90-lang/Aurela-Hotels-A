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

VALID_STATUSES = ("open", "investigating", "resolved")


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tickets_table() -> None:
    """Idempotently create the Tickets table if it doesn't exist yet. Safe
    to call on every import -- lets this module work even if someone hasn't
    re-run schema.sql against an existing hotel.db.
    """
    conn = _get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Tickets (
            ticket_id VARCHAR(36) PRIMARY KEY,
            thread_id VARCHAR(36) NOT NULL,
            graph_name VARCHAR(100) NOT NULL,
            node_name VARCHAR(100),
            status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved')),
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
