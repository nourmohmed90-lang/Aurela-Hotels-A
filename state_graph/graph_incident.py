"""
Graph #3: Incident & Maintenance Escalation.

Uses the same SqliteSaver (hotel.db) as Graph 1 and Graph 2 — single shared
checkpoint store, with thread IDs prefixed 'incident-' to distinguish runs
from the other graphs in the checkpoints table.

Routing logic:
    analyze_incident
        ↓
    generate_resolution
        ├─ FAILED_TICKET  → create_ticket → END
        ├─ requires_human_approval  → hitl_approval
        │       ├─ APPROVED  → END (COMPLETED)
        │       └─ DECLINED  → END (DECLINED)
        └─ (low/medium, no approval needed)  → END (COMPLETED)

Failure handling: analyze_incident_and_policy may raise IncidentProcessingError
on RAG failures. run_incident_graph() (below) is the single catch point — same
pattern as run_complaint_graph() in graph_complaint.py. The node itself raises
rather than self-catching, so the checkpoint sits at the last successful step
and is genuinely resumable.
"""

import os
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .schemas import IncidentEscalationState
from .nodes_incident import (
    analyze_incident_and_policy,
    generate_resolution_plan,
    incident_hitl_node,
    create_incident_ticket_node,
    IncidentProcessingError,
    GRAPH_NAME,
)

# Same hotel.db as the other two graphs.
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "hotel.db"))
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_analysis(state: IncidentEscalationState) -> str:
    """After Node 1: bail to ticket queue on failure, else continue."""
    if state.get("status") == "FAILED_TICKET":
        return "create_ticket"
    return "generate_resolution"


def route_after_resolution(state: IncidentEscalationState) -> str:
    """After Node 2: HITL gate if the issue warrants manager approval."""
    if state.get("status") == "FAILED_TICKET":
        return "create_ticket"
    if state.get("requires_human_approval"):
        return "hitl_approval"
    return END


def route_after_hitl(state: IncidentEscalationState) -> str:
    """After the HITL interrupt resolves: approved → COMPLETED, else DECLINED → END."""
    # Both paths go to END; the status field carries the outcome.
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_incident_graph():
    builder = StateGraph(IncidentEscalationState)

    builder.add_node("analyze_incident", analyze_incident_and_policy)
    builder.add_node("generate_resolution", generate_resolution_plan)
    builder.add_node("hitl_approval", incident_hitl_node)
    builder.add_node("create_ticket", create_incident_ticket_node)

    builder.set_entry_point("analyze_incident")

    builder.add_conditional_edges(
        "analyze_incident",
        route_after_analysis,
        {
            "generate_resolution": "generate_resolution",
            "create_ticket": "create_ticket",
        },
    )

    builder.add_conditional_edges(
        "generate_resolution",
        route_after_resolution,
        {
            "hitl_approval": "hitl_approval",
            "create_ticket": "create_ticket",
            END: END,
        },
    )

    # hitl_approval uses interrupt() internally — resume continues from here.
    # Both approved and declined paths go to END; the status field distinguishes.
    builder.add_conditional_edges(
        "hitl_approval",
        route_after_hitl,
        {END: END},
    )

    builder.add_edge("create_ticket", END)

    return builder.compile(checkpointer=_checkpointer)


incident_escalation_graph = build_incident_graph()


# ---------------------------------------------------------------------------
# Public run wrapper (single catch point for unplanned failures)
# ---------------------------------------------------------------------------

def run_incident_graph(graph_input, config):
    """All call sites (start, HITL approve, ticket resolve) go through this
    wrapper. An IncidentProcessingError from analyze_incident_and_policy is
    caught exactly once, here, and turned into a real Ticket row — never
    swallowed, never duplicated.

    Returns (state_dict, ticket_id_or_None).
    """
    from .tickets import create_ticket as _create_ticket

    thread_id = config.get("configurable", {}).get("thread_id")
    try:
        result = incident_escalation_graph.invoke(graph_input, config=config)
        return result, None
    except IncidentProcessingError as e:
        ticket_id = _create_ticket(
            thread_id=thread_id,
            graph_name=GRAPH_NAME,
            node_name="analyze_incident_and_policy",
            error_message=str(e),
        )
        # Best-effort snapshot; the Ticket row is the authoritative record.
        try:
            snapshot = incident_escalation_graph.get_state(config)
            state = dict(snapshot.values) if snapshot and snapshot.values else {}
        except Exception:
            state = {}
        state["status"] = "FAILED_TICKET"
        state["ticket_id"] = ticket_id
        state["error_message"] = str(e)
        return state, ticket_id
