import os
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .schemas import ComplaintResolutionState
from .nodes_complaint import (
    intake_and_draft_offer,
    hitl_manager_approval_node,
    mark_rounds_exhausted,
    explore_next_offer,
    await_guest_response_node,
    finalize_compensation_node,
)

# Same hotel.db as the VIP booking graph and the ticket system -- this graph
# is a new agent scope sitting next to the existing ones, not a parallel
# database (per the assignment's "don't rebuild your existing database").
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "hotel.db"))
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


def route_after_intake(state: ComplaintResolutionState) -> str:
    return "hitl_manager_approval" if state.get("requires_manager_approval") else "await_guest_response"


def route_after_guest_response(state: ComplaintResolutionState) -> str:
    """The real cycle: reject/counter routes back into another offer round
    instead of ending the run, which a DAG-shaped planner cannot express."""
    decision = state.get("guest_decision")
    round_number = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", 3)

    if decision == "accept":
        return "finalize_compensation"
    if decision in ("reject", "counter") and round_number < max_rounds:
        return "explore_next_offer"
    # Rejected/countered but negotiation rounds are exhausted -- escalate to
    # a human for the final call, rather than looping forever.
    return "mark_rounds_exhausted"


def route_after_offer_round(state: ComplaintResolutionState) -> str:
    return "hitl_manager_approval" if state.get("requires_manager_approval") else "await_guest_response"


def route_after_hitl(state: ComplaintResolutionState) -> str:
    """Where control goes once a manager has acted on a HITL pause depends
    on *why* the pause happened (state["hitl_reason"], set by whichever node
    routed into this one) -- an amount-threshold gate resumes toward the
    guest; a rounds-exhausted escalation finalizes or ends the run directly.
    """
    approved = state.get("is_approved")
    reason = state.get("hitl_reason")

    if reason == "rounds_exhausted":
        return "finalize_compensation" if approved else END
    # reason == "amount_threshold"
    return "await_guest_response" if approved else END


def build_complaint_resolution_graph():
    builder = StateGraph(ComplaintResolutionState)

    builder.add_node("intake_and_draft_offer", intake_and_draft_offer)
    builder.add_node("hitl_manager_approval", hitl_manager_approval_node)
    builder.add_node("mark_rounds_exhausted", mark_rounds_exhausted)
    builder.add_node("explore_next_offer", explore_next_offer)
    builder.add_node("await_guest_response", await_guest_response_node)
    builder.add_node("finalize_compensation", finalize_compensation_node)

    builder.set_entry_point("intake_and_draft_offer")

    builder.add_conditional_edges(
        "intake_and_draft_offer",
        route_after_intake,
        {"hitl_manager_approval": "hitl_manager_approval", "await_guest_response": "await_guest_response"},
    )

    # await_guest_response and hitl_manager_approval both pause *inside the
    # node itself* via interrupt() (see nodes_complaint.py) -- resume
    # continues execution right after the interrupt() call, not from
    # set_entry_point(), so earlier rounds/state are never silently re-run.
    builder.add_conditional_edges(
        "await_guest_response",
        route_after_guest_response,
        {
            "finalize_compensation": "finalize_compensation",
            "explore_next_offer": "explore_next_offer",
            "mark_rounds_exhausted": "mark_rounds_exhausted",
        },
    )

    builder.add_edge("mark_rounds_exhausted", "hitl_manager_approval")

    builder.add_conditional_edges(
        "explore_next_offer",
        route_after_offer_round,
        {"hitl_manager_approval": "hitl_manager_approval", "await_guest_response": "await_guest_response"},
    )

    builder.add_conditional_edges(
        "hitl_manager_approval",
        route_after_hitl,
        {"await_guest_response": "await_guest_response", "finalize_compensation": "finalize_compensation", END: END},
    )

    # finalize_compensation has NO edge to a ticket node. On success it ends
    # normally. On failure it *raises* (see CompensationRecordingError in
    # nodes_complaint.py) rather than self-catching -- so LangGraph never
    # marks this node "completed," and the checkpoint sits at the last
    # successful step. run_complaint_graph() below is where that exception
    # is actually caught and turned into a ticket, keeping resume genuinely
    # resumable from the exact failed step instead of the graph's entry point.
    builder.add_edge("finalize_compensation", END)

    return builder.compile(checkpointer=_checkpointer)


complaint_resolution_graph = build_complaint_resolution_graph()


def run_complaint_graph(graph_input, config):
    """Every call site (start, HITL approve, guest-response, ticket resolve)
    should go through this wrapper instead of calling
    complaint_resolution_graph.invoke() directly, so a genuine failure
    (CompensationRecordingError) is caught exactly once, in exactly one
    place, and turned into a real ticket -- never silently swallowed, never
    duplicated into two different failure-handling code paths.

    Returns (state_dict, ticket_id_or_None).
    """
    from .nodes_complaint import CompensationRecordingError, GRAPH_NAME
    from .tickets import create_ticket

    thread_id = config.get("configurable", {}).get("thread_id")
    try:
        result = complaint_resolution_graph.invoke(graph_input, config=config)
        return result, None
    except CompensationRecordingError as e:
        ticket_id = create_ticket(
            thread_id=thread_id,
            graph_name=GRAPH_NAME,
            node_name="finalize_compensation_node",
            error_message=str(e),
        )
        # Best-effort state snapshot for the caller; the ticket row is the
        # authoritative record of the failure, not this dict.
        snapshot = complaint_resolution_graph.get_state(config)
        state = dict(snapshot.values) if snapshot and snapshot.values else {}
        state["status"] = "FAILED_TICKET"
        state["ticket_id"] = ticket_id
        state["error_message"] = str(e)
        return state, ticket_id
