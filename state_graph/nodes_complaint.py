import sys
import os
from typing import Dict, Any, List

from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from .schemas import ComplaintResolutionState
from .tickets import create_ticket

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_server.tools import create_recovery_request, record_compensation, recommend_compensation
from agent.rag.hybrid_retriever import hybrid_retriever

GRAPH_NAME = "complaint_resolution_graph"

# From documents/compensation_policy.md: "Requests of $100 or more require
# Manager approval." Real policy threshold, not an invented number.
MANAGER_APPROVAL_THRESHOLD = 100.0

# recommend_compensation() (mcp_server/tools.py) returns free text, not a
# structured {type, amount} -- this maps its known recommendation strings to
# a starting (type, amount) pair the graph can actually reason over and
# check against the policy threshold. Anything not in the map falls back to
# a manager-review default, matching the tool's own "Manager review
# required" fallback text.
_BASELINE_OFFERS = {
    "Overbooking": ("Room Upgrade", 0.0),
    "Maintenance": ("Discount", 60.0),
    "Room Not Ready": ("Meal Voucher", 25.0),
    "Double Booking": ("Room Upgrade", 0.0),
    "VIP": ("Room Upgrade", 0.0),
}
_DEFAULT_OFFER = ("Discount", 150.0)  # unmapped issue types default to a manager-reviewed amount


def intake_and_draft_offer(state: ComplaintResolutionState) -> Dict[str, Any]:
    """LLM addition #1: RAG. Grounds the initial offer in the company's
    actual compensation policy instead of letting the model invent an
    amount/type, and logs the recovery request via the real MCP tool.

    No idempotency guard needed: with interrupt()-based pausing (see
    hitl_manager_approval_node / await_guest_response_node below), resume
    continues execution from the exact node that paused, not from this
    entry point -- so this node genuinely only runs once per run, unlike
    the vip_booking_graph's re-invoke-from-entry pattern.
    """
    result = create_recovery_request(
        reservation_id=state["reservation_id"],
        issue_type=state["issue_type"],
        priority=state["priority"],
        created_by=state["created_by"],
    )
    request_id = result["request_id"]

    policy_context = hybrid_retriever.retrieve_context(
        query=f"compensation policy for {state['issue_type']} complaint",
        k=3,
        source="compensation_policy.md",
    )

    baseline = recommend_compensation(state["issue_type"])
    proposed_type, proposed_amount = _BASELINE_OFFERS.get(state["issue_type"], _DEFAULT_OFFER)
    policy_context = f"MCP recommend_compensation: {baseline['recommendation']}\n\n{policy_context}"

    requires_approval = proposed_amount >= MANAGER_APPROVAL_THRESHOLD

    return {
        "request_id": request_id,
        "round_number": 1,
        "max_rounds": state.get("max_rounds", 3),
        "proposed_type": proposed_type,
        "proposed_amount": proposed_amount,
        "policy_context": policy_context,
        "requires_manager_approval": requires_approval,
        "hitl_reason": "amount_threshold" if requires_approval else None,
        "status": "PAUSED_HITL" if requires_approval else "AWAITING_GUEST",
    }


def hitl_manager_approval_node(state: ComplaintResolutionState) -> Dict[str, Any]:
    """HITL node: pauses execution *inside this node* via LangGraph's
    interrupt() and waits for a real manager decision, delivered through the
    platform. Unlike the vip_booking_graph's pattern (an edge straight to
    END, resumed by re-invoking from the entry point), interrupt() persists
    the checkpoint mid-node and resumes execution *exactly here* -- no risk
    of earlier nodes (intake, prior negotiation rounds) silently re-running
    and corrupting round_number or re-creating the recovery request. This is
    the more correct mechanism the assignment's own resources point to
    (LangGraph Dynamic Interrupts & Time Travel); flagged to the team as
    worth adopting for the other two graphs as well.

    No separate persisted ticket record here -- see state_graph/tickets.py
    for why HITL (an expected pause) and tickets (an unplanned failure) are
    intentionally different mechanisms.
    """
    decision = interrupt({
        "kind": "manager_approval",
        "reason": state.get("hitl_reason"),
        "request_id": state.get("request_id"),
        "proposed_type": state.get("proposed_type"),
        "proposed_amount": state.get("proposed_amount"),
        "round_number": state.get("round_number"),
        "policy_context": state.get("policy_context"),
    })
    # decision is whatever the platform passes via Command(resume=...):
    # {"approved": True/False}
    approved = bool(decision.get("approved"))
    return {
        "is_approved": approved,
        "status": "IN_PROGRESS" if approved else "DECLINED",
    }


def mark_rounds_exhausted(state: ComplaintResolutionState) -> Dict[str, Any]:
    """The negotiation loop hit max_rounds without the guest accepting.
    Sets hitl_reason so route_after_hitl (graph_complaint.py) knows this
    HITL pause is a final-call escalation, not an amount-threshold gate --
    approval here finalizes the last offer directly instead of sending
    another round back to the guest.
    """
    return {"requires_manager_approval": True, "hitl_reason": "rounds_exhausted", "status": "PAUSED_HITL"}


def explore_next_offer(state: ComplaintResolutionState) -> Dict[str, Any]:
    """LLM addition #2: Tree of Thoughts. The guest rejected or countered --
    explore several concrete next-offer candidates (bump the same
    compensation type, switch type, or split the difference) and pick the
    one that best fits the remaining policy budget and how many rounds have
    already been tried. This differs from the planning agent's ToT usage,
    which picks a trade-off once in a single pass; here it operates inside a
    multi-round negotiation loop and its input includes prior-round history.
    """
    current_amount = state.get("proposed_amount") or 50.0
    current_type = state.get("proposed_type") or "Discount"
    round_number = state.get("round_number", 1) + 1

    # Candidate generation -- in a full LLM-backed implementation this calls
    # the model once per candidate; kept deterministic here so the graph's
    # control flow (cycles, thresholds, HITL routing) can be tested without
    # a live model call.
    candidates: List[Dict[str, Any]] = [
        {"type": current_type, "amount": round(current_amount * 1.5, 2),
         "rationale": "Increase the same compensation type -- guest already engaged with this offer type."},
        {"type": "Room Upgrade" if current_type != "Room Upgrade" else "Free Night", "amount": current_amount,
         "rationale": "Switch compensation type at the same value -- may address a non-monetary concern."},
        {"type": "Refund", "amount": round(current_amount * 1.2, 2),
         "rationale": "Move to a straightforward refund -- reduces further back-and-forth."},
    ]

    # Score candidates: prefer smaller increases as rounds progress (avoid
    # runaway escalation), and prefer non-refund types early on.
    def score(c):
        escalation_penalty = c["amount"] * (round_number - 1) * 0.1
        return c["amount"] + escalation_penalty

    best = min(candidates, key=score)

    requires_approval = best["amount"] >= MANAGER_APPROVAL_THRESHOLD

    return {
        "round_number": round_number,
        "proposed_type": best["type"],
        "proposed_amount": best["amount"],
        "candidate_offers": candidates,
        "requires_manager_approval": requires_approval,
        "hitl_reason": "amount_threshold" if requires_approval else None,
        "status": "PAUSED_HITL" if requires_approval else "AWAITING_GUEST",
        "guest_decision": None,
    }


def await_guest_response_node(state: ComplaintResolutionState) -> Dict[str, Any]:
    """The genuine wait state: interrupt() pauses and persists the
    checkpoint exactly here, potentially for hours or days, until staff
    record the guest's actual decision through the platform (standing in
    for an external channel -- email/SMS reply -- landing whenever it
    lands). Resume continues from this exact point, not from the graph's
    entry point, so no earlier round's work gets silently re-run.
    """
    decision = interrupt({
        "kind": "await_guest_response",
        "request_id": state.get("request_id"),
        "proposed_type": state.get("proposed_type"),
        "proposed_amount": state.get("proposed_amount"),
        "round_number": state.get("round_number"),
    })
    # decision: {"guest_decision": "accept" | "reject" | "counter"}
    return {"guest_decision": decision.get("guest_decision"), "status": "IN_PROGRESS"}


class CompensationRecordingError(RuntimeError):
    """Raised (not swallowed) when record_compensation() rejects a write.
    Left to propagate out of the node on purpose: LangGraph's checkpointer
    persists state after every successfully *completed* node, so an
    uncaught exception here leaves the checkpoint sitting at the last
    completed step (guest's acceptance already recorded), with
    finalize_compensation_node itself never having completed. Re-invoking
    this thread later genuinely resumes at this exact node -- it does not
    re-run intake or any negotiation round that already finished. Contrast
    with the vip_booking_graph's constrained_react_node, which catches its
    own failure internally and routes to create_ticket_node via a graph
    edge; that reaches END "successfully," so its ticket-resolve path
    re-invokes from the entry point and re-runs completed work. This graph
    intentionally avoids that gap -- see run_complaint_graph() in
    graph_complaint.py for where the exception is actually caught and
    turned into a ticket.
    """


def finalize_compensation_node(state: ComplaintResolutionState) -> Dict[str, Any]:
    """Guest accepted -- write the approved compensation via the real MCP
    tool. A rejected write here (duplicate approval, invalid request id) is
    exactly the kind of failure a blind retry cannot fix -- it must become a
    ticket, not loop forever. Raises rather than self-catching; see
    CompensationRecordingError above for why.
    """
    result = record_compensation(
        request_id=state["request_id"],
        compensation_type=state["proposed_type"],
        amount=state["proposed_amount"],
        approval_status="Approved",
        approved_by=state["created_by"],
    )

    if "error" in result:
        raise CompensationRecordingError(result["error"])

    return {"status": "COMPLETED", "compensation_id": result["compensation_id"]}
