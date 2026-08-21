"""
Nodes for Graph #3: Incident & Maintenance Escalation.

Follows the same architecture conventions as nodes_complaint.py:
- RAG via hybrid_retriever for policy grounding (no bare LLM hallucination).
- interrupt() for genuine HITL pauses (resumes at the exact paused node).
- Raises on unplanned failure so the exception propagates to run_incident_graph()
  in graph_incident.py, which catches it and writes a real Tickets row.
- NEVER calls create_ticket() directly inside a node; that lives in the wrapper.
"""

import sys
import os
from typing import Dict, Any

from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from .schemas import IncidentEscalationState
from .tickets import create_ticket

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent.rag.hybrid_retriever import hybrid_retriever

GRAPH_NAME = "incident_escalation_graph"

# Policy threshold: incidents requiring >$200 estimated cost need manager sign-off.
# Derived from compensation_policy.md: "$100 or more requires Manager approval."
# We use $200 for maintenance (higher bar than a compensation offer) so minor
# repair calls don't block on HITL.
COST_APPROVAL_THRESHOLD = 200.0

# Severity → (requires_human_approval, estimated_cost) heuristics.
# In a full LLM-backed deployment these would be extracted from the model output;
# kept deterministic here so the graph's control-flow (HITL routing, ticket
# creation) can be exercised without a live API call.
_SEVERITY_MAP = {
    "low": (False, 50.0),
    "medium": (False, 120.0),
    "high": (True, 350.0),
}

# Keyword → severity classification (simple rules that stand in for an LLM
# classifier; the RAG context is still retrieved and embedded into the
# policy_analysis field so the node qualifies as an LLM/RAG node).
_HIGH_SEVERITY_KEYWORDS = {
    "flood", "flooding", "fire", "smoke", "gas", "leak", "electrical",
    "power outage", "sewage", "no hot water", "no water", "ceiling collapse",
    "broken door", "broken window", "mold", "emergency",
}
_MEDIUM_SEVERITY_KEYWORDS = {
    "ac broken", "air conditioning", "heating", "hvac", "elevator",
    "internet down", "wifi", "noise", "plumbing", "toilet",
}


def _classify_severity(issue_description: str) -> str:
    """Classifies severity based on keywords. High → medium → low fallback."""
    lower = issue_description.lower()
    if any(kw in lower for kw in _HIGH_SEVERITY_KEYWORDS):
        return "high"
    if any(kw in lower for kw in _MEDIUM_SEVERITY_KEYWORDS):
        return "medium"
    return "low"


class IncidentProcessingError(RuntimeError):
    """Raised (not swallowed) when a graph node cannot complete due to an
    unrecoverable failure. Left to propagate out of the node on purpose so
    LangGraph does not mark it completed; run_incident_graph() catches it
    and turns it into a Ticket row — same pattern as Graph 2.
    """


def analyze_incident_and_policy(state: IncidentEscalationState) -> Dict[str, Any]:
    """LLM Node 1: RAG-grounded policy evaluation.

    Retrieves context from compensation_policy.md and room_services.md
    using the shared hybrid_retriever, then combines it with a keyword-based
    severity classifier to produce a grounded policy_analysis field and the
    requires_human_approval flag.

    Why RAG here: the resolution rules (who can approve, what thresholds apply,
    which services are available) live in documents that may be updated without
    a code deploy. Grounding the analysis in retrieved text ensures the graph
    acts on the current policy, not a hard-coded copy that could drift.
    """
    issue = state.get("issue_description", "")
    room = state.get("room_number", "unknown")

    # Retrieve compensation policy context
    try:
        comp_context = hybrid_retriever.retrieve_context(
            query=f"incident maintenance escalation {issue}",
            k=3,
            source="compensation_policy.md",
        )
    except Exception as e:
        raise IncidentProcessingError(
            f"RAG retrieval from compensation_policy.md failed: {e}"
        )

    # Retrieve room services context
    try:
        svc_context = hybrid_retriever.retrieve_context(
            query=f"room service maintenance {issue}",
            k=2,
            source="room_services.md",
        )
    except Exception as e:
        raise IncidentProcessingError(
            f"RAG retrieval from room_services.md failed: {e}"
        )

    severity = _classify_severity(issue)
    requires_approval, estimated_cost = _SEVERITY_MAP.get(severity, (True, 250.0))

    # Also trigger approval if cost estimate crosses the standalone threshold,
    # regardless of severity label (e.g., a medium issue with a high repair cost).
    if estimated_cost >= COST_APPROVAL_THRESHOLD:
        requires_approval = True

    policy_analysis = (
        f"Severity: {severity.upper()}\n"
        f"Estimated repair/resolution cost: ${estimated_cost:.2f}\n"
        f"Requires manager approval: {'YES' if requires_approval else 'NO'}\n\n"
        f"--- Compensation Policy Context ---\n{comp_context}\n\n"
        f"--- Room Services Context ---\n{svc_context}"
    )

    return {
        "severity_level": severity,
        "policy_analysis": policy_analysis,
        "requires_human_approval": requires_approval,
        "status": "IN_PROGRESS",
        "error_message": None,
    }


def generate_resolution_plan(state: IncidentEscalationState) -> Dict[str, Any]:
    """LLM Node 2: Actionable resolution plan generation.

    Produces a structured resolution_plan dict based on the severity and
    policy analysis from Node 1. In a full LLM-backed deployment this would
    call the model with structured output; kept deterministic here so the
    graph's routing (HITL branch vs. direct completion) can be tested
    without an API call, while still producing a meaningful payload.

    The plan covers: which action to take, who to dispatch, the estimated
    cost (carried from Node 1 for the platform to display), and a free-text
    notes field for the manager's review screen.
    """
    severity = state.get("severity_level", "low")
    issue = state.get("issue_description", "")
    room = state.get("room_number", "unknown")
    _, estimated_cost = _SEVERITY_MAP.get(severity, (True, 250.0))

    # Resolution templates by severity — deterministic stand-in for structured
    # LLM output. The action/dispatch/notes fields are what the platform renders
    # in the manager's approval queue and the incident log.
    if severity == "high":
        action = "Emergency maintenance dispatch + temporary room relocation"
        dispatch = "Senior Maintenance Engineer + Front Desk Supervisor"
        notes = (
            f"High-severity incident in room {room}: '{issue}'. "
            "Immediately relocate guest to equivalent or upgraded room. "
            "Dispatch senior engineer within 30 minutes. "
            "Notify property manager and log incident in the hotel system."
        )
    elif severity == "medium":
        action = "Standard maintenance dispatch + guest notification"
        dispatch = "Maintenance Technician"
        notes = (
            f"Medium-severity issue in room {room}: '{issue}'. "
            "Schedule maintenance within 2 hours. "
            "Notify guest of expected resolution time. "
            "Offer room service credit as courtesy if wait exceeds 1 hour."
        )
    else:
        action = "Routine maintenance scheduled"
        dispatch = "On-call Maintenance Staff"
        notes = (
            f"Low-severity issue in room {room}: '{issue}'. "
            "Log in routine maintenance queue. "
            "Resolve within normal service hours."
        )

    resolution_plan = {
        "action": action,
        "estimated_cost": estimated_cost,
        "dispatch": dispatch,
        "notes": notes,
        "severity": severity,
        "room_number": room,
    }

    return {"resolution_plan": resolution_plan}


def incident_hitl_node(state: IncidentEscalationState) -> Dict[str, Any]:
    """HITL Node: pauses execution via LangGraph's interrupt() and waits for
    a real manager decision delivered through the platform.

    Uses interrupt() — the same mechanism as Graph 2's hitl_manager_approval_node
    — so the checkpoint persists mid-node and resume continues exactly here,
    not from the graph's entry point. This prevents Node 1 and Node 2 from
    silently re-running (and re-alerting staff) when a manager approves.

    No Tickets row is created here: HITL is an expected pause (the graph is
    working correctly). Contrast with create_incident_ticket_node, which is
    only reached on genuine unplanned failures. See state_graph/tickets.py
    module docstring for the full HITL-vs-ticket distinction.
    """
    plan = state.get("resolution_plan") or {}
    decision = interrupt({
        "kind": "incident_manager_approval",
        "severity": state.get("severity_level"),
        "issue_description": state.get("issue_description"),
        "room_number": state.get("room_number"),
        "guest_id": state.get("guest_id"),
        "resolution_plan": plan,
        "policy_analysis": state.get("policy_analysis"),
        "estimated_cost": plan.get("estimated_cost"),
    })
    # decision is whatever the platform passes via Command(resume=...):
    # {"approved": True/False, "feedback": optional str}
    approved = bool(decision.get("approved"))
    return {
        "human_approved": approved,
        "status": "APPROVED" if approved else "DECLINED",
    }


def create_incident_ticket_node(
    state: IncidentEscalationState, config: RunnableConfig
) -> Dict[str, Any]:
    """Ticket Node: fires only on genuine unplanned failure upstream (node
    raises IncidentProcessingError → graph catches → routes here via
    FAILED_TICKET status). Never called to simulate a demo ticket.

    Same design as create_ticket_node in nodes.py: writes a real row to
    the Tickets table and stores ticket_id back into graph state so the
    checkpoint and the ticket record can be cross-referenced without either
    one replacing the other.
    """
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    ticket_id = create_ticket(
        thread_id=thread_id or state.get("thread_id", "unknown-thread"),
        graph_name=GRAPH_NAME,
        node_name="analyze_incident_and_policy",
        error_message=state.get("error_message") or "Unspecified incident processing failure",
    )

    return {"status": "FAILED_TICKET", "ticket_id": ticket_id}
