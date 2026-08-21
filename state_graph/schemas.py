from typing import TypedDict, List, Dict, Any, Optional

class VIPBookingState(TypedDict):
    booking_id: str
    guest_id: str
    user_request: str
    
    # Task Decomposition node output
    subtasks: List[Dict[str, Any]]
    
    # ReAct execution output
    itinerary_items: List[Dict[str, Any]]
    total_price: float
    
    # Workflow & HITL flags
    requires_approval: bool
    is_approved: Optional[bool]
    
    # Failure & Ticket tracking
    status: str  # "IN_PROGRESS", "PAUSED_HITL", "FAILED_TICKET", "COMPLETED"
    error_message: Optional[str]
    ticket_id: Optional[str]  # set by create_ticket_node; foreign key into the Tickets table (state_graph/tickets.py)


class ComplaintResolutionState(TypedDict):
    """
    Graph #2 (Person B): Guest Complaint Escalation & Compensation Resolution.

    Why this needs a state graph (not a single-pass DAG):
    - Genuinely spans more than one sitting: after an offer is drafted, the
      graph pauses at `await_guest_response` for the guest's reply, which
      may arrive hours or days later, via staff recording it on the platform.
    - Real branch outside the model's control: the path taken (accept /
      reject / counter) is the guest's decision, not something the model
      chooses.
    - Real failure a retry can't fix: record_compensation() can reject a
      write (duplicate approval, invalid request id) -- retrying the exact
      same write just fails again; it needs a human to look at it.
    """
    request_id: Optional[int]
    reservation_id: int
    guest_id: int
    issue_type: str
    priority: str
    created_by: int  # staff_id who logged the complaint

    # Negotiation loop state
    round_number: int
    max_rounds: int
    proposed_type: Optional[str]
    proposed_amount: Optional[float]
    policy_context: Optional[str]        # RAG-retrieved compensation policy text grounding the offer
    candidate_offers: List[Dict[str, Any]]  # Tree-of-Thoughts candidates for this round

    # External wait -- set by staff via the platform when the guest replies
    guest_decision: Optional[str]  # "accept" | "reject" | "counter" | None (still waiting)

    # HITL
    requires_manager_approval: bool
    hitl_reason: Optional[str]  # "amount_threshold" | "rounds_exhausted" -- decides where control goes after approval
    is_approved: Optional[bool]

    # Terminal outcome
    compensation_id: Optional[int]
    status: str  # "DRAFTING", "PAUSED_HITL", "AWAITING_GUEST", "DECLINED", "FAILED_TICKET", "COMPLETED"
    error_message: Optional[str]
    ticket_id: Optional[str]


class IncidentEscalationState(TypedDict):
    """
    Graph #3 (Person C): Incident & Maintenance Escalation.

    Why this needs a state graph (not a single-pass DAG):
    - Requires multi-step policy checks: the graph must first retrieve and
      evaluate compensation and room-service policy context (RAG), then
      generate an actionable resolution plan — two distinct, ordered steps
      whose outputs feed each other and cannot be collapsed into one prompt.
    - Conditional human approval for high severity/cost: the path taken
      (dispatch technician immediately vs. escalate to manager) is determined
      by a policy evaluation the model performs at runtime, not something
      known at graph-construction time. Only a conditional edge on live state
      can express this correctly.
    - State persistence across worker failures: maintenance incidents can
      take hours to resolve. The graph must survive process termination and
      resume from the exact checkpointed node — a DAG that re-runs from
      scratch would re-create tickets and re-alert staff on every restart.
    """
    thread_id: str
    guest_id: str
    room_number: str
    issue_description: str

    # LLM Node 1 outputs
    severity_level: Optional[str]        # "low" | "medium" | "high"
    policy_analysis: Optional[str]       # RAG-grounded policy context + cost estimate
    requires_human_approval: bool        # True when severity is "high" or cost exceeds threshold

    # LLM Node 2 output
    resolution_plan: Optional[Dict[str, Any]]  # {"action", "estimated_cost", "dispatch", "notes"}

    # HITL
    human_approved: Optional[bool]       # Set by manager via platform after interrupt()

    # Ticket tracking (failure path only — not HITL)
    ticket_id: Optional[str]

    # Lifecycle
    status: str   # "IN_PROGRESS" | "PAUSED_HITL" | "APPROVED" | "DECLINED" | "FAILED_TICKET" | "COMPLETED"
    error_message: Optional[str]
