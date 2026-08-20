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