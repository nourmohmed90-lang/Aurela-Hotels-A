import os
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .schemas import VIPBookingState
from .nodes import (
    task_decomposition_node,
    constrained_react_node,
    hitl_pause_node,
    create_ticket_node
)

# Routing Logic
def route_after_react(state: VIPBookingState) -> str:
    if state.get("status") == "FAILED_TICKET":
        return "create_ticket"
    if state.get("requires_approval") and not state.get("is_approved"):
        return "hitl_pause"
    return "finalize_booking"

# Graph Construction
builder = StateGraph(VIPBookingState)

builder.add_node("decompose", task_decomposition_node)
builder.add_node("execute_react", constrained_react_node)
builder.add_node("hitl_pause", hitl_pause_node)
builder.add_node("create_ticket", create_ticket_node)

builder.set_entry_point("decompose")
builder.add_edge("decompose", "execute_react")

builder.add_conditional_edges(
    "execute_react",
    route_after_react,
    {
        "create_ticket": "create_ticket",
        "hitl_pause": "hitl_pause",
        "finalize_booking": END
    }
)

builder.add_edge("hitl_pause", END)
builder.add_edge("create_ticket", END)

# Durable Checkpointing pointing directly to existing database/hotel.db
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "hotel.db"))
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

vip_booking_graph = builder.compile(checkpointer=memory)