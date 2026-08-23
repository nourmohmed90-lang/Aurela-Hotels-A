import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from .schemas import VIPBookingState
from .tickets import create_ticket

# Connect to existing mcp_server tools directly
BASE_DIR = Path(__file__).resolve().parent.parent
AGENT_ENV_PATH = BASE_DIR / "agent" / ".env"
load_dotenv(dotenv_path=AGENT_ENV_PATH)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from planning.algorithms.decomposition import decompose_goal
from mcp_server.tools import search_available_rooms, analyze_reservation
from langchain_google_genai import ChatGoogleGenerativeAI

GRAPH_NAME = "vip_booking_graph"

# ---------------------------------------------------------------------------
# Gemini LLM Initialization
# ---------------------------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(f"GEMINI_API_KEY not found in {AGENT_ENV_PATH}")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=api_key,
    temperature=0
)

# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def task_decomposition_node(state: VIPBookingState) -> Dict[str, Any]:
    """Uses planning/algorithms/decomposition.py to dynamically create task subtasks with Gemini."""
    user_request = state.get("user_request", "")
    print(f"[DECOMPOSE NODE] Running Gemini decomposition for: '{user_request}'")

    try:
        plan = decompose_goal(goal=user_request, llm=llm)

        formatted_subtasks = []
        for task in plan.tasks:
            instruction_lower = task.instruction.lower()
            room_type = "Suite" if "suite" in instruction_lower else ("Deluxe" if "deluxe" in instruction_lower else "ALL")
            
            formatted_subtasks.append({
                "id": task.id,
                "instruction": task.instruction,
                "action": "search_available_rooms",
                "room_type": room_type,
                "depends_on": task.depends_on
            })

        print(f"[DECOMPOSE NODE] Gemini generated {len(formatted_subtasks)} subtasks successfully.")
        return {"subtasks": formatted_subtasks, "status": "DECOMPOSED"}

    except Exception as e:
        print(f"[DECOMPOSE ERROR] Failed to decompose: {e}")
        return {"status": "FAILED_TICKET", "error_message": str(e)}


def constrained_react_node(state: VIPBookingState) -> Dict[str, Any]:
    """Executes tool calls directly against mcp_server/tools.py."""
    try:
        calculated_total = 0.0
        items = []

        for task in state.get("subtasks", []):
            if task.get("action") == "search_available_rooms":
                target_room = task.get("room_type", "ALL")
                result = search_available_rooms(target_room)
                
                # Fallback to searching all rooms if specified type is unavailable
                if not result.get("success") or not result.get("rooms"):
                    result = search_available_rooms("ALL")

                if result.get("success") and result.get("rooms"):
                    for room in result["rooms"][:2]:
                        items.append({
                            "hotel": room["hotel"], 
                            "type": room["type"], 
                            "price": room["price"]
                        })
                        calculated_total += room["price"]

        if calculated_total == 0.0:
            raise Exception("No available rooms found across any category in hotel.db.")

        requires_approval = calculated_total > 1500.0 or len(items) > 0

        return {
            "itinerary_items": items,
            "total_price": calculated_total,
            "requires_approval": requires_approval
        }

    except Exception as e:
        return {
            "status": "FAILED_TICKET",
            "error_message": str(e)
        }


def hitl_pause_node(state: VIPBookingState) -> Dict[str, Any]:
    """HITL Node: pauses execution state for manager approval."""
    return {"status": "PAUSED_HITL"}


def create_ticket_node(state: VIPBookingState, config: RunnableConfig) -> Dict[str, Any]:
    """Ticket Node: fires only on a genuine unplanned failure upstream."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    ticket_id = create_ticket(
        thread_id=thread_id or state.get("booking_id", "unknown-thread"),
        graph_name=GRAPH_NAME,
        node_name="constrained_react_node",
        error_message=state.get("error_message") or "Unspecified failure",
    )

    return {"status": "FAILED_TICKET", "ticket_id": ticket_id}


def shared_hitl_approval_node(state: dict) -> dict:
    """Generic interrupt()-based HITL node reusable by any state graph."""
    _FORWARD_KEYS = {
        "kind", "severity_level", "severity", "issue_description", "room_number",
        "guest_id", "booking_id", "reservation_id", "issue_type",
        "proposed_type", "proposed_amount", "total_price", "itinerary_items",
        "resolution_plan", "policy_analysis", "policy_context",
        "round_number", "max_rounds", "hitl_reason", "request_id",
        "estimated_cost",
    }
    payload = {k: v for k, v in state.items() if k in _FORWARD_KEYS and v is not None}
    if "kind" not in payload:
        payload["kind"] = "generic_hitl_approval"

    decision = interrupt(payload)

    approved = bool(decision.get("approved"))
    feedback = decision.get("feedback") or ""

    return {
        "is_approved": approved,
        "human_approved": approved,
        "status": "APPROVED" if approved else "DECLINED",
        "hitl_feedback": feedback,
    }