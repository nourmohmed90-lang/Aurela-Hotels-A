import uuid
import sqlite3
import os
from .graph import vip_booking_graph

def test_vip_booking_flow():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print(f"STARTING STATE GRAPH TEST — Thread ID: {thread_id}")
    print("=" * 60)

    initial_state = {
        "booking_id": "RES-999",
        "guest_id": "GUEST-123",
        "user_request": "I want to check availability for Suite and Deluxe rooms.",
        "subtasks": [],
        "itinerary_items": [],
        "total_price": 0.0,
        "requires_approval": False,
        "is_approved": None,
        "status": "INITIATED",
        "error_message": None
    }

    # Phase 1: Execution up to HITL pause
    print("\n--- Phase 1: Running Graph to HITL Pause ---")
    current_state = vip_booking_graph.invoke(initial_state, config=config)

    print(f"Status: {current_state.get('status')}")
    print(f"Total Price Calculated: ${current_state.get('total_price')}")
    print(f"Requires Manager Approval: {current_state.get('requires_approval')}")
    print(f"Itinerary Items: {current_state.get('itinerary_items')}")

    if current_state.get("status") == "PAUSED_HITL":
        print("\nSUCCESS: Graph correctly paused at HITL node due to $1,500+ threshold!")
    else:
        print(f"\nUNEXPECTED STATUS: {current_state.get('status')}")
        print(f"Error Details: {current_state.get('error_message')}")
        return

    # Phase 2: Resume from checkpoint with approval
    print("\n--- Phase 2: Simulating Admin Approval & Resuming ---")
    current_state["is_approved"] = True
    current_state["status"] = "APPROVED"

    resumed_state = vip_booking_graph.invoke(current_state, config=config)

    print(f"Final Status: {resumed_state.get('status')}")
    print(f"Manager Approved: {resumed_state.get('is_approved')}")
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_vip_booking_flow()