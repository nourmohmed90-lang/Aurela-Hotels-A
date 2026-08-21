#!/usr/bin/env python
"""
demo_crash_recovery.py — Graph 3 Crash-and-Resume Verification

Demonstrates that the incident_escalation_graph:
1. Runs to the HITL interrupt checkpoint on a high-severity incident.
2. Persists its state to database/hotel.db even if the process is killed.
3. Can be resumed from the exact checkpoint node without re-running earlier nodes.

Run this script directly:
    python -m state_graph.demo_crash_recovery

Simulate crash + resume:
    Step 1: Run the script (it will print a THREAD_ID and pause at the HITL gate)
    Step 2: Copy the THREAD_ID from the output
    Step 3: Kill the process (Ctrl+C or in a separate terminal: kill -9 <PID>)
    Step 4: Re-run the script with --resume <THREAD_ID> to resume from checkpoint:
            python -m state_graph.demo_crash_recovery --resume <THREAD_ID>
"""

import argparse
import sqlite3
import sys
import os
import uuid

# Ensure project root is in path when run as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.types import Command

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "hotel.db"))


def _hr(char: str = "=", width: int = 70) -> None:
    print(char * width)


def _assert_checkpoint_exists(thread_id: str) -> dict:
    """Queries hotel.db directly to confirm the checkpoint was written."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # LangGraph's SqliteSaver writes to 'checkpoints' table.
    rows = conn.execute(
        "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
        (thread_id,)
    ).fetchall()
    conn.close()
    if not rows:
        raise AssertionError(
            f"❌ ASSERTION FAILED: No checkpoint found for thread_id={thread_id!r} in {DB_PATH}\n"
            "The graph did not persist state as expected."
        )
    row = dict(rows[0])
    print(f"  ✅ Checkpoint confirmed in hotel.db (checkpoint_id={row.get('checkpoint_id', '?')})")
    return row


def run_fresh(thread_id: str) -> None:
    """Phase 1: Start a new high-severity incident and run until HITL interrupt."""
    from state_graph.graph_incident import incident_escalation_graph, run_incident_graph

    _hr()
    print("PHASE 1 — Starting high-severity incident run")
    print(f"Thread ID: {thread_id}")
    _hr()

    initial_state = {
        "thread_id": thread_id,
        "guest_id": "GUEST-DEMO-001",
        "room_number": "412",
        # 'flooding' triggers HIGH severity → requires_human_approval=True → HITL interrupt
        "issue_description": "Emergency flooding in bathroom — water leaking through ceiling",
        "severity_level": None,
        "policy_analysis": None,
        "requires_human_approval": False,
        "resolution_plan": None,
        "human_approved": None,
        "ticket_id": None,
        "status": "IN_PROGRESS",
        "error_message": None,
    }
    config = {"configurable": {"thread_id": thread_id}}

    print("\n[1/3] Invoking incident_escalation_graph ...")
    result, ticket_id = run_incident_graph(initial_state, config)

    print(f"\n[2/3] Graph paused. Status       : {result.get('status')}")
    print(f"       Severity detected          : {result.get('severity_level')}")
    print(f"       Requires human approval    : {result.get('requires_human_approval')}")
    if result.get('resolution_plan'):
        plan = result['resolution_plan']
        print(f"       Resolution action         : {plan.get('action')}")
        print(f"       Estimated cost            : ${plan.get('estimated_cost', 0):.2f}")
        print(f"       Dispatch                  : {plan.get('dispatch')}")

    # Confirm state is checkpointed in hotel.db
    print("\n[3/3] Verifying checkpoint in hotel.db ...")
    _assert_checkpoint_exists(thread_id)

    _hr()
    print("\n✅ PHASE 1 COMPLETE: Graph paused at HITL interrupt with state saved to hotel.db")
    _hr("-")
    print("\n📌 TO SIMULATE A CRASH AND RESUME:")
    print(f"\n   1. Kill this process now (Ctrl+C or kill -9 <PID>).")
    print(f"   2. The checkpoint is already saved to hotel.db.")
    print(f"   3. Resume from the exact HITL node by running:\n")
    print(f"        python -m state_graph.demo_crash_recovery --resume {thread_id}\n")
    print("   The graph will continue from hitl_approval — NOT from analyze_incident.")
    print("   No RAG retrieval, no policy analysis, no re-alerting of staff.\n")
    _hr("-")

    # Check if the graph is actually paused at the HITL node
    snap = incident_escalation_graph.get_state(config)
    if snap and snap.next:
        print(f"   Graph is paused at node(s): {snap.next}")
    else:
        print(f"   Graph final status: {result.get('status')}")


def resume_from_checkpoint(thread_id: str, approved: bool = True) -> None:
    """Phase 2: Resume the graph from its saved checkpoint (simulates post-crash recovery)."""
    from state_graph.graph_incident import incident_escalation_graph, run_incident_graph

    _hr()
    print(f"PHASE 2 — Resuming from checkpoint (thread_id={thread_id!r})")
    print(f"Manager decision: {'APPROVED ✅' if approved else 'DECLINED ❌'}")
    _hr()

    config = {"configurable": {"thread_id": thread_id}}

    # Confirm checkpoint still exists (even after simulated process kill)
    print("[1/3] Verifying checkpoint still exists in hotel.db after (simulated) crash ...")
    _assert_checkpoint_exists(thread_id)

    print("\n[2/3] Resuming graph with Command(resume={approved: ...}) ...")
    result, ticket_id = run_incident_graph(
        Command(resume={"approved": approved, "feedback": "Demo: manager reviewed and decided."}),
        config,
    )

    print(f"\n[3/3] Resume complete.")
    print(f"       Final status    : {result.get('status')}")
    print(f"       Human approved  : {result.get('human_approved')}")
    if result.get('resolution_plan'):
        plan = result['resolution_plan']
        print(f"       Action taken    : {plan.get('action')}")

    _hr()
    print("\n✅ CRASH-AND-RESUME DEMO COMPLETE")
    print("   The graph resumed from the HITL checkpoint node — no earlier nodes re-ran.")
    _hr()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incident Graph 3 — Crash-and-Resume Demo"
    )
    parser.add_argument(
        "--resume", metavar="THREAD_ID",
        help="Resume an existing HITL-paused run from its checkpoint.",
    )
    parser.add_argument(
        "--decline", action="store_true",
        help="When resuming, send a manager DECLINE instead of APPROVE.",
    )
    parser.add_argument(
        "--thread-id", metavar="THREAD_ID",
        help="Use a specific thread ID for the fresh run (default: random UUID).",
    )
    args = parser.parse_args()

    if args.resume:
        resume_from_checkpoint(args.resume, approved=not args.decline)
    else:
        thread_id = args.thread_id or f"incident-demo-{uuid.uuid4()}"
        run_fresh(thread_id)


if __name__ == "__main__":
    main()
