"""
Decomposition Divergence Demo
==============================
Feeds the SAME goal through BOTH planning methods and captures where they diverge.

Goal: "Guest delayed 5 hours due to flight, adjust check-in, spa, and conference schedule."

Run with:
    python planning_eval/test_divergence.py
Or as a pytest:
    pytest planning_eval/test_divergence.py -v -s
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

from planning.algorithms.decomposition import decompose_goal
from planning.algorithms.dynamic_decomposition import dynamic_decomposition, DynamicDecision
from planning.models import Plan

# ──────────────────────────────────────────────
# Shared goal used for both methods
# ──────────────────────────────────────────────
SHARED_GOAL = (
    "Guest delayed 5 hours due to flight, "
    "adjust check-in, spa, and conference schedule."
)

# ──────────────────────────────────────────────
# Mock LLM that returns deterministic sub-tasks
# so we can see the structural difference between
# Decomposition-first and Dynamic without LLM calls.
# ──────────────────────────────────────────────
class DivergenceMockLLM:
    """
    Decomposition-first path: Returns a full upfront DAG plan.
    Dynamic path:             Returns one step at a time (reactive).
    """
    def __init__(self):
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        prompt = messages[-1][1] if isinstance(messages[-1], tuple) else str(messages[-1])
        # Execution calls for decomposition.execute_plan()
        return SimpleNamespace(content=f"[Executed: {prompt[:60]}...]")

    class _Structured:
        def __init__(self, owner, schema):
            self.owner = owner
            self.schema = schema
            self._step = 0

        def invoke(self, messages, **kwargs):
            self.owner.calls += 1
            self._step += 1

            # ── Decomposition-first: return full upfront DAG ──────────────────
            if self.schema.__name__ == "GeneratedPlan":
                return self.schema.model_validate({
                    "goal": SHARED_GOAL,
                    "tasks": [
                        {"id": "t1", "instruction": "Notify front desk of delayed arrival time", "depends_on": []},
                        {"id": "t2", "instruction": "Reschedule spa appointment by 5 hours",     "depends_on": []},
                        {"id": "t3", "instruction": "Reschedule conference room booking",          "depends_on": []},
                        {"id": "t4", "instruction": "Prepare late check-in welcome package",       "depends_on": ["t1"]},
                        {"id": "t5", "instruction": "Confirm all adjustments with guest via SMS", "depends_on": ["t2", "t3", "t4"]},
                    ]
                })

            # ── Dynamic path: return one step at a time ────────────────────────
            if self.schema.__name__ == "DynamicDecision":
                steps = [
                    DynamicDecision(done=False, next_task="Check current check-in queue and flag late arrival"),
                    DynamicDecision(done=False, next_task="Contact spa to push appointment 5 hours forward"),
                    DynamicDecision(done=False, next_task="Alert conference coordinator of revised slot"),
                    DynamicDecision(done=True,  next_task=""),
                ]
                idx = min(self._step - 1, len(steps) - 1)
                return steps[idx]

            return self.schema()

    def with_structured_output(self, schema, *, method="json_schema"):
        return self._Structured(self, schema)


# ──────────────────────────────────────────────
# The actual divergence test
# ──────────────────────────────────────────────
def run_divergence_demo() -> dict:
    llm = DivergenceMockLLM()

    # ── Method A: Decomposition-first ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("METHOD A: Decomposition-First (Full upfront DAG)")
    print("=" * 65)
    plan: Plan = decompose_goal(SHARED_GOAL, llm)
    batches = plan.execution_batches()
    print(f"  Goal       : {plan.goal}")
    print(f"  Total tasks: {len(plan.tasks)}")
    print(f"  Exec order (batches):")
    for i, batch in enumerate(batches, 1):
        tasks_detail = [(t, plan.task(t).instruction) for t in batch]
        for tid, instruction in tasks_detail:
            print(f"    Batch {i} | {tid}: {instruction}")

    decomp_trace = [
        {"batch": i, "task_id": tid, "instruction": plan.task(tid).instruction}
        for i, batch in enumerate(batches, 1)
        for tid in batch
    ]

    # ── Method B: Dynamic Decomposition ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("METHOD B: Dynamic / Interleaved Decomposition (Reactive)")
    print("=" * 65)
    llm2 = DivergenceMockLLM()
    history = dynamic_decomposition(SHARED_GOAL, llm2, max_steps=4)
    for i, (task, result) in enumerate(history, 1):
        print(f"  Step {i}: {task}")
        print(f"         → {result[:70]}...")

    dynamic_trace = [
        {"step": i, "task": task, "result_preview": result[:80]}
        for i, (task, result) in enumerate(history, 1)
    ]

    # ── Divergence Analysis ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DIVERGENCE ANALYSIS")
    print("=" * 65)
    print("""
  Decomposition-First:
  - Plans ALL sub-tasks upfront as a DAG before any execution begins.
  - Tasks t2 (spa) and t3 (conference) run IN PARALLEL (same batch).
  - The synthesis task t5 waits for all branches to complete.
  - Rigid: if guest cancels the conference room, the plan still executes t3.

  Dynamic Decomposition:
  - Decides the NEXT single task based on what has already happened.
  - No parallelism: spa and conference are handled sequentially.
  - Adaptive: if step 2 reveals the spa is closed, step 3 can change.
  - Terminates as soon as `done=True` — no wasted synthesis step.

  KEY DIVERGENCE POINT: Task ordering and parallelism.
  - Decomp-first batches [t2, t3] in parallel.
  - Dynamic handles them sequentially, one at a time, based on observations.
""")

    return {
        "goal": SHARED_GOAL,
        "decomposition_first": {
            "method": "decompose_goal → execute_plan",
            "total_tasks": len(plan.tasks),
            "parallel_batch_count": len(batches),
            "execution_trace": decomp_trace,
        },
        "dynamic_decomposition": {
            "method": "dynamic_decomposition (reactive, step-by-step)",
            "total_steps": len(history),
            "execution_trace": dynamic_trace,
        },
        "divergence_summary": (
            "Decomposition-First creates a full DAG upfront with parallel branches "
            "(t2+t3 run simultaneously). Dynamic Decomposition is strictly sequential "
            "and reactive — it only picks the next task after observing the previous "
            "result, allowing mid-stream adaptation but sacrificing parallelism."
        )
    }


def test_decomposition_divergence():
    """Pytest-compatible: verifies both methods produce different task orderings."""
    result = run_divergence_demo()

    decomp = result["decomposition_first"]
    dynamic = result["dynamic_decomposition"]

    # Decomposition-first should produce a multi-batch parallel DAG
    assert decomp["parallel_batch_count"] >= 2, "Decomp-first must have at least 2 execution batches"
    assert decomp["total_tasks"] >= 4, "Decomp-first must plan at least 4 tasks upfront"

    # Dynamic should produce a strictly sequential, shorter chain
    assert dynamic["total_steps"] >= 2, "Dynamic must take at least 2 steps"

    # Key divergence: decomp has more tasks than dynamic steps (full plan vs reactive)
    assert decomp["total_tasks"] > dynamic["total_steps"], (
        "Decomp-first should plan more tasks upfront than dynamic executes steps"
    )

    # Confirm the methods actually produce DIFFERENT task sequences
    decomp_instructions = [t["instruction"] for t in decomp["execution_trace"]]
    dynamic_tasks = [t["task"] for t in dynamic["execution_trace"]]
    assert decomp_instructions != dynamic_tasks, "The two methods must produce different task sequences"

    print("\n✅ Divergence test passed: the two methods produce structurally different plans.")


if __name__ == "__main__":
    result = run_divergence_demo()
    # Save trace to artifact
    out = Path(__file__).parent / "artifacts" / "divergence_trace.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved divergence trace → {out}")
