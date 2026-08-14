from langchain_core.language_models.chat_models import BaseChatModel

from .algorithms.plan_and_solve import plan_and_solve, _extract_text
from .algorithms.tree_of_thoughts import tree_of_thoughts
from .algorithms.lats import lats
from .algorithms.reflexion import reflexion
from .algorithms.self_refine import reflect_and_refine
from .algorithms.environment import Environment


def route_task(question: str, llm: BaseChatModel, db_path: str | None = None) -> dict:
    """
    Inspects sub-tasks/goals and dispatches them to the most suitable planning strategy,
    then applies self-correction at the appropriate scope:

    Strategy routing (by task shape):
    - LATS (MCTS + Reflexion):  High-cost financial/crisis decisions (VIP, leaks, emergency refunds).
      Self-correction: Reflexion retry loop wraps LATS when environment score is low.
    - Tree of Thoughts:         Multi-option trade-off negotiations (vouchers, alternatives, compensation).
      Self-correction: Self-Refine applied to the best thought output.
    - Plan-and-Solve:           Standard linear requests (laundry, amenities, extra beds).
      Self-correction: Self-Refine applied as post-processing refinement step.
    """
    q_lower = question.lower()
    env = Environment(db_path=db_path)

    # ── 1. High-cost financial / crisis decisions → LATS + Reflexion ──────────────
    if any(k in q_lower for k in ["vip", "leak", "emergency", "refund", "crisis", "financial", "broken", "water leak"]):
        strategy = "LATS (MCTS + Reflexion)"

        # Primary LATS search
        lats_result = lats(question, llm, env, iterations=2, n_actions=2)

        # If LATS did not fully succeed, apply Reflexion as an outer retry loop
        if not lats_result.success:
            reflexion_result = reflexion(question, llm, env, max_trials=2, memory_size=3)
            final_output = reflexion_result.output
            final_success = reflexion_result.success
            correction_note = f"[Reflexion applied: {len(reflexion_result.trials)} trial(s)]"
        else:
            final_output = lats_result.output
            final_success = lats_result.success
            correction_note = "[LATS succeeded — no Reflexion retry needed]"

        # Apply Self-Refine as a final polish pass on the chosen output
        refined = reflect_and_refine(question, final_output, llm)

        return {
            "strategy": strategy,
            "success": final_success,
            "output": refined.revised,
            "raw_output": final_output,
            "self_correction": correction_note,
            "self_refine_critique": refined.critique,
            "best_score": lats_result.best_score,
            "iterations": lats_result.iterations,
        }

    # ── 2. Multi-option trade-offs → Tree of Thoughts + Self-Refine ───────────────
    elif any(k in q_lower for k in ["option", "alternative", "compensat", "voucher", "trade-off", "negotiat", "compare"]):
        strategy = "Tree of Thoughts"
        thoughts = tree_of_thoughts(question, llm, depth=2, beam_width=2)
        best_thought = thoughts[0].state if thoughts else "No thoughts generated."

        # Self-Refine: polish the best thought into a final deliverable
        refined = reflect_and_refine(question, best_thought, llm)

        return {
            "strategy": strategy,
            "success": bool(thoughts),
            "output": refined.revised,
            "raw_output": best_thought,
            "self_correction": "[Self-Refine applied to best Tree-of-Thoughts branch]",
            "self_refine_critique": refined.critique,
            "thoughts": [t.model_dump() for t in thoughts],
        }

    # ── 3. Standard linear requests → Plan-and-Solve + Self-Refine ────────────────
    else:
        strategy = "Plan-and-Solve"
        raw_output = plan_and_solve(question, llm)

        # Self-Refine: verify completeness and improve the plan
        refined = reflect_and_refine(question, raw_output, llm)

        return {
            "strategy": strategy,
            "success": bool(refined.revised),
            "output": refined.revised,
            "raw_output": raw_output,
            "self_correction": "[Self-Refine applied as post-processing refinement]",
            "self_refine_critique": refined.critique,
        }
