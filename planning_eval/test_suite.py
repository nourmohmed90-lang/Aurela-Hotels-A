import pytest
import sqlite3
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Add current workspace to system path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from planning.models import Plan, EnvironmentFeedback
from planning.algorithms.environment import Environment
from planning.algorithms.plan_and_solve import plan_and_solve
from planning.algorithms.tree_of_thoughts import tree_of_thoughts, ThoughtCandidates, ThoughtEvaluation
from planning.algorithms.dynamic_decomposition import dynamic_decomposition, DynamicDecision
from planning.algorithms.lats import lats, LATSActionBatch, ValueEstimate
from planning.router import route_task

# Set up test database path
DB_PATH = str(Path(__file__).resolve().parents[1] / "database" / "hotel.db")

# A Recording / Mock LLM that returns pre-defined structures or strings to avoid hitting actual LLM token quotas.
class MockLLM:
    def __init__(self, mode="default"):
        self.mode = mode
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        prompt = messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1].content
        prompt_lower = prompt.lower()

        # Self-Refine critique call → return PASS so the draft is accepted unchanged
        if "rubric" in prompt_lower and "draft" in prompt_lower and "list concrete issues" in prompt_lower:
            return SimpleNamespace(content="PASS")

        # Self-Refine revision call → return the draft as-is
        if "return only the improved deliverable" in prompt_lower:
            # Extract original draft from the prompt
            if "laundry" in prompt_lower or "pillow" in prompt_lower:
                return SimpleNamespace(content="PLAN: Verify reservation, call housekeeping.\nSOLUTION: Bed and laundry service scheduled for guest Sara Mohamed under reservation 2.")
            if "vip" in prompt_lower or "203" in prompt_lower or "leak" in prompt_lower:
                return SimpleNamespace(content="Guest Sara Mohamed VIP checked. Reservation 2 is confirmed. Room 103 has leak. Transfer VIP guest to Suite 203 which is available. Refund approved: $1000.")
            return SimpleNamespace(content="Mocked refined output.")

        # Return mock outputs matching the expected structure or narrative
        if "laundry" in prompt_lower or "bed" in prompt_lower:
            return SimpleNamespace(content="PLAN: Verify reservation, call housekeeping.\nSOLUTION: Bed and laundry service scheduled for guest Sara Mohamed under reservation 2.")

        if "water leak" in prompt_lower or "vip" in prompt_lower:
            # LATS reflection call or general task response
            return SimpleNamespace(content="I should check the database to verify if VIP guest Sara Mohamed has reservation 2, and then transfer to an available Suite room since room 103 has a leak. Refund limit is under $4000.")

        if "flight" in prompt_lower or "schedule" in prompt_lower:
            return SimpleNamespace(content="Rescheduled guest arrival details to late check-in protocol.")

        return SimpleNamespace(content="Mocked general success output.")

    class Structured:
        def __init__(self, owner, schema):
            self.owner = owner
            self.schema = schema

        def invoke(self, messages, **kwargs):
            self.owner.calls += 1
            # Return appropriate structured schemas
            if self.schema == ThoughtCandidates:
                return ThoughtCandidates(candidates=["Offer alternative Deluxe Room 202", "Offer $500 voucher compensation"])
            elif self.schema == ThoughtEvaluation:
                return ThoughtEvaluation(score=0.9, rationale="Balances guest satisfaction and branch occupancy perfectly.")
            elif self.schema == DynamicDecision:
                # First step next_task, second step done
                if self.owner.calls <= 1:
                    return DynamicDecision(done=False, next_task="Check hotel check-in time limit for late arrivals")
                else:
                    return DynamicDecision(done=True, next_task="")
            elif self.schema == LATSActionBatch:
                return LATSActionBatch(actions=[
                    {"action": "Check reservation and transfer VIP", "state": "Guest Sara Mohamed VIP checked. Reservation 2 is confirmed. Room 103 has leak. Transfer VIP guest to Suite 203 which is available. Refund approved: $1000."},
                    {"action": "Exceed refund limit", "state": "Refund approved: $5000."}
                ])
            elif self.schema == ValueEstimate:
                return ValueEstimate(score=0.95)
            
            return self.schema()

    def with_structured_output(self, schema, *, method="json_schema"):
        return self.Structured(self, schema)


def test_scenario_c_linear_request():
    """
    Scenario C (Linear Request): Laundry service or extra bed request.
    Uses Plan-and-Solve.
    """
    llm = MockLLM()
    # Let's route the task
    res = route_task("Guest Sara Mohamed requests extra pillows and laundry service for reservation 2.", llm, db_path=DB_PATH)
    assert res["strategy"] == "Plan-and-Solve"
    assert res["success"] is True
    assert "laundry" in res["output"].lower()


def test_scenario_b_dynamic_ambiguity():
    """
    Scenario B (Dynamic Ambiguity): Late flight arrival requiring dynamic schedule adjustment.
    Uses Dynamic Interleaved Decomposition.
    """
    llm = MockLLM()
    # Dynamic decomposition test
    history = dynamic_decomposition("Guest Ahmed Ali arrives 5 hours late due to flight delay; adjust schedule.", llm, max_steps=2)
    assert len(history) > 0
    assert history[0][0] == "Check hotel check-in time limit for late arrivals"


def test_scenario_a_crisis_vip_leak():
    """
    Scenario A (Crisis/Financial): VIP Room leak requiring emergency refund/rebooking.
    Uses LATS (MCTS + Reflexion) + Grounded database check.
    """
    llm = MockLLM()
    # Should route to LATS due to key terms "vip", "leak"
    res = route_task("VIP Room leak for Guest Sara Mohamed under reservation 2. Requires emergency rebooking and $1000 refund.", llm, db_path=DB_PATH)
    assert res["strategy"] == "LATS (MCTS + Reflexion)"
    assert res["success"] is True
    # The first state from LATSActionBatch mock contains the transfer to suite 203, and $1000 refund, which is valid.
    assert "203" in res["output"]
    assert "1000" in res["output"]


def test_grounded_environment_vip_check():
    """
    Verify that the Grounded Environment correctly retrieves details from hotel.db
    and flags VIP protocol or refund violations.
    """
    env = Environment(db_path=DB_PATH)
    
    # 1. Test VIP guest mentioned but VIP protocols missing
    feedback = env.evaluate("Ahmed Ali under reservation 1 has standard check-in.")
    # Ahmed Ali is Gold, not VIP. Let's see if VIP Sara is detected
    feedback_vip = env.evaluate("Sara Mohamed under reservation 2 has standard check-in.")
    # Sara is VIP. It should flag VIP protocol missing.
    assert any("VIP" in detail for detail in feedback_vip.details)

    # 2. Test Refund limit violation
    feedback_limit = env.evaluate("Guest Ahmed Ali refund amount is $5000 due to cancel.")
    assert feedback_limit.success is False
    assert any("exceeds the permitted limit" in detail for detail in feedback_limit.details)

    # 3. Test Room availability
    # Standard, Deluxe, Suite check
    feedback_rooms = env.evaluate("We will book standard room.")
    assert any("Standard" in detail for detail in feedback_rooms.details)
