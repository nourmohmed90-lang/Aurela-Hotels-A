"""
test_consolidation.py
-------------------------
Exercises memory/ltm.py end-to-end against two scenarios:

1. Promote-or-drop routing (process_overflow) -- a quick sanity check that
   small talk is forgotten and a noteworthy event is promoted to episodic
   memory, with zero direct writes to semantic memory from the router.

2. The main scenario requested: a real semantic-memory conflict.
     - Existing active fact: guest checkout preference = "2 PM" (version 1).
     - New episodic event: guest explicitly changed it to "4 PM".
     - consolidate_semantic_memory() is run with a scripted llm_call_fn that
       deterministically returns the expected FactUpdate (no network calls,
       fully reproducible).
     - We assert: the old fact is deactivated (is_active=False), the new
       fact is inserted as version 2 with conflict_notes populated, and it
       is now the only active fact for that key.

Standalone stores are defined here (not imported from agent.client) so this
test has no dependency on the MCP/Gemini client stack and can run in
isolation as a pure unit test over the memory layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memory.ltm import process_overflow, consolidate_semantic_memory, FactUpdate


# ---------------------------------------------------------------------------
# Minimal standalone stores (same interface as agent.client's, duplicated
# here on purpose to keep this test dependency-free).
# ---------------------------------------------------------------------------
class InMemoryEpisodicStore:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._next_id = 1

    def insert(self, record: dict[str, Any]) -> int:
        record = dict(record)
        record["id"] = self._next_id
        record["consolidated"] = False
        self._records.append(record)
        self._next_id += 1
        return record["id"]

    def get_unconsolidated(self, user_id: str) -> list[dict[str, Any]]:
        return [
            r for r in self._records
            if r.get("metadata", {}).get("user_id") == user_id and not r["consolidated"]
        ]

    def mark_consolidated(self, ids: list[int]) -> None:
        id_set = set(ids)
        for r in self._records:
            if r["id"] in id_set:
                r["consolidated"] = True


class InMemorySemanticStore:
    def __init__(self) -> None:
        self._facts: dict[int, dict[str, Any]] = {}
        self._next_id = 1

    def get_active_facts(self, user_id: str) -> list[dict[str, Any]]:
        return [f for f in self._facts.values() if f["user_id"] == user_id and f["is_active"]]

    def get_fact(self, user_id: str, fact_key: str) -> dict[str, Any] | None:
        for f in self._facts.values():
            if f["user_id"] == user_id and f["fact_key"] == fact_key and f["is_active"]:
                return f
        return None

    def update_status(self, fact_id: int, is_active: bool) -> None:
        if fact_id in self._facts:
            self._facts[fact_id]["is_active"] = is_active

    def insert_fact(self, fact: dict[str, Any]) -> int:
        fact = dict(fact)
        fact["id"] = self._next_id
        self._facts[self._next_id] = fact
        self._next_id += 1
        return fact["id"]

    def facts_for_key(self, user_id: str, fact_key: str) -> list[dict[str, Any]]:
        return [f for f in self._facts.values() if f["user_id"] == user_id and f["fact_key"] == fact_key]


USER_ID = "guest-sara-mohamed"


# ---------------------------------------------------------------------------
# Test 1: promote-or-drop routing sanity check
# ---------------------------------------------------------------------------
def scripted_routing_llm_fn(prompt: str, response_schema: Any) -> str:
    """Deterministic stand-in for the LLM: routes anything mentioning
    'checkout' or 'compensation' to episodic, everything else is forgotten.

    Only inspects the actual evicted-item payload (the text after the
    'Evicted Item:' marker in the prompt template), not the surrounding
    instructions -- otherwise words like "tool errors" in the criteria
    description itself would cause false positives.
    """
    item_text = prompt.split("Evicted Item:", 1)[-1]
    lowered = item_text.lower()
    if "checkout" in lowered or "compensation" in lowered or "error" in lowered:
        decision = {
            "reasoning": "Scripted test decision: event concerns a guest preference/decision worth keeping.",
            "destination": "episodic",
            "event_summary": "Guest checkout preference change captured.",
            "context": "Reservation follow-up",
            "outcome": "Preference updated in system",
        }
    else:
        decision = {
            "reasoning": "Scripted test decision: small talk, not worth retaining.",
            "destination": "forget",
            "event_summary": None,
            "context": None,
            "outcome": None,
        }
    import json
    return json.dumps(decision)


def test_promote_or_drop_routing() -> None:
    print("\n" + "-" * 70)
    print("TEST 1: Promote-or-Drop routing (process_overflow)")
    print("-" * 70)

    episodic_store = InMemoryEpisodicStore()

    small_talk_item = {"role": "user", "content": "Thanks so much, have a great day!"}
    process_overflow(small_talk_item, episodic_store, scripted_routing_llm_fn, USER_ID)
    assert episodic_store.get_unconsolidated(USER_ID) == [], (
        "Small talk must NOT be written to episodic memory"
    )
    print("PASS: small talk correctly forgotten (no episodic write).")

    noteworthy_item = {
        "role": "user",
        "content": "Guest requested checkout preference updated to 4 PM instead of 2 PM.",
    }
    process_overflow(noteworthy_item, episodic_store, scripted_routing_llm_fn, USER_ID)
    unconsolidated = episodic_store.get_unconsolidated(USER_ID)
    assert len(unconsolidated) == 1, "Noteworthy event must be promoted to episodic memory"
    print(f"PASS: noteworthy event promoted to episodic memory -> {unconsolidated[0]['event_summary']!r}")

    print("PASS: router never touched semantic memory directly (no semantic_store reference exists here).")


# ---------------------------------------------------------------------------
# Test 2: the main scenario -- semantic-memory conflict resolution
# ---------------------------------------------------------------------------
def scripted_consolidation_llm_fn(prompt: str, response_schema: Any) -> list[FactUpdate]:
    """Deterministic stand-in for the consolidation LLM. Recognizes the
    checkout-time conflict and returns the exact update we expect the
    consolidation pass to apply."""
    return [
        FactUpdate(
            fact_key="checkout_preference",
            action="UPDATE",
            new_value="4 PM",
            conflict_resolved=True,
            conflict_notes=(
                "Existing fact said checkout preference is 2 PM (version 1). "
                "A newer episodic event shows the guest explicitly changed the "
                "request to 4 PM after their flight moved; the later event "
                "takes precedence."
            ),
        )
    ]


def test_semantic_conflict_resolution() -> None:
    print("\n" + "-" * 70)
    print("TEST 2: Semantic memory conflict resolution (consolidate_semantic_memory)")
    print("-" * 70)

    episodic_store = InMemoryEpisodicStore()
    semantic_store = InMemorySemanticStore()

    now = datetime.now(timezone.utc).isoformat()

    # Seed an existing, already-active semantic fact: checkout at 2 PM (v1).
    original_fact_id = semantic_store.insert_fact({
        "user_id": USER_ID,
        "fact_key": "checkout_preference",
        "value": "2 PM",
        "version": 1,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
        "conflict_notes": None,
    })
    print(f"Seeded fact #{original_fact_id}: checkout_preference = '2 PM' (v1, active)")

    # Event 1: original preference stated (reflected in the existing fact already).
    episodic_store.insert({
        "timestamp": now,
        "event_summary": "Guest prefers late checkout at 2 PM",
        "context": "Initial booking",
        "outcome": "Preference recorded",
        "routing_reasoning": "Booking-time preference capture",
        "metadata": {"user_id": USER_ID},
    })

    # Event 2: guest updates the preference -- this is the conflicting event.
    episodic_store.insert({
        "timestamp": now,
        "event_summary": "Guest updated checkout preference to 4 PM",
        "context": "Follow-up call after flight change",
        "outcome": "Preference change requested",
        "routing_reasoning": "Flight schedule changed, guest needs later checkout",
        "metadata": {"user_id": USER_ID},
    })

    unconsolidated_before = episodic_store.get_unconsolidated(USER_ID)
    assert len(unconsolidated_before) == 2, "Both episodic events should be unconsolidated going in"

    consolidate_semantic_memory(
        user_id=USER_ID,
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        llm_call_fn=scripted_consolidation_llm_fn,
    )

    # --- Assertions -----------------------------------------------------
    original_fact = next(
        f for f in semantic_store.facts_for_key(USER_ID, "checkout_preference") if f["id"] == original_fact_id
    )
    assert original_fact["is_active"] is False, "Old fact (v1) must be deactivated after the update"

    active_facts = semantic_store.get_active_facts(USER_ID)
    checkout_facts = [f for f in active_facts if f["fact_key"] == "checkout_preference"]
    assert len(checkout_facts) == 1, "Exactly one active checkout_preference fact should remain"

    new_fact = checkout_facts[0]
    assert new_fact["version"] == 2, f"Expected version 2, got {new_fact['version']}"
    assert new_fact["value"] == "4 PM", f"Expected value '4 PM', got {new_fact['value']!r}"
    assert new_fact["is_active"] is True
    assert new_fact["conflict_notes"], "conflict_notes must be populated on a resolved conflict"

    remaining_unconsolidated = episodic_store.get_unconsolidated(USER_ID)
    assert remaining_unconsolidated == [], "All processed episodic records must be marked consolidated"

    print(f"PASS: old fact #{original_fact_id} deactivated (is_active=False)")
    print(f"PASS: new fact #{new_fact['id']} -> version={new_fact['version']}, value={new_fact['value']!r}")
    print(f"PASS: conflict_notes populated -> {new_fact['conflict_notes']!r}")
    print("PASS: episodic events marked consolidated after the pass.")


def main() -> None:
    test_promote_or_drop_routing()
    test_semantic_conflict_resolution()
    print("\n" + "=" * 70)
    print("ALL CONSOLIDATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
