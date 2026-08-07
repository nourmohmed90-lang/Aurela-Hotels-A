from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime, timezone

# ---------------------------------------------------------------------
# 1. PROMOTE-OR-DROP ROUTER (FORGET vs. EPISODIC ONLY)
# ---------------------------------------------------------------------

class MemoryRoutingDecision(BaseModel):
    reasoning: str = Field(..., description="Mandatory detailed explanation for why this fate was chosen.")
    destination: Literal["forget", "episodic"] = Field(..., description="Restricted strictly to 'forget' or 'episodic'.")
    event_summary: Optional[str] = None
    context: Optional[str] = None
    outcome: Optional[str] = None

ROUTING_PROMPT = """An item is being evicted from short-term memory.
Decide if it should be forgotten or promoted to episodic memory.

Criteria:
- FORGET: Small talk, one-off clarifications, intermediate non-essential steps.
- EPISODIC: Specific events, tool errors, key decisions, or milestones.

You MUST provide clear reasoning for graders to inspect.

Evicted Item:
{item}
"""

def route_evicted_item(item: Dict[str, Any], llm_call_fn) -> MemoryRoutingDecision:
    prompt = ROUTING_PROMPT.format(item=str(item))
    raw_json = llm_call_fn(prompt=prompt, response_schema=MemoryRoutingDecision)
    return MemoryRoutingDecision.model_validate_json(raw_json)


def process_overflow(item: Dict[str, Any], episodic_store, llm_call_fn, user_id: str):
    """Processes evicted item, logs reasoning, and inserts into episodic_store if promoted."""
    decision = route_evicted_item(item, llm_call_fn)
    
    # Reasoning log for evaluation/grading
    print(f"[ROUTER LOG] Destination: {decision.destination.upper()} | Reasoning: {decision.reasoning}")
    
    if decision.destination == "forget":
        return
        
    elif decision.destination == "episodic":
        episodic_store.insert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_summary": decision.event_summary,
            "context": decision.context,
            "outcome": decision.outcome,
            "routing_reasoning": decision.reasoning,
            "metadata": {"user_id": user_id}
        })


# ---------------------------------------------------------------------
# 2. SEMANTIC CONSOLIDATION LAYER (OFFLINE/PERIODIC PASS)
# ---------------------------------------------------------------------

class FactUpdate(BaseModel):
    fact_key: str
    action: Literal["NEW", "UPDATE", "EXPIRE", "NO_CHANGE"]
    new_value: Optional[str] = None
    conflict_resolved: bool = False
    conflict_notes: Optional[str] = None

CONSOLIDATION_PROMPT = """Analyze recent episodic logs against active semantic facts for a user.
Produce updated semantic facts.

Current Active Facts:
{current_facts}

New Episodic Events:
{recent_episodes}

Detect updates, fact expirations, or conflicts between episodes and existing facts.
"""

class SemanticFact(BaseModel):
    user_id: str
    fact_key: str
    value: str
    version: int = 1
    is_active: bool = True
    created_at: str
    updated_at: str
    conflict_notes: Optional[str] = None


def consolidate_semantic_memory(user_id: str, episodic_store, semantic_store, llm_call_fn):
    """
    Periodic Consolidation Pass:
    Reads episodic store, resolves conflicts/staleness, and updates versioned facts in semantic store.
    """
    recent_episodes = episodic_store.get_unconsolidated(user_id=user_id)
    if not recent_episodes:
        return

    current_facts = semantic_store.get_active_facts(user_id=user_id)
    
    prompt = CONSOLIDATION_PROMPT.format(
        current_facts=str(current_facts),
        recent_episodes=str(recent_episodes)
    )
    
    updates: List[FactUpdate] = llm_call_fn(prompt=prompt, response_schema=List[FactUpdate])
    now = datetime.now(timezone.utc).isoformat()
    
    for update in updates:
        if update.action == "NO_CHANGE":
            continue
            
        existing_fact = semantic_store.get_fact(user_id, update.fact_key)
        
        # Mark old version inactive for versioning audit trail
        if update.action in ["UPDATE", "EXPIRE"] and existing_fact:
            semantic_store.update_status(existing_fact["id"], is_active=False)
            
        if update.action in ["NEW", "UPDATE"]:
            new_version = (existing_fact["version"] + 1) if existing_fact else 1
            
            new_fact = SemanticFact(
                user_id=user_id,
                fact_key=update.fact_key,
                value=update.new_value,
                version=new_version,
                is_active=True,
                created_at=existing_fact["created_at"] if existing_fact else now,
                updated_at=now,
                conflict_notes=update.conflict_notes if update.conflict_resolved else None
            )
            semantic_store.insert_fact(new_fact.dict())
            
    # Mark batch of episodic records as consolidated
    episodic_store.mark_consolidated([e["id"] for e in recent_episodes])