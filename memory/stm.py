import copy
from typing import List, Dict, Any, Optional

class ShortTermMemory:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.messages: List[Dict[str, Any]] = []
        # Distinct Scratchpad (preserves working state independently of transcript pruning)
        self.scratchpad: Dict[str, Any] = {
            "plan": None,
            "current_subgoal": None,
            "working_state": {}
        }

    def add(self, role: str, content: str, **kwargs) -> Optional[Dict[str, Any]]:
        msg = {"role": role, "content": content, **kwargs}
        self.messages.append(msg)
        
        # When context threshold is hit, evict oldest turn for LTM evaluation
        if len(self.messages) > self.max_turns:
            evicted = self.messages.pop(0)
            while self.messages and self.messages[0]["role"] in ["tool", "assistant"]:
                self.messages.pop(0)
            return evicted
        return None

    def update_scratchpad(self, plan: str = None, subgoal: str = None, state: dict = None):
        if plan is not None:
            self.scratchpad["plan"] = plan
        if subgoal is not None:
            self.scratchpad["current_subgoal"] = subgoal
        if state is not None:
            self.scratchpad["working_state"].update(state)

    def get_context(self) -> List[Dict[str, Any]]:
        return self.messages


# =====================================================================
# THE 4 REQUIRED CONTEXT WINDOW MANAGEMENT STRATEGIES
# =====================================================================

def strategy_sliding_window(messages: List[Dict[str, Any]], window_size: int = 6) -> List[Dict[str, Any]]:
    """Strategy 1: Keeps only the most recent N turns."""
    return messages[-window_size:]


def strategy_tool_masking(messages: List[Dict[str, Any]], keep_recent_tools: int = 2) -> List[Dict[str, Any]]:
    """Strategy 2: Replaces raw tool outputs older than keep_recent_tools with placeholders."""
    processed = copy.deepcopy(messages)
    tool_count = 0
    
    for msg in reversed(processed):
        if msg.get("role") == "tool":
            tool_count += 1
            if tool_count > keep_recent_tools:
                msg["content"] = "[tool output omitted — see reasoning above]"
    return processed


def strategy_recursive_summarization(messages: List[Dict[str, Any]], summary_llm_fn) -> List[Dict[str, Any]]:
    """Strategy 3: Recursively summarizes older turns into a single summary block."""
    if len(messages) <= 4:
        return messages
    
    old_turns = messages[:-4]
    recent_turns = messages[-4:]
    
    summary_text = summary_llm_fn(old_turns)
    
    return [
        {"role": "system", "content": f"Previous Context Summary: {summary_text}"}
    ] + recent_turns


def strategy_zone_pruning(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strategy 4: Zone-based pruning.
    - Zone 1 (System/Initial Intent): Retained.
    - Zone 2 (Middle/Tool Execution Logs): Pruned aggressively.
    - Zone 3 (Fresh Dialogue): Retained.
    """
    if len(messages) <= 4:
        return messages
        
    zone1_pinned = [m for m in messages[:2] if m.get("role") in ["system", "user"]]
    zone3_fresh = messages[-3:]
    
    zone2_middle = []
    for m in messages[2:-3]:
        if m.get("role") != "tool":
            zone2_middle.append(m)
            
    return zone1_pinned + zone2_middle + zone3_fresh