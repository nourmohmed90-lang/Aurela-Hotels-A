from typing import Optional, Literal
from pydantic import BaseModel
# Local stubs for missing modules
def llm_call(system: str, messages: list, response_schema: type) -> str:
    """Stub for LLM call reasoning."""
    raise NotImplementedError("llm_call is not implemented.")

def execute_tool(action_input: dict | None) -> str:
    """Stub for tool execution."""
    raise NotImplementedError("execute_tool is not implemented.")
class ShortTermMemory:
    def __init__(self, max_turns=20):
        self.max_turns = max_turns #Max turns to remember
        self.messages = [] #List of messages
        self.scratchpad = {} #Scratchpad for storing the agent's current plan and task results

    def add(self, role: str, content: str, **kwargs): #Add message to memory
        msg = {"role": role, "content": content, **kwargs} #Add message to memory
        self.messages.append(msg) #Append message to list
        self._truncate_safely() #Keep within bounds without breaking tool-call / tool-response pairs

    def _truncate_safely(self): 
        if len(self.messages) > self.max_turns:#Keep within bounds without breaking tool-call / tool-response pairs
            self.messages = self.messages[-self.max_turns:]# slice only the last N turns
            while self.messages and self.messages[0]["role"] in ["tool", "assistant"]:#Ensure we don't start memory with an orphaned 'tool' or 'assistant' completion
                self.messages.pop(0) #Remove the oldest message if it's a tool or assistant completion

    def get_context(self): #Get context
        return self.messages #Return messages


class AgentStep(BaseModel): #Agent Step Schema
    thought: str # Assistant's reasoning
    action: Literal["call_tool", "respond", "replan"] # Actions the agent can take
    action_input: Optional[dict] = None # Input for the action
    final_answer: Optional[str] = None 
    plan_updated: bool = False # if plan was updated
    new_plan: Optional[str] = None 
    next_subgoal: Optional[str] = None 


def agent_step(memory: ShortTermMemory, user_input: str, max_internal_steps: int = 5) -> AgentStep: #Execute one agent step
    # 1. Add new user input to context
    memory.add("user", user_input) #Add user input to memory
    
    # Internal agent execution loop for multi-step reasoning / tool calls
    for _ in range(max_internal_steps): # Number of internal steps
        system_prompt = f"""Current plan: {memory.scratchpad.get('plan', 'None')} 
        Sub-goal: {memory.scratchpad.get('current_subgoal', 'None')} """ # Current plan and sub-goal

        raw = llm_call( # Call LLM for reasoning
            system=system_prompt, 
            messages=memory.get_context(), 
            response_schema=AgentStep 
        )
        response = AgentStep.model_validate_json(raw) 

        # Update scratchpad state if requested
        if response.plan_updated:
            memory.scratchpad["plan"] = response.new_plan
            memory.scratchpad["current_subgoal"] = response.next_subgoal

        # Log assistant's thought/action turn into short-term memory
        memory.add("assistant", f"Thought: {response.thought}\nAction: {response.action}") # Add assistant's thought/action turn into short-term memory

        # Execute action
        if response.action == "call_tool":
            result = execute_tool(response.action_input) # Execute tool
            
            # Mask or truncate large tool outputs before saving to memory
            truncated_result = str(result)[:2000] 
            memory.add("tool", truncated_result)
            
            # Continue internal loop so the LLM processes the tool output immediately
            continue 

        elif response.action == "respond":
            memory.add("assistant", response.final_answer)
            return response

    return response