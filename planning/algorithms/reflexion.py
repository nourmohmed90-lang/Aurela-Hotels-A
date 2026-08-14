from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from ..models import EnvironmentFeedback
from .environment import Environment


def _extract_text(content) -> str:
    """Handle both plain string and Gemini list-of-blocks content format."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b["text"] if isinstance(b, dict) and "text" in b else str(b)
            for b in content
        ).strip()
    return ""


@dataclass
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]


def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    max_trials: int = 3,
    memory_size: int = 3,
) -> ReflexionResult:
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive")
    memory: list[str] = []
    trials: list[ReflexionTrial] = []
    best_attempt = ""
    best_score = -1.0
    for number in range(1, max_trials + 1):
        recalled = "\n".join(f"- {item}" for item in memory[-memory_size:]) or "- No prior trials."
        response = llm.invoke([
            ("system", "You are the acting agent in a Reflexion loop. Attempt the entire task again."),
            ("human", f"""Task: {task}
Episodic memory from previous failed trials:
{recalled}

Produce the complete deliverable. Apply remembered lessons without discussing them."""),
        ])
        attempt = _extract_text(response.content)
        if not attempt:
            raise RuntimeError("The chat model returned an empty or unsupported response")
        feedback = environment.evaluate(attempt)
        trial = ReflexionTrial(number=number, attempt=attempt, feedback=feedback)
        if feedback.score > best_score:
            best_attempt, best_score = attempt, feedback.score
        if feedback.success:
            trials.append(trial)
            return ReflexionResult(True, attempt, trials, memory[-memory_size:])
        response = llm.invoke([
            ("system", "Generate a concise first-person Reflexion memory, not a revised answer."),
            ("human", f"""Task: {task}
Failed attempt:
{attempt}

External environment feedback (score {feedback.score}):
{chr(10).join('- ' + item for item in feedback.details)}

State what I did wrong and the specific strategy I should use next trial. Start with 'I'."""),
        ])
        reflection = _extract_text(response.content)
        if not reflection:
            raise RuntimeError("The chat model returned an empty or unsupported response")
        trial.reflection = reflection
        trials.append(trial)
        memory.append(reflection)
    return ReflexionResult(False, best_attempt, trials, memory[-memory_size:])
