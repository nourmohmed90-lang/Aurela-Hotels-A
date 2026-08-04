from __future__ import annotations

from typing import Any

from mcp import ClientSession, types
from mcp.shared.context import RequestContext

from agent.helpers import coerce


async def handle_elicitation(
    agent: Any,
    context: RequestContext[ClientSession, Any],
    params: types.ElicitRequestFormParams | types.ElicitRequestURLParams,
) -> types.ElicitResult | types.ErrorData:
    """Answer a server-initiated elicitation request (e.g., manager sign-off forms)."""
    if isinstance(params, types.ElicitRequestURLParams):
        return types.ErrorData(
            code=types.INVALID_REQUEST,
            message="Aurelia's agent only supports form-mode elicitation.",
        )

    print("\n" + "=" * 60)
    print("MANAGER SIGN-OFF REQUIRED")
    print(params.message)
    schema_props = (params.requestedSchema or {}).get("properties", {})
    required = set((params.requestedSchema or {}).get("required", []))
    print("=" * 60)

    if agent._elicitation_queue:
        answers = agent._elicitation_queue.pop(0)
        print(f"[elicitation] using scripted response: {answers}")
    elif agent.config.interactive_elicitation:
        answers = prompt_console(agent, schema_props, required)
    else:
        # Fail safe by declining if no human is at terminal and no script remains
        return types.ElicitResult(action="decline")

    action = answers.pop("__action__", "accept")
    if action != "accept":
        return types.ElicitResult(action=action)  # type: ignore[arg-type]
    return types.ElicitResult(action="accept", content=answers)


def prompt_console(
    agent: Any, schema_props: dict[str, Any], required: set[str]
) -> dict[str, Any]:
    """Collect requested fields from interactive terminal input."""
    decision = input("Approve this compensation? [y/n]: ").strip().lower()
    if decision not in {"y", "yes"}:
        return {"__action__": "decline"}

    answers: dict[str, Any] = {}
    for field_name, field_schema in schema_props.items():
        label = field_schema.get("title", field_name)
        raw = input(f"  {label}{' (required)' if field_name in required else ''}: ")
        answers[field_name] = coerce(raw, field_schema.get("type", "string"))

    return answers