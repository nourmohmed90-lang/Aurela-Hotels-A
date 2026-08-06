import json
from typing import Any
from mcp import types


def content_to_text(content: Any) -> str:
    """Extract string content from various MCP content structures."""
    if isinstance(content, list):
        return "\n".join(content_to_text(c) for c in content)
    if isinstance(content, types.TextContent):
        return content.text
    return str(content)


def tool_result_to_text(result: types.CallToolResult) -> str:
    """Convert CallToolResult block contents into plain text or JSON string."""
    parts = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        else:
            parts.append(json.dumps(block.model_dump(mode="json")))
    return "\n".join(parts) or "(empty tool result)"


def coerce(raw: str, json_type: str) -> Any:
    """Coerce string inputs into target JSON schema types."""
    if json_type == "number":
        return float(raw)
    if json_type == "integer":
        return int(raw)
    if json_type == "boolean":
        return raw.strip().lower() in {"y", "yes", "true", "1"}
    return raw


def mcp_tools_to_openai(tools: list[types.Tool]) -> list[dict[str, Any]]:
    """Convert MCP tools list into OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            }
        }
        for tool in tools
    ]


async def call_tool_with_progress(
    agent, name: str, arguments: dict[str, Any]
) -> types.CallToolResult:
    """Call an MCP tool with real-time execution progress updates printed."""
    assert agent.session is not None

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        pct = f"{(progress / total) * 100:.0f}%" if total else f"{progress:g}"
        print(f"  [progress] {pct}{f' - {message}' if message else ''}")

    return await agent.session.call_tool(name, arguments, progress_callback=on_progress)