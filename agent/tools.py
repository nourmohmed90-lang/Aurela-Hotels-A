
async def refresh_catalog(agent) -> None:
        assert agent.session is not None
        previous_names = {t.name for t in agent.tools}

        tools_result = await agent.session.list_tools()
        agent.tools = tools_result.tools

        if agent.negotiated and agent.negotiated.supports_resources:
            agent.resources = (await agent.session.list_resources()).resources
        if agent.negotiated and agent.negotiated.supports_prompts:
            agent.prompts = (await agent.session.list_prompts()).prompts

        new_names = {t.name for t in agent.tools}
        added, removed = new_names - previous_names, previous_names - new_names
        if added or removed:
            if added:
                print(f"[catalog] tools now available: {sorted(added)}")
            if removed:
                print(f"[catalog] tools no longer available: {sorted(removed)}")
        else:
            print(f"[catalog] {len(agent.tools)} tool(s): {sorted(new_names)}")

from mcp import types as mcp_types

mcp_types.Tool(
    name="decompose_and_search",
    description="Decomposes compound questions into sub-questions and searches the knowledge base for each part.",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The complex or multi-part user question to search."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to return per sub-question.",
                "default": 3
            }
        },
        "required": ["query"]
    }
)            