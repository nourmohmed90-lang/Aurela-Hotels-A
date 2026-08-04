from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from google import genai
from google.genai import types as gtypes
from mcp import ClientSession, types as mcp_types

import agent.elicitation
import agent.notification
import agent.sampling
import agent.tools
from agent.config import AgentConfig
from agent.handshake import NegotiatedCapabilities, perform_handshake
from agent.helpers import (
    call_tool_with_progress,
    mcp_tools_to_gemini,
    tool_result_to_text,
)
from agent.transport import open_transport


class AureliaAgent:
    """Core agent orchestrator: connects transport, handles MCP capabilities, and runs Gemini reasoning loop."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.negotiated: NegotiatedCapabilities | None = None
        self.tools: list[mcp_types.Tool] = []
        self.resources: list[mcp_types.Resource] = []
        self.prompts: list[mcp_types.Prompt] = []

        self._elicitation_queue: list[dict[str, Any]] = list(
            config.scripted_elicitation_responses
        )
        self._pending_notification_tasks: list[asyncio.Task[None]] = []

        self._llm = (
            genai.Client(api_key=config.gemini_api_key)
            if config.gemini_api_key
            else None
        )

    # Lifecycle Management
    async def __aenter__(self) -> AureliaAgent:
        read, write, get_session_id = await self._exit_stack.enter_async_context(
            open_transport(self.config)
        )
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(
                read,
                write,
                sampling_callback=lambda ctx, params: agent.sampling.handle_sampling(
                    self, ctx, params
                ),
                elicitation_callback=lambda ctx, params: agent.elicitation.handle_elicitation(
                    self, ctx, params
                ),
                message_handler=lambda *args, **kwargs: agent.notification.handle_notification(
                 self, *args, **kwargs
                ),
                client_info=mcp_types.Implementation(
                    name=self.config.client_name,
                    version=self.config.client_version,
                ),
            )
        )
        self.negotiated = await perform_handshake(self.session)
        self._get_session_id = get_session_id
        await agent.tools.refresh_catalog(self)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        try:
            await self._exit_stack.aclose()
        except BaseExceptionGroup as eg:  # Python 3.11+
            import anyio

            if not all(isinstance(e, anyio.BrokenResourceError) for e in eg.exceptions):
                raise

    async def wait_for_pending_notifications(self) -> None:
        """Await background tasks (e.g., catalog refresh triggered by notifications)."""
        tasks, self._pending_notification_tasks = self._pending_notification_tasks, []
        if tasks:
            await asyncio.gather(*tasks)

    # Resources / Prompts Helpers
    async def read_resource_text(self, uri: str) -> str:
        """Fetch and combine text resource contents."""
        assert self.session is not None
        if self.negotiated and not self.negotiated.supports_resources:
            self.negotiated.require("resources", needed_for=f"reading {uri}")
        result = await self.session.read_resource(uri)  # type: ignore[arg-type]
        parts = [
            c.text for c in result.contents if isinstance(c, mcp_types.TextResourceContents)
        ]
        return "\n".join(parts)

    async def get_prompt_messages(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[mcp_types.PromptMessage]:
        """Fetch prompt messages template from server."""
        assert self.session is not None
        result = await self.session.get_prompt(name, arguments)
        return result.messages

    # Reasoning Loop (Gemini LLM)
    async def run_turn(self, user_message: str, *, system: str | None = None) -> str:
        """Run single conversational turn with tool execution loop."""
        if self._llm is None:
            raise RuntimeError("GEMINI_API_KEY is required to run the agent loop.")
        assert self.session is not None
        if system is None:
            system = """
         are Aurelia Hotel's AI recovery assistant.
        Rules:
        - Use the minimum number of tool calls required.
        - Prefer search_available_rooms before any other room search tool.
        - If search_available_rooms returns available rooms, use that information directly.
        - Only call find_alternative_branch if no rooms are available.
        - Only call search_all_branches when the user explicitly asks to search every branch or when availability must be checked across all branches.
        - Never contradict previous tool results.
        - Never invent hotel availability.
        - Base every answer strictly on tool outputs.
        - If the user asks what rooms are available, use search_available_rooms with room_type="all".
        - Do not call search_all_branches unless the user explicitly asks to search all branches or no rooms are found.
        """
        config = gtypes.GenerateContentConfig(
            system_instruction=system,
            tools=mcp_tools_to_gemini(self.tools),
        )

        contents: list[Any] = [user_message]

        while True:
            response = await self._llm.aio.models.generate_content(
                model=self.config.gemini_model,
                contents=contents,
                config=config,
            )

            # Return final response text if no tools need to be called
            if not response.function_calls:
                return response.text or ""

            # Keep model thoughts/calls in turn context
            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

            # Execute tool calls
            function_responses = []
            for call in response.function_calls:
                print(f"[agent] calling tool: {call.name}({call.args})")
                args = dict(call.args) if call.args else {}
                result = await call_tool_with_progress(self, call.name, args)

                function_responses.append(
                    gtypes.Part.from_function_response(
                        name=call.name,
                        response={
                            "content": tool_result_to_text(result),
                            "is_error": result.isError or False,
                        },
                    )
                )

            # Feed tool results back to Gemini
            contents.append(gtypes.Content(role="user", parts=function_responses))