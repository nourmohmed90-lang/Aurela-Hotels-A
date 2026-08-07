from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any

from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, types as mcp_types

import agent.elicitation
import agent.notification
import agent.sampling
import agent.tools
from agent.config import AgentConfig
from agent.handshake import NegotiatedCapabilities, perform_handshake
from agent.helpers import (
    call_tool_with_progress,
    mcp_tools_to_openai,
    tool_result_to_text,
)
from agent.transport import open_transport

# ---------------------------------------------------------------------------
# Memory subsystem imports. Repos that keep stm.py/ltm.py under a top-level
# `memory/` package should hit the first branch; a flat layout falls back to
# the second. Adjust to match your actual package location if neither fits.
# ---------------------------------------------------------------------------
try:
    from memory.stm import ShortTermMemory
    from memory.ltm import (
        process_overflow,
        consolidate_semantic_memory,
        MemoryRoutingDecision,
        FactUpdate,
    )
except ImportError:  # pragma: no cover - fallback for flat repo layout
    from stm import ShortTermMemory
    from ltm import (
        process_overflow,
        consolidate_semantic_memory,
        MemoryRoutingDecision,
        FactUpdate,
    )


# ---------------------------------------------------------------------------
# Lightweight in-memory stores backing episodic / semantic memory.
#
# These satisfy exactly the interface that memory/ltm.py expects
# (episodic_store.insert / get_unconsolidated / mark_consolidated and
# semantic_store.get_active_facts / get_fact / update_status / insert_fact).
# Swap these for a real database-backed implementation in production; the
# agent only ever talks to them through this interface.
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
            r
            for r in self._records
            if r.get("metadata", {}).get("user_id") == user_id and not r["consolidated"]
        ]

    def mark_consolidated(self, ids: list[int]) -> None:
        id_set = set(ids)
        for r in self._records:
            if r["id"] in id_set:
                r["consolidated"] = True

    def all_records(self) -> list[dict[str, Any]]:
        return list(self._records)


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

    def all_facts(self) -> list[dict[str, Any]]:
        return list(self._facts.values())


class AureliaAgent:
    """Core agent orchestrator: connects transport, handles MCP capabilities, and runs Gemini reasoning loop.

    Also owns the 3-tier memory stack:
      - Short-term memory (STM) + scratchpad: a rolling transcript buffer plus a
        pruning-immune working-state dict, injected into every Gemini call.
      - Episodic memory: durable log of noteworthy events, populated by the
        promote-or-drop router whenever STM evicts an old turn.
      - Semantic memory: versioned, conflict-resolved facts, produced by a
        periodic consolidation pass over unconsolidated episodic records.
    """

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

        api_key = os.getenv("GEMINI_API_KEY") or config.gemini_api_key
        self._client = genai.Client(api_key=api_key) if api_key else None
        self._openai_client = self._client
        self._llm = self._client

        # ---- Memory subsystem -------------------------------------------------
        stm_max_turns = getattr(config, "stm_max_turns", 20)
        self.stm = ShortTermMemory(max_turns=stm_max_turns)
        self.episodic_store = InMemoryEpisodicStore()
        self.semantic_store = InMemorySemanticStore()
        # Single logical "user" for this session; wire to a real guest/session
        # id if the transport carries one.
        self._user_id = getattr(config, "user_id", None) or "demo-guest-session"
        self._turns_since_consolidation = 0
        self._consolidation_interval = getattr(config, "consolidation_every_n_turns", 5)

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
        # Best-effort flush: promote anything still sitting unconsolidated in
        # episodic memory before the session goes away. Never let this block
        # or fail teardown.
        try:
            if self.episodic_store.get_unconsolidated(self._user_id):
                await self._run_consolidation()
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] shutdown consolidation skipped: {exc}")

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

    # -----------------------------------------------------------------------
    # Scratchpad convenience API (kept distinct from the pruned transcript)
    # -----------------------------------------------------------------------
    def set_plan(self, plan: str) -> None:
        self.stm.update_scratchpad(plan=plan)

    def set_subgoal(self, subgoal: str) -> None:
        self.stm.update_scratchpad(subgoal=subgoal)

    def update_working_state(self, **state: Any) -> None:
        self.stm.update_scratchpad(state=state)

    # -----------------------------------------------------------------------
    # STM eviction -> Promote-or-Drop routing (Episodic memory)
    # -----------------------------------------------------------------------
    async def _stm_add_and_handle(self, role: str, content: str, **kwargs: Any) -> None:
        """Push a message into STM; if this evicts the oldest turn, hand it to
        the promote-or-drop router. The router only ever writes to episodic
        memory (never semantic) -- see memory/ltm.py::process_overflow.
        """
        evicted = self.stm.add(role, content, **kwargs)
        if evicted is not None:
            await self._handle_eviction(evicted)

    async def _handle_eviction(self, evicted: dict[str, Any]) -> None:
        # process_overflow is a blocking/sync function (it may call out to an
        # LLM); run it off the event loop so a slow routing decision never
        # stalls the reasoning loop.
        await asyncio.to_thread(
            process_overflow,
            evicted,
            self.episodic_store,
            self._llm_routing_call,
            self._user_id,
        )

    # -----------------------------------------------------------------------
    # Periodic semantic consolidation
    # -----------------------------------------------------------------------
    async def _run_consolidation(self) -> None:
        await asyncio.to_thread(
            consolidate_semantic_memory,
            self._user_id,
            self.episodic_store,
            self.semantic_store,
            self._llm_consolidation_call,
        )

    async def _maybe_consolidate(self) -> None:
        self._turns_since_consolidation += 1
        if self._turns_since_consolidation >= self._consolidation_interval:
            await self._run_consolidation()
            self._turns_since_consolidation = 0

    # -----------------------------------------------------------------------
    # LLM call adapters used by memory/ltm.py.
    #
    # NOTE: the two functions in ltm.py have different contracts --
    # route_evicted_item() parses the return value as a JSON string via
    # `MemoryRoutingDecision.model_validate_json(raw_json)`, while
    # consolidate_semantic_memory() assigns the return value directly to a
    # `List[FactUpdate]`. These two adapters mirror that split exactly.
    # Both run synchronously (called via asyncio.to_thread) and both fall
    # back to a deterministic heuristic if no Gemini key is configured or the
    # call fails, so the memory pipeline degrades gracefully instead of
    # crashing the agent.
    # -----------------------------------------------------------------------
    def _model_name(self) -> str:
        model_name = os.getenv("GEMINI_MODEL") or self.config.gemini_model or "gemini-2.5-flash"
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        return model_name

    def _llm_routing_call(self, prompt: str, response_schema: Any) -> str:
        if self._client is None:
            return self._fallback_routing_decision(prompt)
        try:
            response = self._client.models.generate_content(
                model=self._model_name(),
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            return response.text or self._fallback_routing_decision(prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] routing LLM call failed, using heuristic fallback: {exc}")
            return self._fallback_routing_decision(prompt)

    def _fallback_routing_decision(self, prompt: str) -> str:
        # Only inspect the actual evicted-item payload, not the surrounding
        # ROUTING_PROMPT instructions -- the template text itself contains
        # words like "tool errors" and "key decisions", which would
        # otherwise make every item match.
        item_text = prompt.split("Evicted Item:", 1)[-1]
        lowered = item_text.lower()
        keep_signals = (
            "error",
            "approved",
            "denied",
            "compensation",
            "decision",
            "escalat",
            "manager",
            "policy",
            "overbook",
        )
        destination = "episodic" if any(sig in lowered for sig in keep_signals) else "forget"
        decision = MemoryRoutingDecision(
            reasoning=(
                "Heuristic fallback (no Gemini key configured): matched a "
                "keyword signal (error/decision/approval/policy) indicating "
                "a noteworthy event worth retaining."
                if destination == "episodic"
                else "Heuristic fallback (no Gemini key configured): no "
                "significant-event keywords found; treated as disposable "
                "small talk."
            ),
            destination=destination,
            event_summary=None if destination == "forget" else "Auto-captured event (heuristic routing).",
            context=None,
            outcome=None,
        )
        return decision.model_dump_json()

    def _llm_consolidation_call(self, prompt: str, response_schema: Any) -> list[FactUpdate]:
        if self._client is None:
            return self._fallback_consolidation()
        try:
            response = self._client.models.generate_content(
                model=self._model_name(),
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            raw = response.text or "[]"
            parsed = json.loads(raw)
            return [FactUpdate(**item) for item in parsed]
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] consolidation LLM call failed, using heuristic fallback: {exc}")
            return self._fallback_consolidation()

    def _fallback_consolidation(self) -> list[FactUpdate]:
        # Without an LLM we cannot reliably diff free-text episodic events
        # against existing facts, so the safe fallback is a no-op pass rather
        # than guessing. Real conflict handling is exercised deterministically
        # in test_consolidation.py via a scripted llm_call_fn.
        return []

    # Reasoning Loop (Gemini LLM)
    async def run_turn(self, user_message: str, *, system: str | None = None) -> str:
        """Run single conversational turn with tool execution loop."""
        if self._client is None:
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

        # Inject the live scratchpad (plan / subgoal / working state). This is
        # pulled straight from STM's scratchpad dict, which is never touched
        # by transcript pruning -- it survives even when old turns are
        # evicted and routed to episodic memory.
        scratch = self.stm.scratchpad
        scratch_lines: list[str] = []
        if scratch.get("plan"):
            scratch_lines.append(f"Active Plan: {scratch['plan']}")
        if scratch.get("current_subgoal"):
            scratch_lines.append(f"Current Subgoal: {scratch['current_subgoal']}")
        if scratch.get("working_state"):
            scratch_lines.append(f"Working State: {json.dumps(scratch['working_state'])}")
        if scratch_lines:
            system = (
                system
                + "\n\nScratchpad Context (persists independently of conversation pruning):\n"
                + "\n".join(scratch_lines)
            )

        # Read the model name from GEMINI_MODEL env var, defaulting to self.config.gemini_model, then gemini-2.5-flash
        model_name = self._model_name()

        # Convert MCP tools to google-genai tools format
        tools_list = []
        if self.tools:
            decls = []
            for tool in self.tools:
                decls.append(
                    genai_types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=tool.inputSchema,
                    )
                )
            tools_list.append(genai_types.Tool(function_declarations=decls))

        config_kwargs = {}
        if system:
            config_kwargs["system_instruction"] = system
        if tools_list:
            config_kwargs["tools"] = tools_list

        config = genai_types.GenerateContentConfig(**config_kwargs)

        messages: list[genai_types.Content] = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)]
            )
        ]

        # STM: record the incoming user turn. May trigger an eviction ->
        # promote-or-drop routing into episodic memory.
        await self._stm_add_and_handle("user", user_message)

        final_text = ""
        while True:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=messages,
                config=config,
            )

            if not response.candidates:
                final_text = ""
                break
            candidate = response.candidates[0]
            model_content = candidate.content
            if not model_content:
                final_text = ""
                break

            # Append the model's content (preserving thought_signature) directly to messages history
            messages.append(model_content)

            if not response.function_calls:
                final_text = response.text or ""
                break

            # Execute tool calls
            for tool_call in response.function_calls:
                name = tool_call.name
                args = tool_call.args or {}
                print(f"[agent] calling tool: {name}({json.dumps(args)})")

                result = await call_tool_with_progress(self, name, args)
                result_text = tool_result_to_text(result)

                messages.append(
                    genai_types.Content(
                        role="tool",
                        parts=[
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=name,
                                    response={"result": result_text}
                                )
                            )
                        ]
                    )
                )

                # STM: record the tool call/result as its own turn. This is
                # what strategy_tool_masking() and the eviction router act on.
                await self._stm_add_and_handle(
                    "tool", result_text, tool_name=name
                )

        # STM: record the final assistant reply.
        await self._stm_add_and_handle("assistant", final_text)

        # A "turn" is complete (user -> tool calls -> assistant reply). Every
        # `_consolidation_interval` completed turns, run the offline-style
        # semantic consolidation pass over whatever episodic memory has
        # accumulated since the last pass.
        await self._maybe_consolidate()

        return final_text