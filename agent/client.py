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
    tool_result_to_text,
)
from agent.transport import open_transport

# ---- RAG & Memory Imports ----------------------------------------------------
from agent.rag.hybrid_retriever import hybrid_retriever
from agent.rag.hybrid_rag import HybridRAG
from agent.rag.agentic_rag import AgenticRAG
from agent.rag.self_rag import SelfRAG
from agent.rag.router import classify_query_intent

try:
    from memory.stm import ShortTermMemory
    from memory.ltm import (
        process_overflow,
        consolidate_semantic_memory,
        MemoryRoutingDecision,
        FactUpdate,
    )
except ImportError:
    from stm import ShortTermMemory
    from ltm import (
        process_overflow,
        consolidate_semantic_memory,
        MemoryRoutingDecision,
        FactUpdate,
    )


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
    """Core agent orchestrator connecting transport, MCP, 3-tier memory, and dynamic RAG router."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.negotiated: NegotiatedCapabilities | None = None
        self.tools: list[mcp_types.Tool] = []
        self.resources: list[mcp_types.Resource] = []
        self.prompts: list[mcp_types.Prompt] = []

        self._elicitation_queue: list[dict[str, Any]] = list(config.scripted_elicitation_responses)
        self._pending_notification_tasks: list[asyncio.Task[None]] = []

        api_key = os.getenv("GEMINI_API_KEY") or config.gemini_api_key
        self._client = genai.Client(api_key=api_key) if api_key else None
        self._openai_client = self._client
        self._llm = self._client

        # ---- RAG Subsystems ----------------------------------------------------
        model_name = self._model_name()
        self.hybrid_rag = HybridRAG(api_key=api_key, model=model_name) if api_key else None
        self.agentic_rag = AgenticRAG(api_key=api_key, model=model_name) if api_key else None
        self.self_rag = SelfRAG(api_key=api_key, model=model_name) if api_key else None

        # ---- Memory Subsystem -------------------------------------------------
        stm_max_turns = getattr(config, "stm_max_turns", 20)
        self.stm = ShortTermMemory(max_turns=stm_max_turns)
        self.episodic_store = InMemoryEpisodicStore()
        self.semantic_store = InMemorySemanticStore()

        self._user_id = getattr(config, "user_id", None) or "demo-guest-session"
        self._turns_since_consolidation = 0
        self._consolidation_interval = getattr(config, "consolidation_every_n_turns", 5)

    async def __aenter__(self) -> AureliaAgent:
        read, write, get_session_id = await self._exit_stack.enter_async_context(
            open_transport(self.config)
        )
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(
                read,
                write,
                sampling_callback=lambda ctx, params: agent.sampling.handle_sampling(self, ctx, params),
                elicitation_callback=lambda ctx, params: agent.elicitation.handle_elicitation(self, ctx, params),
                message_handler=lambda *args, **kwargs: agent.notification.handle_notification(self, *args, **kwargs),
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
            if self.episodic_store.get_unconsolidated(self._user_id):
                await self._run_consolidation()
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] shutdown consolidation skipped: {exc}")

        try:
            await self._exit_stack.aclose()
        except BaseExceptionGroup as eg:
            import anyio
            if not all(isinstance(e, anyio.BrokenResourceError) for e in eg.exceptions):
                raise

    async def wait_for_pending_notifications(self) -> None:
        tasks, self._pending_notification_tasks = self._pending_notification_tasks, []
        if tasks:
            await asyncio.gather(*tasks)

    # RAG Direct Invocation
    def search_knowledge_base(self, query: str, k: int = 3, source: str | None = None) -> str:
        return hybrid_retriever.retrieve_context(query=query, k=k, source=source)

    def ask_agentic_rag(self, question: str, k: int = 3) -> dict[str, Any]:
        if not self.agentic_rag:
            raise RuntimeError("GEMINI_API_KEY is required to run AgenticRAG.")
        return self.agentic_rag.ask(question=question, k=k)

    def ask_self_rag(self, question: str, k: int = 3) -> dict[str, Any]:
        if not self.self_rag:
            raise RuntimeError("GEMINI_API_KEY is required to run SelfRAG.")
        return self.self_rag.ask(question=question, k=k)

    # Resource & Prompt Helpers
    async def read_resource_text(self, uri: str) -> str:
        assert self.session is not None
        if self.negotiated and not self.negotiated.supports_resources:
            self.negotiated.require("resources", needed_for=f"reading {uri}")
        result = await self.session.read_resource(uri)  # type: ignore[arg-type]
        parts = [c.text for c in result.contents if isinstance(c, mcp_types.TextResourceContents)]
        return "\n".join(parts)

    async def get_prompt_messages(self, name: str, arguments: dict[str, str] | None = None) -> list[mcp_types.PromptMessage]:
        assert self.session is not None
        result = await self.session.get_prompt(name, arguments)
        return result.messages

    # Memory Management
    async def _stm_add_and_handle(self, role: str, content: str, **kwargs: Any) -> None:
        evicted = self.stm.add(role, content, **kwargs)
        if evicted is not None:
            await asyncio.to_thread(
                process_overflow,
                evicted,
                self.episodic_store,
                self._llm_routing_call,
                self._user_id,
            )

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
            print(f"[memory] routing LLM call failed: {exc}")
            return self._fallback_routing_decision(prompt)

    def _fallback_routing_decision(self, prompt: str) -> str:
        item_text = prompt.split("Evicted Item:", 1)[-1]
        lowered = item_text.lower()
        keep_signals = ("error", "approved", "denied", "compensation", "decision", "policy", "overbook")
        destination = "episodic" if any(sig in lowered for sig in keep_signals) else "forget"
        decision = MemoryRoutingDecision(
            reasoning="Heuristic fallback routing decision.",
            destination=destination,
            event_summary=None if destination == "forget" else "Auto-captured event.",
            context=None,
            outcome=None,
        )
        return decision.model_dump_json()

    def _llm_consolidation_call(self, prompt: str, response_schema: Any) -> list[FactUpdate]:
        if self._client is None:
            return []
        try:
            response = self._client.models.generate_content(
                model=self._model_name(),
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            parsed = json.loads(response.text or "[]")
            return [FactUpdate(**item) for item in parsed]
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] consolidation call failed: {exc}")
            return []

    # Dynamic Agent Reasoning Loop
    async def run_turn(self, user_message: str, *, system: str | None = None) -> str:
        if self._client is None:
            raise RuntimeError("GEMINI_API_KEY is required to run the agent loop.")
        assert self.session is not None

        # 1. Dynamic Query Intent Classification & RAG Routing
        route = classify_query_intent(user_message)
        print(f"[router] Dynamic RAG Route selected: {route.upper()}")

        if route == "agentic" and self.agentic_rag:
            rag_result = self.agentic_rag.ask(user_message)
            retrieved_context = rag_result.get("context", "")
        else:
            retrieved_context = hybrid_retriever.retrieve_context(query=user_message, k=3)

        if system is None:
            system = "You are Aurelia Hotel's AI recovery assistant."

        if retrieved_context.strip():
            system += f"\n\nRetrieved Knowledge Context ({route.upper()} Route):\n{retrieved_context}"

        # Inject Scratchpad
        scratch = self.stm.scratchpad
        scratch_lines = []
        if scratch.get("plan"):
            scratch_lines.append(f"Active Plan: {scratch['plan']}")
        if scratch.get("current_subgoal"):
            scratch_lines.append(f"Current Subgoal: {scratch['current_subgoal']}")
        if scratch.get("working_state"):
            scratch_lines.append(f"Working State: {json.dumps(scratch['working_state'])}")

        if scratch_lines:
            system += "\n\nScratchpad Context:\n" + "\n".join(scratch_lines)

        model_name = self._model_name()

        tools_list = []
        if self.tools:
            decls = [
                genai_types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema,
                )
                for tool in self.tools
            ]
            tools_list.append(genai_types.Tool(function_declarations=decls))

        config_kwargs = {}
        if system:
            config_kwargs["system_instruction"] = system
        if tools_list:
            config_kwargs["tools"] = tools_list

        config = genai_types.GenerateContentConfig(**config_kwargs)

        messages = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)]
            )
        ]

        await self._stm_add_and_handle("user", user_message)

        final_text = ""
        while True:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=messages,
                config=config,
            )

            if not response.candidates:
                break
            candidate = response.candidates[0]
            model_content = candidate.content
            if not model_content:
                break

            messages.append(model_content)

            if not response.function_calls:
                final_text = response.text or ""
                break

            for tool_call in response.function_calls:
                name = tool_call.name
                args = tool_call.args or {}
                print(f"[agent] executing tool: {name}({json.dumps(args)})")

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

                await self._stm_add_and_handle("tool", result_text, tool_name=name)

        await self._stm_add_and_handle("assistant", final_text)
        await self._maybe_consolidate()

        return final_text