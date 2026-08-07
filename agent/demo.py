"""
agent/demo.py
-------------------------
Demonstration suite for Aurelia Hotel Recovery Server tools, resources,
prompts, memory tracking, and RAG knowledge retrieval strategies.
"""

from __future__ import annotations

import argparse
import asyncio
from agent.client import AureliaAgent
from agent.config import load_config
from agent.helpers import call_tool_with_progress
from dotenv import load_dotenv

load_dotenv(override=True)


def _section(title: str) -> None:
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


async def run(auto: bool) -> None:
    config = load_config()
    if auto:
        object.__setattr__(
            config,
            "scripted_elicitation_responses",
            ({"confirm": True, "manager_note": "approved (scripted demo run)"},),
        )

    async with AureliaAgent(config) as agent:
        # 1. Capability negotiation
        _section("1) CAPABILITY NEGOTIATION")
        assert agent.negotiated is not None
        print(f"Server declares tools.listChanged = {agent.negotiated.supports_tools_list_changed}")

        # 2. Baseline tool set
        _section("2) BASELINE TOOL SET")
        print("Available Tools:", sorted(t.name for t in agent.tools))

        # 3. Vector Database & RAG Retrieval Check
        _section("3) RAG SUBSYSTEM: Hybrid Retrieval Test")
        rag_context = agent.search_knowledge_base("compensation policy for maintenance issues")
        print("Retrieved Knowledge Context:\n", rag_context or "No context found.")

        # 4. Progress tracking: search_all_branches
        _section("4) PROGRESS TRACKING: search_all_branches")
        result = await call_tool_with_progress(
            agent,
            "search_all_branches",
            {"room_type": "Deluxe"},
        )
        print(result.content[0].text)

        # 5. Defensive design: approve_compensation validation failure
        _section("5) DEFENSIVE DESIGN: Invalid Compensation Approval")
        denied = await call_tool_with_progress(
            agent,
            "approve_compensation",
            {
                "request_id": 4,
                "approved_by": 1,
                "amount": 5000.0,
            },
        )
        print("Defensive Guard Output:", denied.content[0].text)

        # 6. Role elevation & Notifications
        _section("6) NOTIFICATIONS: promote_to_manager")
        result = await call_tool_with_progress(
            agent,
            "promote_to_manager",
            {},
        )
        print("Promotion Output:", result.content[0].text)
        await agent.wait_for_pending_notifications()

        # 7. Action after elevation: Valid Compensation Approval
        _section("7) SUCCESSFUL MANAGER ACTION: approve_compensation")
        approved = await call_tool_with_progress(
            agent,
            "approve_compensation",
            {
                "request_id": 4,
                "approved_by": 1,
                "amount": 3000.0,
            },
        )
        print("Approval Result:", approved.content[0].text)

        # 8. LLM Reasoning / Turn Check with RAG context
        _section("8) TESTING LLM INTELLIGENCE (RUN_TURN)")
        if agent._openai_client:
            print("Sending query to Gemini...")
            reply = await agent.run_turn("What compensation policies apply for room maintenance issues?")
            print("Gemini Answer:\n", reply)
        else:
            print("Sampling skipped: No LLM API key configured.")

        # 9. Resources Read Check
        _section("9) RESOURCES: guest compensation policy")
        policy_text = await agent.read_resource_text("policy://guest-compensation")
        print(policy_text)

        # 10. Prompts Execution Check
        _section("10) PROMPTS: draft_guest_apology")
        messages = await agent.get_prompt_messages(
            "draft_guest_apology",
            {
                "guest_name": "Sara Mohamed",
                "issue": "Overbooking Conflict",
            },
        )
        print("Prompt Generated Text:\n", messages[0].content.text)

        _section("DONE")
        print(f"Transport used this run: {config.transport_mode.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use scripted parameters.",
    )
    args = parser.parse_args()
    asyncio.run(run(auto=args.auto))


if __name__ == "__main__":
    main()