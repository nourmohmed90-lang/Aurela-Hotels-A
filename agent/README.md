# Agent Module Overview

The `agent` module houses the primary orchestration logic for **AureliaAgent**, an AI recovery assistant for Aurelia Hotels. It manages communication via the **Model Context Protocol (MCP)**, connects to Google Gemini for reasoning, and maintains continuous state using the 3-tier memory architecture.

---

## Capabilities & Execution Flow

* **Lifecycle Management:** Handles connection setup and teardown over transport protocols (Stdio / Streamable HTTP).
* **Capability Negotiation:** Verifies server capabilities on handshake (tools, resources, prompts, logging).
* **Dynamic Catalog Updates:** Listens for server notifications (e.g., tool list updates) and dynamically refreshes available capabilities without restarting.
* **Interactive Elicitation:** Supports server-initiated interaction forms (such as manager sign-off approvals for guest compensation).

---

## Reasoning & Memory Loop

When the agent executes a turn:

```text
User Message Input
       │
       ▼
1. Push Turn to Short-Term Memory ──► [If Overflow] ──► Promote-or-Drop Router
       │                                                     (FORGET vs. EPISODIC)
       ▼
2. Load Active Scratchpad & System Guidelines
       │
       ▼
3. Apply Context Window Strategy (Tool Masking)
       │
       ▼
4. Generate Content (Gemini API Call)
       │
       ├─► Function Call? ──► Execute Tool ──► Append Result to STM
       │
       └─► Final Response? ──► Return Output
                                    │
                                    ▼
                     5. Periodic Semantic Consolidation Pass