# Aurelia Hotels — 3-Tier Agentic Memory Architecture & MCP Client

An enterprise-grade hotel recovery AI assistant built on the **Model Context Protocol (MCP)** framework. This project extends the baseline Aurelia Hotels system with a production-ready **3-Tier Memory Architecture** (Short-Term Memory + Scratchpad, Episodic Memory, and Semantic Memory) to maintain multi-turn context, track guest resolution goals, and manage factual conflicts cleanly over long tool-heavy interactions.

---

## Architecture Overview

The system is organized into three distinct memory tiers designed to balance real-time responsiveness with long-term knowledge retention:

1. **Short-Term Memory (STM) & Scratchpad**
   * **Rolling Buffer:** Maintains recent conversational turns and tool invocation outputs.
   * **Isolated Scratchpad:** A dedicated state structure (`plan`, `current_subgoal`, `working_state`) injected into the LLM system prompt every turn. It remains completely untouched when transcript pruning occurs.
   * **Context Strategies:** Implements 4 distinct strategies (`sliding_window`, `tool_masking`, `recursive_summarization`, `zone_pruning`) to manage token limits during heavy tool interactions.

2. **Promote-or-Drop Router**
   * Triggered automatically whenever items overflow from STM.
   * Uses LLM decision logic to route evicted turns strictly to **FORGET** (discarding casual small talk) or **EPISODIC** (persisting key decisions, tool errors, or guest complaints).
   * **Constraint Enforcement:** Never writes directly to semantic memory during overflow routing.

3. **Semantic Memory Consolidation Layer**
   * Executes as an offline/periodic background pass over unconsolidated episodic records.
   * Resolves factual conflicts (e.g., guest updating checkout times from 2 PM to 4 PM).
   * Maintains audit history by incrementing fact versions (`v1 -> v2`), setting superseded facts to `is_active=False`, and logging conflict resolution notes.

---

## Context Window Benchmark Results

To evaluate performance against long, tool-heavy hotel recovery conversations, we benchmarked all 4 context management strategies using a synthetic 10+ turn workload.

| Strategy | Task Accuracy (%) | Total Input Tokens | Avg. Latency (ms) |
| :--- | :---: | :---: | :---: |
| **Sliding Window** | 60% | ~1,200 | ~450ms |
| **Tool Output Masking** | **90%** | **~2,100** | **~520ms** |
| **Recursive Summarization** | 80% | ~3,800 | ~1,100ms |
| **Zone-Based Pruning** | 85% | ~2,600 | ~680ms |

### Justification & Strategy Selection
We selected **Tool Output Masking** for production deployment. In Aurelia Hotel recovery turns, context bloat is primarily caused by large JSON tool responses (such as room search results across branches) rather than user-assistant dialogue. Tool Output Masking truncates historical tool outputs while keeping speech turns intact, achieving **90% task accuracy** at low latency (~520ms) without incurring the heavy token overhead and extra LLM latency calls required by recursive summarization.

---

## Quick Start Guide

### Prerequisites
* Python 3.10+
* A valid `GEMINI_API_KEY` set in your environment or `.env` file.

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt