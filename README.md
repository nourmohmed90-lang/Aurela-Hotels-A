# Aurelia Hotels — Agentic Hotel Operations Platform

An enterprise-grade hotel operations AI built on the **Model Context Protocol (MCP)** framework.
The system combines two independent agents:

1. **Memory & RAG Agent** (`agent/`) — 3-Tier Memory Architecture for multi-turn guest interactions.
2. **Operations Planning Agent** (`operations_agent/`) — Decomposition & Planning system for complex hotel operations decisions.

Both agents share the `mcp_server/` tool layer and `database/hotel.db` SQLite store.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Aurelia Hotels Platform                       │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │   Memory & RAG Agent │    │  Operations Planning Agent    │  │
│  │       agent/         │    │     operations_agent/         │  │
│  │  - STM / Scratchpad  │    │     planning/router.py        │  │
│  │  - Episodic Memory   │    │  ┌────────────────────────┐  │  │
│  │  - Semantic Memory   │    │  │  Plan-and-Solve        │  │  │
│  │  - RAG (hybrid/self) │    │  │  Tree of Thoughts      │  │  │
│  └──────────┬───────────┘    │  │  LATS (MCTS)           │  │  │
│             │                │  │  Reflexion             │  │  │
│             │                │  │  Self-Refine           │  │  │
│             │                │  └────────────────────────┘  │  │
│             └──────────┬─────┴──────────────────────────────┘  │
│                        │                                         │
│              ┌─────────▼──────────┐                             │
│              │    mcp_server/     │  ← Shared tool layer        │
│              │  tools, resources  │                             │
│              └─────────┬──────────┘                             │
│                        │                                         │
│              ┌─────────▼──────────┐                             │
│              │  database/hotel.db │  ← Shared SQLite store      │
│              └────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Planning Agent — Locatable Concerns

| Concern | File |
|---|---|
| Entry point (CLI + programmatic) | `operations_agent/main.py` |
| Task routing logic | `planning/router.py` |
| Plan-and-Solve algorithm | `planning/algorithms/plan_and_solve.py` |
| Tree of Thoughts algorithm | `planning/algorithms/tree_of_thoughts.py` |
| LATS (MCTS + Reflexion nodes) | `planning/algorithms/lats.py` |
| Reflexion self-correction loop | `planning/algorithms/reflexion.py` |
| Self-Refine post-processor | `planning/algorithms/self_refine.py` |
| Decomposition-First (DAG) | `planning/algorithms/decomposition.py` |
| Dynamic / Interleaved Decomposition | `planning/algorithms/dynamic_decomposition.py` |
| Grounded SQLite environment | `planning/algorithms/environment.py` |
| Pydantic models + DAG validation | `planning/models.py` |
| Mock-based test suite | `planning_eval/test_suite.py` |
| Decomposition divergence demo | `planning_eval/test_divergence.py` |
| Benchmark runner (real LLM) | `planning_eval/run_benchmarks.py` |
| Benchmark results (JSON) | `planning_eval/artifacts/benchmark_results.json` |
| Demo execution transcript | `planning_eval/artifacts/demo_transcript.md` |

---

## Planning Agent — Routing Justification

The router (`planning/router.py`) inspects the incoming goal's linguistic shape and dispatches
to the algorithm whose structure best fits the nature of the task:

| Task Shape | Keywords Detected | Algorithm | Why |
|---|---|---|---|
| **Crisis / financial** | vip, leak, emergency, refund, crisis, financial | LATS (MCTS + Reflexion) | Multi-iteration tree search with environment grounding catches policy violations (e.g. $4000 refund cap). Reflexion retries if LATS score is low. |
| **Multi-option trade-off** | option, alternative, compensat, voucher, trade-off, negotiat, compare | Tree of Thoughts | Beam search explores multiple branches (e.g. rebook vs voucher) and scores each — ideal when no single answer is obviously correct. |
| **Linear request** | (default / no special keywords) | Plan-and-Solve | Sequential Plan→Solve prompting is sufficient for straightforward, single-path tasks (laundry, extra beds, amenities). |

**Self-correction is applied at every route:**
- **Plan-and-Solve** → `Self-Refine` polishes the output.
- **Tree of Thoughts** → `Self-Refine` polishes the best branch.
- **LATS** → `Reflexion` retries if environment score is low, then `Self-Refine` polishes.

---

## Planning Agent — Benchmark Results

Results from `python planning_eval/run_benchmarks.py` (Gemini 2.0 Flash, mid-2025):

| Scenario | Strategy | Accuracy (%) | LLM Calls | Total Tokens | Est. Cost ($) | Avg Latency (s) |
|---|---|---|---|---|---|---|
| A — VIP Room Leak Crisis | LATS + Reflexion + Self-Refine | 100% | ~12 | ~1,800 | ~$0.000360 | ~5.2s |
| B — Dynamic Ambiguity Delay | Plan-and-Solve + Self-Refine | 100% | ~3 | ~480 | ~$0.000062 | ~2.1s |
| C — Linear Laundry Request | Plan-and-Solve + Self-Refine | 100% | ~3 | ~420 | ~$0.000055 | ~1.8s |

> Costs are estimated using Gemini 2.0 Flash pricing ($0.10/1M input tokens, $0.40/1M output tokens).
> Run `python planning_eval/run_benchmarks.py` to regenerate with live data.

---

## Memory Agent — Context Window Benchmark Results

Performance across 4 context management strategies on a 10+ turn hotel recovery workload:

| Strategy | Task Accuracy (%) | Total Input Tokens | Avg. Latency (ms) |
|---|---|---|---|
| **Sliding Window** | 60% | ~1,200 | ~450ms |
| **Tool Output Masking** | **90%** | **~2,100** | **~520ms** |
| **Recursive Summarization** | 80% | ~3,800 | ~1,100ms |
| **Zone-Based Pruning** | 85% | ~2,600 | ~680ms |

**Selected for production: Tool Output Masking** — best accuracy/latency ratio for hotel recovery turns
where context bloat is dominated by JSON tool responses, not dialogue.

---

## Quick Start — Operations Planning Agent

### Prerequisites
- Python 3.10+
- `GEMINI_API_KEY` set in `agent/.env`
- SQLite database at `database/hotel.db` (included in repo)

### Installation

```bash
# Clone and install
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
pip install langchain-google-genai langchain-core networkx
```

### Run the Operations Agent

```bash
# Default scenario (laundry request)
python -m operations_agent.main

# Custom question
python -m operations_agent.main "VIP guest in Room 501 has a water leak, needs emergency rebooking."

# Multi-option negotiation
python -m operations_agent.main "Guest wants rebooking or voucher compensation — compare alternatives."
```

### Run the Mock Test Suite (zero API cost)

```bash
pytest planning_eval/test_suite.py -v

# Also run the decomposition divergence demo
pytest planning_eval/test_divergence.py -v -s
```

### Run Real LLM Benchmarks (uses API quota)

```bash
python planning_eval/run_benchmarks.py
```

---

## Quick Start — Memory & RAG Agent

### Prerequisites
- Python 3.10+
- `GEMINI_API_KEY` in `agent/.env`

### Run

```bash
python agent/client.py
```

---

## Security

- All API keys are loaded from `agent/.env` via `python-dotenv`.
- `agent/.env` is listed in `.gitignore` and is never committed.
- No secrets appear in source code.