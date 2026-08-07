# Memory Subsystem (`memory/`)

This module implements the **3-Tier Agentic Memory Architecture** for the Aurelia Hotels recovery assistant. It enables the agent to maintain multi-turn context, retain guest goals across transcript pruning, and handle factual updates without state pollution.

---

## Architecture Overview

```text
       ┌─────────────────────────────────────────┐
       │         Short-Term Memory (STM)          │
       │   - Rolling Message Buffer               │
       │   - Isolated Scratchpad (Plan/Goals)     │
       └────────────────────┬────────────────────┘
                            │
                      (STM Overflow)
                            │
                            ▼
       ┌─────────────────────────────────────────┐
       │         Promote-or-Drop Router          │
       │   - Evaluates Evicted Items             │
       │   - Routes to FORGET or EPISODIC       │
       └────────────────────┬────────────────────┘
                            │
                    (Episodic Events)
                            │
                            ▼
       ┌─────────────────────────────────────────┐
       │      Semantic Memory Consolidation      │
       │   - Offline/Periodic Pass               │
       │   - Resolves Fact Conflicts             │
       │   - Versions & Deactivates Superceded   │
       └─────────────────────────────────────────┘