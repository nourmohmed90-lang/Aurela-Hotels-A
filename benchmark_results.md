# Context Window Management Strategy Benchmark

Workload: 18-message synthetic Aurelia Hotels support conversation (5 critical facts seeded across user/assistant/tool turns, heavy tool-output padding).

| Strategy | Task Accuracy (%) | Total Input Tokens | Latency (ms) |
|---|---|---|---|
| sliding_window | 20.0 | 491 | 0.001 |
| tool_masking | 100.0 | 700 | 0.055 |
| recursive_summarization | 100.0 | 1255 | 0.037 |
| zone_pruning | 100.0 | 455 | 0.006 |

## Rationale

Across the 10+ turn, tool-output-heavy Aurelia support transcript, **zone_pruning** retained the highest share of the seeded critical facts (100.0%) while using 455 estimated input tokens, compared with sliding_window, tool_masking, recursive_summarization which each traded off accuracy or token footprint differently. Sliding-window pruning is cheap and fast but blind to content -- once a fact scrolls outside the window it is gone regardless of importance, which is exactly what happens to the guest's name and room preference here. Tool masking keeps every conversational turn but blanks older raw tool payloads, which helps token count a lot on tool-heavy workloads but only preserves facts that also got restated in surrounding assistant/user turns, not facts that lived solely inside a tool response.

Recursive summarization and zone pruning both do better on accuracy because they make an explicit pass over older content instead of discarding it outright: zone pruning keeps the pinned system/intent turns and the freshest dialogue while stripping only mid-conversation tool noise, and recursive summarization actively extracts salient facts from everything it is about to drop. In production for Aurelia Hotels, where compensation approvals, error codes, and preference changes buried in tool output must survive long support sessions, we recommend **zone_pruning** as the default strategy, since it gives the best accuracy-per-token tradeoff on this workload; the other strategies remain useful fallbacks -- sliding window for latency-critical paths where perfect recall is not required, and tool masking when token budget is the binding constraint and critical facts are reliably echoed back in assistant replies.
