from __future__ import annotations


def classify_query_intent(query: str) -> str:
    """Classifies user query intent to determine optimal RAG path.

    - Returns 'hybrid' for fast, citation, or direct single-topic lookups.
    - Returns 'agentic' for complex multi-part, multi-hop, or reasoning queries.
    """
    lowered = query.lower()

    # Multi-part or decomposition indicators
    agentic_signals = [
        " and ", " both ", " along with ", " compare ", 
        " as well as ", " multi ", " step ", " first then "
    ]

    matches = sum(1 for signal in agentic_signals if signal in lowered)

    # Route long prompts or prompts with multiple sub-questions to agentic loop
    if len(query.split()) > 15 or matches >= 1 or lowered.count("?") > 1:
        return "agentic"

    return "hybrid"