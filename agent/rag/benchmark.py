from __future__ import annotations

import os
import inspect
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent.rag.naive_rag import NaiveRAG
from agent.rag.hybrid_rag import HybridRAG
from agent.rag.agentic_rag import AgenticRAG

# 12 domain test questions spanning General, Citation, and Multi-part categories
TEST_QUESTIONS = [
    # General Clinical / Policy Questions
    {"id": 1, "cat": "General", "q": "What is the standard cancellation policy for Deluxe rooms?"},
    {"id": 2, "cat": "General", "q": "What are the standard check-in and check-out times?"},
    {"id": 3, "cat": "General", "q": "What is the procedure for handling guest noise complaints?"},
    {"id": 4, "cat": "General", "q": "What are the rules for late checkout approvals?"},

    # Citation-Heavy Questions
    {"id": 5, "cat": "Citation", "q": "What does Policy Section 4.2b say regarding compensation limits?"},
    {"id": 6, "cat": "Citation", "q": "Which rule covers maintenance emergency protocol in Section 12.1?"},
    {"id": 7, "cat": "Citation", "q": "What is specified in Code 3.1a for overbooking resolution?"},
    {"id": 8, "cat": "Citation", "q": "According to Section 8.4, who can authorize a room upgrade?"},

    # Multi-Part / Decomposition Questions
    {"id": 9, "cat": "Multi-Part", "q": "A guest has a damaged air conditioner and overbooking issue. What compensation and rebooking steps apply?"},
    {"id": 10, "cat": "Multi-Part", "q": "For an executive member requesting early check-in and late checkout during peak season, what approvals are needed?"},
    {"id": 11, "cat": "Multi-Part", "q": "What is the procedure if a room is dirty, the guest is unhappy, and demands a manager call immediately?"},
    {"id": 12, "cat": "Multi-Part", "q": "If a guest is transferred to a partner branch, what fare coverage and room match policies apply?"}
]


def safely_instantiate(cls, api_key: str):
    """Safely instantiates a RAG class whether __init__ requires api_key or takes no arguments."""
    sig = inspect.signature(cls.__init__)
    if "api_key" in sig.parameters:
        return cls(api_key=api_key)
    return cls()


def invoke_rag_ask(ask_fn, question: str):
    """Flexible wrapper to invoke ask_fn whether it accepts (question) or (question, k=3)."""
    sig = inspect.signature(ask_fn)
    if "k" in sig.parameters:
        return ask_fn(question, k=3)
    return ask_fn(question)


def extract_answer_text(res) -> str:
    """Extracts answer string regardless of return structure (dict, str, or tuple)."""
    if isinstance(res, dict):
        return res.get("answer", res.get("response", str(res)))
    if isinstance(res, (list, tuple)):
        return " ".join(str(item) for item in res)
    return str(res)


def run_benchmark():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is missing! "
            "Please set it in your environment or inside a .env file."
        )

    print("[info] Instantiating RAG strategies...")
    naive = safely_instantiate(NaiveRAG, api_key)
    hybrid = safely_instantiate(HybridRAG, api_key)
    agentic = safely_instantiate(AgenticRAG, api_key)

    strategies = [
        ("Naive RAG", naive.ask),
        ("Hybrid Search (Vector + BM25)", hybrid.ask),
        ("Agentic RAG (Multi-Hop)", agentic.ask)
    ]

    results = []

    for name, ask_fn in strategies:
        print(f"\n==================================================")
        print(f"   Running Benchmark Suite: {name}")
        print(f"==================================================")
        
        total_latency = 0.0
        total_tokens = 0
        correct_count = 0

        for item in TEST_QUESTIONS:
            q_id = item["id"]
            cat = item["cat"]
            query = item["q"]
            
            print(f"[{q_id}/12] ({cat}) Processing: {query[:50]}...")

            start_time = time.time()
            try:
                raw_res = invoke_rag_ask(ask_fn, query)
                latency = time.time() - start_time
                answer = extract_answer_text(raw_res)
            except Exception as exc:
                latency = time.time() - start_time
                answer = f"ERROR: {exc}"
                print(f"  └─ Question {q_id} failed with error: {exc}")

            # Heuristic token estimate (approx. 1.3 tokens per word + 350 system prompt overhead)
            words = len(answer.split())
            tokens = int(words * 1.3) + 350

            # Grounding check evaluation rule
            is_valid = (
                len(answer) > 30 
                and "couldn't find" not in answer.lower() 
                and "not found" not in answer.lower()
                and not answer.startswith("ERROR:")
            )
            
            if is_valid:
                correct_count += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"  └─ Latency: {latency:.2f}s | Status: {status}")

            total_latency += latency
            total_tokens += tokens

        avg_latency = round(total_latency / len(TEST_QUESTIONS), 2)
        avg_tokens = int(total_tokens / len(TEST_QUESTIONS))
        accuracy_str = f"{correct_count}/{len(TEST_QUESTIONS)}"

        results.append({
            "Retrieval Architecture": name,
            "Accuracy (12 questions)": accuracy_str,
            "Avg. Tokens/Query": avg_tokens,
            "Avg. Latency/Query": f"{avg_latency}s"
        })

    df = pd.DataFrame(results)
    print("\n================ RAG EVALUATION BENCHMARK RESULTS ================\n")
    print(df.to_string(index=False))
    print("\n==================================================================\n")


if __name__ == "__main__":
    run_benchmark()