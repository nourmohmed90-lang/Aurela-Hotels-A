import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from planning.router import route_task
from operations_agent.main import get_gemini_llm

# Gemini 2.0 Flash pricing (USD per 1M tokens, as of mid-2025)
COST_PER_1M_INPUT_TOKENS  = 0.10
COST_PER_1M_OUTPUT_TOKENS = 0.40
AVG_TOKENS_PER_WORD = 1.35   # rough approximation

# Scenarios to benchmark
SCENARIOS = [
    {
        "id": "A",
        "name": "VIP Room Leak Crisis",
        "question": "VIP Room leak for Guest Sara Mohamed under reservation 2. Requires emergency rebooking and $1000 refund."
    },
    {
        "id": "B",
        "name": "Dynamic Ambiguity Delay",
        "question": "Guest Ahmed Ali arrives 5 hours late due to flight delay; adjust schedule and coordinate check-in."
    },
    {
        "id": "C",
        "name": "Linear Request Bed/Laundry",
        "question": "Guest Sara Mohamed requests extra pillows and laundry service for reservation 2."
    }
]


class CallCountingLLM:
    """
    Transparent wrapper around any BaseChatModel that counts every invoke() call
    and accumulates estimated token usage.
    """
    def __init__(self, base_llm: BaseChatModel):
        self._llm = base_llm
        self.call_count = 0
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * AVG_TOKENS_PER_WORD)

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b["text"] if isinstance(b, dict) and "text" in b else str(b)
                for b in content
            )
        return ""

    def invoke(self, messages, **kwargs):
        self.call_count += 1
        prompt_text = " ".join(
            m[1] if isinstance(m, tuple) else str(m) for m in messages
        )
        self.estimated_input_tokens += self._estimate_tokens(prompt_text)
        result = self._llm.invoke(messages, **kwargs)
        self.estimated_output_tokens += self._estimate_tokens(self._extract_text(result.content))
        return result

    def with_structured_output(self, schema, **kwargs):
        inner = self._llm.with_structured_output(schema, **kwargs)
        return _CountingStructured(self, inner)

    # Proxy everything else to the underlying LLM
    def __getattr__(self, name):
        return getattr(self._llm, name)


class _CountingStructured:
    """Proxy for structured-output chains that still counts calls."""
    def __init__(self, counter: CallCountingLLM, inner):
        self._counter = counter
        self._inner = inner

    def invoke(self, messages, **kwargs):
        self._counter.call_count += 1
        prompt_text = " ".join(
            m[1] if isinstance(m, tuple) else str(m) for m in messages
        )
        self._counter.estimated_input_tokens += self._counter._estimate_tokens(prompt_text)
        result = self._inner.invoke(messages, **kwargs)
        # Structured outputs don't have a .content, so we estimate from repr
        self._counter.estimated_output_tokens += self._counter._estimate_tokens(str(result))
        return result


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1_000_000 * COST_PER_1M_INPUT_TOKENS +
        output_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS
    )


def run_benchmarks():
    print("Starting Aurelia Hotels Real LLM Benchmarks...")

    # Load env and initialize base LLM
    try:
        base_llm = get_gemini_llm()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Please check your .env key.")
        return

    results = []
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    for sc in SCENARIOS:
        print(f"\nRunning Scenario {sc['id']}: {sc['name']}")

        # Fresh counter per scenario
        llm = CallCountingLLM(base_llm)
        start_time = time.time()

        try:
            res = route_task(sc["question"], llm)
            latency = round(time.time() - start_time, 2)

            total_tokens = llm.estimated_input_tokens + llm.estimated_output_tokens
            cost = estimate_cost(llm.estimated_input_tokens, llm.estimated_output_tokens)

            # Accuracy heuristic based on keyword presence in output
            out_lower = res.get("output", "").lower()
            accuracy = 0.0
            if sc["id"] == "A":
                if "transfer" in out_lower or "rebook" in out_lower or "203" in out_lower or "suite" in out_lower:
                    accuracy += 50.0
                if "refund" in out_lower or "1000" in out_lower or "compensation" in out_lower:
                    accuracy += 50.0
            elif sc["id"] == "B":
                if any(k in out_lower for k in ["late", "check-in", "delay", "resched", "plan", "adjust"]):
                    accuracy = 100.0
                else:
                    accuracy = 50.0
            elif sc["id"] == "C":
                if any(k in out_lower for k in ["laundry", "pillow", "schedule", "plan", "housekeep"]):
                    accuracy = 100.0
                else:
                    accuracy = 50.0

            trace = {
                "scenario_id": sc["id"],
                "scenario_name": sc["name"],
                "question": sc["question"],
                "selected_strategy": res["strategy"],
                "output": res["output"],
                "self_correction": res.get("self_correction", "N/A"),
                "latency_sec": latency,
                "total_llm_calls": llm.call_count,
                "estimated_input_tokens": llm.estimated_input_tokens,
                "estimated_output_tokens": llm.estimated_output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(cost, 6),
                "accuracy_pct": accuracy,
            }
            results.append(trace)

            print(f"  Strategy      : {res['strategy']}")
            print(f"  Latency       : {latency}s")
            print(f"  LLM Calls     : {llm.call_count}")
            print(f"  Total Tokens  : {total_tokens}")
            print(f"  Est. Cost     : ${cost:.6f}")
            print(f"  Accuracy      : {accuracy}%")
            print(f"  Self-Correction: {res.get('self_correction', 'N/A')}")

        except Exception as err:
            latency = round(time.time() - start_time, 2)
            print(f"  Failed: {err}")
            results.append({
                "scenario_id": sc["id"],
                "scenario_name": sc["name"],
                "question": sc["question"],
                "error": str(err),
                "latency_sec": latency,
                "total_llm_calls": llm.call_count,
                "estimated_input_tokens": llm.estimated_input_tokens,
                "estimated_output_tokens": llm.estimated_output_tokens,
                "total_tokens": llm.estimated_input_tokens + llm.estimated_output_tokens,
                "estimated_cost_usd": 0.0,
                "accuracy_pct": 0.0,
            })

    # ── Summary Table ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Scenario':<28} {'Strategy':<26} {'Acc%':>5} {'Calls':>6} {'Tokens':>8} {'Cost($)':>10} {'Lat(s)':>7}")
    print("-" * 95)
    for r in results:
        name  = r["scenario_name"][:27]
        strat = r.get("selected_strategy", "ERROR")[:25]
        acc   = r.get("accuracy_pct", 0.0)
        calls = r.get("total_llm_calls", 0)
        toks  = r.get("total_tokens", 0)
        cost  = r.get("estimated_cost_usd", 0.0)
        lat   = r.get("latency_sec", 0.0)
        print(f"{name:<28} {strat:<26} {acc:>5.0f} {calls:>6} {toks:>8} {cost:>10.6f} {lat:>7.2f}")
    print("=" * 95)

    # Save traces
    out_file = artifact_dir / "benchmark_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark traces to {out_file}")


if __name__ == "__main__":
    run_benchmarks()
