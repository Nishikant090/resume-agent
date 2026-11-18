"""
Eval suite for the Resume Agent.

Runs every question in evals/questions.json through the agent and checks:
  1. Tool choice  - did the agent call the expected tool?
  2. Grounding    - does the output contain the facts it must (e.g. "40%", "7,043")?
  3. Hallucination traps (LLM mode) - does it refuse to invent experience?

`must_include` entries are case-insensitive; "a|b" means either one is enough.

Usage
-----
    python eval.py                 # offline agent (keyword router), no key needed
    python eval.py --llm           # LLM agent (needs GROQ_API_KEY or OPENAI_API_KEY)
    python eval.py --threshold 0.9 # exit code 1 if pass rate is below 90%
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import agent

HERE = Path(__file__).parent
CASES_FILE = HERE / "evals" / "questions.json"
RESULTS_FILE = HERE / "evals" / "results.json"


def _has(text, option):
    return any(alt.strip().lower() in text for alt in option.split("|"))


def run_case(case, llm=None):
    """Run one question; return (tool_used, output_text, latency_s)."""
    start = time.perf_counter()
    if llm:
        client, model = llm
        tools = []
        history = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
        with contextlib.redirect_stdout(io.StringIO()):  # hide [tool] prints
            output = agent.run_llm_agent(client, model, history, case["question"], tool_log=tools)
        tool = tools[0] if tools else "none"
    else:
        tool, output = agent.route_offline(case["question"])
    return tool, output or "", time.perf_counter() - start


def grade(case, tool, output, llm_mode):
    text = output.lower()
    failures = []
    if tool != case["expected_tool"]:
        failures.append(f"tool: expected {case['expected_tool']}, got {tool}")
    must = list(case.get("must_include", []))
    must_not = []
    if llm_mode:
        must += case.get("llm_must_include", [])
        must_not += case.get("llm_must_not_include", [])
    failures += [f"missing: {m}" for m in must if not _has(text, m)]
    failures += [f"should not say: {m}" for m in must_not if _has(text, m)]
    return failures


def main():
    ap = argparse.ArgumentParser(description="Evaluate the Resume Agent")
    ap.add_argument("--llm", action="store_true", help="evaluate the LLM agent instead of the offline router")
    ap.add_argument("--threshold", type=float, default=0.8, help="minimum pass rate (0-1)")
    args = ap.parse_args()

    llm = None
    if args.llm:
        client, model = agent._llm_client()
        if not client:
            raise SystemExit("--llm needs `pip install openai` and GROQ_API_KEY or OPENAI_API_KEY")
        llm = (client, model)

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    mode = f"LLM ({llm[1]})" if llm else "offline router"
    print(f"Evaluating {len(cases)} cases | mode: {mode}\n")

    results, tool_ok, latencies = [], 0, []
    for case in cases:
        try:
            tool, output, secs = run_case(case, llm)
            failures = grade(case, tool, output, bool(llm))
        except Exception as e:  # an API error counts as a failure, not a crash
            tool, output, secs, failures = "error", "", 0.0, [f"error: {e}"]
        tool_ok += tool == case["expected_tool"]
        latencies.append(secs)
        passed = not failures
        results.append({"id": case["id"], "question": case["question"], "tool": tool,
                        "passed": passed, "failures": failures, "latency_s": round(secs, 3)})
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark}  {case['id']:<13} {tool:<14} {'; '.join(failures)}")

    n = len(cases)
    passed = sum(r["passed"] for r in results)
    summary = {
        "mode": mode,
        "cases": n,
        "pass_rate": round(passed / n, 3),
        "tool_accuracy": round(tool_ok / n, 3),
        "avg_latency_s": round(sum(latencies) / n, 3),
    }
    print(f"\nPass rate:     {passed}/{n} ({summary['pass_rate']:.0%})")
    print(f"Tool accuracy: {tool_ok}/{n} ({summary['tool_accuracy']:.0%})")
    print(f"Avg latency:   {summary['avg_latency_s']}s")

    RESULTS_FILE.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved: {RESULTS_FILE.relative_to(HERE)}")
    raise SystemExit(0 if summary["pass_rate"] >= args.threshold else 1)


if __name__ == "__main__":
    main()
