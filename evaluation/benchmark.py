
import json
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agents.financial_agent import FinancialAgent
from src.config import MODEL, CONTEXT_WINDOW, TEMPERATURE
from evaluation.test_cases import TEST_CASES


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def normalize_arguments(arguments):

    normalized = dict(arguments)

    if isinstance(normalized.get("symbol"), str):
        normalized["symbol"] = (
            normalized["symbol"].strip().upper()
        )

    return normalized


def evaluate_calls(expected_calls, actual_calls):

    expected_tools = Counter(
        call["tool"] for call in expected_calls
    )

    actual_tools = Counter(
        call["tool"] for call in actual_calls
    )

    tool_selection_correct = (
        expected_tools == actual_tools
    )

    unmatched = list(actual_calls)

    matched_arguments = 0

    for expected in expected_calls:

        expected_arguments = normalize_arguments(
            expected["arguments"]
        )

        for index, actual in enumerate(unmatched):

            if expected["tool"] != actual["tool"]:
                continue

            actual_arguments = normalize_arguments(
                actual["arguments"]
            )

            if actual["tool"] == "analyze_stock":
                actual_arguments.setdefault("period", "2y")

            arguments_match = all(
                actual_arguments.get(key) == value
                for key, value in expected_arguments.items()
            )

            if arguments_match:

                matched_arguments += 1

                unmatched.pop(index)

                break

    argument_accuracy = (
        matched_arguments / len(expected_calls)
        if expected_calls
        else None
    )

    arguments_correct = (
        matched_arguments == len(expected_calls)
        and not unmatched
    )

    execution_success = all(
        call.get("success", False)
        for call in actual_calls
    )

    return {
        "tool_selection_correct": tool_selection_correct,
        "arguments_correct": arguments_correct,
        "argument_accuracy": argument_accuracy,
        "execution_success": execution_success,
        "passed": (
            tool_selection_correct
            and arguments_correct
            and execution_success
        )
    }


def run_test_case(agent, test_case):

    agent.clear_history()

    start_time = time.perf_counter()

    error = None
    answer = ""

    try:

        answer = agent.chat(
            test_case["prompt"]
        )

    except Exception as exception:

        error = (
            f"{type(exception).__name__}: {exception}"
        )

    latency = time.perf_counter() - start_time

    actual_calls = agent.get_tool_logs()

    metrics = agent.get_metrics()

    evaluation = evaluate_calls(
        test_case["expected_calls"],
        actual_calls
    )

    if error is not None:
        evaluation["passed"] = False

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "prompt": test_case["prompt"],

        "expected_calls": test_case["expected_calls"],
        "actual_calls": actual_calls,

        "answer": answer,
        "error": error,

        **evaluation,

        "latency": latency,
        "llm_calls": metrics.get("llm_calls", 0),
        "tool_calls": metrics.get("tool_calls", 0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "llm_latency": metrics.get("llm_latency", 0),
        "tool_execution_time": metrics.get(
            "tool_execution_time", 0
        )
    }


def summarize_results(results):

    df = pd.DataFrame(results)

    tool_tests = df[
        df["argument_accuracy"].notna()
    ]

    total_generation_tokens = int(
        df["output_tokens"].sum()
    )

    return {
        "total_tests": len(df),

        "passed_tests": int(df["passed"].sum()),

        "pass_rate": float(df["passed"].mean()),

        "tool_selection_accuracy": float(
            df["tool_selection_correct"].mean()
        ),

        "argument_accuracy": (
            float(tool_tests["argument_accuracy"].mean())
            if not tool_tests.empty
            else None
        ),

        "average_latency": float(
            df["latency"].mean()
        ),

        "median_latency": float(
            df["latency"].median()
        ),

        "p95_latency": float(
            df["latency"].quantile(0.95)
        ),

        "average_llm_calls": float(
            df["llm_calls"].mean()
        ),

        "total_input_tokens": int(
            df["input_tokens"].sum()
        ),

        "total_output_tokens": total_generation_tokens,

        "total_benchmark_time": float(
            df["latency"].sum()
        )
    }


def save_results(results, summary):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    json_path = (
        RESULTS_DIR / f"benchmark_{timestamp}.json"
    )

    csv_path = (
        RESULTS_DIR / f"benchmark_{timestamp}.csv"
    )

    report = {
        "timestamp": timestamp,
        "model": MODEL,
        "context_window": CONTEXT_WINDOW,
        "temperature": TEMPERATURE,
        "summary": summary,
        "results": results
    }

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    pd.DataFrame(results).to_csv(
        csv_path,
        index=False
    )

    print("\nResults saved:")
    print(json_path)
    print(csv_path)


def main():

    agent = FinancialAgent()

    results = []

    print("=== Financial Agent Benchmark ===")
    print(f"Model: {MODEL}")
    print(f"Total tests: {len(TEST_CASES)}")

    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        print(
            f"\n[{index}/{len(TEST_CASES)}] "
            f"{test_case['id']}"
        )

        print(test_case["prompt"], flush=True)

        result = run_test_case(
            agent,
            test_case
        )

        results.append(result)

        status = (
            "PASS" if result["passed"] else "FAIL"
        )

        print(f"Result: {status}")

        print(
            f"Latency: {result['latency']:.2f} sec"
        )

        print(
            f"LLM calls: {result['llm_calls']}"
        )

        print(
            f"Tool calls: {result['tool_calls']}"
        )

        if result["error"]:
            print(f"Error: {result['error']}")

    summary = summarize_results(results)

    print("\n=== Benchmark Summary ===")

    print(
        json.dumps(
            summary,
            indent=4,
            ensure_ascii=False
        )
    )

    save_results(
        results,
        summary
    )


if __name__ == "__main__":
    main()
