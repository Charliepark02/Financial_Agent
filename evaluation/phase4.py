"""Offline scenario evaluation plus an optional controlled live cache/stream probe."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import unittest

from src.agents.financial_agent import FinancialAgent
from src.grounding import claim_matches
from src.memory.persistent import AnalysisArchive
from src.recovery import RetryPolicy

OUT = Path(__file__).parent / "results"


def suites():
    groups = []
    for pattern in ("test_phase4_flow.py", "test_grounding.py", "test_archive.py",
                    "test_stream.py", "test_recovery.py", "test_context_manager.py",
                    "test_tool_execution.py", "test_tool_registry.py"):
        suite = unittest.defaultTestLoader.discover("tests", pattern=pattern)
        for group in suite:
            for cls in group:
                for case in cls:
                    result = unittest.TestResult()
                    case.run(result)
                    groups.append({"case": case.id(), "passed": result.wasSuccessful(),
                                   "details": [detail for _, detail in result.errors + result.failures]})
    return groups


def probes():
    samples = [(.12, 12, "%", "total_return"), (-.23, -23, "%", "maximum_drawdown"),
               (.12, 18, "%", "total_return"), (1.5, 150, "%", "sharpe_ratio"),
               (None, 0, "ratio", "rsi_14")]
    decisions = [claim_matches(*sample) for sample in samples]
    with TemporaryDirectory() as tmp:
        archive = AnalysisArchive(Path(tmp) / "probe.sqlite3")
        # Explicit synthetic fixture confined to a temporary database.
        archive.save("fixture", [{"arguments":{"symbol":"TEST"}, "success":True,
                     "tool":"get_latest_price", "data":{"symbol":"TEST","price":1,
                     "date":"2020-01-01","data_source":"Yahoo Finance via yfinance"}}])
        expectations = [("TEST", 1), ("ABSENT", 0)]
        retrieval = [len(archive.search(ticker)) == count for ticker, count in expectations]
    attempts = iter([TimeoutError(), 1])
    def io():
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    recovery = {}
    RetryPolicy(sleep=lambda _: None).run(io, recovery)
    return {"claims_submitted": len(samples), "claims_with_reference": 4,
            "correct_supported_claims": sum(decisions), "unsupported_or_unavailable": len(samples)-sum(decisions),
            "classification_decisions_correct": sum(a == b for a,b in zip(decisions,[True,True,False,False,False])),
            "retrieval_queries":len(retrieval), "correct_retrievals":sum(retrieval),
            "transient_requests":1, "recovered_requests":1, "retry_metrics":recovery}


def live_probe(cold_model=False):
    import ollama
    agent = FinancialAgent()
    if cold_model:
        ollama.generate(model=agent.llm.model, prompt="", keep_alive=0)
    rows = []
    for index in range(2):
        agent.clear_history()
        resident = any(m.model == agent.llm.model for m in ollama.ps().models)
        before = agent.registry.get_cache_metrics()
        started = perf_counter()
        output = "".join(agent.stream_chat("What is the latest stock price of Apple?"))
        rows.append({"model_resident_at_start":resident, "cache_before":before,
                     "cache_after":agent.registry.get_cache_metrics(), "wall":perf_counter()-started,
                     "metrics":agent.get_metrics(), "answer":output, "tools":agent.get_tool_logs()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--cold-model", action="store_true", help="Unload the selected local model before the live pair")
    args = parser.parse_args()
    if args.cold_model and not args.live:
        parser.error("--cold-model requires --live")
    report = {"dataset":"phase4-v1", "timestamp":datetime.now(timezone.utc).isoformat(),
              "method":"offline fixtures and optional repeated live streaming price request",
              "cases":suites(), "fixtures":probes()}
    if args.live:
        report["stream_probes"] = live_probe(args.cold_model)
    path = OUT / ("phase4_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    path.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with path.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["case","passed","details"])
        writer.writeheader()
        writer.writerows(report["cases"])
    print(path, len(report["cases"]), "offline cases")
    if not all(case["passed"] for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
