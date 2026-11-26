from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from ollama import Message

from src.agents.financial_agent import FinancialAgent
from src.agents.planner import FinancialPlanner
from src.memory.persistent import AnalysisArchive
from src.recovery import RetryPolicy
from src.tools.registry import ToolRegistry


def message(text="", symbols=()):
    calls = [{"function": {"name": "get_latest_price", "arguments": {"symbol": symbol}}}
             for symbol in symbols]
    return SimpleNamespace(message=Message(role="assistant", content=text, tool_calls=calls or None))


def client(*answers):
    llm = Mock()
    llm.chat.side_effect = list(answers)
    llm.get_metrics.return_value = dict(input_tokens=1, output_tokens=1, generation_time=.1,
                                        end_to_end_latency=.1, load_time=0, prompt_time=0)
    return llm


def quote(symbol):
    if symbol == "BAD":
        raise ValueError("No market data")
    return dict(symbol=symbol, date="2020-01-01", price=10, currency="USD",
                data_source="Yahoo Finance via yfinance")


class Phase4FlowTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(retry=RetryPolicy(sleep=lambda _: None))
        self.registry.tools["get_latest_price"] = quote

    def test_partial_batch_and_fabricated_synthesis(self):
        agent = FinancialAgent(client(message(symbols=["ABC", "BAD"]), message("999 USD")), self.registry)
        answer = agent.chat("Compare ABC and BAD")
        self.assertIn("10 USD", answer)
        self.assertNotIn("999", answer)
        self.assertIn("unavailable", answer)
        self.assertEqual([x["success"] for x in agent.get_tool_logs()], [True, False])

    def test_stream_tool_limit(self):
        agent = FinancialAgent(client(message(symbols=["ABC", "BAD"]), message("999")), self.registry)
        agent.max_tool_calls = 1
        answer = "".join(agent.stream_chat("Compare"))
        self.assertIn("10 USD", answer)
        self.assertEqual(agent.state.tool_call_count, 1)
        self.assertIsNotNone(agent.get_metrics()["first_visible_seconds"])

    def test_generation_failure_terminal_state(self):
        agent = FinancialAgent(client(RuntimeError("Generation failure")), self.registry)
        with self.assertRaises(RuntimeError):
            agent.chat("Hello")
        self.assertEqual(agent.state.status.value, "failed")

    def test_planner_recovers_without_execution(self):
        planner = FinancialPlanner(client(message("bad"), message('{"objective":"Explain risk","tasks":[]}')), self.registry)
        self.assertFalse(planner.create_plan("Explain risk").tasks)
        self.assertEqual(planner.llm.chat.call_count, 2)

    def test_short_term_history(self):
        llm = client(message("One"), message("Two"))
        agent = FinancialAgent(llm, self.registry)
        agent.chat("First")
        agent.chat("Second")
        sent = llm.chat.call_args.kwargs["messages"]
        self.assertTrue(any(row.get("content") == "One" for row in sent))

    def test_history_after_restart_and_live_lookup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "analyses.sqlite3"
            agent = FinancialAgent(client(message(symbols=["ABC"]), message("done")), self.registry,
                                   archive=AnalysisArchive(path))
            agent.chat("quote")
            new = FinancialAgent(client(message(symbols=["ABC"]), message("done")), ToolRegistry(),
                                 archive=AnalysisArchive(path))
            new.registry.tools["get_latest_price"] = quote
            self.assertEqual(len(new.archive.search("ABC")), 1)
            new.chat("latest")
            self.assertEqual(new.registry.get_cache_metrics()["misses"], 1)
            self.assertEqual(len(new.archive.search("ABC")), 2)

    def test_invalid_input_terminal_and_memory_disabled(self):
        agent = FinancialAgent(client(), self.registry)
        with self.assertRaises(ValueError):
            agent.chat(" " * 17000)
        self.assertEqual(agent.state.status.value, "failed")
        self.assertIn("invalid_request", agent.state.error)
        self.assertIsNone(agent.archive)

    def test_invalid_plan_exhausts_before_tool_execution(self):
        planner = FinancialPlanner(client(message("bad"), message("bad again")), self.registry)
        with self.assertRaisesRegex(ValueError, "regeneration"):
            planner.create_plan("Analyze ABC")
        self.assertEqual(planner.llm.chat.call_count, 2)
        self.assertEqual(self.registry.get_cache_metrics()["misses"], 0)

    def test_provider_timeout_after_success_preserves_verified_observation(self):
        agent = FinancialAgent(client(message(symbols=["ABC"]), TimeoutError()), self.registry)
        answer = agent.chat("Check ABC")
        self.assertIn("10 USD", answer)
        self.assertIn("Partial report", answer)
        self.assertEqual(agent.state.status.value, "completed")
        self.assertEqual(self.registry.get_cache_metrics()["misses"], 1)
