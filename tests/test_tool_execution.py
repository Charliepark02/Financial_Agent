import unittest
from threading import Barrier
from unittest.mock import Mock

from src.tools.execution import execute_batch
from src.tools.registry import ToolRegistry


class ToolExecutionTests(unittest.TestCase):
    def test_normalized_cache_expiry_invalidation_and_isolation(self):
        clock = Mock(return_value=10)
        registry = ToolRegistry(cache_ttl=5, clock=clock)
        calls = []

        def price(symbol):
            calls.append(symbol)
            return {"prices": [10]}

        registry.tools["get_latest_price"] = price

        def lookup(symbol):
            return registry.execute("get_latest_price", {"symbol": symbol})["data"]

        lookup(" aapl ")["prices"].clear()
        self.assertEqual(lookup("AAPL"), {"prices": [10]})
        clock.return_value = 15
        lookup("AAPL")
        registry.clear_cache()
        lookup("AAPL")
        self.assertEqual(calls, ["AAPL"] * 3)
        self.assertEqual(registry.get_cache_metrics()["hit_ratio"], 0.25)

    def test_failed_execution_is_retried(self):
        registry = ToolRegistry()
        source = Mock(side_effect=[ValueError("offline"), 10])

        def price(symbol):
            return source(symbol)

        registry.tools["get_latest_price"] = price
        call = ("get_latest_price", {"symbol": "ABC"})
        self.assertFalse(registry.execute(*call)["success"])
        self.assertEqual(registry.execute(*call)["data"], 10)

    def test_independent_calls_overlap_and_keep_order(self):
        registry = ToolRegistry()
        barrier = Barrier(2, timeout=3)

        def price(symbol):
            barrier.wait()
            return symbol

        registry.tools["get_latest_price"] = price
        calls = [("get_latest_price", {"symbol": symbol}) for symbol in ("ABC", "XYZ")]
        results = execute_batch(registry, calls)
        self.assertEqual([result["data"] for result, _ in results], ["ABC", "XYZ"])
