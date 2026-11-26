from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
import unittest
from unittest.mock import patch

from src.tools.registry import ToolRegistry


class CacheRaceTests(unittest.TestCase):
    def test_duplicate_requests_share_external_fetch(self):
        joined = Event()
        started = Event()

        class ObservedFuture(Future):
            def result(self, *args, **kwargs):
                joined.set()
                return super().result(*args, **kwargs)

        def lookup(symbol):
            started.set()
            if not joined.wait(3):
                raise TimeoutError("second caller did not join")
            return {"price": 10}

        registry = ToolRegistry()
        registry.tools["get_latest_price"] = lookup
        with patch("src.tools.registry.Future", ObservedFuture), ThreadPoolExecutor(2) as pool:
            first = pool.submit(registry.execute, "get_latest_price", {"symbol":"ABC"})
            self.assertTrue(started.wait(3))
            second = pool.submit(registry.execute, "get_latest_price", {"symbol":" abc "})
            self.assertEqual(first.result()["data"], second.result()["data"])
        self.assertEqual(registry.get_cache_metrics()["shared_calls"], 1)
        self.assertEqual(registry.get_cache_metrics()["misses"], 1)

    def test_invalidation_during_fetch(self):
        started, release = Event(), Event()
        fetched = []

        def lookup(symbol):
            fetched.append(symbol)
            started.set()
            if not release.wait(3):
                raise TimeoutError("test stalled")
            return {"price": len(fetched)}

        registry = ToolRegistry()
        registry.tools["get_latest_price"] = lookup
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(registry.execute, "get_latest_price", {"symbol":"ABC"})
            self.assertTrue(started.wait(3))
            registry.clear_cache()
            release.set()
            self.assertEqual(first.result()["data"]["price"], 1)
        self.assertEqual(registry.execute("get_latest_price", {"symbol":"ABC"})["data"]["price"], 2)
