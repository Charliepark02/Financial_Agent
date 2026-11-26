
from concurrent.futures import Future
from copy import deepcopy
from threading import Lock
from time import monotonic

import inspect
import math
import re

from src.recovery import RetryPolicy, classify
from src.tools.market_data import get_latest_price
from src.tools.financial_analysis import analyze_stock
from src.tools.schemas import FINANCIAL_TOOL_SCHEMAS


class ToolRegistry:

    def __init__(self, cache_ttl=60.0, clock=monotonic, retry=None):
        self.retry = retry or RetryPolicy()
        if not math.isfinite(cache_ttl) or cache_ttl < 0:
            raise ValueError("Cache TTL must be finite and nonnegative.")
        self.cache_ttl = cache_ttl
        self._clock = clock
        self._cache = {}
        self._pending = {}
        self._lock = Lock()
        self._hits = self._misses = self._shared = 0


        self.tools = {
            "get_latest_price": get_latest_price,
            "analyze_stock": analyze_stock
        }

        self.schemas = FINANCIAL_TOOL_SCHEMAS

    def get_schemas(self) -> list:
        return self.schemas

    def list_tools(self) -> list:
        return list(self.tools.keys())

    def validate_arguments(
        self,
        name: str,
        arguments: dict
    ) -> dict:

        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a dictionary.")

        function = self.tools[name]

        # Validate required and unexpected arguments
        signature = inspect.signature(function)
        bound = signature.bind(**arguments)
        bound.apply_defaults()

        validated = dict(bound.arguments)

        # Validate stock ticker
        symbol = validated.get("symbol")

        if not isinstance(symbol, str):
            raise ValueError("Symbol must be a string.")

        symbol = symbol.strip().upper()

        if not re.fullmatch(
            r"[A-Z0-9.^=\-]{1,20}",
            symbol
        ):
            raise ValueError("Invalid ticker symbol.")

        validated["symbol"] = symbol

        # Validate analysis parameters
        if name == "analyze_stock":

            allowed_periods = {
                "3mo", "6mo", "1y",
                "2y", "5y", "10y"
            }

            period = validated["period"]

            if (
                not isinstance(period, str)
                or period not in allowed_periods
            ):
                raise ValueError("Invalid analysis period.")

            rate = validated["risk_free_rate"]

            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(rate)
                or not 0 <= rate <= 1
            ):
                raise ValueError("Invalid risk-free rate.")

            confidence = validated["confidence_level"]

            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or confidence not in (0.90, 0.95, 0.99)
            ):
                raise ValueError("Invalid confidence level.")

        return validated

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            # In-flight readers may finish, but cannot repopulate this cache.
            self._pending.clear()

    def get_cache_metrics(self):
        with self._lock:
            total = self._hits + self._misses
            return {"hits": self._hits, "misses": self._misses,
                    "shared_calls": self._shared,
                    "hit_ratio": self._hits / total if total else 0.0}

    def _cached_call(self, name, arguments, metrics):
        started = monotonic()
        key = (name, tuple(sorted(arguments.items())))
        with self._lock:
            now = self._clock()
            for expired in [k for k, (expiry, _) in self._cache.items() if expiry <= now]:
                del self._cache[expired]
            metrics["cache_lookup_seconds"] = monotonic() - started
            if key in self._cache:
                self._hits += 1
                return deepcopy(self._cache[key][1])
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                self._misses += 1
                pending = self._pending[key] = Future()
            else:
                self._shared += 1

        if not owner:
            return deepcopy(pending.result())
        try:
            data = self.retry.run(lambda: self.tools[name](**arguments), metrics)
            snapshot = deepcopy(data)
            with self._lock:
                if self._pending.get(key) is pending and self.cache_ttl:
                    self._cache[key] = (self._clock() + self.cache_ttl, snapshot)
                pending.set_result(snapshot)
            return data
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self._lock:
                if self._pending.get(key) is pending:
                    del self._pending[key]

    def execute(
        self,
        name: str,
        arguments: dict
    ) -> dict:

        if name not in self.tools:
            return {
                "success": False,
                "tool": name,
                "error_code": "unsupported_tool",
                "error": "Unknown tool."
            }

        metrics = {"retries": 0, "retry_delay": 0.0}
        validated = None
        try:
            validated = self.validate_arguments(
                name,
                arguments
            )

            result = self._cached_call(name, validated, metrics)

            return {
                "success": True,
                "tool": name,
                "arguments": validated,
                "recovery": metrics,
                "data": result
            }

        except Exception as error:
            return {
                "success": False,
                "tool": name,
                "arguments": validated or {},
                "error_code": (classify(error) if validated is not None else
                               "invalid_symbol" if "symbol" in str(error).lower() else "invalid_arguments"),
                "recovery": metrics,
                "error_type": type(error).__name__,
                "error": str(error)
            }
