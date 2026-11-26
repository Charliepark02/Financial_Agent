"""Bounded retries at the external I/O boundary."""
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from curl_cffi.requests import exceptions as curl_errors
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError, YFInvalidPeriodError


class PlannerResponseError(ValueError):
    pass


class GenerationError(RuntimeError):
    pass


def classify(error):
    if isinstance(error, YFTickerMissingError):
        return "missing_market_data"
    if isinstance(error, YFInvalidPeriodError):
        return "invalid_arguments"
    if isinstance(error, ArithmeticError):
        return "numerical_failure"
    if isinstance(error, PlannerResponseError):
        return "invalid_plan"
    if isinstance(error, GenerationError):
        return "generation_failure"

    if type(error).__name__ == "InvalidRequest":
        return "invalid_request"
    if type(error).__name__ == "ContextBudgetError":
        return "context_overflow"
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None) or getattr(error, "status_code", None)
    if isinstance(error, YFRateLimitError) or status == 429:
        return "rate_limit"
    if isinstance(error, (TimeoutError, httpx.TimeoutException, curl_errors.Timeout)):
        return "timeout"
    if isinstance(error, (ConnectionError, httpx.NetworkError, curl_errors.ConnectionError)):
        return "connection"
    if isinstance(status, int) and 500 <= status < 600:
        return "provider_failure"
    if isinstance(error, ValueError):
        return "invalid_data"
    return "internal_error"


class RetryPolicy:
    def __init__(self, attempts=3, base=0.25, cap=4.0, sleep=time.sleep, jitter=random.random):
        if attempts < 1 or not 0 <= base <= cap:
            raise ValueError("Invalid retry configuration")
        self.attempts, self.base, self.cap = attempts, base, cap
        self.sleep, self.jitter = sleep, jitter

    def run(self, operation, metrics):
        metrics.update(retries=0, retry_delay=0.0)
        for index in range(self.attempts):
            try:
                return operation()
            except Exception as error:
                kind = classify(error)
                if kind not in {"timeout", "connection", "rate_limit", "provider_failure"} or index + 1 == self.attempts:
                    raise
                delay = min(self.cap, self.base * 2**index * (1 + self.jitter()))
                headers = getattr(getattr(error, "response", None), "headers", {})
                hint = headers.get("Retry-After")
                if hint:
                    try:
                        minimum = float(hint)
                    except ValueError:
                        try:
                            minimum = (parsedate_to_datetime(hint) - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            minimum = 0
                    if minimum > self.cap:
                        raise
                    delay = max(delay, minimum)
                metrics["retries"] += 1
                metrics["retry_delay"] += delay
                self.sleep(delay)
