import unittest
from unittest.mock import Mock
from src.recovery import RetryPolicy


class RecoveryTests(unittest.TestCase):
    def test_backoff_exhaustion(self):
        sleep = Mock()
        policy = RetryPolicy(sleep=sleep, jitter=lambda: 0)
        op = Mock(side_effect=TimeoutError())
        metrics = {}
        with self.assertRaises(TimeoutError):
            policy.run(op, metrics)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [.25, .5])
        self.assertEqual(metrics, {"retries": 2, "retry_delay": .75})

    def test_recovery_and_permanent_failure(self):
        policy = RetryPolicy(sleep=lambda _: None)
        self.assertEqual(policy.run(Mock(side_effect=[ConnectionError(), 7]), {}), 7)
        op = Mock(side_effect=ValueError())
        with self.assertRaises(ValueError):
            policy.run(op, {})
        op.assert_called_once()

    def test_provider_delay_larger_than_cap(self):
        import httpx
        response = httpx.Response(429, headers={"Retry-After":"30"})
        error = httpx.HTTPStatusError("rate limit", request=httpx.Request("GET","https://example.test"), response=response)
        operation = Mock(side_effect=error)
        with self.assertRaises(httpx.HTTPStatusError):
            RetryPolicy(sleep=Mock()).run(operation,{})
        operation.assert_called_once()

    def test_provider_retry_after_within_cap(self):
        import httpx
        err = httpx.HTTPStatusError("limited", request=httpx.Request("GET","https://example.test"),
                                    response=httpx.Response(429, headers={"Retry-After":"2"}))
        sleep = Mock()
        result = RetryPolicy(sleep=sleep).run(Mock(side_effect=[err, "ok"]), {})
        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(2)
