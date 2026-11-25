
import unittest

from unittest.mock import patch

from src.tools.registry import ToolRegistry


class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()

    def test_registered_tools(self):

        self.assertEqual(
            set(self.registry.list_tools()),
            {"get_latest_price", "analyze_stock"}
        )

    def test_unknown_tool(self):

        result = self.registry.execute(
            "unknown_tool",
            {}
        )

        self.assertFalse(result["success"])

    def test_missing_symbol(self):

        result = self.registry.execute(
            "get_latest_price",
            {}
        )

        self.assertFalse(result["success"])

    def test_invalid_symbol(self):

        result = self.registry.execute(
            "get_latest_price",
            {"symbol": ""}
        )

        self.assertFalse(result["success"])

    def test_invalid_period(self):

        result = self.registry.execute(
            "analyze_stock",
            {
                "symbol": "AAPL",
                "period": "invalid"
            }
        )

        self.assertFalse(result["success"])

    def test_invalid_risk_free_rate(self):

        result = self.registry.execute(
            "analyze_stock",
            {
                "symbol": "AAPL",
                "risk_free_rate": 2.0
            }
        )

        self.assertFalse(result["success"])

    def test_invalid_confidence_level(self):

        result = self.registry.execute(
            "analyze_stock",
            {
                "symbol": "AAPL",
                "confidence_level": 1.5
            }
        )

        self.assertFalse(result["success"])

    def test_unexpected_argument(self):

        result = self.registry.execute(
            "get_latest_price",
            {
                "symbol": "AAPL",
                "unknown": True
            }
        )

        self.assertFalse(result["success"])

    def test_symbol_normalization(self):

        arguments = self.registry.validate_arguments(
            "get_latest_price",
            {"symbol": " aapl "}
        )

        self.assertEqual(
            arguments["symbol"],
            "AAPL"
        )

    def test_default_arguments(self):

        arguments = self.registry.validate_arguments(
            "analyze_stock",
            {"symbol": "AAPL"}
        )

        self.assertEqual(arguments["period"], "2y")
        self.assertEqual(arguments["risk_free_rate"], 0.04)
        self.assertEqual(arguments["confidence_level"], 0.95)

    def test_successful_execution(self):

        mock_data = {
            "symbol": "AAPL",
            "price": 100.0
        }

        with patch(
            "src.tools.registry.get_latest_price",
            autospec=True,
            return_value=mock_data
        ) as mock_function:

            registry = ToolRegistry()

            result = registry.execute(
                "get_latest_price",
                {"symbol": "AAPL"}
            )

            mock_function.assert_called_once_with(
                symbol="AAPL"
            )

        self.assertTrue(result["success"])

        self.assertEqual(
            result["data"],
            mock_data
        )

    def test_external_api_failure(self):

        with patch(
            "src.tools.registry.get_latest_price",
            autospec=True,
            side_effect=RuntimeError("API unavailable")
        ) as mock_function:

            registry = ToolRegistry()

            result = registry.execute(
                "get_latest_price",
                {"symbol": "AAPL"}
            )

            mock_function.assert_called_once_with(
                symbol="AAPL"
            )

        self.assertFalse(result["success"])

        self.assertEqual(
            result["error_type"],
            "RuntimeError"
        )

        self.assertEqual(
            result["error"],
            "API unavailable"
        )


if __name__ == "__main__":
    unittest.main()
