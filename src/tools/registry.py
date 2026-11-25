
import inspect
import math
import re

from src.tools.market_data import get_latest_price
from src.tools.financial_analysis import analyze_stock
from src.tools.schemas import FINANCIAL_TOOL_SCHEMAS


class ToolRegistry:

    def __init__(self):

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

    def execute(
        self,
        name: str,
        arguments: dict
    ) -> dict:

        if name not in self.tools:
            return {
                "success": False,
                "tool": name,
                "error": "Unknown tool."
            }

        try:
            validated = self.validate_arguments(
                name,
                arguments
            )

            function = self.tools[name]

            result = function(**validated)

            return {
                "success": True,
                "tool": name,
                "arguments": validated,
                "data": result
            }

        except Exception as error:
            return {
                "success": False,
                "tool": name,
                "error_type": type(error).__name__,
                "error": str(error)
            }
