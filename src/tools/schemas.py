
GET_LATEST_PRICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_latest_price",
        "description": (
            "Retrieve the latest available daily closing "
            "price of a stock. This is not a guaranteed "
            "real-time market quote."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "Stock ticker symbol, "
                        "such as AAPL, NVDA, or 005930.KS."
                    )
                }
            },
            "required": ["symbol"],
            "additionalProperties": False
        }
    }
}


ANALYZE_STOCK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_stock",
        "description": (
            "Analyze historical stock performance, "
            "technical indicators, and financial risk. "
            "Includes returns, SMA, EMA, RSI, MACD, "
            "Bollinger Bands, volatility, Sharpe Ratio, "
            "maximum drawdown, VaR and Expected Shortfall."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol."
                },
                "period": {
                    "type": "string",
                    "description": "Historical analysis period.",
                    "enum": [
                        "3mo",
                        "6mo",
                        "1y",
                        "2y",
                        "5y",
                        "10y"
                    ],
                    "default": "2y"
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": (
                        "Assumed annual risk-free rate "
                        "in decimal form. For example, "
                        "0.04 means 4 percent."
                    ),
                    "default": 0.04,
                    "minimum": 0,
                    "maximum": 1
                },
                "confidence_level": {
                    "type": "number",
                    "description": (
                        "Confidence level for historical "
                        "VaR and Expected Shortfall."
                    ),
                    "default": 0.95,
                    "enum": [0.90, 0.95, 0.99]
                }
            },
            "required": ["symbol"],
            "additionalProperties": False
        }
    }
}


FINANCIAL_TOOL_SCHEMAS = [
    GET_LATEST_PRICE_SCHEMA,
    ANALYZE_STOCK_SCHEMA
]
