
import math
from datetime import datetime, timezone

from src.tools.market_data import get_historical_data

from src.tools.preprocessing import (
    clean_market_data,
    calculate_returns
)

from src.tools.indicators import (
    add_technical_indicators
)

from src.tools.risk_metrics import (
    calculate_risk_metrics,
    TRADING_DAYS
)


def sanitize_json(value):
    """
    Replace non-finite numbers with JSON-compatible nulls.
    """

    if isinstance(value, dict):
        return {
            key: sanitize_json(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            sanitize_json(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def analyze_stock(
    symbol: str,
    period: str = "2y",
    risk_free_rate: float = 0.04,
    confidence_level: float = 0.95
) -> dict:
    """
    Analyze a stock using historical daily market data.

    Args:
        symbol: Stock ticker symbol, such as AAPL or NVDA.
        period: Historical period, such as 1y or 2y.
        risk_free_rate: Assumed annual risk-free rate.
        confidence_level: Confidence level for VaR and ES.

    Returns:
        Structured financial analysis report.
    """

    symbol = symbol.strip().upper()

    # 1. Market data acquisition
    raw_data = get_historical_data(
        symbol=symbol,
        period=period,
        interval="1d"
    )

    # 2. Data preprocessing
    df = clean_market_data(raw_data)

    if len(df) < 30:
        raise ValueError(
            "At least 30 valid observations are required."
        )

    df = calculate_returns(df)

    # 3. Technical analysis
    df = add_technical_indicators(df)

    # 4. Risk and performance analysis
    risk_metrics = calculate_risk_metrics(
        df,
        risk_free_rate=risk_free_rate,
        confidence_level=confidence_level
    )

    latest = df.iloc[-1]

    # 5. Structured report
    report = {
        "symbol": symbol,
        "currency": raw_data.attrs.get("currency"),

        "data_source": "Yahoo Finance via yfinance",

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "data": {
            "requested_period": period,
            "interval": "1d",
            "price_type": "adjusted",
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "observations": len(df),
            "removed_observations": len(raw_data) - len(df)
        },

        "price": {
            "latest_close": float(latest["Close"]),
            "latest_volume": float(latest["Volume"])
        },

        "technical_indicators": {
            "sma_20": float(latest["SMA_20"]),
            "sma_50": float(latest["SMA_50"]),
            "sma_200": float(latest["SMA_200"]),

            "ema_12": float(latest["EMA_12"]),
            "ema_26": float(latest["EMA_26"]),
            "ema_20": float(latest["EMA_20"]),

            "rsi_14": float(latest["RSI_14"]),

            "macd": float(latest["MACD"]),
            "macd_signal": float(latest["MACD_Signal"]),
            "macd_histogram": float(
                latest["MACD_Histogram"]
            ),

            "bollinger_middle": float(latest["BB_Middle"]),
            "bollinger_upper": float(latest["BB_Upper"]),
            "bollinger_lower": float(latest["BB_Lower"])
        },

        "risk_and_performance": risk_metrics,

        "assumptions": {
            "trading_days_per_year": TRADING_DAYS,
            "annual_risk_free_rate": risk_free_rate,
            "confidence_level": confidence_level,
            "return_unit": "decimal",
            "var_method": "historical"
        },

        "warnings": [
            "Market data may be delayed or incomplete.",
            "The latest daily bar may be incomplete.",
            "Past performance does not guarantee future results.",
            "Risk-free rate is an assumed constant.",
            "Indicators may be unavailable for short periods."
        ]
    }

    return sanitize_json(report)
