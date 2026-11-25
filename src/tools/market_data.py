
import yfinance as yf
import pandas as pd


def get_historical_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Retrieve historical adjusted OHLCV market data.
    """

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=True,
        raise_errors=True
    )

    if df.empty:
        raise ValueError(
            f"No market data found for {symbol}."
        )

    return df


def get_latest_price(symbol: str) -> dict:
    """
    Retrieve the latest available unadjusted daily close.
    """

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period="5d",
        interval="1d",
        auto_adjust=False,
        raise_errors=True
    )

    if df.empty:
        raise ValueError(
            f"No price data found for {symbol}."
        )

    close = df["Close"].dropna()

    if close.empty:
        raise ValueError("No valid closing price.")

    metadata = ticker.history_metadata

    return {
        "symbol": symbol,
        "price": float(close.iloc[-1]),
        "currency": metadata.get("currency"),
        "date": str(close.index[-1].date()),
        "price_type": "latest_available_daily_close",
        "data_source": "Yahoo Finance via yfinance"
    }
