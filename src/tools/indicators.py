
import numpy as np
import pandas as pd


def calculate_sma(
    close: pd.Series,
    period: int
) -> pd.Series:
    """
    Calculate Simple Moving Average.
    """

    if period < 1:
        raise ValueError("Period must be positive.")

    return close.rolling(window=period).mean()


def calculate_ema(
    close: pd.Series,
    period: int
) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    """

    if period < 1:
        raise ValueError("Period must be positive.")

    return close.ewm(
        span=period,
        adjust=False
    ).mean()


def calculate_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing method.
    """

    if period < 1:
        raise ValueError("Period must be positive.")

    if len(close) <= period:
        return pd.Series(
            np.nan,
            index=close.index,
            dtype=float
        )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(
        np.nan,
        index=close.index,
        dtype=float
    )

    avg_loss = avg_gain.copy()

    avg_gain.iloc[period] = (
        gain.iloc[1:period + 1].mean()
    )

    avg_loss.iloc[period] = (
        loss.iloc[1:period + 1].mean()
    )

    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (
            avg_gain.iloc[i - 1] * (period - 1)
            + gain.iloc[i]
        ) / period

        avg_loss.iloc[i] = (
            avg_loss.iloc[i - 1] * (period - 1)
            + loss.iloc[i]
        ) / period

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0),
        50
    )

    return rsi


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD, signal line and histogram.
    """

    if not 0 < fast < slow or signal < 1:
        raise ValueError("Invalid MACD periods.")

    fast_ema = calculate_ema(close, fast)
    slow_ema = calculate_ema(close, slow)

    macd = fast_ema - slow_ema

    signal_line = calculate_ema(macd, signal)

    histogram = macd - signal_line

    return pd.DataFrame({
        "MACD": macd,
        "MACD_Signal": signal_line,
        "MACD_Histogram": histogram
    })


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    """

    if period < 1 or num_std <= 0:
        raise ValueError("Invalid Bollinger Band parameters.")

    middle = close.rolling(period).mean()

    std = close.rolling(period).std(ddof=0)

    return pd.DataFrame({
        "BB_Middle": middle,
        "BB_Upper": middle + num_std * std,
        "BB_Lower": middle - num_std * std
    })


def add_technical_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add technical indicators to market data.
    """

    data = df.copy()

    close = data["Close"]

    data["SMA_20"] = calculate_sma(close, 20)
    data["SMA_50"] = calculate_sma(close, 50)
    data["SMA_200"] = calculate_sma(close, 200)

    data["EMA_12"] = calculate_ema(close, 12)
    data["EMA_26"] = calculate_ema(close, 26)
    data["EMA_20"] = calculate_ema(close, 20)

    data["RSI_14"] = calculate_rsi(close, 14)

    macd = calculate_macd(close)

    bands = calculate_bollinger_bands(close)

    data = data.join(macd)
    data = data.join(bands)

    return data
