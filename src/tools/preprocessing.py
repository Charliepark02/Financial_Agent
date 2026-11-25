
import numpy as np
import pandas as pd


OHLCV_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


def clean_market_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean and validate OHLCV market data.
    """

    missing_columns = [
        column
        for column in OHLCV_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[OHLCV_COLUMNS].copy()

    data = data.sort_index()

    data = data.loc[
        ~data.index.duplicated(keep="last")
    ]

    for column in OHLCV_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=OHLCV_COLUMNS
    )

    valid_prices = (
        data[["Open", "High", "Low", "Close"]] > 0
    ).all(axis=1)

    valid_volume = data["Volume"] >= 0

    valid_high = (
        data["High"]
        >= data[["Open", "Close", "Low"]].max(axis=1)
    )

    valid_low = (
        data["Low"]
        <= data[["Open", "Close", "High"]].min(axis=1)
    )

    data = data.loc[
        valid_prices
        & valid_volume
        & valid_high
        & valid_low
    ].copy()

    if len(data) < 2:
        raise ValueError(
            "Insufficient valid market data."
        )

    return data


def calculate_returns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate simple, logarithmic and cumulative returns.
    """

    data = df.copy()

    data["Simple_Return"] = (
        data["Close"].pct_change(fill_method=None)
    )

    data["Log_Return"] = (
        np.log(data["Close"]).diff()
    )

    data["Cumulative_Return"] = (
        (1 + data["Simple_Return"]).cumprod() - 1
    )

    data.loc[
        data.index[0],
        "Cumulative_Return"
    ] = 0.0

    return data
