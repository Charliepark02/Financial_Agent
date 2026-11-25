
import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_volatility(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS
) -> float:
    """
    Calculate annualized return volatility.
    """

    returns = returns.dropna()

    return float(
        returns.std(ddof=1) * np.sqrt(trading_days)
    )


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = TRADING_DAYS
) -> float:
    """
    Calculate annualized Sharpe Ratio.
    """

    if risk_free_rate <= -1:
        raise ValueError("Invalid risk-free rate.")

    returns = returns.dropna()

    daily_rf = (
        (1 + risk_free_rate) ** (1 / trading_days)
        - 1
    )

    excess_returns = returns - daily_rf

    std = excess_returns.std(ddof=1)

    if not np.isfinite(std) or std <= 0:
        return float("nan")

    sharpe = (
        excess_returns.mean()
        / std
        * np.sqrt(trading_days)
    )

    return float(sharpe)


def calculate_max_drawdown(
    close: pd.Series
) -> float:
    """
    Calculate maximum drawdown as a nonpositive return.
    """

    running_peak = close.cummax()

    drawdown = close / running_peak - 1

    return float(drawdown.min())


def calculate_cagr(
    close: pd.Series
) -> float:
    """
    Calculate compound annual growth rate.
    """

    if len(close) < 2:
        return float("nan")

    years = (
        close.index[-1] - close.index[0]
    ).total_seconds() / (365.25 * 24 * 60 * 60)

    if years <= 0:
        return float("nan")

    cagr = (
        (close.iloc[-1] / close.iloc[0])
        ** (1 / years) - 1
    )

    return float(cagr)


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate historical Value at Risk.
    """

    if not 0 < confidence_level < 1:
        raise ValueError("Invalid confidence level.")

    losses = -returns.dropna()

    return float(
        losses.quantile(confidence_level)
    )


def calculate_expected_shortfall(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Estimate Expected Shortfall using historical tail losses.
    """

    losses = -returns.dropna()

    var = calculate_var(
        returns,
        confidence_level
    )

    tail_losses = losses[losses >= var]

    return float(tail_losses.mean())


def calculate_risk_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    confidence_level: float = 0.95
) -> dict:
    """
    Calculate performance and risk metrics from daily data.
    """

    close = df["Close"]

    returns = df["Simple_Return"].dropna()

    total_return = (
        close.iloc[-1] / close.iloc[0] - 1
    )

    annualized_mean_return = (
        returns.mean() * TRADING_DAYS
    )

    return {
        "total_return": float(total_return),

        "annualized_mean_return": float(
            annualized_mean_return
        ),

        "cagr": calculate_cagr(close),

        "annualized_volatility": calculate_volatility(
            returns
        ),

        "sharpe_ratio": calculate_sharpe_ratio(
            returns,
            risk_free_rate
        ),

        "maximum_drawdown": calculate_max_drawdown(
            close
        ),

        "daily_var": calculate_var(
            returns,
            confidence_level
        ),

        "daily_expected_shortfall":
            calculate_expected_shortfall(
                returns,
                confidence_level
            )
    }
