
import unittest

import numpy as np
import pandas as pd

from src.tools.preprocessing import (
    clean_market_data,
    calculate_returns
)

from src.tools.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd
)

from src.tools.risk_metrics import (
    calculate_max_drawdown,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_var,
    calculate_expected_shortfall
)


class TestFinancialTools(unittest.TestCase):

    def test_simple_return(self):
        df = pd.DataFrame({
            "Close": [100.0, 110.0, 121.0]
        })

        result = calculate_returns(df)

        self.assertAlmostEqual(
            result["Simple_Return"].iloc[1],
            0.10
        )

        self.assertAlmostEqual(
            result["Cumulative_Return"].iloc[-1],
            0.21
        )

    def test_log_return(self):
        df = pd.DataFrame({
            "Close": [100.0, 110.0]
        })

        result = calculate_returns(df)

        self.assertAlmostEqual(
            result["Log_Return"].iloc[-1],
            np.log(1.1)
        )

    def test_sma(self):
        close = pd.Series([
            100.0, 110.0, 120.0
        ])

        sma = calculate_sma(close, 3)

        self.assertAlmostEqual(
            sma.iloc[-1],
            110.0
        )

    def test_ema(self):
        close = pd.Series([
            100.0, 110.0, 120.0
        ])

        ema = calculate_ema(close, 3)

        self.assertAlmostEqual(
            ema.iloc[-1],
            112.5
        )

    def test_rsi_uptrend(self):
        close = pd.Series(
            np.arange(100.0, 130.0)
        )

        rsi = calculate_rsi(close)

        self.assertAlmostEqual(
            rsi.iloc[-1],
            100.0
        )

    def test_rsi_downtrend(self):
        close = pd.Series(
            np.arange(130.0, 100.0, -1.0)
        )

        rsi = calculate_rsi(close)

        self.assertAlmostEqual(
            rsi.iloc[-1],
            0.0
        )

    def test_rsi_flat_price(self):
        close = pd.Series([100.0] * 30)

        rsi = calculate_rsi(close)

        self.assertAlmostEqual(
            rsi.iloc[-1],
            50.0
        )

    def test_macd_flat_price(self):
        close = pd.Series([100.0] * 50)

        macd = calculate_macd(close)

        self.assertAlmostEqual(
            macd["MACD"].iloc[-1],
            0.0
        )

    def test_max_drawdown(self):
        close = pd.Series([
            100.0, 150.0, 100.0, 160.0
        ])

        mdd = calculate_max_drawdown(close)

        self.assertAlmostEqual(
            mdd,
            -1 / 3
        )

    def test_volatility(self):
        returns = pd.Series([
            -0.01, 0.0, 0.01
        ])

        volatility = calculate_volatility(returns)

        self.assertAlmostEqual(
            volatility,
            0.01 * np.sqrt(252)
        )

    def test_sharpe_ratio(self):
        returns = pd.Series([
            0.0, 0.01, 0.02
        ])

        sharpe = calculate_sharpe_ratio(
            returns,
            risk_free_rate=0.0
        )

        self.assertAlmostEqual(
            sharpe,
            np.sqrt(252)
        )

    def test_var_and_es(self):
        returns = pd.Series([
            -0.05, -0.03, -0.01, 0.01, 0.03
        ])

        var = calculate_var(
            returns,
            confidence_level=0.75
        )

        es = calculate_expected_shortfall(
            returns,
            confidence_level=0.75
        )

        self.assertAlmostEqual(var, 0.03)
        self.assertAlmostEqual(es, 0.04)

    def test_clean_market_data(self):
        df = pd.DataFrame({
            "Open": [100.0, 110.0, -10.0],
            "High": [105.0, 115.0, 5.0],
            "Low": [95.0, 105.0, -20.0],
            "Close": [102.0, 112.0, -5.0],
            "Volume": [1000, 2000, 3000]
        })

        result = clean_market_data(df)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
