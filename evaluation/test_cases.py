
TEST_CASES = [
    {
        "id": "T01",
        "category": "price_lookup",
        "prompt": "What is the latest stock price of Apple?",
        "expected_calls": [
            {
                "tool": "get_latest_price",
                "arguments": {"symbol": "AAPL"}
            }
        ]
    },
    {
        "id": "T02",
        "category": "price_lookup",
        "prompt": "엔비디아의 최신 주가를 알려줘.",
        "expected_calls": [
            {
                "tool": "get_latest_price",
                "arguments": {"symbol": "NVDA"}
            }
        ]
    },
    {
        "id": "T03",
        "category": "price_lookup",
        "prompt": "삼성전자의 최신 주가를 알려줘.",
        "expected_calls": [
            {
                "tool": "get_latest_price",
                "arguments": {"symbol": "005930.KS"}
            }
        ]
    },
    {
        "id": "T04",
        "category": "financial_analysis",
        "prompt": "애플의 최근 1년간 Sharpe Ratio를 계산해 줘.",
        "expected_calls": [
            {
                "tool": "analyze_stock",
                "arguments": {
                    "symbol": "AAPL",
                    "period": "1y"
                }
            }
        ]
    },
    {
        "id": "T05",
        "category": "financial_analysis",
        "prompt": "Analyze Microsoft's volatility over the past 2 years.",
        "expected_calls": [
            {
                "tool": "analyze_stock",
                "arguments": {
                    "symbol": "MSFT",
                    "period": "2y"
                }
            }
        ]
    },
    {
        "id": "T06",
        "category": "financial_analysis",
        "prompt": "엔비디아의 최근 6개월간 최대 낙폭을 분석해 줘.",
        "expected_calls": [
            {
                "tool": "analyze_stock",
                "arguments": {
                    "symbol": "NVDA",
                    "period": "6mo"
                }
            }
        ]
    },
    {
        "id": "T07",
        "category": "multi_asset",
        "prompt": "애플과 엔비디아의 최근 1년간 변동성을 비교해 줘.",
        "expected_calls": [
            {
                "tool": "analyze_stock",
                "arguments": {
                    "symbol": "AAPL",
                    "period": "1y"
                }
            },
            {
                "tool": "analyze_stock",
                "arguments": {
                    "symbol": "NVDA",
                    "period": "1y"
                }
            }
        ]
    },
    {
        "id": "T08",
        "category": "conceptual",
        "prompt": "Sharpe Ratio의 수학적 정의를 설명해 줘.",
        "expected_calls": []
    },
    {
        "id": "T09",
        "category": "conceptual",
        "prompt": "Explain the difference between simple and log returns.",
        "expected_calls": []
    },
    {
        "id": "T10",
        "category": "conceptual",
        "prompt": "Maximum Drawdown이 무엇인지 설명해 줘.",
        "expected_calls": []
    }
]
