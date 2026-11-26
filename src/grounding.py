"""Render registered financial fields only. Provider prose is never executable."""
import math
import re
from datetime import date, timedelta

PERCENT = {"total_return", "annualized_mean_return", "cagr", "annualized_volatility",
           "maximum_drawdown", "daily_var", "daily_expected_shortfall"}
RATIOS = {"sharpe_ratio", "rsi_14"}


class InvalidRequest(ValueError):
    pass


def validate_request(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
        raise InvalidRequest("Expected a non-empty request of at most 16000 characters.")
    return text.strip()


def claim_matches(actual, displayed, unit, metric, digits=2):
    if actual is None or not all(type(x) in (int, float) and math.isfinite(x) for x in (actual, displayed)):
        return False
    if metric not in PERCENT | RATIOS or unit != ("%" if metric in PERCENT else "ratio"):
        return False
    expected = actual * 100 if unit == "%" else actual
    return abs(expected - displayed) <= .5 * 10**-digits + 1e-12


def render_result(result):
    args = result.get("arguments", {})
    symbol = args.get("symbol", "")
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9.^=\-]{1,20}", symbol):
        symbol = "?"
    if not result.get("success"):
        return f"{symbol} · data unavailable"
    data = result["data"]
    if data["symbol"] != symbol or data["data_source"] != "Yahoo Finance via yfinance":
        raise ValueError("Unverified source or symbol")
    latest = result["tool"] == "get_latest_price"
    day = date.fromisoformat(data["date"] if latest else data["data"]["end_date"])
    if day > date.today():
        raise ValueError("Future observation date")
    close = data["price"] if latest else data["price"]["latest_close"]
    if type(close) not in (float, int) or not math.isfinite(close) or close <= 0:
        raise ValueError("Invalid closing price")
    currency = data.get("currency")
    if currency is not None and not re.fullmatch(r"[A-Za-z]{3}", currency):
        raise ValueError("Invalid currency")
    lines = [f"{symbol} · {day.isoformat()} · Yahoo Finance via yfinance",
             f"{'Unadjusted' if latest else 'Adjusted'} daily close: {close:g} {currency or '(currency unavailable)'}"]
    if day < date.today() - timedelta(days=7):
        lines.append("Historical observation more than seven calendar days old.")
    if not latest:
        assumptions = data["assumptions"]
        if assumptions["return_unit"] != "decimal" or data["data"]["requested_period"] != args["period"]:
            raise ValueError("Invalid analysis provenance")
        if (assumptions["annual_risk_free_rate"] != args["risk_free_rate"] or
                assumptions["confidence_level"] != args["confidence_level"] or
                data["data"]["price_type"] != "adjusted" or data["data"]["observations"] < 30):
            raise ValueError("Analysis assumptions or sample are inconsistent")
        lines.append(f"Period {args['period']} · risk-free {args['risk_free_rate']:.2%} · confidence {args['confidence_level']:.0%} · {int(assumptions['trading_days_per_year'])} trading days/year")
        series = {**data["risk_and_performance"], **data["technical_indicators"]}
        for metric in sorted(PERCENT | RATIOS):
            value = series.get(metric)
            if value is None:
                formatted = "unavailable"
            elif type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Invalid financial value")
            else:
                formatted = format(value, ".2%" if metric in PERCENT else ".2f")
            lines.append(f"{metric}: {formatted}")
    return "\n".join(lines)


def verified_answer(results):
    lines = []
    for result in results:
        try:
            lines.append(render_result(result))
        except (KeyError, ValueError, TypeError, OverflowError):
            lines.append("Financial result unavailable: validation failed.")
    return "\n\n".join(lines)


def check_claim(claim, results):
    """Compare one typed claim to a matching asset, date, period and unit."""
    for item in results:
        if not item.get("success"):
            continue
        try:
            render_result(item)
            data = item["data"]
            latest = item["tool"] == "get_latest_price"
            day = data["date"] if latest else data["data"]["end_date"]
            if (claim["symbol"] != data["symbol"] or claim["date"] != day or
                    claim.get("period") != item["arguments"].get("period")):
                continue
            metric = claim["metric"]
            if metric == "daily_close":
                expected = data["price"] if latest else data["price"]["latest_close"]
                value = claim["value"]
                if claim.get("basis") != ("unadjusted" if latest else "adjusted") or not data.get("currency"):
                    continue
                if claim["unit"] != data["currency"]:
                    return "unsupported"
                return "verified" if type(value) in (int,float) and math.isfinite(value) and abs(value-expected) <= .005000000001 else "unsupported"
            fields = data.get("risk_and_performance", {}) | data.get("technical_indicators", {})
            if fields.get(metric) is None:
                continue
            if claim_matches(fields[metric], claim["value"], claim["unit"], metric):
                return "verified"
            return "unsupported"
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    return "unavailable"


def safe_explanation(text):
    """Flag explicit execution/guarantee assertions in ungrounded prose."""
    patterns = (
        r"\bI (?:have )?(?:bought|sold|executed|placed|submitted)\b",
        r"\b(?:I|we) guarantee (?:a |your |positive )?(?:returns?|profits?)\b",
        r"(?:주문을? (?:체결|실행|완료)했|수익을 보장(?:합니다|해요))",
    )
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
        return "This analysis agent can explain market data. It cannot confirm trades or guarantee returns."
    return text
