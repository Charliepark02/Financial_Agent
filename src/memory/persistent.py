"""Opt-in local archive of verified historical financial results."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.grounding import PERCENT, RATIOS, render_result


class AnalysisArchive:
    def __init__(self, path, session=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.session = session or str(uuid4())
        with closing(sqlite3.connect(path)) as db, db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported archive version")
            db.execute("""CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY, session TEXT NOT NULL, request TEXT NOT NULL,
                created TEXT NOT NULL, symbol TEXT NOT NULL, observed TEXT NOT NULL,
                category TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS by_symbol_date ON analyses(symbol, observed)")
            db.execute("PRAGMA user_version=1")

    def save(self, request, results):
        with closing(sqlite3.connect(self.path)) as db, db:
            for item in results:
                if not item.get("success"):
                    continue
                try:
                    report = render_result(item)
                except (ValueError, TypeError, KeyError, OverflowError):
                    continue
                data = item["data"]
                observed = data.get("date") or data["data"]["end_date"]
                # Only persist renderer-allowlisted data. No prompt, provider prose or credentials.
                payload = {"report": report, "source": data["data_source"]}
                fields = data.get("risk_and_performance", {}) | data.get("technical_indicators", {})
                payload["values"] = {key: fields[key] for key in PERCENT | RATIOS if key in fields}
                payload["arguments"] = {key: item["arguments"][key] for key in ("symbol", "period", "risk_free_rate", "confidence_level") if key in item["arguments"]}
                payload["currency"] = data.get("currency")
                payload["price_type"] = "daily_unadjusted" if item["tool"] == "get_latest_price" else "daily_adjusted"
                payload["close"] = data["price"] if item["tool"] == "get_latest_price" else data["price"]["latest_close"]
                db.execute("INSERT INTO analyses VALUES(NULL,?,?,?,?,?,?,?)",
                           (self.session, request, datetime.now(timezone.utc).isoformat(),
                            data["symbol"], observed, item["tool"], json.dumps(payload, allow_nan=False)))

    def search(self, symbol="", since="", session=None, limit=10):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("""
                SELECT * FROM analyses WHERE (?='' OR symbol=?) AND observed>=?
                AND (? IS NULL OR session=?) ORDER BY id DESC LIMIT ?""",
                (symbol, symbol.upper(), since, session, session, min(max(limit, 1), 50)))]

    def delete(self, record_id):
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute("DELETE FROM analyses WHERE id=?", (record_id,)).rowcount

    def clear(self, confirmation):
        if confirmation != "CONFIRM":
            raise ValueError("Use memory clear CONFIRM")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM analyses")
