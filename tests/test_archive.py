import tempfile
import unittest
from pathlib import Path
from src.memory.persistent import AnalysisArchive
from src.memory.context_manager import ContextManager


class ArchiveTests(unittest.TestCase):
    def test_restart_filters_budget_and_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.sqlite3"
            first = AnalysisArchive(path)
            item = {"tool": "get_latest_price", "arguments": {"symbol": "ABC"}, "success": True,
                    "data": {"symbol": "ABC", "price": 10, "date": "2020-01-01",
                             "data_source": "Yahoo Finance via yfinance"}}
            first.save("request", [item])
            second = AnalysisArchive(path)
            self.assertFalse(second.search(session=second.session))
            self.assertFalse(second.search("ABC", since="2021-01-01"))
            rows = second.search("abc")
            self.assertIn("Historical", ContextManager().historical_context(rows))
            self.assertEqual(ContextManager().historical_context(rows, 2), "")
            with self.assertRaises(ValueError):
                second.clear("no")
            self.assertEqual(second.delete(rows[0]["id"]), 1)
            self.assertFalse(second.search())

    def test_rollback_on_storage_failure(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            archive = AnalysisArchive(Path(tmp) / "data.sqlite3")
            item = {"success":True, "arguments":{"symbol":"ABC"}, "tool":"get_latest_price",
                    "data":{"symbol":"ABC","price":1,"date":"2020-01-01","data_source":"Yahoo Finance via yfinance"}}
            with sqlite3.connect(archive.path) as db:
                db.execute("""CREATE TRIGGER fail_second BEFORE INSERT ON analyses
                    WHEN (SELECT COUNT(*) FROM analyses)>0 BEGIN SELECT RAISE(ABORT,'test rollback'); END""")
            with self.assertRaises(sqlite3.IntegrityError):
                archive.save("request", [item, item])
            self.assertFalse(archive.search())
