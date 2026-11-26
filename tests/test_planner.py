
import json
import unittest

from unittest.mock import Mock
from types import SimpleNamespace

from src.agents.planner import FinancialPlanner


class TestFinancialPlanner(unittest.TestCase):

    def setUp(self):

        self.planner = FinancialPlanner()

    def sample_plan(self):

        return {
            "objective": "Analyze Apple",
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "Analyze AAPL",
                    "tool": "analyze_stock",
                    "arguments": {
                        "symbol": "AAPL",
                        "period": "1y"
                    }
                }
            ]
        }

    def test_valid_plan(self):

        plan = self.planner.validate_plan(
            self.sample_plan()
        )

        self.assertEqual(
            len(plan.tasks),
            1
        )

        self.assertEqual(
            plan.tasks[0].tool,
            "analyze_stock"
        )

    def test_empty_tasks(self):

        plan = self.planner.validate_plan({
            "objective": "Explain Sharpe Ratio",
            "tasks": []
        })

        self.assertEqual(
            len(plan.tasks),
            0
        )

    def test_invalid_json(self):

        with self.assertRaises(ValueError):

            self.planner.parse_json(
                "invalid JSON"
            )

    def test_markdown_json(self):

        content = (
            "```json\n"
            + json.dumps(self.sample_plan())
            + "\n```"
        )

        result = self.planner.parse_json(
            content
        )

        self.assertIn(
            "tasks",
            result
        )

    def test_missing_objective(self):

        raw_plan = self.sample_plan()

        del raw_plan["objective"]

        with self.assertRaises(ValueError):

            self.planner.validate_plan(
                raw_plan
            )

    def test_unknown_tool(self):

        raw_plan = self.sample_plan()

        raw_plan["tasks"][0]["tool"] = (
            "unknown_tool"
        )

        with self.assertRaises(ValueError):

            self.planner.validate_plan(
                raw_plan
            )

    def test_invalid_arguments(self):

        raw_plan = self.sample_plan()

        raw_plan["tasks"][0]["arguments"][
            "period"
        ] = "invalid"

        with self.assertRaises(ValueError):

            self.planner.validate_plan(
                raw_plan
            )

    def test_duplicate_task_id(self):

        raw_plan = self.sample_plan()

        raw_plan["tasks"].append({
            "task_id": "T1",
            "description": "Analyze NVDA",
            "tool": "analyze_stock",
            "arguments": {
                "symbol": "NVDA",
                "period": "1y"
            }
        })

        with self.assertRaises(ValueError):

            self.planner.validate_plan(
                raw_plan
            )

    def test_duplicate_tool_task(self):

        raw_plan = self.sample_plan()

        raw_plan["tasks"].append({
            "task_id": "T2",
            "description": "Analyze Apple again",
            "tool": "analyze_stock",
            "arguments": {
                "symbol": "aapl",
                "period": "1y"
            }
        })

        with self.assertRaises(ValueError):

            self.planner.validate_plan(
                raw_plan
            )

    def test_maximum_tasks(self):

        raw_plan = self.sample_plan()

        raw_plan["tasks"] = [
            {
                "task_id": f"T{i}",
                "description": "Analyze stock",
                "tool": "analyze_stock",
                "arguments": {
                    "symbol": f"TEST{i}"
                }
            }
            for i in range(
                self.planner.max_tasks + 1
            )
        ]

        with self.assertRaises(ValueError):

            self.planner.validate_plan(
                raw_plan
            )

    def test_symbol_normalization(self):

        raw_plan = self.sample_plan()

        raw_plan["tasks"][0]["arguments"][
            "symbol"
        ] = " aapl "

        plan = self.planner.validate_plan(
            raw_plan
        )

        self.assertEqual(
            plan.tasks[0].arguments["symbol"],
            "AAPL"
        )

    def test_default_arguments(self):

        raw_plan = self.sample_plan()

        raw_plan["tasks"][0]["arguments"] = {
            "symbol": "AAPL"
        }

        plan = self.planner.validate_plan(
            raw_plan
        )

        arguments = plan.tasks[0].arguments

        self.assertEqual(
            arguments["period"],
            "2y"
        )

        self.assertEqual(
            arguments["risk_free_rate"],
            0.04
        )

        self.assertEqual(
            arguments["confidence_level"],
            0.95
        )

    def test_create_plan_with_mock_llm(self):

        mock_llm = Mock()

        mock_llm.chat.return_value = (
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        self.sample_plan()
                    )
                )
            )
        )

        planner = FinancialPlanner(
            llm=mock_llm
        )

        plan = planner.create_plan(
            "Analyze Apple over 1 year."
        )

        self.assertEqual(
            len(plan.tasks),
            1
        )

        self.assertEqual(
            plan.tasks[0].arguments["symbol"],
            "AAPL"
        )

        mock_llm.chat.assert_called_once()


if __name__ == "__main__":

    unittest.main()
