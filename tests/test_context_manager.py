
import json
import unittest

from copy import deepcopy

from src.memory.context_manager import (
    ContextManager,
    ContextBudgetError
)


class TestContextManager(unittest.TestCase):

    def setUp(self):

        self.manager = ContextManager(
            context_window=4096,
            reserved_output_tokens=1024,
            safety_margin=256,
            max_history_turns=2,
            max_tool_result_chars=100
        )

    def sample_messages(self):

        return [
            {
                "role": "system",
                "content": "Financial assistant"
            },
            {
                "role": "user",
                "content": "Question 1"
            },
            {
                "role": "assistant",
                "content": "Answer 1"
            },
            {
                "role": "user",
                "content": "Question 2"
            },
            {
                "role": "assistant",
                "content": "Answer 2"
            },
            {
                "role": "user",
                "content": "Question 3"
            },
            {
                "role": "assistant",
                "content": "Answer 3"
            },
            {
                "role": "user",
                "content": "Current question"
            }
        ]

    def test_input_budget(self):

        self.assertEqual(
            self.manager.input_budget,
            2816
        )

    def test_invalid_budget(self):

        with self.assertRaises(ValueError):

            ContextManager(
                context_window=100,
                reserved_output_tokens=100,
                safety_margin=10
            )

    def test_empty_token_estimation(self):

        self.assertEqual(
            self.manager.estimate_tokens(""),
            0
        )

    def test_token_estimation(self):

        tokens = self.manager.estimate_tokens(
            "Financial analysis"
        )

        self.assertGreater(tokens, 0)

    def test_non_ascii_estimation(self):

        tokens = self.manager.estimate_tokens(
            "금융 데이터 분석"
        )

        self.assertGreater(tokens, 0)

    def test_split_history(self):

        system, turns = (
            self.manager.split_history(
                self.sample_messages()
            )
        )

        self.assertEqual(len(system), 1)

        self.assertEqual(len(turns), 4)

    def test_system_prompt_preserved(self):

        messages = self.sample_messages()

        prepared = self.manager.prepare_messages(
            messages
        )

        self.assertEqual(
            prepared[0],
            messages[0]
        )

    def test_current_question_preserved(self):

        messages = self.sample_messages()

        prepared = self.manager.prepare_messages(
            messages
        )

        self.assertEqual(
            prepared[-1],
            messages[-1]
        )

    def test_sliding_window(self):

        prepared = self.manager.prepare_messages(
            self.sample_messages()
        )

        contents = [
            message["content"]
            for message in prepared
        ]

        self.assertNotIn(
            "Question 1",
            contents
        )

        self.assertIn(
            "Question 3",
            contents
        )

    def test_original_messages_unchanged(self):

        messages = self.sample_messages()

        original = deepcopy(messages)

        self.manager.prepare_messages(messages)

        self.assertEqual(
            messages,
            original
        )

    def test_tool_schema_budget(self):

        messages = self.sample_messages()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_stock"
                }
            }
        ]

        without_tools = (
            self.manager.estimate_context_tokens(
                messages
            )
        )

        with_tools = (
            self.manager.estimate_context_tokens(
                messages,
                tools
            )
        )

        self.assertGreater(
            with_tools,
            without_tools
        )

    def test_tool_result_compression(self):

        result = {
            "success": True,
            "tool": "analyze_stock",
            "data": {
                "symbol": "AAPL",
                "risk_and_performance": {
                    "sharpe_ratio": 1.23456789
                },
                "large_unused_field": "x" * 2000
            }
        }

        message = {
            "role": "tool",
            "content": json.dumps(result)
        }

        compressed = (
            self.manager.compress_tool_result(
                message
            )
        )

        data = json.loads(
            compressed["content"]
        )

        self.assertEqual(
            data["data"]["symbol"],
            "AAPL"
        )

        self.assertEqual(
            data["data"]["risk_and_performance"][
                "sharpe_ratio"
            ],
            1.23456789
        )

        self.assertNotIn(
            "large_unused_field",
            data["data"]
        )

        self.assertTrue(
            data["compression"]["compressed"]
        )

    def test_failed_tool_result_preserved(self):

        message = {
            "role": "tool",
            "content": json.dumps({
                "success": False,
                "error": "API unavailable"
            })
        }

        compressed = (
            self.manager.compress_tool_result(
                message,
                force=True
            )
        )

        self.assertEqual(
            compressed,
            message
        )

    def test_context_overflow(self):

        messages = [
            {
                "role": "system",
                "content": "Financial assistant"
            },
            {
                "role": "user",
                "content": "x" * 20000
            }
        ]

        with self.assertRaises(
            ContextBudgetError
        ):

            self.manager.prepare_messages(
                messages
            )

    def test_context_metrics(self):

        self.manager.prepare_messages(
            self.sample_messages()
        )

        metrics = self.manager.get_metrics()

        self.assertEqual(
            metrics["status"],
            "ready"
        )

        self.assertLessEqual(
            metrics["estimated_final_tokens"],
            metrics["input_budget"]
        )

        self.assertEqual(
            metrics["removed_history_turns"],
            1
        )

    def test_active_tool_turn_preserved(self):

        messages = [
            {
                "role": "system",
                "content": "Financial assistant"
            },
            {
                "role": "user",
                "content": "Analyze AAPL"
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "analyze_stock",
                            "arguments": {
                                "symbol": "AAPL"
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "success": True,
                    "data": {
                        "symbol": "AAPL"
                    }
                })
            }
        ]

        prepared = self.manager.prepare_messages(
            messages
        )

        self.assertEqual(
            len(prepared),
            4
        )

        self.assertEqual(
            prepared[-1]["role"],
            "tool"
        )

        self.assertIn(
            "tool_calls",
            prepared[-2]
        )


if __name__ == "__main__":

    unittest.main()
