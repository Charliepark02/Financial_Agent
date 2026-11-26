from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from src.LLM.client import LLMClient


def chunk(text, done):
    return SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=[]), done=done,
                           total_duration=1, load_duration=0, prompt_eval_duration=1,
                           eval_duration=1, prompt_eval_count=2, eval_count=2)


class StreamTests(unittest.TestCase):
    def test_buffered_stream_and_timing(self):
        client = LLMClient()
        client.transport = Mock()
        client.transport.chat.return_value = (x for x in [chunk("one", False), chunk(" two", True)])
        self.assertEqual(client.chat([], stream=True).message.content, "one two")
        self.assertIsNotNone(client.metrics["ttft"])

    def test_incomplete_stream(self):
        client = LLMClient()
        client.transport = Mock()
        client.transport.chat.return_value = (x for x in [chunk("partial", False)])
        with self.assertRaises(RuntimeError):
            client.chat([], stream=True)
