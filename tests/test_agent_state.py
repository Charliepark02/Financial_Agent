
import unittest

from src.agents.state import (
    AgentState,
    AgentStatus
)


class TestAgentState(unittest.TestCase):

    def test_initial_state(self):

        state = AgentState()

        self.assertEqual(
            state.status,
            AgentStatus.IDLE
        )

        self.assertEqual(
            state.tool_call_count,
            0
        )

        self.assertFalse(
            state.is_terminal()
        )

    def test_unique_request_id(self):

        first = AgentState()
        second = AgentState()

        self.assertNotEqual(
            first.request_id,
            second.request_id
        )

    def test_valid_transition(self):

        state = AgentState()

        state.transition(
            AgentStatus.REASONING
        )

        self.assertEqual(
            state.status,
            AgentStatus.REASONING
        )

    def test_invalid_transition(self):

        state = AgentState()

        with self.assertRaises(ValueError):

            state.transition(
                AgentStatus.COMPLETED
            )

    def test_invalid_status_type(self):

        state = AgentState()

        with self.assertRaises(TypeError):

            state.transition("reasoning")

    def test_tool_execution_cycle(self):

        state = AgentState()

        state.transition(
            AgentStatus.REASONING
        )

        state.transition(
            AgentStatus.EXECUTING_TOOLS
        )

        state.transition(
            AgentStatus.REASONING
        )

        self.assertEqual(
            state.status,
            AgentStatus.REASONING
        )

        self.assertEqual(
            len(state.transitions),
            3
        )

    def test_completed_state(self):

        state = AgentState()

        state.transition(
            AgentStatus.REASONING
        )

        state.transition(
            AgentStatus.COMPLETED
        )

        self.assertTrue(
            state.is_terminal()
        )

        self.assertIsNotNone(
            state.completed_at
        )

    def test_failed_state(self):

        state = AgentState()

        state.transition(
            AgentStatus.REASONING
        )

        state.transition(
            AgentStatus.FAILED
        )

        self.assertTrue(
            state.is_terminal()
        )

    def test_terminal_state_transition(self):

        state = AgentState()

        state.transition(
            AgentStatus.REASONING
        )

        state.transition(
            AgentStatus.COMPLETED
        )

        with self.assertRaises(ValueError):

            state.transition(
                AgentStatus.REASONING
            )

    def test_independent_state_data(self):

        first = AgentState()
        second = AgentState()

        first.tool_logs.append({
            "tool": "get_latest_price"
        })

        self.assertEqual(
            len(second.tool_logs),
            0
        )

    def test_state_serialization(self):

        state = AgentState(
            user_input="Analyze AAPL"
        )

        state.transition(
            AgentStatus.REASONING
        )

        result = state.to_dict()

        self.assertEqual(
            result["status"],
            "reasoning"
        )

        self.assertEqual(
            result["user_input"],
            "Analyze AAPL"
        )

    def test_transition_history(self):

        state = AgentState()

        state.transition(
            AgentStatus.REASONING
        )

        transition = state.transitions[0]

        self.assertEqual(
            transition["from"],
            "idle"
        )

        self.assertEqual(
            transition["to"],
            "reasoning"
        )

        self.assertIn(
            "timestamp",
            transition
        )

    def test_final_generation_transition(self):

        state = AgentState()

        state.transition(
            AgentStatus.REASONING
        )

        state.transition(
            AgentStatus.EXECUTING_TOOLS
        )

        state.transition(
            AgentStatus.GENERATING_RESPONSE
        )

        state.transition(
            AgentStatus.COMPLETED
        )

        self.assertTrue(
            state.is_terminal()
        )


if __name__ == "__main__":
    unittest.main()
