
import json
import math

from copy import deepcopy

from src.config import (
    CONTEXT_WINDOW,
    RESERVED_OUTPUT_TOKENS,
    CONTEXT_SAFETY_MARGIN,
    MAX_HISTORY_TURNS,
    MAX_TOOL_RESULT_CHARS
)


class ContextBudgetError(RuntimeError):
    pass


class ContextManager:

    def __init__(
        self,
        context_window=CONTEXT_WINDOW,
        reserved_output_tokens=RESERVED_OUTPUT_TOKENS,
        safety_margin=CONTEXT_SAFETY_MARGIN,
        max_history_turns=MAX_HISTORY_TURNS,
        max_tool_result_chars=MAX_TOOL_RESULT_CHARS
    ):

        self.context_window = context_window

        self.reserved_output_tokens = (
            reserved_output_tokens
        )

        self.safety_margin = safety_margin

        self.max_history_turns = max_history_turns

        self.max_tool_result_chars = (
            max_tool_result_chars
        )

        self.input_budget = (
            context_window
            - reserved_output_tokens
            - safety_margin
        )

        if self.input_budget <= 0:
            raise ValueError(
                "Invalid context budget."
            )

        if max_history_turns < 0:
            raise ValueError(
                "Invalid history turn limit."
            )

        if max_tool_result_chars <= 0:
            raise ValueError(
                "Invalid tool result size limit."
            )

        self.metrics = {}

    def estimate_tokens(self, text: str) -> int:

        if not text:
            return 0

        ascii_count = sum(
            character.isascii()
            for character in text
        )

        non_ascii_count = (
            len(text) - ascii_count
        )

        estimated_tokens = (
            ascii_count / 3
            + non_ascii_count * 2
        )

        return math.ceil(estimated_tokens)

    def estimate_message_tokens(
        self,
        message: dict
    ) -> int:

        serialized = json.dumps(
            message,
            ensure_ascii=False,
            default=str
        )

        return (
            self.estimate_tokens(serialized)
            + 8
        )

    def estimate_context_tokens(
        self,
        messages: list,
        tools: list | None = None
    ) -> int:

        message_tokens = sum(
            self.estimate_message_tokens(message)
            for message in messages
        )

        tool_tokens = 0

        if tools:

            serialized_tools = json.dumps(
                tools,
                ensure_ascii=False
            )

            tool_tokens = self.estimate_tokens(
                serialized_tools
            )

        return message_tokens + tool_tokens

    def split_history(self, messages: list):

        system_messages = []

        turns = []

        current_turn = []

        for message in messages:

            role = message.get("role")

            if role == "system":

                system_messages.append(message)

            elif role == "user":

                if current_turn:
                    turns.append(current_turn)

                current_turn = [message]

            else:

                current_turn.append(message)

        if current_turn:

            turns.append(current_turn)

        return system_messages, turns

    def compress_tool_result(
        self,
        message: dict,
        force: bool = False
    ) -> dict:

        compressed_message = deepcopy(message)

        if message.get("role") != "tool":

            return compressed_message

        content = message.get("content", "")

        if not isinstance(content, str):

            return compressed_message

        if (
            not force
            and len(content)
            <= self.max_tool_result_chars
        ):

            return compressed_message

        try:

            result = json.loads(content)

        except json.JSONDecodeError:

            return compressed_message

        if not isinstance(result, dict):

            return compressed_message

        # Preserve errors without modification.

        if not result.get("success", False):

            return compressed_message

        data = result.get("data")

        if not isinstance(data, dict):

            return compressed_message

        important_fields = {
            "symbol",
            "price",
            "currency",
            "date",
            "price_type",
            "data_source",
            "generated_at",
            "data",
            "technical_indicators",
            "risk_and_performance",
            "assumptions",
            "warnings"
        }

        reduced_data = {
            key: value
            for key, value in data.items()
            if key in important_fields
        }

        omitted_fields = sorted(
            set(data) - set(reduced_data)
        )

        if not omitted_fields:

            return compressed_message

        compressed_result = {
            **result,
            "data": reduced_data,
            "compression": {
                "compressed": True,
                "omitted_fields": omitted_fields
            }
        }

        new_content = json.dumps(
            compressed_result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":")
        )

        if len(new_content) < len(content):

            compressed_message["content"] = (
                new_content
            )

        return compressed_message

    def compact_messages(
        self,
        messages: list
    ) -> list:

        compacted = deepcopy(messages)

        for message in compacted:

            if message.get("role") != "tool":
                continue

            content = message.get("content", "")

            try:

                data = json.loads(content)

                message["content"] = json.dumps(
                    data,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":")
                )

            except (ValueError, TypeError):

                continue

        return compacted

    def prepare_messages(
        self,
        messages: list,
        tools: list | None = None
    ) -> list:

        self.metrics = {}

        original_messages = deepcopy(messages)

        original_tokens = (
            self.estimate_context_tokens(
                original_messages,
                tools
            )
        )

        compacted_messages = (
            self.compact_messages(
                original_messages
            )
        )

        system_messages, turns = (
            self.split_history(
                compacted_messages
            )
        )

        if not turns:

            raise ValueError(
                "No conversation turns available."
            )

        current_turn = [
            self.compress_tool_result(message)
            for message in turns[-1]
        ]

        previous_turns = turns[:-1]

        required_messages = (
            system_messages + current_turn
        )

        required_tokens = (
            self.estimate_context_tokens(
                required_messages,
                tools
            )
        )

        # Never silently discard the active turn.

        if required_tokens > self.input_budget:

            current_turn = [
                self.compress_tool_result(
                    message,
                    force=True
                )
                for message in current_turn
            ]

            required_messages = (
                system_messages + current_turn
            )

            required_tokens = (
                self.estimate_context_tokens(
                    required_messages,
                    tools
                )
            )

        if required_tokens > self.input_budget:

            self.metrics = {
                "status": "overflow",
                "estimated_original_tokens":
                    original_tokens,
                "required_tokens":
                    required_tokens,
                "input_budget":
                    self.input_budget
            }

            raise ContextBudgetError(
                f"Required context exceeds budget: "
                f"{required_tokens} > "
                f"{self.input_budget}. "
                "Reduce the request size or increase "
                "the context window."
            )

        selected_turns = []

        recent_turns = (
            previous_turns[-self.max_history_turns:]
            if self.max_history_turns > 0
            else []
        )

        for turn in reversed(recent_turns):

            compressed_turn = [
                self.compress_tool_result(
                    message,
                    force=True
                )
                for message in turn
            ]

            candidate_turns = (
                [compressed_turn]
                + selected_turns
            )

            candidate_messages = (
                system_messages
                + [
                    message
                    for selected_turn in candidate_turns
                    for message in selected_turn
                ]
                + current_turn
            )

            candidate_tokens = (
                self.estimate_context_tokens(
                    candidate_messages,
                    tools
                )
            )

            if candidate_tokens > self.input_budget:

                break

            selected_turns = candidate_turns

        prepared_messages = (
            system_messages
            + [
                message
                for turn in selected_turns
                for message in turn
            ]
            + current_turn
        )

        final_tokens = (
            self.estimate_context_tokens(
                prepared_messages,
                tools
            )
        )

        self.metrics = {
            "status": "ready",

            "context_window":
                self.context_window,

            "input_budget":
                self.input_budget,

            "estimated_original_tokens":
                original_tokens,

            "estimated_final_tokens":
                final_tokens,

            "estimated_tokens_saved":
                original_tokens - final_tokens,

            "original_messages":
                len(original_messages),

            "final_messages":
                len(prepared_messages),

            "removed_messages":
                len(original_messages)
                - len(prepared_messages),

            "retained_history_turns":
                len(selected_turns),

            "removed_history_turns":
                len(previous_turns)
                - len(selected_turns),

            "context_utilization":
                final_tokens / self.input_budget
        }

        return prepared_messages

    def historical_context(self, records, budget=512):
        """Select complete historical records for explicit historical requests."""
        selected = []
        remaining = min(budget, self.input_budget)
        for record in records:
            report = json.loads(record["payload"])["report"]
            block = f"Historical record #{record['id']} (saved {record['created']})\n{report}"
            cost = self.estimate_tokens(block) + 2
            if cost <= remaining:
                selected.append(block)
                remaining -= cost
        return "\n\n".join(selected)

    def get_metrics(self) -> dict:

        return self.metrics.copy()
