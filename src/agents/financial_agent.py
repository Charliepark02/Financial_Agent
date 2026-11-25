
import json
import time

from src.config import (
    SYSTEM_PROMPT,
    MAX_TOOL_ROUNDS,
    MAX_TOOL_CALLS
)

from src.LLM.client import LLMClient
from src.tools.registry import ToolRegistry


class FinancialAgent:

    def __init__(self):

        self.llm = LLMClient()
        self.registry = ToolRegistry()

        self.max_tool_rounds = MAX_TOOL_ROUNDS
        self.max_tool_calls = MAX_TOOL_CALLS

        self.metrics = {}

        self.tool_logs = []

        self.clear_history()

    def clear_history(self):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        self.metrics = {}
        self.tool_logs = []

    def chat(self, user_input: str) -> str:

        user_input = user_input.strip()

        if not user_input:
            raise ValueError("User input cannot be empty.")

        start_time = time.perf_counter()

        self.tool_logs = []

        llm_metrics = []

        tool_call_count = 0
        tool_round_count = 0

        tool_execution_time = 0.0

        # Work on a temporary conversation history

        messages = self.messages.copy()

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:

            while True:

                # 1. Determine whether tools are available

                tools_enabled = (
                    tool_round_count < self.max_tool_rounds
                    and tool_call_count < self.max_tool_calls
                )

                schemas = (
                    self.registry.get_schemas()
                    if tools_enabled
                    else None
                )

                # 2. Request an LLM response

                response = self.llm.chat(
                    messages=messages,
                    tools=schemas
                )

                llm_metrics.append(
                    self.llm.get_metrics()
                )

                assistant_message = response.message

                tool_calls = (
                    assistant_message.tool_calls or []
                )

                # 3. Return the final answer

                if not tool_calls:

                    final_answer = (
                        assistant_message.content or ""
                    ).strip()

                    if not final_answer:
                        final_answer = (
                            "The model returned an empty response."
                        )

                    messages.append({
                        "role": "assistant",
                        "content": final_answer
                    })

                    break

                # 4. Prevent calls beyond the configured limits

                if not tools_enabled:

                    final_answer = (
                        "Tool execution limit reached. "
                        "Unable to complete the analysis."
                    )

                    messages.append({
                        "role": "assistant",
                        "content": final_answer
                    })

                    break

                # 5. Preserve the assistant's tool requests

                messages.append(
                    assistant_message.model_dump(
                        exclude_none=True
                    )
                )

                tool_round_count += 1

                # 6. Execute requested tools

                for tool_call in tool_calls:

                    name = tool_call.function.name

                    arguments = dict(
                        tool_call.function.arguments
                    )

                    if tool_call_count >= self.max_tool_calls:

                        result = {
                            "success": False,
                            "tool": name,
                            "error": "Tool call limit reached."
                        }

                        duration = 0.0

                    else:

                        tool_start = time.perf_counter()

                        result = self.registry.execute(
                            name=name,
                            arguments=arguments
                        )

                        duration = (
                            time.perf_counter() - tool_start
                        )

                        tool_execution_time += duration

                        tool_call_count += 1

                    # 7. Record tool execution logs

                    self.tool_logs.append({
                        "tool": name,
                        "arguments": arguments,
                        "success": result.get(
                            "success",
                            False
                        ),
                        "execution_time": duration
                    })

                    # 8. Return the tool result to the LLM

                    tool_result_json = json.dumps(
                        result,
                        ensure_ascii=False,
                        allow_nan=False
                    )

                    messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": tool_result_json
                    })

                # 9. Request a final answer when limits are reached

                if (
                    tool_round_count >= self.max_tool_rounds
                    or tool_call_count >= self.max_tool_calls
                ):

                    messages.append({
                        "role": "user",
                        "content": (
                            "The tool execution limit has been reached. "
                            "Provide a final answer using only the "
                            "available tool results. Clearly explain "
                            "any missing data."
                        )
                    })

            # Save conversation only after successful completion

            self.messages = messages

            return final_answer

        finally:

            end_time = time.perf_counter()

            self.metrics = self.aggregate_metrics(
                llm_metrics=llm_metrics,
                tool_call_count=tool_call_count,
                tool_round_count=tool_round_count,
                tool_execution_time=tool_execution_time,
                total_latency=end_time - start_time
            )

    def aggregate_metrics(
        self,
        llm_metrics: list,
        tool_call_count: int,
        tool_round_count: int,
        tool_execution_time: float,
        total_latency: float
    ) -> dict:

        total_input_tokens = sum(
            metric["input_tokens"]
            for metric in llm_metrics
        )

        total_output_tokens = sum(
            metric["output_tokens"]
            for metric in llm_metrics
        )

        total_generation_time = sum(
            metric["generation_time"]
            for metric in llm_metrics
        )

        total_llm_latency = sum(
            metric["end_to_end_latency"]
            for metric in llm_metrics
        )

        total_load_time = sum(
            metric["load_time"]
            for metric in llm_metrics
        )

        total_prompt_time = sum(
            metric["prompt_time"]
            for metric in llm_metrics
        )

        tokens_per_second = (
            total_output_tokens / total_generation_time
            if total_generation_time > 0
            else 0
        )

        tpot = (
            total_generation_time / total_output_tokens
            if total_output_tokens > 0
            else 0
        )

        return {
            "llm_calls": len(llm_metrics),
            "tool_calls": tool_call_count,
            "tool_rounds": tool_round_count,

            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,

            "load_time": total_load_time,
            "prompt_time": total_prompt_time,
            "generation_time": total_generation_time,

            "tokens_per_second": tokens_per_second,
            "tpot": tpot,

            "llm_latency": total_llm_latency,
            "tool_execution_time": tool_execution_time,
            "total_latency": total_latency,

            "history_messages": len(self.messages)
        }

    def get_metrics(self):

        return self.metrics.copy()

    def get_tool_logs(self):

        return self.tool_logs.copy()
