from src.recovery import classify, GenerationError
from src.grounding import validate_request, verified_answer, safe_explanation

import json
import time

from copy import deepcopy

from src.config import (
    SYSTEM_PROMPT,
    MAX_TOOL_ROUNDS,
    MAX_TOOL_CALLS
)

from src.LLM.client import LLMClient

from src.tools.execution import execute_batch
from src.tools.registry import ToolRegistry

from src.agents.state import (
    AgentState,
    AgentStatus
)

from src.memory.context_manager import (
    ContextManager
)


class FinancialAgent:

    def __init__(
        self,
        llm=None,
        registry=None,
        context_manager=None,
        archive=None
    ):

        self.llm = (
            llm if llm is not None
            else LLMClient()
        )

        self.registry = (
            registry if registry is not None
            else ToolRegistry()
        )

        self.context_manager = (
            context_manager
            if context_manager is not None
            else ContextManager()
        )

        self.request_started = time.perf_counter()
        self.archive = archive
        self.streaming = False
        self.max_tool_rounds = MAX_TOOL_ROUNDS

        self.max_tool_calls = MAX_TOOL_CALLS

        self.clear_history()

    def clear_history(self):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        self.state = AgentState()

        self.metrics = {}

        self.context_logs = []

        self.context_manager.metrics = {}

    def create_state(self, user_input: str):

        messages = deepcopy(self.messages)

        messages.append({
            "role": "user",
            "content": user_input
        })

        self.state = AgentState(
            user_input=user_input,
            messages=messages
        )

        self.metrics = {}

        self.context_logs = []

        self.context_manager.metrics = {}

    def tools_available(self) -> bool:

        return (
            self.state.tool_round_count
            < self.max_tool_rounds
            and
            self.state.tool_call_count
            < self.max_tool_calls
        )

    def call_llm(
        self,
        tools=None,
        final_only=False
    ):

        # Keep temporary instructions out of
        # permanent conversation history.

        source_messages = deepcopy(
            self.state.messages
        )

        if final_only:

            source_messages[0]["content"] += (
                "\nThe tool execution limit has "
                "been reached. Provide a final "
                "answer using only available "
                "tool results. Explain missing data."
            )

        context_start = time.perf_counter()
        try:

            prepared_messages = (
                self.context_manager.prepare_messages(
                    messages=source_messages,
                    tools=tools
                )
            )

        finally:

            self.context_logs.append(dict(self.context_manager.get_metrics(), seconds=time.perf_counter()-context_start))

        call_offset = time.perf_counter() - self.request_started
        response = self.llm.chat(
            messages=prepared_messages,
            tools=tools,
            **({"stream": True} if self.streaming and isinstance(self.llm, LLMClient) else {})
        )

        metrics = self.llm.get_metrics()

        if metrics.get("ttft") is not None:
            metrics["request_first_text_seconds"] = call_offset + metrics["ttft"]
        self.state.llm_metrics.append(metrics)

        if metrics.get("done_reason") == "length":

            raise GenerationError(
                "Model output exceeded the "
                "generation token limit."
            )

        return response.message

    def execute_tool(self, tool_call):

        self._execute_calls([tool_call])

    def _execute_calls(self, tool_calls):
        calls = [(call.function.name, dict(call.function.arguments)) for call in tool_calls]
        remaining = max(0, self.max_tool_calls - self.state.tool_call_count)
        allowed = calls[:remaining]
        start = time.perf_counter()
        outcomes = execute_batch(self.registry, allowed)
        if allowed:
            self.state.tool_execution_time += time.perf_counter() - start
        self.state.tool_call_count += len(allowed)
        for index, (name, arguments) in enumerate(calls):
            if index < len(outcomes):
                result, duration = outcomes[index]
            else:
                result = {"success": False, "tool": name, "error": "Tool call limit reached."}
                duration = 0.0
            self._record_tool(name, arguments, result, duration)

    def _record_tool(self, name, arguments, result, duration):
        self.state.tool_logs.append({
            "tool": name,

            "arguments": arguments,

            "success": result.get(
                "success", False
            ),

            "execution_time": duration
        })

        self.state.tool_results.append(
            deepcopy(result)
        )

        self.state.messages.append({
            "role": "tool",

            "tool_name": name,

            "content": json.dumps(
                {"success": result.get("success", False), "verified_report": verified_answer([result])},
                ensure_ascii=False,
                allow_nan=False
            )
        })

    def execute_tools(self, assistant_message):

        self.state.transition(
            AgentStatus.EXECUTING_TOOLS
        )

        self.state.messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        self.state.tool_round_count += 1

        self._execute_calls(assistant_message.tool_calls)

    def complete_response(self, answer: str):

        validation_start = time.perf_counter()
        answer = verified_answer(self.state.tool_results) if self.state.tool_results else safe_explanation(answer.strip())

        if self.state.tool_results and self.state.error:
            answer += "\n\nPartial report: model processing stopped after these tool results."
        self.validation_seconds = time.perf_counter() - validation_start
        if not answer:

            raise GenerationError(
                "The model returned an empty response."
            )

        if self.archive is not None:
            self.archive.save(self.state.request_id, self.state.tool_results)
        self.state.final_answer = answer

        self.state.messages.append({
            "role": "assistant",
            "content": answer
        })

        self.state.transition(
            AgentStatus.COMPLETED
        )

        self.messages = deepcopy(
            self.state.messages
        )

        return answer

    def generate_final_response(self):

        self.state.transition(
            AgentStatus.GENERATING_RESPONSE
        )

        if self.state.tool_results:
            return self.complete_response("")

        assistant_message = self.call_llm(
            tools=None,
            final_only=True
        )

        if assistant_message.tool_calls:

            raise GenerationError(
                "Unexpected tool call after "
                "tool execution limit."
            )

        return self.complete_response(
            assistant_message.content or ""
        )

    def stream_chat(self, user_input):
        """Buffer model output until the tool protocol and financial checks finish."""
        started = time.perf_counter()
        self.streaming = True
        try:
            answer = self.chat(user_input)
            self.metrics["first_visible_seconds"] = time.perf_counter() - started
            yield from answer.splitlines(keepends=True)
        finally:
            self.streaming = False

    def chat(self, user_input: str) -> str:

        self.request_started = time.perf_counter()
        self.validation_seconds = 0.0
        self.create_state(user_input if isinstance(user_input, str) and len(user_input) <= 16000 else "")
        start_time = time.perf_counter()
        try:
            validate_request(user_input)
            self.state.transition(
                AgentStatus.REASONING
            )

            while True:

                if not self.tools_available():

                    return self.generate_final_response()

                assistant_message = self.call_llm(
                    tools=self.registry.get_schemas()
                )

                tool_calls = (
                    assistant_message.tool_calls or []
                )

                if not tool_calls:

                    return self.complete_response(
                        assistant_message.content or ""
                    )

                self.execute_tools(
                    assistant_message
                )

                if not self.tools_available():

                    return self.generate_final_response()

                self.state.transition(
                    AgentStatus.REASONING
                )

        except (Exception, KeyboardInterrupt) as error:

            self.state.error = (
                f"{classify(error)}: {type(error).__name__}: {error}"
            )

            if (self.state.tool_results and self.state.status == AgentStatus.REASONING and
                    classify(error) in {"timeout", "connection", "provider_failure", "generation_failure", "rate_limit", "context_overflow"}):
                return self.complete_response("")
            if not self.state.is_terminal():

                self.state.transition(
                    AgentStatus.FAILED
                )

            raise

        finally:

            total_latency = (
                time.perf_counter() - start_time
            )

            self.metrics = self.aggregate_metrics(
                total_latency
            )

    def aggregate_metrics(
        self,
        total_latency: float
    ) -> dict:

        llm_metrics = self.state.llm_metrics

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
            total_output_tokens
            / total_generation_time
            if total_generation_time > 0
            else 0
        )

        tpot = (
            total_generation_time
            / total_output_tokens
            if total_output_tokens > 0
            else 0
        )

        context_tokens_saved = sum(
            log.get(
                "estimated_tokens_saved", 0
            )
            for log in self.context_logs
        )

        peak_context_tokens = max(
            (
                log.get(
                    "estimated_final_tokens",
                    log.get("required_tokens", 0)
                )
                for log in self.context_logs
            ),
            default=0
        )

        return {
            "llm_calls": len(llm_metrics),

            "tool_calls":
                self.state.tool_call_count,

            "tool_rounds":
                self.state.tool_round_count,

            "input_tokens": total_input_tokens,

            "output_tokens": total_output_tokens,

            "load_time": total_load_time,

            "prompt_time": total_prompt_time,

            "generation_time":
                total_generation_time,

            "tokens_per_second":
                tokens_per_second,

            "tpot": tpot,

            "llm_latency": total_llm_latency,

            "tool_execution_time":
                self.state.tool_execution_time,

            "total_latency": total_latency,

            "history_messages":
                len(self.messages),

            "request_id":
                self.state.request_id,

            "status":
                self.state.status.value,

            "state_transitions":
                len(self.state.transitions),

            "estimated_context_tokens_saved":
                context_tokens_saved,

            "peak_estimated_context_tokens":
                peak_context_tokens,

            "validation_seconds": self.validation_seconds,
            "ttft": next((m["request_first_text_seconds"] for m in llm_metrics
                          if m.get("request_first_text_seconds") is not None), None),
            "context_preparation_seconds": sum(m["seconds"] for m in self.context_logs),
            "context_estimation_errors": [c.get("estimated_final_tokens", 0)-m["input_tokens"]
                                          for c, m in zip(self.context_logs, llm_metrics)],
            "context_input_budget":
                self.context_manager.input_budget
        }

    def get_metrics(self):

        return self.metrics.copy()

    def get_tool_logs(self):

        return deepcopy(
            self.state.tool_logs
        )

    def get_tool_results(self):

        return deepcopy(
            self.state.tool_results
        )

    def get_state(self):

        return self.state.to_dict()

    def get_context_logs(self):

        return deepcopy(
            self.context_logs
        )

    def get_context_metrics(self):

        return self.context_manager.get_metrics()
