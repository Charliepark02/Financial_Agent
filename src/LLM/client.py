
import time

import ollama
from src.recovery import RetryPolicy, GenerationError

from src.config import (
    MODEL,
    THINK,
    CONTEXT_WINDOW,
    TEMPERATURE,
    KEEP_ALIVE,
    RESERVED_OUTPUT_TOKENS
)


class LLMClient:

    def __init__(self, timeout=120, retry=None):
        self.transport = ollama.Client(timeout=timeout)
        self.retry = retry or RetryPolicy()

        self.model = MODEL

        self.think = THINK

        self.context_window = CONTEXT_WINDOW

        self.temperature = TEMPERATURE

        self.keep_alive = KEEP_ALIVE

        self.max_output_tokens = (
            RESERVED_OUTPUT_TOKENS
        )

        self.metrics = {}

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        stream=False
    ):

        self.metrics = {}

        start_time = time.perf_counter()

        recovery = {}
        first_token = None

        def generate():
            nonlocal first_token
            response = self.transport.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                think=self.think,
                stream=stream,
                keep_alive=self.keep_alive,
                options={
                    "num_ctx": self.context_window,
                    "temperature": self.temperature,
                    "num_predict": self.max_output_tokens
                }
            )
            if not stream:
                return response
            pieces = []
            calls = []
            last = None
            try:
                for chunk in response:
                    last = chunk
                    if chunk.message.content:
                        if first_token is None:
                            first_token = time.perf_counter() - start_time
                        pieces.append(chunk.message.content)
                    calls.extend(chunk.message.tool_calls or [])
            finally:
                response.close()
            if last is None or not last.done:
                raise GenerationError("Incomplete model stream")
            last.message.content = "".join(pieces)
            last.message.tool_calls = calls or None
            return last

        try:
            response = self.retry.run(generate, recovery)
        except Exception:
            self.metrics.update(recovery, end_to_end_latency=time.perf_counter()-start_time)
            raise

        end_time = time.perf_counter()

        self.metrics = self.calculate_metrics(
            response
        )

        self.metrics["end_to_end_latency"] = (
            end_time - start_time
        )

        self.metrics.update(recovery, ttft=first_token)
        return response

    def calculate_metrics(self, response):

        total_time = (
            response.total_duration or 0
        ) / 1e9

        load_time = (
            response.load_duration or 0
        ) / 1e9

        prompt_time = (
            response.prompt_eval_duration or 0
        ) / 1e9

        generation_time = (
            response.eval_duration or 0
        ) / 1e9

        input_tokens = (
            response.prompt_eval_count or 0
        )

        output_tokens = (
            response.eval_count or 0
        )

        tokens_per_second = (
            output_tokens / generation_time
            if generation_time > 0
            else 0
        )

        tpot = (
            generation_time / output_tokens
            if output_tokens > 0
            else 0
        )

        return {
            "total_time": total_time,

            "load_time": load_time,

            "prompt_time": prompt_time,

            "generation_time": generation_time,

            "input_tokens": input_tokens,

            "output_tokens": output_tokens,

            "tokens_per_second":
                tokens_per_second,

            "tpot": tpot,

            "done_reason": getattr(
                response,
                "done_reason",
                None
            )
        }

    def get_metrics(self):

        return self.metrics.copy()
