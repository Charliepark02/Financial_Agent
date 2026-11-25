
import time
import ollama

from src.config import (
    MODEL,
    THINK,
    CONTEXT_WINDOW,
    TEMPERATURE,
    KEEP_ALIVE
)


class LLMClient:

    def __init__(self):

        self.model = MODEL
        self.think = THINK
        self.context_window = CONTEXT_WINDOW
        self.temperature = TEMPERATURE
        self.keep_alive = KEEP_ALIVE

        self.metrics = {}

    def chat(
        self,
        messages: list,
        tools: list | None = None
    ):

        self.metrics = {}

        start_time = time.perf_counter()

        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            think=self.think,
            stream=False,
            keep_alive=self.keep_alive,
            options={
                "num_ctx": self.context_window,
                "temperature": self.temperature
            }
        )

        end_time = time.perf_counter()

        self.metrics = self.calculate_metrics(
            response
        )

        self.metrics["end_to_end_latency"] = (
            end_time - start_time
        )

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

        tps = (
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
            "tokens_per_second": tps,
            "tpot": tpot
        }

    def get_metrics(self):

        return self.metrics.copy()
