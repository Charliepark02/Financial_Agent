
import time
import ollama

from src.config import (
    MODEL,
    THINK,
    CONTEXT_WINDOW,
    SYSTEM_PROMPT
)


class LLMClient:

    def __init__(self):
        self.model = MODEL
        self.think = THINK
        self.context_window = CONTEXT_WINDOW
        self.system_prompt = SYSTEM_PROMPT
        self.metrics = {}

        self.clear_history()

    def clear_history(self):
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        self.metrics = {}

    def chat(self, user_input):
        self.metrics = {}

        user_message = {
            "role": "user",
            "content": user_input
        }

        # Start latency measurement
        start_time = time.perf_counter()
        first_token_time = None

        stream = ollama.chat(
            model=self.model,
            messages=self.messages + [user_message],
            think=self.think,
            stream=True,
            options={
                "num_ctx": self.context_window
            }
        )

        full_response = ""
        final_chunk = None

        for chunk in stream:
            content = chunk.message.content or ""

            # Measure time to first visible output
            if content and first_token_time is None:
                first_token_time = time.perf_counter()

            full_response += content

            if content:
                yield content

            if chunk.done:
                final_chunk = chunk

        # End latency measurement
        end_time = time.perf_counter()

        if final_chunk is not None:
            self.messages.extend([
                user_message,
                {
                    "role": "assistant",
                    "content": full_response
                }
            ])

            self.metrics = self.calculate_metrics(final_chunk)

            # Time to First Token
            self.metrics["ttft"] = (
                first_token_time - start_time
                if first_token_time is not None
                else None
            )

            # End-to-End Latency
            self.metrics["end_to_end_latency"] = (
                end_time - start_time
            )

    def calculate_metrics(self, response):
        total_time = (response.total_duration or 0) / 1e9
        load_time = (response.load_duration or 0) / 1e9
        prompt_time = (response.prompt_eval_duration or 0) / 1e9
        generation_time = (response.eval_duration or 0) / 1e9

        input_tokens = response.prompt_eval_count or 0
        output_tokens = response.eval_count or 0

        # Tokens per Second
        tps = (
            output_tokens / generation_time
            if generation_time > 0
            else 0
        )

        # Average Time per Output Token
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
            "tpot": tpot,
            "history_messages": len(self.messages)
        }

    def get_metrics(self):
        return self.metrics.copy()
