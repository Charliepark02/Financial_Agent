from src.grounding import validate_request, verified_answer, safe_explanation

import time

from copy import deepcopy
from uuid import uuid4

from src.config import SYSTEM_PROMPT
from src.LLM.client import LLMClient
from src.tools.execution import execute_batch
from src.tools.registry import ToolRegistry
from src.agents.planner import FinancialPlanner


class PlanningAgent:

    def __init__(self, llm=None, registry=None, archive=None):
        self.archive = archive

        self.llm = (
            llm if llm is not None
            else LLMClient()
        )

        self.registry = (
            registry if registry is not None
            else ToolRegistry()
        )

        self.planner = FinancialPlanner(
            llm=self.llm,
            registry=self.registry
        )

        self.clear()

    def clear(self):

        self.plan = None

        self.results = []

        self.metrics = {}

    def execute_plan(self, plan) -> list:

        results = []

        calls = [(task.tool, task.arguments) for task in plan.tasks]
        outcomes = execute_batch(self.registry, calls)
        for task, (result, execution_time) in zip(plan.tasks, outcomes):
            results.append({
                "task_id": task.task_id,
                "description": task.description,
                "tool": task.tool,
                "arguments": task.arguments.copy(),
                "success": result.get(
                    "success",
                    False
                ),
                "execution_time": execution_time,
                "result": result
            })

        return results

    def synthesize(
        self,
        user_input: str,
        plan,
        results: list
    ) -> str:

        if not results:
            response = self.llm.chat(messages=[{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":user_input}], tools=None)
            if response.message.tool_calls or not response.message.content:
                raise ValueError("Invalid conceptual answer")
            return safe_explanation(response.message.content.strip())
        return verified_answer([item["result"] for item in results])

    def chat(self, user_input: str) -> str:

        self.clear()

        user_input = validate_request(user_input)

        if not user_input:

            raise ValueError(
                "User input cannot be empty."
            )

        total_start = time.perf_counter()

        planning_time = 0.0
        execution_time = 0.0
        synthesis_time = 0.0

        planning_metrics = {}
        synthesis_metrics = {}

        status = "failed"

        try:

            planning_start = time.perf_counter()

            self.plan = self.planner.create_plan(
                user_input
            )

            planning_time = (
                time.perf_counter() - planning_start
            )

            planning_metrics = self.llm.get_metrics()

            execution_start = time.perf_counter()

            self.results = self.execute_plan(
                self.plan
            )

            execution_time = (
                time.perf_counter() - execution_start
            )

            synthesis_start = time.perf_counter()

            answer = self.synthesize(
                user_input,
                self.plan,
                self.results
            )

            synthesis_time = (
                time.perf_counter() - synthesis_start
            )

            synthesis_metrics = {} if self.results else self.llm.get_metrics()

            status = (
                "completed"
                if all(
                    result["success"]
                    for result in self.results
                )
                else "completed_with_errors"
            )

            if self.archive is not None:
                self.archive.save(str(uuid4()), [item["result"] for item in self.results])
            return answer

        finally:

            total_latency = (
                time.perf_counter() - total_start
            )

            self.metrics = {
                "status": status,

                "planning_time": planning_time,

                "execution_time": execution_time,

                "synthesis_time": synthesis_time,

                "total_latency": total_latency,

                "total_tasks": (
                    len(self.plan.tasks)
                    if self.plan is not None
                    else 0
                ),

                "successful_tasks": sum(
                    result["success"]
                    for result in self.results
                ),

                "failed_tasks": sum(
                    not result["success"]
                    for result in self.results
                ),

                "input_tokens": (
                    planning_metrics.get(
                        "input_tokens", 0
                    )
                    +
                    synthesis_metrics.get(
                        "input_tokens", 0
                    )
                ),

                "output_tokens": (
                    planning_metrics.get(
                        "output_tokens", 0
                    )
                    +
                    synthesis_metrics.get(
                        "output_tokens", 0
                    )
                )
            }

    def get_plan(self):

        if self.plan is None:

            return None

        return deepcopy(
            self.plan.to_dict()
        )

    def get_results(self):

        return deepcopy(
            self.results
        )

    def get_metrics(self):

        return self.metrics.copy()
