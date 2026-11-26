from src.recovery import PlannerResponseError
from src.grounding import validate_request

import json

from dataclasses import dataclass, field

from src.config import MAX_TOOL_CALLS
from src.LLM.client import LLMClient
from src.tools.registry import ToolRegistry


PLANNER_PROMPT = """
You are a financial task planner.

Convert the user's request into a structured execution plan.

Available tools:

1. get_latest_price
   Arguments:
   - symbol: string

2. analyze_stock
   Arguments:
   - symbol: string
   - period: 3mo, 6mo, 1y, 2y, 5y, 10y
   - risk_free_rate: number, optional
   - confidence_level: number, optional

Rules:

- Return only valid JSON.
- Do not use Markdown code fences.
- Do not invent financial data.
- Use valid ticker symbols.
- Preserve the requested analysis period.
- Create one task per necessary tool call.
- Avoid duplicate tasks.
- Use no tasks for conceptual questions.
- Use analyze_stock when financial metrics are required.
- Do not create separate tasks for metrics that
  analyze_stock already calculates.
- Tasks must be independently executable.
- Do not create dependencies between tasks.
- If the request cannot be fulfilled with available tools,
  explain the limitation in the objective.
- Do not answer the user's question directly.

Return this JSON structure:

{
    "objective": "Description of the objective",
    "tasks": [
        {
            "task_id": "T1",
            "description": "Description of task",
            "tool": "analyze_stock",
            "arguments": {
                "symbol": "AAPL",
                "period": "2y"
            }
        }
    ]
}
"""


@dataclass
class PlanTask:

    task_id: str

    description: str

    tool: str

    arguments: dict = field(default_factory=dict)

    def to_dict(self) -> dict:

        return {
            "task_id": self.task_id,
            "description": self.description,
            "tool": self.tool,
            "arguments": self.arguments.copy()
        }


@dataclass
class ExecutionPlan:

    objective: str

    tasks: list[PlanTask] = field(default_factory=list)

    def to_dict(self) -> dict:

        return {
            "objective": self.objective,
            "tasks": [
                task.to_dict()
                for task in self.tasks
            ]
        }


class FinancialPlanner:

    def __init__(self, llm=None, registry=None):

        self.llm = (
            llm if llm is not None
            else LLMClient()
        )

        self.registry = (
            registry if registry is not None
            else ToolRegistry()
        )

        self.max_tasks = MAX_TOOL_CALLS

    def create_plan(
        self,
        user_input: str
    ) -> ExecutionPlan:

        user_input = validate_request(user_input)

        if not user_input:

            raise ValueError(
                "User input cannot be empty."
            )

        messages = [
            {
                "role": "system",
                "content": PLANNER_PROMPT
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        for attempt in range(2):
            response = self.llm.chat(messages=messages, tools=None)
            try:
                return self.validate_plan(self.parse_json(response.message.content or ""))
            except (ValueError, TypeError, KeyError) as error:
                if attempt:
                    raise PlannerResponseError("Invalid planner response after regeneration") from error
                messages.append({"role": "user", "content": "Regenerate the complete plan using the specified JSON schema."})

    def parse_json(self, content: str) -> dict:

        content = content.strip()

        if content.startswith("```"):

            lines = content.splitlines()

            if (
                len(lines) >= 3
                and lines[-1].strip() == "```"
            ):

                content = "\n".join(
                    lines[1:-1]
                ).strip()

        try:

            result = json.loads(content)

        except json.JSONDecodeError as error:

            raise ValueError(
                "Planner returned invalid JSON."
            ) from error

        if not isinstance(result, dict):

            raise ValueError(
                "Plan must be a JSON object."
            )

        return result

    def validate_plan(
        self,
        raw_plan: dict
    ) -> ExecutionPlan:

        if not isinstance(raw_plan, dict):

            raise ValueError(
                "Plan must be a dictionary."
            )

        objective = raw_plan.get("objective")

        raw_tasks = raw_plan.get("tasks")

        if (
            not isinstance(objective, str)
            or not objective.strip()
        ):

            raise ValueError(
                "Plan objective is required."
            )

        if not isinstance(raw_tasks, list):

            raise ValueError(
                "Plan tasks must be a list."
            )

        if len(raw_tasks) > self.max_tasks:

            raise ValueError(
                "Plan exceeds maximum task limit."
            )

        validated_tasks = []

        task_ids = set()

        task_signatures = set()

        for raw_task in raw_tasks:

            if not isinstance(raw_task, dict):

                raise ValueError(
                    "Each task must be a dictionary."
                )

            task_id = raw_task.get("task_id")

            description = raw_task.get("description")

            tool = raw_task.get("tool")

            arguments = raw_task.get("arguments")

            if (
                not isinstance(task_id, str)
                or not task_id.strip()
            ):

                raise ValueError(
                    "Invalid task ID."
                )

            task_id = task_id.strip()

            if task_id in task_ids:

                raise ValueError(
                    "Duplicate task ID."
                )

            if (
                not isinstance(description, str)
                or not description.strip()
            ):

                raise ValueError(
                    "Task description is required."
                )

            if (
                not isinstance(tool, str)
                or tool not in self.registry.list_tools()
            ):

                raise ValueError(
                    f"Unknown tool: {tool}"
                )

            validated_arguments = (
                self.registry.validate_arguments(
                    tool,
                    arguments
                )
            )

            signature = json.dumps(
                {
                    "tool": tool,
                    "arguments": validated_arguments
                },
                sort_keys=True,
                allow_nan=False
            )

            if signature in task_signatures:

                raise ValueError(
                    "Duplicate tool task."
                )

            task_ids.add(task_id)

            task_signatures.add(signature)

            validated_tasks.append(
                PlanTask(
                    task_id=task_id,
                    description=description.strip(),
                    tool=tool,
                    arguments=validated_arguments
                )
            )

        return ExecutionPlan(
            objective=objective.strip(),
            tasks=validated_tasks
        )
