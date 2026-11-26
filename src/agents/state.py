
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class AgentStatus(str, Enum):

    IDLE = "idle"
    REASONING = "reasoning"
    EXECUTING_TOOLS = "executing_tools"
    GENERATING_RESPONSE = "generating_response"
    COMPLETED = "completed"
    FAILED = "failed"


ALLOWED_TRANSITIONS = {
    AgentStatus.IDLE: {
        AgentStatus.REASONING,
        AgentStatus.FAILED
    },
    AgentStatus.REASONING: {
        AgentStatus.EXECUTING_TOOLS,
        AgentStatus.GENERATING_RESPONSE,
        AgentStatus.COMPLETED,
        AgentStatus.FAILED
    },
    AgentStatus.EXECUTING_TOOLS: {
        AgentStatus.REASONING,
        AgentStatus.GENERATING_RESPONSE,
        AgentStatus.FAILED
    },
    AgentStatus.GENERATING_RESPONSE: {
        AgentStatus.COMPLETED,
        AgentStatus.FAILED
    },
    AgentStatus.COMPLETED: set(),
    AgentStatus.FAILED: set()
}


@dataclass
class AgentState:

    request_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    user_input: str = ""

    status: AgentStatus = AgentStatus.IDLE

    messages: list = field(default_factory=list)

    tool_logs: list = field(default_factory=list)

    tool_results: list = field(default_factory=list)

    llm_metrics: list = field(default_factory=list)

    transitions: list = field(default_factory=list)

    tool_call_count: int = 0

    tool_round_count: int = 0

    tool_execution_time: float = 0.0

    final_answer: str = ""

    error: str | None = None

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    completed_at: str | None = None

    def transition(self, new_status: AgentStatus):

        if not isinstance(new_status, AgentStatus):
            raise TypeError(
                "new_status must be an AgentStatus."
            )

        if new_status not in ALLOWED_TRANSITIONS[self.status]:

            raise ValueError(
                f"Invalid state transition: "
                f"{self.status.value} -> "
                f"{new_status.value}"
            )

        previous_status = self.status

        self.status = new_status

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        self.transitions.append({
            "from": previous_status.value,
            "to": new_status.value,
            "timestamp": timestamp
        })

        if new_status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED
        }:
            self.completed_at = timestamp

    def is_terminal(self) -> bool:

        return self.status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED
        }

    def to_dict(self) -> dict:

        return {
            "request_id": self.request_id,
            "user_input": self.user_input,
            "status": self.status.value,
            "tool_call_count": self.tool_call_count,
            "tool_round_count": self.tool_round_count,
            "tool_execution_time": self.tool_execution_time,
            "final_answer": self.final_answer,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "transitions": [
                transition.copy()
                for transition in self.transitions
            ]
        }
