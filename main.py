from pathlib import Path
from src.memory.persistent import AnalysisArchive
from src.agents.planning_agent import PlanningAgent

import json

from src.agents.financial_agent import FinancialAgent


def print_metrics(metrics):

    if not metrics:
        return

    print("\n--- Agent Performance Metrics ---")

    print(f"LLM calls: {metrics['llm_calls']}")
    print(f"Tool calls: {metrics['tool_calls']}")
    print(f"Tool rounds: {metrics['tool_rounds']}")

    print(
        f"\nInput tokens: "
        f"{metrics['input_tokens']}"
    )

    print(
        f"Output tokens: "
        f"{metrics['output_tokens']}"
    )

    print(
        f"Tokens per second: "
        f"{metrics['tokens_per_second']:.2f}"
    )

    print(
        f"TPOT: "
        f"{metrics['tpot'] * 1000:.2f} ms"
    )

    print(
        f"\nLoad time: "
        f"{metrics['load_time']:.4f} sec"
    )

    print(
        f"Prompt time: "
        f"{metrics['prompt_time']:.4f} sec"
    )

    print(
        f"Generation time: "
        f"{metrics['generation_time']:.2f} sec"
    )

    print(
        f"LLM latency: "
        f"{metrics['llm_latency']:.2f} sec"
    )

    print(
        f"Tool execution time: "
        f"{metrics['tool_execution_time']:.2f} sec"
    )

    print(
        f"Total agent latency: "
        f"{metrics['total_latency']:.2f} sec"
    )

    print(
        f"History messages: "
        f"{metrics['history_messages']}"
    )


def print_tool_logs(logs):

    if not logs:
        print("\nNo tools were executed.")
        return

    print("\n--- Tool Execution Logs ---")

    for index, log in enumerate(logs, start=1):

        print(f"\nTool #{index}")

        print(f"Name: {log['tool']}")

        print(
            "Arguments:",
            json.dumps(
                log["arguments"],
                ensure_ascii=False
            )
        )

        print(f"Success: {log['success']}")

        print(
            f"Execution time: "
            f"{log['execution_time']:.4f} sec"
        )


def memory_command(agent, command):
    words = command.split()
    if words == ["memory", "on"]:
        agent.archive = AnalysisArchive(Path.home() / ".financial-agent" / "analyses.sqlite3")
        return "Memory enabled."
    if words == ["memory", "off"]:
        agent.archive = None
        return "Memory disabled."
    if agent.archive is None:
        return "Memory disabled. Use memory on."
    if words[:2] == ["memory", "clear"] and len(words) == 3:
        agent.archive.clear(words[2])
        return "Archive cleared."
    if words[:2] == ["memory", "delete"] and len(words) == 3:
        return f"Deleted {agent.archive.delete(int(words[2]))} record(s)."
    if words == ["memory"] or (words[:2] == ["memory", "search"] and len(words) == 3):
        rows = agent.archive.search(words[2] if len(words) == 3 else "")
        return agent.context_manager.historical_context(rows) or "No records within display budget."
    raise ValueError("memory on/off | memory | memory search SYMBOL | memory delete ID | memory clear CONFIRM")


def main():

    agent = FinancialAgent()

    print("=== Financial Agentic AI ===")

    print("Model: Qwen3-8B")

    print(
        "Available tools:",
        agent.registry.list_tools()
    )

    print("\nCommands:")
    print("  exit    - Quit the application")
    print("  clear   - Clear conversation history")
    print("  tools   - List available tools")
    print("  metrics - Show latest performance metrics")
    print("  logs    - Show latest tool execution logs")

    print("  plan REQUEST - Plan independent tasks explicitly")
    print("  memory on/off | memory | memory search SYMBOL | memory delete ID | memory clear CONFIRM")

    while True:

        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        command = user_input.lower()

        if command == "exit":

            print("Goodbye!")
            break

        if command == "clear":

            agent.clear_history()

            print("Conversation history cleared.")

            continue

        if command == "tools":

            print(agent.registry.list_tools())

            continue

        if command == "metrics":

            print_metrics(
                agent.get_metrics()
            )

            continue

        if command == "logs":

            print_tool_logs(
                agent.get_tool_logs()
            )

            continue

        try:

            if user_input.split()[0] == "memory":
                print(memory_command(agent, user_input))
                continue
            if user_input.startswith("plan "):
                print(PlanningAgent(registry=agent.registry, archive=agent.archive).chat(user_input[5:]))
                continue
            print("\nAgent is processing your request...")

            chunks = agent.stream_chat(user_input)

            print("\nAssistant:\n")

            for chunk in chunks:
                print(chunk, end="", flush=True)
            print()

            print_tool_logs(
                agent.get_tool_logs()
            )

            print_metrics(
                agent.get_metrics()
            )

        except KeyboardInterrupt:

            print("\nRequest interrupted.")

        except Exception as error:

            print(
                f"\nError: "
                f"{type(error).__name__}: {error}"
            )


if __name__ == "__main__":

    main()
