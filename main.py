
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

            print("\nAgent is processing your request...")

            answer = agent.chat(user_input)

            print("\nAssistant:\n")

            print(answer)

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
