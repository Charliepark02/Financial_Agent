
import json
import ollama

from src.config import MODEL, CONTEXT_WINDOW
from src.tools.registry import ToolRegistry


def main():

    registry = ToolRegistry()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial AI assistant. "
                "Use the provided tools whenever actual "
                "financial market data or quantitative "
                "calculations are required. "
                "Never fabricate financial data. "
                "Respond in the user's language."
            )
        }
    ]

    print("=== Function Calling Test ===")
    print("Available tools:", registry.list_tools())
    print("Type 'exit' to quit.")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        request_messages = messages + [
            {
                "role": "user",
                "content": user_input
            }
        ]

        response = ollama.chat(
            model=MODEL,
            messages=request_messages,
            tools=registry.get_schemas(),
            think=False,
            stream=False,
            options={
                "num_ctx": CONTEXT_WINDOW,
                "temperature": 0
            }
        )

        print("\n--- Raw LLM Response ---")

        print(
            response.message.model_dump_json(
                indent=4
            )
        )

        tool_calls = response.message.tool_calls

        if not tool_calls:

            print("\nNo tool call requested.")

            print(
                "\nAssistant:",
                response.message.content
            )

            continue

        for tool_call in tool_calls:

            name = tool_call.function.name

            arguments = dict(
                tool_call.function.arguments
            )

            print("\n--- Tool Call Detected ---")
            print("Function:", name)

            print(
                "Arguments:",
                json.dumps(
                    arguments,
                    indent=4,
                    ensure_ascii=False
                )
            )

            print("\n--- Executing Tool ---")

            result = registry.execute(
                name,
                arguments
            )

            print(
                json.dumps(
                    result,
                    indent=4,
                    ensure_ascii=False,
                    allow_nan=False
                )
            )


if __name__ == "__main__":
    main()
