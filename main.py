
from src.LLM.client import LLMClient


def main():
    client = LLMClient()

    print("=== Financial AI Assistant ===")
    print("Type 'exit' to quit.")
    print("Type 'clear' to reset conversation history.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            client.clear_history()
            print("Conversation history cleared.")
            continue

        if not user_input:
            continue

        print("\nAssistant: ", end="", flush=True)

        for content in client.chat(user_input):
            print(content, end="", flush=True)

        print()

        metrics = client.get_metrics()

        if metrics:
            print("\n--- Performance Metrics ---")

            # Time metrics
            print(f"Total time: {metrics['total_time']:.2f} sec")
            print(f"Load time: {metrics['load_time']:.4f} sec")
            print(f"Prompt time: {metrics['prompt_time']:.4f} sec")
            print(f"Generation time: {metrics['generation_time']:.2f} sec")

            # Token metrics
            print(f"Input tokens: {metrics['input_tokens']}")
            print(f"Output tokens: {metrics['output_tokens']}")
            print(f"Tokens per second: {metrics['tokens_per_second']:.2f}")

            # Latency metrics
            if metrics["ttft"] is not None:
                print(f"TTFT: {metrics['ttft']:.4f} sec")

            print(f"TPOT: {metrics['tpot'] * 1000:.2f} ms")
            print(f"End-to-End Latency: {metrics['end_to_end_latency']:.2f} sec")

            # Conversation metrics
            print(f"History messages: {metrics['history_messages']}")


if __name__ == "__main__":
    main()
