"""Interactive CLI chat with the LLM."""
import argparse

from .config import MODEL
from .llm import chat
from .metrics import describe, last, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM chat with per-request metrics.")
    parser.add_argument("--metrics", action="store_true", help="print aggregate metrics summary on exit")
    args = parser.parse_args()

    print(f"Модель: {MODEL}. Введите сообщение или 'exit' для выхода.")
    history: list[dict] = []

    while True:
        user = input("\nВы: ").strip()
        if user.lower() in {"exit", "quit", "выход"}:
            break
        history.append({"role": "user", "content": user})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"\nLLM: {reply}")
        print(describe(last()))

    if args.metrics:
        print("\nСводка сессии:", summary())


if __name__ == "__main__":
    main()
