"""Interactive CLI chat with the LLM."""
from .config import MODEL
from .llm import chat


def main() -> None:
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


if __name__ == "__main__":
    main()
