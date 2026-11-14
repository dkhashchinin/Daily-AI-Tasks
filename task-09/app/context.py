"""Context management strategies ("Управление контекстом").

Two modes:
- "full"    — the whole stored history is sent to the LLM as-is;
- "summary" — everything older than the last KEEP messages is replaced by a
  rolling summary. The summary is stored separately (SQLite table
  ``context_state``) and substituted into the request as a system message
  instead of the full history. Only the last KEEP messages ever go "as is":
  as soon as messages fall out of the window they are folded into the summary.
"""
from . import store
from .llm import chat
from .tokens import count_tokens

KEEP = 10  # последние сообщения уходят в запрос "как есть"

SUMMARY_SYSTEM = (
    "Ты ведёшь краткую сводку диалога. Суммируй переписку, сохраняя важные "
    "факты, имена, числа, договорённости и незавершённые задачи. "
    "В ответе — только сама сводка, без вступлений и комментариев."
)


def safe_window_start(messages: list[dict], keep: int = KEEP) -> int:
    """Index where the "as is" tail may start without breaking tool sequences.

    The window must not begin with a tool result: a tool message has to follow
    its assistant tool_calls message, otherwise the API rejects the request.
    """
    start = max(0, len(messages) - keep)
    while start > 0 and messages[start].get("role") == "tool":
        start -= 1
    return start


def _transcript(messages: list[dict]) -> str:
    """Render messages as plain text for the summarizer."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            lines.append(f"Пользователь: {content}")
        elif role == "assistant":
            if m.get("tool_calls"):
                names = ", ".join(
                    (tc.get("function") or {}).get("name", "?")
                    for tc in m["tool_calls"]
                )
                head = f"Ассистент (вызвал инструменты: {names})"
                lines.append(f"{head}: {content}" if content else head)
            elif content:
                lines.append(f"Ассистент: {content}")
        elif role == "tool":
            lines.append(f"Результат инструмента: {content}")
    return "\n".join(lines)


def update_summary() -> tuple[str | None, int, dict | None]:
    """Fold messages that fell out of the window into the stored summary.

    Returns ``(summary, covered, usage)`` where ``covered`` is how many oldest
    messages the summary already replaces and ``usage`` is the summarizer's
    token usage (or None). On summarizer failure the previous state is kept,
    so the agent keeps working.
    """
    messages = store.load_messages()
    summary, covered = store.load_summary()
    window = safe_window_start(messages)

    if covered > window:  # defensive: never claim more than exists
        covered = window
        store.save_summary(summary, covered)
        return summary, covered, None

    # Any message older than the window that is not covered by the summary yet
    # must be folded in right away: the request keeps only the last KEEP
    # messages as-is.
    fresh = messages[covered:window]
    if not fresh:
        return summary, covered, None

    parts = []
    if summary:
        parts.append("Предыдущая сводка диалога:\n" + summary)
    parts.append("Новые сообщения диалога:\n" + _transcript(fresh))
    parts.append("Объедини всё это в одну обновлённую краткую сводку.")

    try:
        response = chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
        )
        new_summary = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[context] не удалось обновить summary: {exc}")
        return summary, covered, None

    if not new_summary:
        return summary, covered, None
    store.save_summary(new_summary, window)
    usage = getattr(response, "usage", None)
    return new_summary, window, {
        "prompt": getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
        "completion": getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
        "cost": float(getattr(usage, "cost", 0) or 0) if usage else 0.0,
    }


def build_context(mode: str) -> tuple[list[dict], dict]:
    """Build the message list to send to the LLM for the given mode.

    Returns ``(context_messages, info)``. In "summary" mode it is a system
    message with the summary (if any) followed by the recent messages kept
    as is.
    """
    messages = store.load_messages()
    if mode != "summary":
        return messages, {"mode": "full", "summary_used": False}

    summary, covered, summary_usage = update_summary()
    context: list[dict] = []
    if summary:
        context.append({
            "role": "system",
            "content": "Сводка более ранней части диалога "
                       "(старые сообщения заменены сводкой):\n" + summary,
        })
    context.extend(messages[covered:])
    return context, {
        "mode": "summary",
        "summary_used": bool(summary),
        "summary_tokens": count_tokens(summary) if summary else 0,
        "as_is": len(messages) - covered,
        "summary_usage": summary_usage,
    }
