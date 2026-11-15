"""Context management strategies ("Управление контекстом").

Three modes ("summary" is the default):
- "summary" — everything older than the last KEEP messages is replaced by a
  rolling summary. The summary is stored separately (SQLite table
  ``context_state``) and substituted into the request as a system message
  instead of the full history. Only the last KEEP messages ever go "as is":
  as soon as messages fall out of the window they are folded into the summary.
- "window"  — sliding window: only the last KEEP messages are sent to the LLM,
  everything older is simply dropped from the request (the DB still keeps the
  whole history).
- "facts"   — sticky facts / key-value memory: a separate JSON block "facts"
  stores the important details of the dialog (goal, constraints, preferences,
  decisions, agreements). It is refreshed after every user message and sent
  as a system message, followed by the last KEEP messages "as is".
"""
import json

from . import store
from .llm import chat
from .tokens import count_tokens

KEEP = 10  # последние сообщения уходят в запрос "как есть"

SUMMARY_SYSTEM = (
    "Ты ведёшь краткую сводку диалога. Суммируй переписку, сохраняя важные "
    "факты, имена, числа, договорённости и незавершённые задачи. "
    "В ответе — только сама сводка, без вступлений и комментариев."
)

FACTS_SYSTEM = (
    "Ты ведёшь блок «факты» диалога в формате JSON (ключ -> значение). "
    "Храни только важное: цель, ограничения, предпочтения, решения, "
    "договорённости, имена, числа, даты. Ничего не выдумывай. Обнови блок "
    "с учётом нового сообщения пользователя: добавь новое, исправь "
    "устаревшее, удали неактуальное. В ответе — только JSON без пояснений."
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


def _usage_dict(usage) -> dict:
    """Extract token usage of an auxiliary LLM call (summarizer / facts)."""
    return {
        "prompt": getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
        "completion": getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
        "cost": float(getattr(usage, "cost", 0) or 0) if usage else 0.0,
    }


def update_summary(branch_id: int = store.ROOT_BRANCH_ID) -> tuple[str | None, int, dict | None]:
    """Fold messages that fell out of the window into the stored summary.

    The summary is kept per branch. Returns ``(summary, covered, usage)``
    where ``covered`` is how many oldest messages the summary already replaces
    and ``usage`` is the summarizer's token usage (or None). On summarizer
    failure the previous state is kept, so the agent keeps working.
    """
    messages = store.load_messages(branch_id)
    summary, covered = store.load_summary(branch_id)
    window = safe_window_start(messages)

    if covered > window:  # defensive: never claim more than exists
        covered = window
        store.save_summary(summary, covered, branch_id)
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
    store.save_summary(new_summary, window, branch_id)
    return new_summary, window, _usage_dict(getattr(response, "usage", None))


def _parse_facts(raw: str | None) -> dict | None:
    """Parse the updater's reply into a facts dict (tolerates ``` fences)."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def update_facts(user_text: str,
                 branch_id: int = store.ROOT_BRANCH_ID) -> tuple[dict, dict | None]:
    """Merge new information from the user message into the sticky facts.

    The facts block is kept per branch. Returns ``(facts, usage)``. On updater
    failure the previous facts are kept unchanged, so the agent keeps working.
    """
    facts = store.load_facts(branch_id)
    parts = ["Текущие факты (JSON; может быть пустым):",
             json.dumps(facts, ensure_ascii=False) if facts else "{}",
             "Новое сообщение пользователя:", user_text,
             "Верни обновлённый блок фактов в формате JSON."]

    try:
        response = chat(
            [
                {"role": "system", "content": FACTS_SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[context] не удалось обновить facts: {exc}")
        return facts, None

    new_facts = _parse_facts(raw)
    if new_facts is None:  # updater replied garbage — keep the old block
        print("[context] facts: не удалось разобрать JSON ответа, блок не изменён")
        return facts, None
    store.save_facts(new_facts, branch_id)
    return new_facts, _usage_dict(getattr(response, "usage", None))


def build_context(mode: str, new_user_text: str | None = None,
                  branch_id: int = store.ROOT_BRANCH_ID) -> tuple[list[dict], dict]:
    """Build the message list to send to the LLM for the given branch/mode.

    Returns ``(context_messages, info)``. In "summary" mode it is a system
    message with the summary (if any) followed by the recent messages kept
    as is. In "window" mode it is only the last KEEP messages. In "facts"
    mode the sticky facts block is refreshed with ``new_user_text`` first
    and sent as a system message before the last KEEP messages. Summary and
    facts state is per branch.
    """
    messages = store.load_messages(branch_id)

    if mode == "window":
        start = safe_window_start(messages)
        context = messages[start:]
        return context, {"mode": "window", "as_is": len(context)}

    if mode == "facts":
        if new_user_text:
            facts, facts_usage = update_facts(new_user_text, branch_id)
        else:  # no new message — send the stored block as is
            facts, facts_usage = store.load_facts(branch_id), None
        start = safe_window_start(messages)
        context: list[dict] = []
        if facts:
            context.append({
                "role": "system",
                "content": "Важные факты диалога (ключ -> значение):\n"
                           + json.dumps(facts, ensure_ascii=False, indent=2),
            })
        context.extend(messages[start:])
        return context, {
            "mode": "facts",
            "facts_used": bool(facts),
            "facts_tokens": count_tokens(json.dumps(facts, ensure_ascii=False)) if facts else 0,
            "as_is": len(messages) - start,
            "facts_usage": facts_usage,
        }

    # "summary" (режим по умолчанию; неизвестный режим тоже попадает сюда)
    summary, covered, summary_usage = update_summary(branch_id)
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
