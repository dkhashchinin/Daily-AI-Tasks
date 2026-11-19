"""Context assembly: the prompt is built from the three memory layers.

There is a single context strategy — the memory model itself (see
``app/memory.py``). Each layer contributes its part of the prompt by its own
rule (``PROMPT_RULES``):

- **LTM** (долговременная) — rendered first as a system message, always;
- **WM**  (рабочая) — the current task's facts block, refreshed after every
  user message by :func:`update_facts` and sent as the second system message;
- **STM** (краткосрочная) — only the last KEEP dialog messages go "as is";
  everything older stays in the DB but is not sent.

:func:`build_context` returns the message list plus per-layer token counts so
the UI can show exactly how many tokens each layer adds to the request.
"""
import json

from . import memory, store
from .llm import chat
from .tokens import count_tokens

KEEP = 10  # последние сообщения уходят в запрос "как есть"

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


def _usage_dict(usage) -> dict:
    """Extract token usage of an auxiliary LLM call (facts updater)."""
    return {
        "prompt": getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
        "completion": getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
        "cost": float(getattr(usage, "cost", 0) or 0) if usage else 0.0,
    }


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
    """Merge new information from the user message into the working memory.

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


def build_context(new_user_text: str | None = None,
                  branch_id: int = store.ROOT_BRANCH_ID) -> tuple[list[dict], dict]:
    """Build the message list to send to the LLM from the three memory layers.

    Returns ``(context_messages, info)``. The layers enter the prompt by their
    rules: LTM first (always), then the working facts block (when present),
    then the last KEEP messages of the dialog "as is" (STM window). Each
    layer's token contribution is counted in ``info``.
    """
    messages = store.load_messages(branch_id)

    if new_user_text:
        facts, facts_usage = update_facts(new_user_text, branch_id)
    else:  # no new message — send the stored block as is
        facts, facts_usage = store.load_facts(branch_id), None

    ltm_text = memory.render_ltm()
    wm_text = memory.render_working(facts)
    start = safe_window_start(messages)
    stm_messages = messages[start:]
    stm_text = "".join(
        (m.get("content") or "") for m in stm_messages if m.get("role") != "assistant"
    )

    context: list[dict] = []
    if ltm_text:
        context.append({"role": "system", "content": ltm_text})
    if wm_text:
        context.append({"role": "system", "content": wm_text})
    context.extend(stm_messages)
    return context, {
        "mode": "memory",
        "ltm_used": bool(ltm_text),
        "ltm_tokens": count_tokens(ltm_text) if ltm_text else 0,
        "facts_used": bool(wm_text),
        "facts_tokens": count_tokens(wm_text) if wm_text else 0,
        "stm_tokens": count_tokens(stm_text),
        "as_is": len(stm_messages),
        "facts_usage": facts_usage,
    }
