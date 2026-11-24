"""Context assembly: the prompt is built from the profile + three memory layers.

There is a single context strategy — the memory model plus the user profile
(see ``app/memory.py`` / ``app/profile.py``). Each part contributes its own
section of the prompt by its own rule (``PROMPT_RULES``):

- **Профиль** (personalization) — the active profile's preferences rendered
  first as a system message, always: it defines HOW to answer;
- **LTM** (долговременная) — second system message, always;
- **WM**  (рабочая) — the current task's facts block, refreshed after every
  user message by :func:`update_facts`;
- **STM** (краткосрочная) — only the last KEEP dialog messages go "as is";
  everything older stays in the DB but is not sent.

Above all of them (and before the profile) come the **invariants**
(``app/invariants.py``) — the project rules the assistant must not violate;
they are injected as the very first system message of every request.

:func:`build_context` returns the message list plus per-section token counts
so the UI can show exactly how many tokens each part adds to the request.
"""
import json

from . import invariants, memory, profile, store
from .llm import chat, parse_json_object
from .tokens import count_tokens

KEEP = 10  # последние сообщения уходят в запрос "как есть"

FACTS_SYSTEM = (
    "Ты ведёшь блок «факты» диалога в формате JSON (ключ -> значение). "
    "Храни только важное: цель, ограничения, предпочтения, решения, "
    "договорённости, имена, числа, даты. Блок ОБНОВЛЯЕТСЯ, а не только "
    "пополняется: новое значение факта (срок, статус, версия, требование, "
    "подход) заменяет прежнее на том же ключе, отменённое — удаляется; "
    "рядом со старым значением новое не дублируется. Ничего не выдумывай. "
    "В ответе — только JSON без пояснений."
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
    """Parse the updater's reply into a facts dict (thin alias)."""
    return parse_json_object(raw)


def update_facts(user_text: str,
                 branch_id: int = store.ROOT_BRANCH_ID) -> tuple[dict, dict | None]:
    """Merge new information from the user message into the working memory.

    The facts block is kept per branch. When the new message CHANGES a stored
    value (срок сдвинулся, требование отменили…), the new value REPLACES the
    old one — the block is stored whole, and every replacement / removal /
    addition is journaled to ``memory_log`` (:func:`memory.log_working_changes`)
    as "что было → что стало" with both write times. Returns ``(facts,
    usage)``. On updater failure the previous facts are kept unchanged, so the
    agent keeps working.
    """
    old_full = store.load_facts_full(branch_id)
    old_facts = {k: v["value"] for k, v in old_full.items()}
    parts = ["Текущие факты (JSON; может быть пустым):",
             json.dumps(old_facts, ensure_ascii=False) if old_facts else "{}",
             "Новое сообщение пользователя:", user_text,
             "Верни обновлённый блок фактов в формате JSON. Правила: если новое "
             "сообщение меняет значение факта (срок, статус, требование, подход) — "
             "запиши НОВОЕ значение на место старого, прежнее значение НЕ сохраняй; "
             "отменённые требования удаляй из блока; ничего не выдумывай."]

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
        return old_facts, None

    new_facts = _parse_facts(raw)
    if new_facts is None:  # updater replied garbage — keep the old block
        print("[context] facts: не удалось разобрать JSON ответа, блок не изменён")
        return old_facts, None
    store.save_facts(new_facts, branch_id)
    try:
        memory.log_working_changes(old_full, new_facts, branch_id)
    except Exception as exc:  # журнал не должен ломать ход агента
        print(f"[context] не удалось записать журнал памяти: {exc}")
    return new_facts, _usage_dict(getattr(response, "usage", None))


def build_context(new_user_text: str | None = None,
                  branch_id: int = store.ROOT_BRANCH_ID) -> tuple[list[dict], dict]:
    """Build the message list to send to the LLM from profile + memory layers.

    Returns ``(context_messages, info)``. The sections enter the prompt by
    their rules: the active user profile first (always — it defines HOW to
    answer), then LTM, then the working facts block (when present), then the
    last KEEP messages of the dialog "as is" (STM window). Each section's
    token contribution is counted in ``info``.
    """
    messages = store.load_messages(branch_id)

    if new_user_text:
        facts, facts_usage = update_facts(new_user_text, branch_id)
    else:  # no new message — send the stored block as is
        facts, facts_usage = store.load_facts(branch_id), None
    facts_full = store.load_facts_full(branch_id)

    profile_text = profile.render()
    inv_text = invariants.render()
    ltm_text = memory.render_ltm()
    wm_text = memory.render_working(facts_full)
    start = safe_window_start(messages)
    stm_messages = messages[start:]
    stm_text = "".join(
        (m.get("content") or "") for m in stm_messages if m.get("role") != "assistant"
    )

    context: list[dict] = []
    # Инварианты — САМЫМ ПЕРВЫМ system-сообщением: они выше профиля и памяти,
    # потому что определяют, что ассистент не имеет права предлагать вообще.
    if inv_text:
        context.append({"role": "system", "content": inv_text})
    if profile_text:
        context.append({"role": "system", "content": profile_text})
    if ltm_text:
        context.append({"role": "system", "content": ltm_text})
    if wm_text:
        context.append({"role": "system", "content": wm_text})
    context.extend(stm_messages)
    return context, {
        "mode": "memory",
        "invariants_used": bool(inv_text),
        "invariants_tokens": count_tokens(inv_text) if inv_text else 0,
        "profile_used": bool(profile_text),
        "profile_tokens": count_tokens(profile_text) if profile_text else 0,
        "ltm_used": bool(ltm_text),
        "ltm_tokens": count_tokens(ltm_text) if ltm_text else 0,
        "facts_used": bool(wm_text),
        "facts_tokens": count_tokens(wm_text) if wm_text else 0,
        "stm_tokens": count_tokens(stm_text),
        "as_is": len(stm_messages),
        "facts_usage": facts_usage,
    }
