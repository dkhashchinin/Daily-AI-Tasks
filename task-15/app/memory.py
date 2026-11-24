"""Three-layer memory: short-term / working / long-term — with UPDATING.

The agent's knowledge is split into three memory layers, each stored in its
own SQLite table (separate storage):

**STM  — краткосрочная (текущий диалог)**
    `app/store.py` ``messages`` table (per branch). The raw dialog, kept whole.

**WM   — рабочая (данные текущей задачи)**
    `app/store.py` ``task_memory`` table (per branch). Key-value facts of the
    current task: goal, constraints, preferences, decisions, names, numbers,
    dates. Extracted from the dialog by :func:`context.update_facts` after each
    user message; every fact carries its write time. Cleared when the task
    finishes (:func:`on_task_finished`).

**LTM  — долговременная (профиль, решения, знания)**
    `app/store.py` ``long_term_memory`` table (global). Reusable facts that
    outlive a single dialog. NOT append-only: every entry has an update
    policy (``POLICIES``) and a write time, and a fact that conflicts with a
    stored one REPLACES it (:func:`store.upsert_long_term` — later write time
    wins), with the replacement journaled in ``memory_log`` as
    "что было → что стало".

Update rules per policy (``classify_policy`` decides automatically, an
explicit choice always wins):

- **latest**  — перезаписывается новым значением: срок, статус, версия,
  приоритет, выбранный подход… Хранится ровно одно актуальное значение.
- **append**  — накапливается списком с датами: принятые решения, история
  изменений. Новое значение дописывается, старые остаются в списке.
- **ttl**     — удаляется после завершения (или отмены) задачи: черновики и
  временные договорённости текущей задачи.

Every fact is written with a timestamp; in the prompt the write time is shown
next to the value, so the model can always tell the current value from an
outdated one — and outdated values are physically removed from memory, not
just marked stale.

What goes into the prompt — one rule per layer (``PROMPT_RULES``), enforced in
:func:`app.context.build_context`:

- **STM** →  только последние KEEP сообщений идут «как есть»; более ранние
  хранятся в БД, но в промпт не попадают.
- **WM**  →  весь блок фактов текущей задачи идёт system-сообщением (с
  временем записи каждого факта); очищается по завершении задачи.
- **LTM** →  вся долговременная память идёт system-сообщением (каждый факт —
  с временем записи и политикой обновления).

Explicit saving — what goes where is chosen, not automatic:
- dialog → WM: :func:`context.update_facts` (extract on each user turn);
- WM → LTM: :func:`promote_working` (pick keys / category, or the UI button);
- manual: :func:`save_long_term`.
"""
from datetime import datetime

from . import store

# LTM entries are grouped by these categories.
CATEGORIES = ("profile", "decision", "knowledge")
CATEGORY_LABELS = {
    "profile": "Профиль",
    "decision": "Решения",
    "knowledge": "Знания",
}

# --- Update policies (how a fact behaves when a new value arrives) -----------

POLICIES = ("latest", "append", "ttl")
POLICY_LABELS = {
    "latest": "перезапись",
    "append": "список",
    "ttl": "до конца задачи",
}
POLICY_RULES = {
    "latest": ("перезаписывается новым значением: хранится одно актуальное "
               "значение (срок, статус, версия, приоритет, выбранный подход)"),
    "append": ("накапливается списком с датами: принятые решения и история "
               "изменений дописываются, старые записи не удаляются"),
    "ttl": ("удаляется после завершения или отмены задачи: черновики и "
            "временные договорённости"),
}

# Key substrings → policy live in ``store.guess_policy`` (single source of
# truth, used by both the memory layer and the store's upsert).


def classify_policy(key: str, value="") -> str:
    """Guess the update policy of a fact from its key/value.

    Facts with mutable single-slot meaning (срок, статус, версия, подход…)
    default to ``latest`` — a newer value replaces the older one. History-like
    keys accumulate (``append``), task-scoped drafts expire (``ttl``).
    """
    return store.guess_policy(key, value)


def _policy_or_guess(policy: str | None, key: str, value="") -> str:
    if policy in POLICIES:
        return policy
    return classify_policy(key, value)


def _fmt_ts(ts: str | None) -> str:
    """'2026-11-21T15:40:12.345' → '2026-11-21 15:40'."""
    if not ts:
        return "?"
    return ts.replace("T", " ")[:16]


def render_ltm() -> str:
    """Render the whole long-term memory for injection into the prompt.

    Grouped by category (Профиль / Решения / Знания). Every fact is shown
    WITH its write time and (for non-default policies) its update rule, so
    the model always knows which value is current. ``append`` facts render as
    a dated list. Empty memory renders as "".
    """
    entries = store.list_long_term()
    if not entries:
        return ""
    lines = [
        "ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (профиль, решения, знания):",
        "Правила памяти: у каждого факта есть время записи; актуальным считается "
        "значение с самой поздней датой записи. Когда пользователь сообщает новое "
        "значение (срок, статус, версия, подход) — оно ЗАМЕНЯЕТ прежнее, прежнее "
        "устарело и использоваться не должно; решения и история изменений "
        "накапливаются списком.",
    ]
    by_cat: dict[str, list] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    for cat in CATEGORIES:
        items = by_cat.get(cat)
        if not items:
            continue
        lines.append(f"{CATEGORY_LABELS.get(cat, cat)}:")
        rendered_keys: set[str] = set()
        for e in items:  # already ordered by write time, oldest first
            if e["key"] in rendered_keys:
                continue
            rendered_keys.add(e["key"])
            same_key = [x for x in items if x["key"] == e["key"]]
            if e["policy"] == "append" and len(same_key) > 1:
                lines.append(f"- {e['key']} (список, дополняется):")
                for x in same_key:
                    lines.append(f"  · {_fmt_ts(x['updated_at'])} — {x['value']}")
            else:
                badge = "" if e["policy"] == "latest" \
                    else f", {POLICY_RULES[e['policy']]}"
                lines.append(f"- {e['key']}: {e['value']} "
                             f"(записано {_fmt_ts(e['updated_at'])}{badge})")
    return "\n".join(lines)


def render_working(facts_full: dict) -> str:
    """Render the current task's facts (working memory) as a system message.

    ``facts_full`` is :func:`store.load_facts_full` output — key → value with
    write time. The write time is shown next to every fact: after an update
    the model must use the value with the latest time and never quote the
    outdated one.
    """
    if not facts_full:
        return ""
    lines = [
        "ДАННЫЕ ТЕКУЩЕЙ ЗАДАЧИ (рабочая память):",
        "Это актуальные данные задачи с временем записи каждого факта; если "
        "сообщение пользователя противоречит какому-то значению — используй "
        "новое значение, прежнее устарело.",
    ]
    for key in sorted(facts_full):
        item = facts_full[key]
        value = item["value"] if isinstance(item, dict) else item
        ts = item.get("updated_at") if isinstance(item, dict) else None
        lines.append(f"- {key}: {value} (записано {_fmt_ts(ts)})")
    return "\n".join(lines)


def promote_working(branch_id: int = store.ROOT_BRANCH_ID,
                    keys: list[str] | None = None,
                    category: str | None = None,
                    policy: str | None = None,
                    remove: bool = False) -> int:
    """Explicitly move working-memory facts into long-term memory.

    ``keys`` picks which facts to promote (None = all). Each fact is stored
    via :func:`store.upsert_long_term` — so promoting an already-known fact
    with a new value UPDATES it (conflict resolution), instead of adding a
    duplicate. ``category`` forces one category, otherwise it is guessed per
    fact; ``policy`` forces the update policy, otherwise it is guessed from
    the key. ``remove`` optionally drops the fact from working memory after
    the move. Returns how many entries were written.
    """
    facts_full = store.load_facts_full(branch_id)
    if keys is None:
        selected = facts_full
    else:
        selected = {k: facts_full[k] for k in keys if k in facts_full}
    written = 0
    for key, item in selected.items():
        value = item["value"] if isinstance(item, dict) else item
        store.upsert_long_term(
            category or _classify(key, value), key, value,
            source="promoted", policy=_policy_or_guess(policy, key, value))
        written += 1
    if remove and selected:
        remaining = {k: v for k, v in store.load_facts(branch_id).items()
                     if k not in selected}
        store.save_facts(remaining, branch_id)
    return written


def save_long_term(category: str, key: str, value: str,
                   policy: str | None = None) -> int:
    """Manually save one long-term fact (explicit choice).

    Goes through :func:`store.upsert_long_term`: a fact whose key already
    exists is REPLACED by the new value (later write time wins, replacement
    logged), history-like keys accumulate as a list.
    """
    if category not in CATEGORIES:
        category = "knowledge"
    key = str(key).strip()
    value = str(value).strip()
    if not key or not value:
        raise ValueError("Ключ и значение не могут быть пустыми.")
    entry_id, _action = store.upsert_long_term(
        category, key, value, source="manual",
        policy=_policy_or_guess(policy, key, value))
    return entry_id


def log_working_changes(old_full: dict, new_facts: dict,
                        branch_id: int = store.ROOT_BRANCH_ID,
                        source: str = "wm-updater") -> int:
    """Journal working-memory changes after :func:`context.update_facts`.

    Compares the previous fact block (with write times) against the new one
    and logs to ``memory_log``: value changed → ``overwrite`` (что было →
    что стало, оба времени записи), fact removed → ``delete``, fact added →
    ``add``. Returns how many log entries were written.
    """
    written = 0
    now = datetime.now().isoformat()
    for key in sorted(set(old_full) | set(new_facts)):
        old_item = old_full.get(key)
        old_value = (old_item["value"] if isinstance(old_item, dict)
                     else old_item) if old_item is not None else None
        old_ts = old_item.get("updated_at") if isinstance(old_item, dict) else None
        if key not in new_facts:
            if old_item is not None:
                store.add_memory_log(
                    "delete", "wm", key, str(old_value), None, old_ts, None,
                    source, f"факт задачи удалён: {key}: {old_value}")
                written += 1
            continue
        new_value = new_facts[key]
        if old_item is None:
            store.add_memory_log(
                "add", "wm", key, None, str(new_value), None, now,
                source, f"новый факт задачи: {key}: {new_value}")
            written += 1
        elif old_value != new_value:
            store.add_memory_log(
                "overwrite", "wm", key, str(old_value), str(new_value),
                old_ts, now, source,
                f"факт задачи {key}: {old_value} → {new_value}")
            written += 1
    return written


def memory_change_log(limit: int = 50) -> list[dict]:
    """Memory change-log entries, newest first (для панели «Память»)."""
    return store.list_memory_log(limit)


def on_task_finished(branch_id: int = store.ROOT_BRANCH_ID,
                     reason: str = "задача завершена") -> int:
    """Cleanup when the current task finishes (done or cancelled).

    Rules for clearing:

    - **рабочая память (WM) ветки очищается целиком** — её факты принадлежали
      завершённой задаче;
    - **долговременные факты с политикой ``ttl``** («до конца задачи»)
      удаляются из LTM;
    - остальная долговременная память (``latest`` / ``append``) сохраняется.

    Every removal is journaled to ``memory_log``. Returns how many facts were
    removed in total.
    """
    removed = 0
    facts = store.load_facts(branch_id)
    store.clear_task_memory(branch_id)
    removed += len(facts)
    for entry in store.list_long_term():
        if entry["policy"] == "ttl":
            store.delete_long_term(entry["id"])
            removed += 1
    if removed:
        store.add_memory_log(
            "cleanup", "wm", "*", None, f"удалено фактов: {removed}",
            None, datetime.now().isoformat(), "task-fsm",
            f"{reason}: рабочая память очищена, ttl-факты LTM удалены")
    return removed


def _classify(key: str, value) -> str:
    """Best-guess category for a fact; used when promoting / seeding."""
    text = f"{key} {value}".lower()
    name_hints = ("имя", "фио", "name", "клиент", "client", "заказчик",
                  "мена", "компания", "company", "должность", "должност")
    decision_hints = ("реши", "отказ", "decision", "согласова", "договор",
                      "правило", "политика", "огранич")
    if any(h in text for h in decision_hints):
        return "decision"
    if any(h in text for h in name_hints):
        return "profile"
    return "knowledge"


# The prompt-inclusion rule for each memory layer. Shown in the UI and
# enforced by build_context: LTM is always injected first, WM is injected when
# present, and STM contributes only the last KEEP messages.
PROMPT_RULES = [
    {
        "layer": "STM",
        "title": "Краткосрочная (текущий диалог)",
        "storage": "таблица messages (на каждую ветку)",
        "rule": "В промпт попадают только последние KEEP сообщений «как есть»; "
                "более ранние лежат в БД, но не отправляются модели.",
    },
    {
        "layer": "WM",
        "title": "Рабочая (данные текущей задачи)",
        "storage": "таблица task_memory (на каждую ветку)",
        "rule": "Весь блок фактов текущей задачи с временем записи каждого "
                "факта уходит system-сообщением сразу после долговременной "
                "памяти. Обновляется после каждой реплики пользователя: новое "
                "значение заменяет устаревшее (замена фиксируется в журнале "
                "памяти); очищается при завершении или отмене задачи.",
    },
    {
        "layer": "LTM",
        "title": "Долговременная (профиль, решения, знания)",
        "storage": "таблица long_term_memory (глобально)",
        "rule": "Вся долговременная память уходит system-сообщением сразу после "
                "профиля пользователя — она задаёт, кто ты и что ты знаешь. "
                "Память обновляется, а не только пополняется: срок/статус/"
                "версия перезаписываются новым значением, решения и история "
                "изменений накапливаются списком, черновики (ttl) удаляются "
                "после завершения задачи; у каждого факта — время записи, "
                "актуально значение с самой поздней датой, конфликт решается "
                "заменой (что было → что стало — в журнале памяти). "
                "Переживает «Сброс» и ветки.",
    },
]
