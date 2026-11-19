"""Three-layer memory: short-term / working / long-term.

The agent's knowledge is split into three memory layers, each stored in its
own SQLite table (separate storage) and promoted explicitly:

**STM  — краткосрочная (текущий диалог)**
    `app/store.py` ``messages`` table (per branch). The raw dialog, kept whole.

**WM   — рабочая (данные текущей задачи)**
    `app/store.py` ``task_memory`` table (per branch). Key-value facts of the
    current task: goal, constraints, preferences, decisions, names, numbers,
    dates. Extracted from the dialog by :func:`context.update_facts` after each
    user message.

**LTM  — долговременная (профиль, решения, знания)**
    `app/store.py` ``long_term_memory`` table (global). Reusable facts that
    outlive a single dialog: who the user/company is, standing decisions,
    domain knowledge. Survives dialog reset and branches.

What goes into the prompt — one rule per layer (``PROMPT_RULES``), enforced in
:func:`app.context.build_context`:

- **STM** →  только последние KEEP сообщений идут «как есть»; более ранние
  хранятся в БД, но в промпт не попадают.
- **WM**  →  весь блок фактов текущей задачи идёт system-сообщением.
- **LTM** →  вся долговременная память идёт первым system-сообщением, всегда.

Explicit saving — what goes where is chosen, not automatic:
- dialog → WM: :func:`context.update_facts` (extract on each user turn);
- WM → LTM: :func:`promote_working` (pick keys / category, or the UI button);
- manual: :func:`save_long_term`.
"""
from . import store

# LTM entries are grouped by these categories.
CATEGORIES = ("profile", "decision", "knowledge")
CATEGORY_LABELS = {
    "profile": "Профиль",
    "decision": "Решения",
    "knowledge": "Знания",
}

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
        "rule": "Весь блок фактов текущей задачи уходит system-сообщением "
                "сразу после долговременной памяти, пока задача (ветка) активна.",
    },
    {
        "layer": "LTM",
        "title": "Долговременная (профиль, решения, знания)",
        "storage": "таблица long_term_memory (глобально)",
        "rule": "Вся долговременная память уходит system-сообщением сразу после "
                "профиля пользователя — она задаёт, кто ты и что ты знаешь. "
                "Переживает «Сброс» и ветки.",
    },
]


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


def render_ltm() -> str:
    """Render the whole long-term memory for injection into the prompt.

    Grouped by category (Профиль / Решения / Знания). Empty memory renders as "".
    """
    entries = store.list_long_term()
    if not entries:
        return ""
    by_cat: dict[str, list] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    lines = ["ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (профиль, решения, знания):"]
    for cat in CATEGORIES:
        items = by_cat.get(cat)
        if not items:
            continue
        lines.append(f"{CATEGORY_LABELS.get(cat, cat)}:")
        for e in items:
            lines.append(f"- {e['key']}: {e['value']}")
    return "\n".join(lines)


def render_working(facts: dict) -> str:
    """Render the current task's facts (working memory) as a system message."""
    if not facts:
        return ""
    lines = ["ДАННЫЕ ТЕКУЩЕЙ ЗАДАЧИ (рабочая память):"]
    for key in sorted(facts):
        lines.append(f"- {key}: {facts[key]}")
    return "\n".join(lines)


def promote_working(branch_id: int = store.ROOT_BRANCH_ID,
                    keys: list[str] | None = None,
                    category: str | None = None,
                    remove: bool = False) -> int:
    """Explicitly move working-memory facts into long-term memory.

    ``keys`` picks which facts to promote (None = all). Each fact is stored as
    an LTM entry; ``category`` forces one category, otherwise it is guessed per
    fact. ``remove`` optionally drops the fact from working memory after the
    move. Returns how many entries were added.
    """
    facts = store.load_facts(branch_id)
    selected = facts if keys is None else {k: facts[k] for k in keys if k in facts}
    added = 0
    for key, value in selected.items():
        store.add_long_term(category or _classify(key, value), key, value)
        added += 1
    if remove and selected:
        remaining = {k: v for k, v in facts.items() if k not in selected}
        store.save_facts(remaining, branch_id)
    return added


def save_long_term(category: str, key: str, value: str) -> int:
    """Manually save one long-term memory entry (explicit choice)."""
    if category not in CATEGORIES:
        category = "knowledge"
    key = str(key).strip()
    value = str(value).strip()
    if not key or not value:
        raise ValueError("Ключ и значение не могут быть пустыми.")
    return store.add_long_term(category, key, value, source="manual")
