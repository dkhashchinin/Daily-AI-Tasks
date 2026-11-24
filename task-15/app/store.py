"""Persistent conversation storage (SQLite).

The whole dialog history — user messages, assistant replies, tool calls and
their results — is stored in a SQLite database so the agent can be restarted
and continue a conversation as if it never stopped. Dialogs are organized in
branches (copy-on-fork).

Memory is split into three layers, each stored in its own table:

- **short-term (STM)** — the current dialog: ``messages`` (per branch).
- **working (WM)** — the current task's facts: ``task_memory`` (per branch).
- **long-term (LTM)** — profile, decisions, knowledge: ``long_term_memory``
  (global). Survives dialog reset and branches. LTM is UPDATABLE: every entry
  carries a ``policy`` (``latest`` — overwritten by a newer value / ``append``
  — accumulated as a dated list / ``ttl`` — dropped when the task finishes),
  a ``version`` counter and write timestamps; a fact conflicting with a
  stored one REPLACES it (later write time wins) and every replacement is
  journaled in ``memory_log`` as "what was → what became".

The per-branch ``context_state`` table only holds legacy keys from earlier
context strategies; they are migrated/cleared and the working-memory facts
block lives in ``task_memory``. The per-branch ``task_state`` table holds the
task lifecycle automaton's state (``app/task_fsm.py``) as one JSON blob.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "conversation.db"

ROOT_BRANCH_ID = 1

# --- Update-policy classification (single source of truth) ---------------------
# Key substrings → policy. Checked in order; an explicit policy argument
# always wins over the guess. Facts with mutable single-slot meaning
# (срок, статус, версия, подход…) default to 'latest' — a newer value
# replaces the older one; history-like keys accumulate ('append');
# task-scoped drafts expire ('ttl').
_APPEND_HINTS = ("решени", "истори", "изменени", "согласован", "журнал", "лог",
                 "log", "decision", "history", "changelog")
_TTL_HINTS = ("черновик", "временн", "tmp", "temp", "промежуточ")


def guess_policy(key: str, value="") -> str:
    """Guess the update policy of a fact from its key/value (see above)."""
    text = f"{key} {value}".lower()
    if any(h in text for h in _APPEND_HINTS):
        return "append"
    if any(h in text for h in _TTL_HINTS):
        return "ttl"
    return "latest"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _encode_fact(value) -> str:
    """Serialize a working-memory fact value to a JSON string (round-trips
    numbers, bools and nested structures)."""
    return json.dumps(value, ensure_ascii=False)


def _decode_fact(raw: str):
    """Parse a fact value stored with :func:`_encode_fact`; raw text on error."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def init_db() -> None:
    """Create tables, migrate old schemas, and seed the root branch."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                role          TEXT NOT NULL,
                content       TEXT,
                tool_calls    TEXT,
                tool_call_id  TEXT,
                created_at    TEXT NOT NULL
            )
            """
        )
        # Legacy per-branch key-value state ("b<id>:summary" and friends from
        # earlier context strategies). Kept only so old databases migrate
        # cleanly; the working-memory facts block lives in ``task_memory``.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        # Dialog branches: a branch copies the first fork_len messages of its
        # parent at creation time and then evolves independently.
        # checkpoint_len is a saved fork point (None = not set).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT NOT NULL,
                parent_id      INTEGER,
                fork_len       INTEGER,
                checkpoint_len INTEGER,
                created_at     TEXT NOT NULL
            )
            """
        )
        # Working memory (WM): facts of the CURRENT TASK, per branch
        # (key -> value). This is the "данные текущей задачи" layer. It is
        # deliberately its own table, separate from the dialog (STM) and from
        # the long-term profile/decisions/knowledge (LTM), because the three
        # memory layers are stored separately and promoted explicitly.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_memory (
                branch_id  INTEGER NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (branch_id, key)
            )
            """
        )
        # Long-term memory (LTM): profile, decisions and knowledge that survive
        # dialog reset and branches. Each entry has a category, a source
        # ('manual' for hand-added entries, 'dialog' for profile rules learned
        # from the dialog) and an UPDATE POLICY:
        #   latest — a single-slot fact (срок, статус, версия): a newer value
        #            REPLACES the old one (conflict resolution, later write
        #            time wins);
        #   append — an accumulating fact (принятые решения, история
        #            изменений): new values are added to the dated list;
        #   ttl    — a task-scoped fact: removed when the task finishes.
        # ``version`` counts overwrites; ``updated_at`` is the write time that
        # decides which value is current.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                category   TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                source     TEXT NOT NULL,
                policy     TEXT NOT NULL DEFAULT 'latest',
                version    INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Memory change log: every add / overwrite / append / delete of a fact
        # (LTM and WM) lands here as "what was → what became", with the write
        # times of both values — the audit trail for conflict resolution.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT NOT NULL,
                action    TEXT NOT NULL,
                layer     TEXT NOT NULL,
                key       TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                old_ts    TEXT,
                new_ts    TEXT,
                source    TEXT NOT NULL,
                detail    TEXT
            )
            """
        )
        # User profiles: personalization on top of the memory model. A profile
        # is a named set of preferences (style, format, tone, constraints).
        # Exactly one profile is active at a time; its rules are injected into
        # every request as the first system message. Rules carry a source:
        # 'manual' (set by hand) or 'dialog' (learned from the conversation).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                is_active   INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_rules (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                kind       TEXT NOT NULL,
                text       TEXT NOT NULL,
                source     TEXT NOT NULL DEFAULT 'manual',
                updated_at TEXT NOT NULL
            )
            """
        )
        # Task lifecycle automaton state (app/task_fsm.py), one JSON blob per
        # branch: stage, current step, plan position, validation checks, history.
        # Kept in SQLite so a task can be paused at any stage and continued
        # after a restart without re-explanations.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_state (
                branch_id  INTEGER PRIMARY KEY,
                state      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Invariants (app/invariants.py): rules the assistant must not violate
        # (architecture / decisions / stack / business rules). Stored in their
        # OWN tables, separate from the dialog: no dialog path can mutate them
        # — only explicit commands (UI panel or /invariant …) go through
        # ``invariants.*`` functions, and every change lands in invariant_log
        # as "what was replaced with what".
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invariants (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                category   TEXT NOT NULL,
                title      TEXT NOT NULL,
                rule       TEXT NOT NULL,
                patterns   TEXT NOT NULL DEFAULT '[]',
                source     TEXT NOT NULL DEFAULT 'seed',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invariant_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT NOT NULL,
                action    TEXT NOT NULL,
                inv_id    INTEGER,
                field     TEXT,
                old_value TEXT,
                new_value TEXT,
                source    TEXT NOT NULL,
                detail    TEXT
            )
            """
        )
        # Migration: old flat history (no branch_id) becomes the root branch.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "branch_id" not in cols:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN branch_id INTEGER NOT NULL DEFAULT 1"
            )
        # Migration: context_state keys become per-branch ("summary" -> "b1:summary").
        old_keys = [r["key"] for r in conn.execute("SELECT key FROM context_state").fetchall()]
        for key in old_keys:
            if ":" in key:
                continue
            row = conn.execute(
                "SELECT value FROM context_state WHERE key = ?", (key,)
            ).fetchone()
            conn.execute(
                "INSERT OR REPLACE INTO context_state (key, value) VALUES (?, ?)",
                (f"b{ROOT_BRANCH_ID}:{key}", row["value"]),
            )
            conn.execute("DELETE FROM context_state WHERE key = ?", (key,))
        # Migration: the sticky facts block used to live in context_state as
        # "bN:facts" (a JSON object). It is the working-memory layer, so it
        # moves to its own table as one row per fact.
        for row in conn.execute(
            "SELECT key, value FROM context_state WHERE key LIKE 'b%:facts'"
        ).fetchall():
            key = row["key"]
            try:
                branch_id = int(key.split(":", 1)[0][1:])
            except ValueError:
                continue
            try:
                facts = json.loads(row["value"]) if row["value"] else {}
            except json.JSONDecodeError:
                continue
            now = datetime.now().isoformat()
            for fkey, fval in facts.items():
                conn.execute(
                    "INSERT OR REPLACE INTO task_memory (branch_id, key, value, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (branch_id, str(fkey), _encode_fact(fval), now),
                )
            conn.execute("DELETE FROM context_state WHERE key = ?", (key,))

        # Migration: long_term_memory gains the update-policy columns
        # (policy / version). Existing rows become 'latest', version 1.
        ltm_cols = {r["name"] for r in conn.execute("PRAGMA table_info(long_term_memory)").fetchall()}
        if "policy" not in ltm_cols:
            conn.execute("ALTER TABLE long_term_memory ADD COLUMN policy TEXT NOT NULL DEFAULT 'latest'")
        if "version" not in ltm_cols:
            conn.execute("ALTER TABLE long_term_memory ADD COLUMN version INTEGER NOT NULL DEFAULT 1")

        if conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO branches (id, name, parent_id, fork_len, checkpoint_len, created_at)"
                " VALUES (?, 'Основной диалог', NULL, 0, NULL, ?)",
                (ROOT_BRANCH_ID, datetime.now().isoformat()),
            )
        # Seed the default profile so the personalization layer is always on.
        if conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO profiles (name, description, is_active, created_at)"
                " VALUES (?, ?, 1, ?)",
                ("Базовый", "Профиль по умолчанию", datetime.now().isoformat()),
            )
            conn.execute(
                "INSERT INTO profile_rules (profile_id, kind, text, source, updated_at)"
                " VALUES (?, 'constraint', 'Отвечай на русском языке', 'manual', ?)",
                (conn.execute("SELECT last_insert_rowid()").fetchone()[0],
                 datetime.now().isoformat()),
            )
        # Seed the project invariants exactly once (app/invariants.py); the
        # marker survives removal of all invariants by explicit commands.
        from . import invariants as _invariants  # local import: avoids a module cycle
        _invariants.seed_into(conn)


def load_messages(branch_id: int = ROOT_BRANCH_ID) -> list[dict]:
    """Return one branch's conversation in order, in OpenAI message format."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages "
            "WHERE branch_id = ? ORDER BY id",
            (branch_id,),
        ).fetchall()

    messages: list[dict] = []
    for row in rows:
        message = {"role": row["role"], "content": row["content"]}
        if row["tool_calls"]:
            message["tool_calls"] = json.loads(row["tool_calls"])
        if row["tool_call_id"]:
            message["tool_call_id"] = row["tool_call_id"]
        messages.append(message)
    return messages


def save_messages(messages: list[dict], branch_id: int = ROOT_BRANCH_ID) -> None:
    """Replace one branch's stored history with the given full message list."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE branch_id = ?", (branch_id,))
        for message in messages:
            conn.execute(
                """
                INSERT INTO messages (branch_id, role, content, tool_calls, tool_call_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    branch_id,
                    message.get("role"),
                    message.get("content"),
                    json.dumps(message["tool_calls"]) if message.get("tool_calls") else None,
                    message.get("tool_call_id"),
                    datetime.now().isoformat(),
                ),
            )


def list_branches() -> list[dict]:
    """All branches with message counts, fork point and checkpoint, in order."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT b.id, b.name, b.parent_id, b.fork_len, b.checkpoint_len,
                   (SELECT COUNT(*) FROM messages m WHERE m.branch_id = b.id) AS count
            FROM branches b ORDER BY b.id
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_branch(branch_id: int) -> dict | None:
    """One branch record (with message count) or None if it does not exist."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT b.id, b.name, b.parent_id, b.fork_len, b.checkpoint_len,
                   (SELECT COUNT(*) FROM messages m WHERE m.branch_id = b.id) AS count
            FROM branches b WHERE b.id = ?
            """,
            (branch_id,),
        ).fetchone()
    return dict(row) if row else None


def set_checkpoint(branch_id: int, length: int) -> None:
    """Remember a checkpoint in the branch: a fork point for new branches."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE branches SET checkpoint_len = ? WHERE id = ?",
            (max(0, length), branch_id),
        )


def create_branch(parent_id: int, name: str | None = None,
                  fork_len: int | None = None) -> dict:
    """Create a branch off ``parent_id`` at its checkpoint (or at the end).

    The first ``fork_len`` parent messages are copied into the new branch
    (copy-on-fork), so it continues the dialog from that exact place and then
    evolves independently. Returns the new branch record.
    """
    init_db()
    parent = get_branch(parent_id)
    if parent is None:
        raise ValueError("Ветка не найдена.")

    if fork_len is None:
        fork_len = parent["checkpoint_len"]
    if fork_len is None:
        fork_len = parent["count"]
    fork_len = max(0, min(fork_len, parent["count"]))

    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO branches (name, parent_id, fork_len, checkpoint_len, created_at)"
            " VALUES (?, ?, ?, NULL, ?)",
            (name or "", parent_id, fork_len, datetime.now().isoformat()),
        )
        new_id = cur.lastrowid
        if not name:
            conn.execute("UPDATE branches SET name = ? WHERE id = ?",
                         (f"Ветка {new_id}", new_id))
        if fork_len:
            conn.execute(
                """
                INSERT INTO messages (branch_id, role, content, tool_calls, tool_call_id, created_at)
                SELECT ?, role, content, tool_calls, tool_call_id, ?
                FROM messages WHERE branch_id = ? ORDER BY id LIMIT ?
                """,
                (new_id, datetime.now().isoformat(), parent_id, fork_len),
            )
    return get_branch(new_id)


def delete_branch(branch_id: int) -> None:
    """Drop a branch with its messages and per-branch context state."""
    init_db()
    if branch_id == ROOT_BRANCH_ID:
        raise ValueError("Основную ветку удалить нельзя.")
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE branch_id = ?", (branch_id,))
        conn.execute("DELETE FROM task_memory WHERE branch_id = ?", (branch_id,))
        conn.execute("DELETE FROM task_state WHERE branch_id = ?", (branch_id,))
        conn.execute("DELETE FROM branches WHERE id = ?", (branch_id,))
        conn.execute("DELETE FROM context_state WHERE key LIKE ?",
                     (f"b{branch_id}:%",))


def load_facts(branch_id: int = ROOT_BRANCH_ID) -> dict:
    """Return the branch's working-memory facts (key -> value).

    Working memory lives in its own table (`task_memory`), one row per fact,
    so it is stored separately from the dialog (STM / `messages`) and from the
    long-term profile/decisions/knowledge (`long_term_memory`).
    """
    return {k: v["value"] for k, v in load_facts_full(branch_id).items()}


def load_facts_full(branch_id: int = ROOT_BRANCH_ID) -> dict:
    """The branch's working-memory facts WITH their write times:
    ``key -> {"value": ..., "updated_at": ...}`` — the timestamp is what lets
    the prompt (and the change log) tell the current value from an outdated
    one."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT key, value, updated_at FROM task_memory WHERE branch_id = ? ORDER BY key",
            (branch_id,),
        ).fetchall()
    return {
        r["key"]: {"value": _decode_fact(r["value"]), "updated_at": r["updated_at"]}
        for r in rows
    }


def save_facts(facts: dict, branch_id: int = ROOT_BRANCH_ID) -> None:
    """Replace the branch's working-memory facts block."""
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM task_memory WHERE branch_id = ?", (branch_id,))
        for key, value in facts.items():
            conn.execute(
                "INSERT OR REPLACE INTO task_memory (branch_id, key, value, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (branch_id, str(key), _encode_fact(value), now),
            )


def clear_task_memory(branch_id: int | None = None) -> None:
    """Drop the working memory of one branch, or all branches when
    ``branch_id`` is None. Does not touch long-term memory."""
    init_db()
    with _connect() as conn:
        if branch_id is None:
            conn.execute("DELETE FROM task_memory")
        else:
            conn.execute("DELETE FROM task_memory WHERE branch_id = ?", (branch_id,))


def delete_fact(branch_id: int, key: str) -> None:
    """Remove a single fact from the branch's working memory."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM task_memory WHERE branch_id = ? AND key = ?",
            (branch_id, key),
        )


# --- Task lifecycle state (finite automaton, per branch) -----------------------


def load_task_state(branch_id: int = ROOT_BRANCH_ID) -> dict | None:
    """The branch's task FSM state (parsed JSON) or None when no task ran."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT state FROM task_state WHERE branch_id = ?", (branch_id,)
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["state"])
    except (json.JSONDecodeError, TypeError):
        return None


def save_task_state(state: dict, branch_id: int = ROOT_BRANCH_ID) -> None:
    """Persist the branch's task FSM state as one JSON blob."""
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO task_state (branch_id, state, updated_at)
            VALUES (?, ?, ?)
            """,
            (branch_id, json.dumps(state, ensure_ascii=False),
             datetime.now().isoformat()),
        )


def clear_task_state(branch_id: int | None = None) -> None:
    """Drop the task FSM state of one branch, or all branches when None."""
    init_db()
    with _connect() as conn:
        if branch_id is None:
            conn.execute("DELETE FROM task_state")
        else:
            conn.execute("DELETE FROM task_state WHERE branch_id = ?", (branch_id,))


def clear_dialog(branch_id: int = ROOT_BRANCH_ID) -> None:
    """Clear one branch's dialog (STM): messages, legacy per-branch state,
    checkpoint.

    Working memory (``task_memory``) and long-term memory are not touched.
    The task automaton is reset to idle: its plan position refers to message
    indexes that no longer exist after the dialog is gone.
    """
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE branch_id = ?", (branch_id,))
        conn.execute("DELETE FROM context_state WHERE key LIKE ?",
                     (f"b{branch_id}:%",))
        conn.execute("UPDATE branches SET checkpoint_len = NULL WHERE id = ?",
                     (branch_id,))
    clear_task_state(branch_id)


# --- Long-term memory (profile, decisions, knowledge) -------------------------


def list_long_term(source: str | None = None) -> list[dict]:
    """Every long-term memory entry (with policy / version / write time),
    oldest write first."""
    init_db()
    with _connect() as conn:
        if source is None:
            rows = conn.execute(
                "SELECT id, category, key, value, source, policy, version,"
                " updated_at FROM long_term_memory ORDER BY updated_at, id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, category, key, value, source, policy, version,"
                " updated_at FROM long_term_memory WHERE source = ?"
                " ORDER BY updated_at, id",
                (source,),
            ).fetchall()
    return [dict(r) for r in rows]


def add_long_term(category: str, key: str, value: str,
                  source: str = "manual", policy: str = "latest") -> int:
    """Add a long-term memory entry (plain insert, logged as 'add') and
    return its id. For conflict-resolving writes use :func:`upsert_long_term`.
    """
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO long_term_memory (category, key, value, source, policy,"
            " version, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (category, str(key), str(value), source, policy, now),
        )
        add_memory_log("add", "ltm", str(key), None, str(value), None, now, source,
                       f"добавлен факт [{category}] {key}: {value}", conn=conn)
        return cur.lastrowid


def upsert_long_term(category: str, key: str, value: str,
                     source: str = "manual", policy: str | None = None) -> tuple[int, str]:
    """Write a long-term fact WITH conflict resolution.

    Rules (the memory is updatable, not append-only):

    - **no entry with this key** → a new row is added (action ``add``);
    - **entry exists and its policy is ``append``** (or the incoming fact is
      ``append``) → the new value JOINS the dated list as its own row
      (action ``append``) — accepted decisions and change histories are
      accumulated, not overwritten;
    - **entry exists otherwise** (``latest`` / ``ttl``) → the row with the
      LATEST write time wins and its value is REPLACED in place (action
      ``overwrite``, ``version`` bumped); no parallel stale copy is kept —
      other same-key rows (e.g. left by older writes) are dropped, so the old
      value never resurfaces in the prompt.

    Every action is journaled to ``memory_log`` with both values and both
    write times ("что было → что стало"). Returns ``(entry_id, action)``.
    """
    init_db()
    key = str(key).strip()
    value = str(value).strip()
    now = datetime.now().isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, category, key, value, policy, version, updated_at"
            " FROM long_term_memory WHERE key = ? ORDER BY updated_at DESC, id DESC",
            (key,),
        ).fetchall()
        if not rows:
            cur = conn.execute(
                "INSERT INTO long_term_memory (category, key, value, source, policy,"
                " version, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                (category, key, value, source, policy or guess_policy(key, value), now),
            )
            add_memory_log("add", "ltm", key, None, value, None, now, source,
                           f"новый факт [{category}] {key}: {value}", conn=conn)
            return cur.lastrowid, "add"

        newest = rows[0]
        chosen_policy = policy or newest["policy"]
        if chosen_policy == "append" or newest["policy"] == "append":
            chosen_policy = chosen_policy if chosen_policy == "append" else newest["policy"]
            cur = conn.execute(
                "INSERT INTO long_term_memory (category, key, value, source, policy,"
                " version, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                (category, key, value, source, "append", now),
            )
            add_memory_log("append", "ltm", key, None, value, None, now, source,
                           f"дополнен список [{category}] {key}: {value}", conn=conn)
            return cur.lastrowid, "append"

        # latest / ttl: the new value REPLACES the old one — one row, bumped
        # version, fresh write time; stale same-key rows are removed so the
        # outdated value cannot enter the prompt alongside the new one.
        old_value, old_ts = newest["value"], newest["updated_at"]
        version = int(newest["version"] or 1) + 1
        conn.execute(
            "UPDATE long_term_memory SET category = ?, value = ?, source = ?,"
            " policy = ?, version = ?, updated_at = ? WHERE id = ?",
            (category, value, source, chosen_policy, version, now, newest["id"]),
        )
        dropped = [r for r in rows if r["id"] != newest["id"]]
        for r in dropped:
            conn.execute("DELETE FROM long_term_memory WHERE id = ?", (r["id"],))
        detail = f"факт [{category}] {key}: {old_value} → {value}"
        if dropped:
            detail += f" (удалено устаревших копий: {len(dropped)})"
        add_memory_log("overwrite", "ltm", key, old_value, value, old_ts, now,
                       source, detail, conn=conn)
        return newest["id"], "overwrite"


def delete_long_term(entry_id: int) -> None:
    """Remove a single long-term memory entry (removal is logged)."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT key, value FROM long_term_memory WHERE id = ?", (entry_id,)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM long_term_memory WHERE id = ?", (entry_id,))
            add_memory_log("delete", "ltm", row["key"], row["value"], None,
                           None, None, "manual",
                           f"удалён факт {row['key']}: {row['value']}", conn=conn)


def clear_long_term(source: str | None = None) -> None:
    """Clear the whole long-term memory, or only entries from ``source``
    (clearing is logged)."""
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM long_term_memory").fetchall() \
            if source is None else conn.execute(
                "SELECT key, value FROM long_term_memory WHERE source = ?", (source,)
            ).fetchall()
        if source is None:
            conn.execute("DELETE FROM long_term_memory")
        else:
            conn.execute("DELETE FROM long_term_memory WHERE source = ?", (source,))
    if rows:
        add_memory_log("clear", "ltm", "*", None,
                       f"удалено записей: {len(rows)}", None, None, "manual",
                       "очищена долговременная память")


# --- Memory change log (what was replaced with what, both write times) --------


def add_memory_log(action: str, layer: str, key: str,
                   old_value: str | None, new_value: str | None,
                   old_ts: str | None, new_ts: str | None,
                   source: str, detail: str, conn=None) -> None:
    """Append one memory-log entry: the action, both values and both write
    times — the audit trail of conflict resolution (LTM and WM).

    ``conn`` lets a caller that already holds a write transaction (upsert /
    delete) journal into the SAME connection — a second connection would
    deadlock on SQLite's write lock.
    """
    if conn is None:
        init_db()
        with _connect() as own:
            _add_memory_log_row(own, action, layer, key, old_value, new_value,
                                old_ts, new_ts, source, detail)
        return
    _add_memory_log_row(conn, action, layer, key, old_value, new_value,
                        old_ts, new_ts, source, detail)


def _add_memory_log_row(conn, action, layer, key, old_value, new_value,
                        old_ts, new_ts, source, detail) -> None:
    conn.execute(
        "INSERT INTO memory_log (ts, action, layer, key, old_value, new_value,"
        " old_ts, new_ts, source, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), action, layer, key, old_value, new_value,
         old_ts, new_ts, source, detail),
    )


def list_memory_log(limit: int = 50) -> list[dict]:
    """Memory change-log entries, newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, action, layer, key, old_value, new_value, old_ts,"
            " new_ts, source, detail FROM memory_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_history() -> None:
    """Drop the dialog (STM) and the working memory (WM), all branches and
    per-branch state; reseed the root branch.

    Long-term memory (``long_term_memory``) is deliberately NOT touched: it is
    the profile / decisions / knowledge layer that survives resets. The
    invariants (and their seeding marker) are not touched either — they are
    project rules, not dialog.
    """
    init_db()
    from . import invariants as _invariants  # local import: avoids a module cycle
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM context_state WHERE key != ?",
                     (_invariants.SEED_MARKER,))
        conn.execute("DELETE FROM task_memory")
        conn.execute("DELETE FROM task_state")
        conn.execute("DELETE FROM branches")
        conn.execute(
            "INSERT INTO branches (id, name, parent_id, fork_len, checkpoint_len, created_at)"
            " VALUES (?, 'Основной диалог', NULL, 0, NULL, ?)",
            (ROOT_BRANCH_ID, datetime.now().isoformat()),
        )


# --- User profiles (personalization) ------------------------------------------


def list_profiles() -> list[dict]:
    """All profiles with their rule counts, active one first is not forced."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.description, p.is_active,
                   (SELECT COUNT(*) FROM profile_rules r WHERE r.profile_id = p.id) AS rules_count
            FROM profiles p ORDER BY p.id
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_profile(profile_id: int) -> dict | None:
    """One profile record or None."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, description, is_active FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()
    return dict(row) if row else None


def get_active_profile() -> dict | None:
    """The active profile or None (there is always one after seeding)."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, description, is_active FROM profiles WHERE is_active = 1"
        ).fetchone()
    return dict(row) if row else None


def set_active_profile(profile_id: int) -> None:
    """Make ``profile_id`` the single active profile."""
    init_db()
    with _connect() as conn:
        conn.execute("UPDATE profiles SET is_active = 0")
        conn.execute("UPDATE profiles SET is_active = 1 WHERE id = ?", (profile_id,))


def create_profile(name: str, description: str = "",
                   rules: list[dict] | None = None,
                   is_active: bool = False) -> int:
    """Create a profile with manual rules; returns its id."""
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO profiles (name, description, is_active, created_at)"
            " VALUES (?, ?, ?, ?)",
            (str(name).strip(), str(description or "").strip(), 1 if is_active else 0, now),
        )
        profile_id = cur.lastrowid
        for rule in rules or []:
            conn.execute(
                "INSERT INTO profile_rules (profile_id, kind, text, source, updated_at)"
                " VALUES (?, ?, ?, 'manual', ?)",
                (profile_id, rule["kind"], str(rule["text"]).strip(), now),
            )
    return profile_id


def update_profile(profile_id: int, name: str, description: str,
                   rules: list[dict] | None) -> None:
    """Update name/description; when ``rules`` is not None, replace all manual
    rules with the given list (dialog-learned rules are kept)."""
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE profiles SET name = ?, description = ? WHERE id = ?",
            (str(name).strip(), str(description or "").strip(), profile_id),
        )
        if rules is not None:
            conn.execute(
                "DELETE FROM profile_rules WHERE profile_id = ? AND source = 'manual'",
                (profile_id,),
            )
            for rule in rules:
                conn.execute(
                    "INSERT INTO profile_rules (profile_id, kind, text, source, updated_at)"
                    " VALUES (?, ?, ?, 'manual', ?)",
                    (profile_id, rule["kind"], str(rule["text"]).strip(), now),
                )


def delete_profile(profile_id: int) -> None:
    """Drop a profile with its rules; activate the first remaining one if the
    deleted profile was active."""
    init_db()
    with _connect() as conn:
        was_active = conn.execute(
            "SELECT is_active FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        conn.execute("DELETE FROM profile_rules WHERE profile_id = ?", (profile_id,))
        conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        if was_active and was_active["is_active"]:
            first = conn.execute(
                "SELECT id FROM profiles ORDER BY id LIMIT 1"
            ).fetchone()
            if first:
                conn.execute("UPDATE profiles SET is_active = 1 WHERE id = ?",
                             (first["id"],))


def list_profile_rules(profile_id: int, source: str | None = None) -> list[dict]:
    """Profile rules in order, optionally filtered by source."""
    init_db()
    with _connect() as conn:
        if source is None:
            rows = conn.execute(
                "SELECT id, kind, text, source, updated_at FROM profile_rules"
                " WHERE profile_id = ? ORDER BY id",
                (profile_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, kind, text, source, updated_at FROM profile_rules"
                " WHERE profile_id = ? AND source = ? ORDER BY id",
                (profile_id, source),
            ).fetchall()
    return [dict(r) for r in rows]


def add_profile_rule(profile_id: int, kind: str, text: str,
                     source: str = "manual") -> int:
    """Append one rule to a profile; returns its id."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO profile_rules (profile_id, kind, text, source, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (profile_id, kind, str(text).strip(), source, datetime.now().isoformat()),
        )
        return cur.lastrowid


def delete_profile_rule(rule_id: int) -> None:
    """Remove a single profile rule."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM profile_rules WHERE id = ?", (rule_id,))


def replace_dialog_rules(profile_id: int, rules: list[dict]) -> None:
    """Replace the profile's dialog-learned rules with ``rules`` (full block)."""
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM profile_rules WHERE profile_id = ? AND source = 'dialog'",
            (profile_id,),
        )
        for rule in rules or []:
            conn.execute(
                "INSERT INTO profile_rules (profile_id, kind, text, source, updated_at)"
                " VALUES (?, ?, ?, 'dialog', ?)",
                (profile_id, rule["kind"], str(rule["text"]).strip(), now),
            )


# --- Invariants (rules the assistant must not violate) -------------------------


def list_invariants() -> list[dict]:
    """All invariants in id order; ``patterns`` decoded from JSON to a list."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, category, title, rule, patterns, source, created_at, updated_at"
            " FROM invariants ORDER BY id"
        ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["patterns"] = json.loads(item["patterns"] or "[]")
        except json.JSONDecodeError:
            item["patterns"] = []
        out.append(item)
    return out


def get_invariant(inv_id: int) -> dict | None:
    """One invariant record (with decoded patterns) or None."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, category, title, rule, patterns, source, created_at, updated_at"
            " FROM invariants WHERE id = ?",
            (inv_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    try:
        item["patterns"] = json.loads(item["patterns"] or "[]")
    except json.JSONDecodeError:
        item["patterns"] = []
    return item


def insert_invariant(category: str, title: str, rule: str, patterns: list[str],
                     source: str) -> int:
    """Add an invariant row and return its id (change is logged by the caller)."""
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO invariants (category, title, rule, patterns, source, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (category, title, rule, json.dumps(patterns, ensure_ascii=False),
             source, now, now),
        )
        return cur.lastrowid


def update_invariant_row(inv_id: int, category: str, title: str, rule: str,
                         patterns: list[str]) -> None:
    """Overwrite an invariant row (change is logged by the caller)."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE invariants SET category = ?, title = ?, rule = ?, patterns = ?,"
            " updated_at = ? WHERE id = ?",
            (category, title, rule, json.dumps(patterns, ensure_ascii=False),
             datetime.now().isoformat(), inv_id),
        )


def delete_invariant(inv_id: int) -> None:
    """Drop an invariant row (change is logged by the caller)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM invariants WHERE id = ?", (inv_id,))


def add_invariant_log(action: str, inv_id: int | None, field: str | None,
                      old_value: str | None, new_value: str | None,
                      source: str, detail: str) -> None:
    """Append one change-log entry: what was replaced with what, by which command."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO invariant_log (ts, action, inv_id, field, old_value,"
            " new_value, source, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), action, inv_id, field,
             old_value, new_value, source, detail),
        )


def list_invariant_log(limit: int = 100) -> list[dict]:
    """Change-log entries, newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, action, inv_id, field, old_value, new_value, source, detail"
            " FROM invariant_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def snapshot_invariants() -> dict:
    """Full dump of the invariants + invariant_log tables (for tests and backups)."""
    init_db()
    with _connect() as conn:
        invariants = [dict(r) for r in conn.execute(
            "SELECT id, category, title, rule, patterns, source, created_at, updated_at"
            " FROM invariants ORDER BY id"
        ).fetchall()]
        log = [dict(r) for r in conn.execute(
            "SELECT id, ts, action, inv_id, field, old_value, new_value, source, detail"
            " FROM invariant_log ORDER BY id"
        ).fetchall()]
    return {"invariants": invariants, "log": log}


def restore_invariants(snap: dict) -> None:
    """Put the invariants tables back exactly as captured by
    :func:`snapshot_invariants` (ids included)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM invariants")
        conn.execute("DELETE FROM invariant_log")
        for i in snap.get("invariants", []):
            conn.execute(
                "INSERT INTO invariants (id, category, title, rule, patterns, source,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (i["id"], i["category"], i["title"], i["rule"], i["patterns"],
                 i["source"], i["created_at"], i["updated_at"]),
            )
        for l in snap.get("log", []):
            conn.execute(
                "INSERT INTO invariant_log (id, ts, action, inv_id, field, old_value,"
                " new_value, source, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (l["id"], l["ts"], l["action"], l["inv_id"], l["field"],
                 l["old_value"], l["new_value"], l["source"], l["detail"]),
            )
