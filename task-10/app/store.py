"""Persistent conversation storage (SQLite).

The whole dialog history — user messages, assistant replies, tool calls and
their results — is stored in a SQLite database so the agent can be restarted
and continue a conversation as if it never stopped.

Dialogs are organized in branches: every message belongs to a branch
(``messages.branch_id``). A branch is created by copying the first
``fork_len`` messages of its parent (copy-on-fork), so branches are
self-contained and evolve independently. The rolling summary and the sticky
facts block are also per-branch: ``context_state`` keys are prefixed with the
branch id ("b3:summary").
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "conversation.db"

ROOT_BRANCH_ID = 1


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
        # Key-value state for the context strategies, per branch (keys are
        # prefixed with the branch id): the rolling summary ("summary" /
        # "summarized") for the "summary + 10 последних" mode and the sticky
        # facts block ("facts") for the "facts + 10 последних" mode.
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
        if conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO branches (id, name, parent_id, fork_len, checkpoint_len, created_at)"
                " VALUES (?, 'Основной диалог', NULL, 0, NULL, ?)",
                (ROOT_BRANCH_ID, datetime.now().isoformat()),
            )


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
        conn.execute("DELETE FROM branches WHERE id = ?", (branch_id,))
        conn.execute("DELETE FROM context_state WHERE key LIKE ?",
                     (f"b{branch_id}:%",))


def load_summary(branch_id: int = ROOT_BRANCH_ID) -> tuple[str | None, int]:
    """Return (summary_text, how many oldest messages it replaces)."""
    init_db()
    with _connect() as conn:
        rows = dict(conn.execute(
            "SELECT key, value FROM context_state WHERE key IN (?, ?)",
            (f"b{branch_id}:summary", f"b{branch_id}:summarized"),
        ).fetchall())

    summary = rows.get(f"b{branch_id}:summary") or None
    try:
        covered = int(rows.get(f"b{branch_id}:summarized") or 0)
    except (TypeError, ValueError):
        covered = 0
    return summary, covered


def save_summary(summary: str | None, covered: int,
                 branch_id: int = ROOT_BRANCH_ID) -> None:
    """Store the rolling summary and the number of messages it covers."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO context_state (key, value) VALUES (?, ?)",
            (f"b{branch_id}:summary", summary),
        )
        conn.execute(
            "INSERT OR REPLACE INTO context_state (key, value) VALUES (?, ?)",
            (f"b{branch_id}:summarized", str(covered)),
        )


def load_facts(branch_id: int = ROOT_BRANCH_ID) -> dict:
    """Return the branch's sticky facts block (key -> value) for 'facts' mode.

    Stored as a JSON string in the same key-value table as the summary.
    A missing or unparsable block reads as empty — the agent keeps working.
    """
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM context_state WHERE key = ?",
            (f"b{branch_id}:facts",),
        ).fetchone()

    if not row or not row["value"]:
        return {}
    try:
        facts = json.loads(row["value"])
    except json.JSONDecodeError:
        return {}
    return facts if isinstance(facts, dict) else {}


def save_facts(facts: dict, branch_id: int = ROOT_BRANCH_ID) -> None:
    """Store the branch's sticky facts block as a JSON string."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO context_state (key, value) VALUES (?, ?)",
            (f"b{branch_id}:facts", json.dumps(facts, ensure_ascii=False)),
        )


def clear_history() -> None:
    """Drop all messages, branches and per-branch state; reseed the root."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM context_state")
        conn.execute("DELETE FROM branches")
        conn.execute(
            "INSERT INTO branches (id, name, parent_id, fork_len, checkpoint_len, created_at)"
            " VALUES (?, 'Основной диалог', NULL, 0, NULL, ?)",
            (ROOT_BRANCH_ID, datetime.now().isoformat()),
        )
