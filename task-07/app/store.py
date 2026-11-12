"""Persistent conversation storage (SQLite).

The whole dialog history — user messages, assistant replies, tool calls and
their results — is stored in a SQLite database so the agent can be restarted
and continue a conversation as if it never stopped.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "conversation.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the messages table if it does not exist."""
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


def load_messages() -> list[dict]:
    """Return the full conversation in order, in OpenAI message format."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages ORDER BY id"
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


def save_messages(messages: list[dict]) -> None:
    """Replace the stored history with the given full message list."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
        for message in messages:
            conn.execute(
                """
                INSERT INTO messages (role, content, tool_calls, tool_call_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message.get("role"),
                    message.get("content"),
                    json.dumps(message["tool_calls"]) if message.get("tool_calls") else None,
                    message.get("tool_call_id"),
                    datetime.now().isoformat(),
                ),
            )


def clear_history() -> None:
    """Drop all stored messages, starting a brand new conversation."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
