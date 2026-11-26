"""SQLite-хранилище снимков курсов: один запуск планировщика = одна строка.

ts хранится в локальном времени в формате "YYYY-MM-DD HH:MM:SS" — так
сравнение с datetime('now', 'localtime') в SQL корректно отбирает период.

Каждое обращение открывает своё соединение — сервер не держит глобального
состояния и безопасен при вызовах из нескольких потоков (FastMCP запускает
синхронные инструменты в threadpool). Журнал в режиме WAL, чтобы веб-интерфейс
мог читать базу параллельно с работающим планировщиком.
"""
import contextlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# <корень проекта>/data/rates.db (mcp_rates_store/store.py -> task-18/)
_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "rates.db"

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT    NOT NULL,
    usd     REAL,
    eur     REAL,
    cny     REAL,
    gbp     REAL,
    payload TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots (ts);
"""


def db_path() -> Path:
    return Path(os.environ.get("RATES_DB_PATH", str(_DEFAULT_DB)))


def _connect(readonly: bool = False) -> sqlite3.Connection:
    path = db_path()
    if not readonly:
        path.parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        # Читаем через URI в режиме only-read: веб-интерфейс не может
        # случайно изменить/создать базу планировщика.
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _write():
    """Соединение для записи: commit при выходе, закрытие всегда."""
    return contextlib.closing(_connect(readonly=False))


def _read():
    return contextlib.closing(_connect(readonly=True))


def init_db() -> None:
    with _write() as conn, conn:  # conn.__exit__ -> commit, closing -> close
        conn.executescript(_SCHEMA)


def save_snapshot(payload: dict) -> dict:
    """Сохранить один снимок. Автоматически ставит ts (локальное время)."""
    ts = datetime.now().strftime(_TS_FORMAT)
    rates = payload.get("rates") or {}
    with _write() as conn, conn:  # commit при выходе, закрытие всегда
        conn.executescript(_SCHEMA)
        cursor = conn.execute(
            "INSERT INTO snapshots (ts, usd, eur, cny, gbp, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (
                ts,
                rates.get("USD"),
                rates.get("EUR"),
                rates.get("CNY"),
                rates.get("GBP"),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        snapshot_id = cursor.lastrowid
    return {"id": snapshot_id, "ts": ts}


def read_snapshots(hours: float, limit: int = 5000) -> list[dict]:
    """Снимки за последние `hours` часов, от старых к новым. hours=0 — вся история.

    Каждая строка: id, ts и курсы всех валют, которые были в снимке
    (ключи в нижнем регистре: usd, eur, jpy, ...). Валюты берутся из
    сохранённого payload, поэтому набор колонок зависит от того, что
    планировщик собирал в этом запуске.
    """
    if not db_path().exists():
        return []
    hours = max(0.0, float(hours))
    with _read() as conn:
        if hours <= 0:
            rows = conn.execute(
                "SELECT id, ts, usd, eur, cny, gbp, payload FROM snapshots "
                "ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ts, usd, eur, cny, gbp, payload FROM snapshots "
                "WHERE ts >= datetime('now', 'localtime', ?) ORDER BY id ASC LIMIT ?",
                (f"-{hours} minutes", limit),
            ).fetchall()

    result: list[dict] = []
    for row in rows:
        item: dict = {"id": row["id"], "ts": row["ts"]}
        try:
            payload_rates = (json.loads(row["payload"]) or {}).get("rates") or {}
        except (json.JSONDecodeError, TypeError):
            payload_rates = {}
        for code, value in payload_rates.items():
            if isinstance(value, (int, float)):
                item[code.lower()] = value
        # Фолбэк для старых снимков с фиксированными колонками.
        for column in ("usd", "eur", "cny", "gbp"):
            if column not in item and row[column] is not None:
                item[column] = row[column]
        result.append(item)
    return result


def count_all() -> int:
    if not db_path().exists():
        return 0
    with _read() as conn:
        return conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]


def clear_all() -> int:
    """Удалить все снимки: история запусков с нуля, id снова с 1.

    Возвращает количество удалённых строк. Отчёты (data/reports/) не
    трогаются — их можно пересобрать из новых снимков заново.
    """
    with _write() as conn, conn:  # commit при выходе, закрытие всегда
        conn.executescript(_SCHEMA)
        removed = conn.execute("DELETE FROM snapshots").rowcount
        try:
            # Сброс счётчика AUTOINCREMENT, чтобы нумерация началась с 1.
            # Таблицы может не быть, если в базу ещё ничего не писали.
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'snapshots'")
        except sqlite3.OperationalError:
            pass
    with _write() as conn:  # вакуум вне транзакции: освобождаем место на диске
        conn.execute("VACUUM")
    return max(removed, 0)


def read_recent(limit: int = 15) -> list[dict]:
    """Последние `limit` снимков, от новых к старым (для веб-панели).

    Формат строк такой же, как у read_snapshots: id, ts и курсы всех
    валют из payload снимка (в нижнем регистре).
    """
    if not db_path().exists():
        return []
    with _read() as conn:
        rows = conn.execute(
            "SELECT id, ts, usd, eur, cny, gbp, payload FROM snapshots "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    result: list[dict] = []
    for row in rows:
        item: dict = {"id": row["id"], "ts": row["ts"]}
        try:
            payload_rates = (json.loads(row["payload"]) or {}).get("rates") or {}
        except (json.JSONDecodeError, TypeError):
            payload_rates = {}
        for code, value in payload_rates.items():
            if isinstance(value, (int, float)):
                item[code.lower()] = value
        for column in ("usd", "eur", "cny", "gbp"):
            if column not in item and row[column] is not None:
                item[column] = row[column]
        result.append(item)
    return result
