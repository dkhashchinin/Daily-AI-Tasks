"""Unit tests for the UPDATING memory layer (app/memory.py + store upsert).

No LLM involved: policies, conflict resolution, the change journal and the
task-finish cleanup are checked against a temporary SQLite database. Run from
the repo root:

    python -m tests.test_memory_updates
"""
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):  # console may be cp1251
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app import context, memory, store, task_fsm  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="memory-updates-test-")
store.DB_PATH = Path(_TMP) / "test.db"


def _backdate(entry_id: int, minutes: int) -> str:
    """Rewrite an entry's write time into the past (simulates an older fact)."""
    old_ts = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    with store._connect() as conn:
        conn.execute("UPDATE long_term_memory SET updated_at = ? WHERE id = ?",
                     (old_ts, entry_id))
    return old_ts


def test_policy_classification():
    assert memory.classify_policy("срок сдачи") == "latest"
    assert memory.classify_policy("дедлайн") == "latest"
    assert memory.classify_policy("статус заявки") == "latest"
    assert memory.classify_policy("версия релиза") == "latest"
    assert memory.classify_policy("выбранный подход") == "latest"
    assert memory.classify_policy("принятые решения") == "append"
    assert memory.classify_policy("история изменений") == "append"
    assert memory.classify_policy("черновик плана") == "ttl"
    # явное правило важнее эвристики
    assert memory._policy_or_guess("append", "срок") == "append"
    assert memory._policy_or_guess(None, "срок") == "latest"


def test_upsert_overwrites_instead_of_duplicating():
    store.init_db()
    id1, action1 = store.upsert_long_term("knowledge", "срок релиза", "15 декабря")
    assert action1 == "add"
    id2, action2 = store.upsert_long_term("knowledge", "срок релиза", "20 декабря")
    assert action2 == "overwrite"
    rows = [e for e in store.list_long_term() if e["key"] == "срок релиза"]
    assert len(rows) == 1, f"факт продублировался: {len(rows)} строк"
    assert rows[0]["value"] == "20 декабря", rows[0]["value"]
    # перезапись заменяет значение В ТОЙ ЖЕ строке (id сохраняется), версию
    # поднимает, а время записи обновляет
    assert rows[0]["id"] == id2 and rows[0]["version"] == 2
    assert rows[0]["id"] == id1
    # старое значение не рендерится в промпт
    rendered = memory.render_ltm()
    assert "20 декабря" in rendered
    assert "15 декабря" not in rendered


def test_later_write_time_wins():
    """Конфликт решает время записи: факт с более поздней датой приоритетен."""
    store.init_db()
    old_id, _ = store.upsert_long_term("decision", "статус задачи", "в работе")
    old_ts = _backdate(old_id, minutes=30)
    new_id, action = store.upsert_long_term("decision", "статус задачи", "готово")
    assert action == "overwrite"
    row = next(e for e in store.list_long_term() if e["key"] == "статус задачи")
    assert row["value"] == "готово" and row["updated_at"] > old_ts
    # дубликат-призрак от старой записи удалён: одна строка на ключ
    rows = [e for e in store.list_long_term() if e["key"] == "статус задачи"]
    assert len(rows) == 1 and rows[0]["id"] == new_id


def test_append_policy_accumulates_list():
    store.init_db()
    store.upsert_long_term("decision", "принятые решения", "хранение в SQLite")
    store.upsert_long_term("decision", "принятые решения", "Flask без сервисов")
    rows = [e for e in store.list_long_term() if e["key"] == "принятые решения"]
    assert len(rows) == 2, "список решений должен накапливаться, а не перезаписываться"
    assert {r["value"] for r in rows} == {"хранение в SQLite", "Flask без сервисов"}
    rendered = memory.render_ltm()
    assert "(список, дополняется)" in rendered
    assert "хранение в SQLite" in rendered and "Flask без сервисов" in rendered


def test_overwrite_is_journaled_what_was_what_became():
    store.init_db()
    store.upsert_long_term("knowledge", "срок", "1 марта", source="dialog")
    _id, _ = store.upsert_long_term("knowledge", "срок", "10 марта", source="dialog")
    entry = store.list_memory_log(10)[0]
    assert entry["action"] == "overwrite" and entry["layer"] == "ltm"
    assert entry["old_value"] == "1 марта" and entry["new_value"] == "10 марта"
    assert entry["old_ts"] and entry["new_ts"]
    assert entry["old_ts"] < entry["new_ts"], "у нового значения время позже"
    assert entry["source"] == "dialog"
    assert "1 марта" in entry["detail"] and "10 марта" in entry["detail"]


def test_delete_and_clear_are_journaled():
    store.init_db()
    entry_id = memory.save_long_term("knowledge", "заметка", "тест")
    store.delete_long_term(entry_id)
    log = store.list_memory_log(5)
    assert log[0]["action"] == "delete" and log[0]["old_value"] == "тест"
    memory.save_long_term("knowledge", "заметка2", "тест2")
    store.clear_long_term()
    log = store.list_memory_log(5)
    assert log[0]["action"] == "clear" and store.list_long_term() == []


def test_working_fact_changes_are_journaled():
    store.init_db()
    b = 501
    store.save_facts({"срок": "15 декабря", "требование": "офлайн"}, b)
    old_full = store.load_facts_full(b)
    # срок изменился (перезапись), требование не тронуто
    written = memory.log_working_changes(old_full, {"срок": "20 декабря",
                                                    "требование": "офлайн"}, b)
    assert written == 1
    entry = store.list_memory_log(5)[0]
    assert entry["action"] == "overwrite" and entry["layer"] == "wm"
    assert entry["old_value"] == "15 декабря" and entry["new_value"] == "20 декабря"
    assert entry["old_ts"] and entry["new_ts"]
    # удаление фактов тоже фиксируется
    written = memory.log_working_changes(store.load_facts_full(b), {}, b)
    assert written == 2  # срок + требование удалены
    actions = [e["action"] for e in store.list_memory_log(3)]
    assert actions.count("delete") == 2, actions


def test_promote_updates_instead_of_duplicating():
    store.init_db()
    b = 502
    store.save_facts({"срок": "15 декабря"}, b)
    assert memory.promote_working(b, keys=["срок"], category="knowledge") == 1
    store.save_facts({"срок": "20 декабря"}, b)
    assert memory.promote_working(b, keys=["срок"], category="knowledge") == 1
    rows = [e for e in store.list_long_term() if e["key"] == "срок"]
    assert len(rows) == 1 and rows[0]["value"] == "20 декабря"
    assert rows[0]["version"] == 2 and rows[0]["source"] == "promoted"


def test_task_done_clears_wm_and_ttl_but_keeps_ltm():
    """Завершение задачи: рабочая память очищена, ttl удалён, LTM сохранён."""
    store.init_db()
    b = 510
    store.save_messages([{"role": "user", "content": "Задача"}], b)
    # память задачи
    store.save_facts({"срок": "15 декабря", "цель": "корпоратив"}, b)
    # долговременная память: постоянный факт, список решений, черновик
    store.upsert_long_term("profile", "имя клиента", "Акме", source="dialog")
    store.upsert_long_term("decision", "принятые решения", "SQLite", source="dialog")
    ttl_id, _ = store.upsert_long_term("knowledge", "черновик бюджета", "v1",
                                       source="dialog", policy="ttl")
    # полный корректный цикл автомата до done
    task_fsm.transition(b, "planning")
    task_fsm.transition(b, "awaiting_approval")
    store.save_messages([
        {"role": "user", "content": "Задача"},
        {"role": "assistant", "content": "план"},
        {"role": "user", "content": "план утверждаю, выполняй"},
    ], b)
    task_fsm.transition(b, "execution")
    task_fsm.transition(b, "validation")
    task_fsm.record_check(b, "проверено", ok=True, evidence="ок")
    task_fsm.transition(b, "done", note="задача закрыта")

    assert task_fsm.load_state(b)["stage"] == "done"
    assert store.load_facts(b) == {}, "рабочая память не очищена после done"
    keys = {e["key"] for e in store.list_long_term()}
    assert "имя клиента" in keys and "принятые решения" in keys
    assert "черновик бюджета" not in keys, "ttl-факт пережил завершение задачи"
    log = store.list_memory_log(3)
    assert log[0]["action"] == "cleanup", log[0]
    assert any(e["action"] == "delete" and e["layer"] == "ltm" for e in log)


def test_task_cancel_clears_wm():
    """Отмена задачи — тот же финал: WM очищена, долговременная сохранена."""
    b = 511
    store.save_messages([{"role": "user", "content": "Задача"}], b)
    store.save_facts({"цель": "отчёт"}, b)
    store.upsert_long_term("knowledge", "знание", "факт", source="dialog")
    task_fsm.transition(b, "planning")
    from app.tools import task_cancel
    reply = task_cancel(note="передумали", branch_id=b)
    assert "очищена" in reply
    assert store.load_facts(b) == {}
    assert any(e["key"] == "знание" for e in store.list_long_term())


def test_outdated_value_never_enters_memory_blocks():
    """Старое значение не попадает в блоки памяти промпта после обновления."""
    store.init_db()
    b = 512
    store.save_facts({"срок": "15 декабря", "формат": "офлайн"}, b)
    store.upsert_long_term("knowledge", "срок", "15 декабря", source="promoted")
    # обновляем и рабочую память, и долговременную
    store.save_facts({"срок": "20 декабря", "формат": "онлайн"}, b)
    store.upsert_long_term("knowledge", "срок", "20 декабря", source="promoted")
    built, info = context.build_context(None, b)
    system_blocks = [m["content"] for m in built if m["role"] == "system"]
    joined = "\n".join(system_blocks)
    assert "15 декабря" not in joined, "устаревший срок попал в промпт"
    assert "офлайн" not in joined
    assert "20 декабря" in joined and "онлайн" in joined
    assert info["facts_used"] and info["ltm_used"]
    # время записи видно модели
    wm_block = next(m for m in system_blocks if m.startswith("ДАННЫЕ ТЕКУЩЕЙ ЗАДАЧИ"))
    assert "записано" in wm_block
    ltm_block = next(m for m in system_blocks if m.startswith("ДОЛГОВРЕМЕННАЯ ПАМЯТЬ"))
    assert "записано" in ltm_block and "Правила памяти" in ltm_block


ALL_TESTS = [
    test_policy_classification,
    test_upsert_overwrites_instead_of_duplicating,
    test_later_write_time_wins,
    test_append_policy_accumulates_list,
    test_overwrite_is_journaled_what_was_what_became,
    test_delete_and_clear_are_journaled,
    test_working_fact_changes_are_journaled,
    test_promote_updates_instead_of_duplicating,
    test_task_done_clears_wm_and_ttl_but_keeps_ltm,
    test_task_cancel_clears_wm,
    test_outdated_value_never_enters_memory_blocks,
]


def main() -> int:
    failed = 0
    for test in ALL_TESTS:
        try:
            test()
            print(f"  ok  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} тестов прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
