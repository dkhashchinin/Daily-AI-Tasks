"""Unit tests for the task lifecycle finite automaton (app/task_fsm.py).

No LLM involved: the automaton's rules, guards and persistence are checked
directly against a temporary SQLite database. Run from the repo root:

    python -m tests.test_task_fsm
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):  # console may be cp1251
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app import store, task_fsm  # noqa: E402
from app.task_fsm import ForbiddenTransition  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="fsm-test-")
store.DB_PATH = Path(_TMP) / "test.db"


def user_msg(text):
    return {"role": "user", "content": text}


def set_messages(messages, branch_id):
    store.save_messages(messages, branch_id)


def expect_reject(branch_id, to_stage, must_contain=()):
    ok, reason = task_fsm.check_transition(branch_id, to_stage)
    assert not ok, f"переход в {to_stage} должен быть отклонён, но разрешён"
    for fragment in must_contain:
        assert fragment in reason, f"в отказе нет «{fragment}»: {reason}"
    try:
        task_fsm.transition(branch_id, to_stage)
    except ForbiddenTransition:
        return reason
    raise AssertionError(f"transition в {to_stage} не поднял ForbiddenTransition")


def test_initial_state_is_idle():
    store.init_db()
    state = task_fsm.load_state(101)
    assert state["stage"] == "idle"
    assert task_fsm.summary(101)["active"] is False


def test_happy_path_with_guards():
    """planning → approval → execution → validation → done, guards enforced."""
    b = 102
    set_messages([user_msg("Задача: посчитай бюджет")], b)
    task_fsm.transition(b, "planning", note="задача начата")
    assert task_fsm.load_state(b)["stage"] == "planning"

    task_fsm.transition(b, "awaiting_approval", note="план готов")
    state = task_fsm.load_state(b)
    assert state["stage"] == "awaiting_approval"
    assert state["plan_presented_at"] == 1  # план появится после этого индекса

    # реализация до утверждённого плана запрещена
    reason = expect_reject(b, "execution", must_contain=("утвержд",))
    assert "план ещё не утверждён" in reason

    # сообщение-одобрение, оставшееся ДО предъявления плана, не считается
    set_messages([user_msg("сделай сразу и да, выполняй")], b)
    expect_reject(b, "execution", must_contain=("план ещё не утверждён",))

    # явное одобрение пользователя ПОСЛЕ плана — код пропускает переход
    set_messages([
        user_msg("Задача: посчитай бюджет"),
        {"role": "assistant", "content": "План: 1) ... 2) ..."},
        user_msg("План утверждаю, выполняй"),
    ], b)
    task_fsm.transition(b, "execution")
    assert task_fsm.load_state(b)["stage"] == "execution"

    task_fsm.transition(b, "validation")
    assert task_fsm.load_state(b)["stage"] == "validation"

    # финал без валидации запрещён
    expect_reject(b, "done", must_contain=("без валидации", "task_check"))

    task_fsm.record_check(b, "итоговая сумма сходится", ok=True, evidence="расчёт в ответе")
    task_fsm.transition(b, "done", note="задача закрыта")
    assert task_fsm.load_state(b)["stage"] == "done"


def test_every_skip_is_rejected():
    """Ни одного ребра «через этап» в автомате нет — код отклоняет всё."""
    def bring_to(stage, b):
        """Legitimately move a branch to the given stage (approval included)."""
        set_messages([user_msg("Задача")], b)
        if stage == "idle":
            return
        task_fsm.transition(b, "planning")
        if stage == "planning":
            return
        task_fsm.transition(b, "awaiting_approval")
        if stage == "awaiting_approval":
            return
        set_messages([
            user_msg("Задача"), {"role": "assistant", "content": "план"},
            user_msg("утверждаю, выполняй"),
        ], b)
        task_fsm.transition(b, "execution")
        if stage == "execution":
            return
        task_fsm.transition(b, "validation")
        if stage == "validation":
            return
        task_fsm.record_check(b, "проверено", ok=True, evidence="ок")
        task_fsm.transition(b, "done")

    cases = {
        201: ("idle", ["execution", "validation", "done", "awaiting_approval"]),
        202: ("planning", ["idle", "execution", "validation", "done"]),
        203: ("awaiting_approval", ["idle", "validation", "done"]),
        204: ("execution", ["idle", "planning", "awaiting_approval", "done"]),
        205: ("validation", ["idle", "planning", "awaiting_approval", "done"]),
        206: ("done", ["idle", "awaiting_approval", "execution", "validation"]),
    }
    for b, (stage, targets) in cases.items():
        bring_to(stage, b)
        for target in targets:
            reason = expect_reject(b, target)
            assert any(w in reason for w in ("запрещён", "отклонён", "неизвестное", "уже на этапе")), reason
        assert task_fsm.load_state(b)["stage"] == stage, "состояние изменилось при отказе"


def test_unknown_and_self_transitions():
    b = 207
    expect_reject(b, "cancelled", must_contain=("неизвестное состояние", "Допустимые состояния"))
    expect_reject(b, "idle", must_contain=("уже на этапе",))
    task_fsm.transition(b, "planning")
    expect_reject(b, "planning", must_contain=("уже на этапе planning",))


def test_rejection_message_lists_allowed_transitions():
    b = 208
    task_fsm.transition(b, "planning")
    reason = expect_reject(b, "done")
    assert "planning → awaiting_approval" in reason


def test_validation_back_to_execution_and_checks():
    """Нашлись проблемы — можно вернуться из validation в execution."""
    b = 209
    set_messages([user_msg("Задача")], b)
    task_fsm.transition(b, "planning")
    task_fsm.transition(b, "awaiting_approval")
    # одобрение пользователя появляется ПОСЛЕ предъявления плана
    set_messages([
        user_msg("Задача"), {"role": "assistant", "content": "план"},
        user_msg("утверждаю"),
    ], b)
    task_fsm.transition(b, "execution")
    task_fsm.transition(b, "validation")
    # проверка вне validation невозможна
    task_fsm.transition(b, "execution")
    try:
        task_fsm.record_check(b, "проверка не на том этапе")
    except ValueError:
        pass
    else:
        raise AssertionError("record_check вне validation должен отклоняться")
    task_fsm.transition(b, "validation")
    task_fsm.record_check(b, "пункт 1 плана выполнен", ok=False, evidence="не сделано")
    task_fsm.transition(b, "execution")  # доработка
    task_fsm.transition(b, "validation")
    task_fsm.record_check(b, "пункт 1 плана выполнен", ok=True, evidence="готово")
    task_fsm.transition(b, "done")
    assert len(task_fsm.load_state(b)["checks"]) == 2


def test_pause_and_continue_without_reexplanation():
    """Пауза на любом этапе: состояние переживает перезапуск (SQLite)."""
    b = 210
    set_messages([user_msg("Задача")], b)
    task_fsm.transition(b, "planning")
    task_fsm.transition(b, "awaiting_approval")

    # «перезапуск процесса»: свежее чтение из БД даёт то же состояние
    reloaded = task_fsm.load_state(b)
    assert reloaded["stage"] == "awaiting_approval"
    assert reloaded["step"] == "ожидание решения пользователя"
    assert reloaded["plan_presented_at"] == 1

    block = task_fsm.render_state_block(b)
    assert "Текущий этап: awaiting_approval" in block
    assert "Ожидаемое действие" in block
    assert "awaiting_approval → planning" in block
    assert "awaiting_approval → execution" in block

    # продолжение после паузы: тот же автомат, те же правила
    set_messages([
        user_msg("Задача"), {"role": "assistant", "content": "План: ..."},
        user_msg("ок, выполняй"),
    ], b)
    task_fsm.transition(b, "execution")
    assert task_fsm.load_state(b)["stage"] == "execution"


def test_rejection_keyword_blocks_approval():
    """«переделай план» — это не одобрение, даже если есть слово «давай»."""
    b = 211
    set_messages([user_msg("Задача")], b)
    task_fsm.transition(b, "planning")
    task_fsm.transition(b, "awaiting_approval")
    set_messages([
        user_msg("Задача"), {"role": "assistant", "content": "План"},
        user_msg("давай, но переделай: убери пункт 2"),
    ], b)
    expect_reject(b, "execution", must_contain=("отклонил план", "planning"))


def test_reset_from_any_stage():
    b = 212
    set_messages([user_msg("Задача")], b)
    task_fsm.transition(b, "planning")
    task_fsm.transition(b, "awaiting_approval")
    state = task_fsm.reset(b, note="пользователь отменил")
    assert state["stage"] == "idle"
    assert task_fsm.load_state(b)["stage"] == "idle"
    # после сброса цикл начинается заново
    task_fsm.transition(b, "planning")
    assert task_fsm.load_state(b)["stage"] == "planning"


def test_dialog_clear_resets_automaton():
    b = 213
    set_messages([user_msg("Задача")], b)
    task_fsm.transition(b, "planning")
    store.clear_dialog(b)
    assert task_fsm.load_state(b)["stage"] == "idle"


def test_branches_are_independent():
    """У каждой ветки свой автомат."""
    b1, b2 = 214, 215
    task_fsm.transition(b1, "planning")
    assert task_fsm.load_state(b2)["stage"] == "idle"
    task_fsm.transition(b2, "planning")
    task_fsm.transition(b2, "awaiting_approval")
    assert task_fsm.load_state(b1)["stage"] == "planning"
    assert task_fsm.load_state(b2)["stage"] == "awaiting_approval"


def test_tools_reject_via_run_tool():
    """Инструменты возвращают отказ строкой (для модели), а не исключение."""
    from app.tools import run_tool
    b = 216
    result = run_tool("task_transition", {"to_stage": "done"}, branch_id=b)
    assert result.startswith("Отказано") and "запрещён" in result
    result = run_tool("task_transition", {"to_stage": "planning"}, branch_id=b)
    assert "Переход выполнен" in result
    result = run_tool("task_check", {"item": "x", "ok": True}, branch_id=b)
    assert result.startswith("Отказано") and "validation" in result
    result = run_tool("task_cancel", {}, branch_id=b)
    assert "отменена" in result or "возвращён в idle" in result
    assert task_fsm.load_state(b)["stage"] == "idle"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ok  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} тестов прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
