"""Live verification: memory UPDATING and conflict resolution on the real model.

Прогоняет диалог, в котором факты меняются по ходу дела, и после каждого
изменения спрашивает агента об актуальном значении:

  1. факты заданы: срок — 15 декабря, формат — офлайн, ведущий — Алексей;
  2. вопрос «какой срок?»      → агент должен назвать актуальное значение;
  3. срок сдвинут на 20 декабря → вопрос снова → счётчик устаревших ответов;
  4. требование «офлайн» отменено (онлайн) → вопрос о формате;
  5. выбрали другой подход: ведущий Алексей → Мария → вопрос о ведущем;
  6. проверка промпта: старые значения НЕ попадают в блоки памяти
     (system-сообщения: LTM + WM + FSM) собранного контекста;
  7. журнал памяти: каждая замена записана «что было → что стало»;
  8. очистка: задача завершается кодом (FSM до done) — рабочая память
     очищена, долговременная (перенесённый туда факт) сохранена.

Счётчик `устаревших значений` — сколько раз в ответах на контрольные вопросы
агент назвал значение, которое к этому моменту уже было заменено. Run from
the repo root (needs a valid .env):

    python -m tests.verify_memory_updates [model]

``model`` — необязательное имя модели (по умолчанию GPUSTACK_MODEL из .env);
полезно, когда дефолтный эндпоинт лимитирует запросы (429).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    RateLimitError)  # noqa: E402

from app import context, memory, store, task_fsm  # noqa: E402
from app.agent import Agent  # noqa: E402

BRANCH_NAME = "Проверка обновления памяти (можно удалить)"

# Эндпоинт лимитирует частоту (429) и эпизодически отдаёт 5xx/сетевые сбои —
# всё это переживаем ретраем с нарастающей паузой.
TRANSIENT = (RateLimitError, InternalServerError, APIConnectionError,
             APITimeoutError)
ASK_ATTEMPTS = 6
ASK_BACKOFF_SEC = 30
TURN_PAUSE_SEC = 20  # пауза между ходами — не будоражим rate limit

# Старые и новые значения фактов. Подстроки — варианты, которыми модель может
# назвать значение в ответе.
FACTS = [
    # (ключ-подстрока, старые варианты, новые варианты, вопрос агенту)
    ("срок", ["15 декабря", "15.12"], ["20 декабря", "20.12"],
     "Какой сейчас срок корпоратива? Ответь одной короткой фразой."),
    ("формат", ["офлайн"], ["онлайн"],
     "В каком формате пройдёт корпоратив? Одна короткая фраза."),
    ("ведущий", ["алексе"], ["мария"],
     "Кто теперь ведущий корпоратива? Одна короткая фраза."),
]


def _mentions(text: str, variants: list[str]) -> int:
    """How many of the value's variants appear in the text."""
    low = (text or "").lower()
    return sum(1 for v in variants if v in low)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else None
    store.init_db()
    branch = store.create_branch(store.ROOT_BRANCH_ID, name=BRANCH_NAME)
    b = branch["id"]
    ltm_before = {e["id"] for e in store.list_long_term()}
    log_before = store.list_memory_log(1)
    log_max_before = log_before[0]["id"] if log_before else 0
    failures = []
    outdated_total = 0  # сколько раз агент назвал устаревшее значение

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" +
              (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def ask(turn_no, text):
        time.sleep(TURN_PAUSE_SEC)  # пауза между ходами: эндпоинт лимитирует rps
        agent = Agent(model=model)  # fresh instance per turn == рестарт; всё из БД
        for attempt in range(1, ASK_ATTEMPTS + 1):
            try:
                result = agent.ask(text, b)
                break
            except TRANSIENT:
                if attempt == ASK_ATTEMPTS:
                    raise
                wait = ASK_BACKOFF_SEC * attempt
                print(f"      (429/5xx, попытка {attempt}/{ASK_ATTEMPTS} — "
                      f"ждём {wait} с)")
                time.sleep(wait)
        print(f"\n--- Ход {turn_no}: пользователь ---\n{text}")
        print(f"--- Ход {turn_no}: ассистент ---\n{result['reply'][:500]}")
        return result

    def fact_blob():
        return "; ".join(f"{k}={v}" for k, v in store.load_facts(b).items())

    def ensure_facts_updated(user_text, must_contain, attempts=6):
        """Подстраховать WM-обновление: служебный вызов update_facts внутри хода
        агента при 429 молча оставляет старый блок — повторяем ТОТ ЖЕ вызов
        (тот же путь, что у агента) до появления нового значения."""
        for attempt in range(1, attempts + 1):
            blob = fact_blob()
            if _mentions(blob, must_contain) > 0:
                return blob
            wait = 30 * attempt
            print(f"      (WM ещё без нового значения, повтор update_facts "
                  f"{attempt}/{attempts} через {wait} с)")
            time.sleep(wait)
            try:
                context.update_facts(user_text, b)
            except TRANSIENT:
                continue
        return fact_blob()

    def journal_since():
        entries = [e for e in store.list_memory_log(50) if e["id"] > log_max_before]
        return list(reversed(entries))  # старые сверху

    try:
        # --- ход 1: факты заданы -------------------------------------------------
        turn1 = ("Готовим корпоратив компании. Зафиксируй данные задачи: срок — "
                 "15 декабря, формат — офлайн в офисе, ведущий — Алексей.")
        ask(1, turn1)
        blob = ensure_facts_updated(turn1, ["15 декабря"])
        check("ход 1: факты попали в рабочую память",
              all(_mentions(blob, old) for _, old, _, _ in FACTS), blob)

        # --- контрольные вопросы и изменения ------------------------------------
        # (idx: 0=срок, 1=формат, 2=ведущий; порядок изменений соответствует им)
        def ask_control(idx, turn_no, count_outdated=True):
            """Задать контрольный вопрос; вернуть (ответ, упоминаний устаревшего,
            названо ли актуальное). ``count_outdated=False`` — для вопроса ДО
            первого изменения: тогда «старое» значение ещё актуально и его
            упоминание устаревшим не считается."""
            _, old_v, new_v, question = FACTS[idx]
            result = ask(turn_no, question)
            reply = result["reply"]
            outdated = _mentions(reply, old_v) if count_outdated else 0
            named_new = _mentions(reply, new_v) > 0
            return reply, outdated, named_new

        # срок: контрольный вопрос ДО первого изменения — актуален пока 15 декабря
        reply, outdated, named_new = ask_control(0, 2, count_outdated=False)
        check("вопрос 1 (срок): агент назвал актуальное значение 15 декабря",
              _mentions(reply, FACTS[0][1]) > 0, reply[:200])
        outdated_total += outdated  # устаревших значений в этот момент не было

        # срок сдвигается
        turn3 = "Срок сдвигаем: корпоратив пройдёт 20 декабря."
        ask(3, turn3)
        blob = ensure_facts_updated(turn3, FACTS[0][2])
        check("ход 3: в рабочей памяти новый срок 20 декабря",
              _mentions(blob, FACTS[0][2]) > 0, blob)
        check("ход 3: старого срока 15 декабря в рабочей памяти нет",
              _mentions(blob, FACTS[0][1]) == 0, blob)
        ow = [e for e in journal_since() if e["action"] == "overwrite"
              and e["layer"] == "wm" and _mentions(e["old_value"] or "", FACTS[0][1])]
        check("ход 3: замена срока зафиксирована в журнале (что было → что стало)",
              bool(ow), str([(e["key"], e["old_value"], e["new_value"]) for e in ow]))

        reply, outdated, named_new = ask_control(0, 4)
        check("вопрос 2 (срок): агент назвал НОВОЕ значение 20 декабря", named_new, reply[:200])
        check("вопрос 2 (срок): агент НЕ назвал устаревшее 15 декабря", outdated == 0, reply[:200])
        outdated_total += outdated

        # требование «офлайн» отменяется
        turn5 = ("Требование «офлайн в офисе» отменяем — корпоратив пройдёт "
                 "полностью онлайн.")
        ask(5, turn5)
        blob = ensure_facts_updated(turn5, FACTS[1][2])
        check("ход 5: формат в рабочей памяти — онлайн",
              _mentions(blob, FACTS[1][2]) > 0, blob)
        check("ход 5: отменённого «офлайн» в рабочей памяти нет",
              _mentions(blob, FACTS[1][1]) == 0, blob)

        reply, outdated, named_new = ask_control(1, 6)
        check("вопрос 3 (формат): агент назвал актуальное значение онлайн", named_new, reply[:200])
        check("вопрос 3 (формат): агент НЕ назвал отменённое офлайн", outdated == 0, reply[:200])
        outdated_total += outdated

        # другой подход к ведущему
        turn7 = ("По ведущему выбираем другой подход: вместо Алексея корпоратив "
                 "будет вести Мария.")
        ask(7, turn7)
        blob = ensure_facts_updated(turn7, FACTS[2][2])
        check("ход 7: ведущий в рабочей памяти — Мария",
              _mentions(blob, FACTS[2][2]) > 0, blob)
        check("ход 7: прежнего ведущего Алексея в рабочей памяти нет",
              _mentions(blob, FACTS[2][1]) == 0, blob)

        reply, outdated, named_new = ask_control(2, 8)
        check("вопрос 4 (ведущий): агент назвал актуальное значение Мария", named_new, reply[:200])
        check("вопрос 4 (ведущий): агент НЕ назвал прежнего Алексея", outdated == 0, reply[:200])
        outdated_total += outdated

        # --- промпт: старые значения не попадают в блоки памяти ------------------
        built, info = context.build_context(None, b)
        system_blocks = [m["content"] for m in built if m["role"] == "system"]
        memory_blocks = "\n".join(m for m in system_blocks
                                  if not m.startswith("ИНВАРИАНТЫ")
                                  and not m.startswith("ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ"))
        stale_in_memory = sum(
            _mentions(memory_blocks, old_v) for _, old_v, _, _ in FACTS)
        check("промпт: в блоках памяти (LTM+WM) нет ни одного устаревшего значения",
              stale_in_memory == 0,
              f"устаревших упоминаний: {stale_in_memory}")
        fresh_in_memory = sum(
            _mentions(memory_blocks, new_v) for _, _, new_v, _ in FACTS)
        check("промпт: все актуальные значения присутствуют в блоках памяти",
              fresh_in_memory == 3, f"актуальных: {fresh_in_memory}/3")
        check("промпт: у фактов памяти видно время записи",
              "записано" in memory_blocks)
        # в STM-окно старые значения попадают законно — это сам диалог
        stm_msgs = [m for m in built if m["role"] in ("user", "assistant", "tool")]
        stm_text = " ".join(m.get("content") or "" for m in stm_msgs)
        print(f"      (в STM-окне диалога старые значения упоминаются "
              f"{sum(_mentions(stm_text, old_v) for _, old_v, _, _ in FACTS)} раз — "
              f"это сырые сообщения диалога, не память)")

        # --- очистка: задача завершена кодом --------------------------------------
        # один факт переносим в долговременную (как через панель «Память»)
        keys = [k for k in store.load_facts(b) if "срок" in k.lower()]
        promoted = memory.promote_working(b, keys=keys[:1], category="knowledge") if keys else 0
        check("очистка: факт «срок» перенесён в долговременную память", promoted == 1)

        # детерминированный финал: полный корректный цикл автомата кодом
        task_fsm.reset(b, note="проверка очистки")
        task_fsm.transition(b, "planning", note="задача — корпоратив")
        task_fsm.transition(b, "awaiting_approval", note="план")
        msgs = store.load_messages(b) + [{"role": "user", "content": "план утверждаю, выполняй"}]
        store.save_messages(msgs, b)
        task_fsm.transition(b, "execution")
        task_fsm.transition(b, "validation")
        task_fsm.record_check(b, "смета сходится", ok=True, evidence="расчёт в диалоге")
        task_fsm.transition(b, "done", note="корпоратив завершён")

        check("очистка: рабочая память очищена после завершения задачи",
              store.load_facts(b) == {}, str(store.load_facts(b)))
        ltm_now = store.list_long_term()
        check("очистка: долговременная память сохранилась",
              any(e["key"] == "срок" for e in ltm_now),
              str([e["key"] for e in ltm_now]))
        cleanup_log = [e for e in journal_since() if e["action"] == "cleanup"]
        check("очистка: очистка задачи зафиксирована в журнале памяти", bool(cleanup_log))
        print(f"\n=== Журнал памяти за сценарий ({len(journal_since())} записей) ===")
        for e in journal_since():
            ts = (e["ts"] or "")[:16].replace("T", " ")
            print(f"  {ts} · {e['layer']} · {e['action']} · {e['key']}: "
                  f"{(e['old_value'] or '—')[:40]} → {(e['new_value'] or '—')[:40]}")

        print(f"\n=== ИТОГ: агент назвал устаревшее значение "
              f"{outdated_total} раз(а) на 4 контрольных вопроса ===")

    finally:
        # прибираем артефакты сценария: ветка, занесённые LTM-факты, журнал
        for e in store.list_long_term():
            if e["id"] not in ltm_before:
                store.delete_long_term(e["id"])
        with store._connect() as conn:
            conn.execute("DELETE FROM memory_log WHERE id > ?", (log_max_before,))
        store.delete_branch(b)
        print("Ветка проверки и её записи памяти удалены.")

    print(f"\n{'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if not failures else 'ПРОВАЛЕНО: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
