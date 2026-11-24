"""Task lifecycle as a finite state machine (FSM).

The lifecycle of the current task is an explicit automaton, not prose:

    idle → planning → awaiting_approval → execution → validation → done

Every stage carries a default current step and the expected action
(``STAGE_INFO``). Transitions are checked BY CODE before they are applied
(:func:`can_transition` is a pure table check; :func:`transition` additionally
runs the per-edge guards and raises :class:`ForbiddenTransition` otherwise),
so the assistant cannot skip stages even if the prompt asks it to:

- **реализация до утверждённого плана** — the edge ``awaiting_approval →
  execution`` has a guard that scans the stored dialog for an explicit user
  approval appearing AFTER the plan was presented (``_guard_plan_approved``);
  without it the transition is rejected;
- **финал без валидации** — the edge ``validation → done`` requires at least
  one recorded validation check (:func:`record_check`); any direct
  ``execution → done`` / ``planning → done`` edge simply does not exist;
- unknown target states, self-transitions and any pair outside
  ``TRANSITIONS`` are rejected with the list of the allowed transitions.

The state (stage, current step, expected action, plan position, validation
checks, transition history) is persisted per branch in SQLite
(``store.task_state``), so a task can be PAUSED at any stage and continued
later without re-explanations: :func:`render_state_block` re-injects the
current stage/step/expected-action into every request.

The FSM is manipulated by the agent only through the ``task_transition`` /
``task_check`` / ``task_status`` tools (``app/tools.py``); ``task_cancel``
returns to idle on explicit user request.
"""
import re
from datetime import datetime

from . import store

# --- States -------------------------------------------------------------------

STATES = ("idle", "planning", "awaiting_approval", "execution", "validation", "done")

STAGE_TITLES = {
    "idle": "задачи нет",
    "planning": "планирование",
    "awaiting_approval": "ожидание утверждения плана",
    "execution": "реализация",
    "validation": "валидация",
    "done": "завершено",
}

# Default current step and expected action per stage. Shown to the model in
# every request and to the user in the UI; ``task_transition`` may override
# the step for the target stage.
STAGE_INFO = {
    "idle": {
        "step": "—",
        "expected": "задач в работе нет; многошаговый запрос — начать: "
                    "task_transition(to_stage='planning', step=…, note='суть задачи')",
    },
    "planning": {
        "step": "составление плана",
        "expected": "изучить задачу, составить план и ОБЯЗАТЕЛЬНО в этом же ответе "
                    "вызвать task_transition(to_stage='awaiting_approval', note='план'), "
                    "затем предъявить план пользователю — план, не проведённый через "
                    "awaiting_approval, считается не предъявленным",
    },
    "awaiting_approval": {
        "step": "ожидание решения пользователя",
        "expected": "предъявить план и дождаться явного одобрения пользователя; "
                    "правки — вернуться в planning: task_transition(to_stage='planning'); "
                    "после одобрения — task_transition(to_stage='execution')",
    },
    "execution": {
        "step": "реализация по утверждённому плану",
        "expected": "выполнять план (инструменты разрешены), затем перевести в "
                    "validation: task_transition(to_stage='validation')",
    },
    "validation": {
        "step": "проверка результата",
        "expected": "проверить результат и зафиксировать каждую проверку инструментом "
                    "task_check(item=…, ok=…, evidence=…); затем — "
                    "task_transition(to_stage='done')",
    },
    "done": {
        "step": "задача завершена",
        "expected": "итоговые результаты уже сообщены; новая задача — начать заново "
                    "с task_transition(to_stage='planning')",
    },
}

# --- Allowed transitions (the automaton's edge table) --------------------------
# Anything not listed here is rejected BY CODE. In particular there is no
# planning → execution / * → done edge: skips are impossible.
TRANSITIONS = {
    "idle": ("planning",),
    "planning": ("awaiting_approval",),
    "awaiting_approval": ("planning", "execution"),  # переработать план | план утверждён
    "execution": ("validation",),
    "validation": ("execution", "done"),             # нашлись проблемы | проверено
    "done": ("planning",),                           # новая задача
}

# Per-edge guards: extra CODE checks on top of the table. Each guard receives
# (branch_id, state) and returns (ok, reason).
GUARDS = {
    ("awaiting_approval", "execution"): "_guard_plan_approved",
    ("validation", "done"): "_guard_validation_done",
}

# --- Approval detection (code, not prompt) -------------------------------------
# A plan counts as approved only when a USER message appearing after the plan
# was presented matches an approval marker and no rejection marker.

_APPROVAL_RE = re.compile(
    r"(утвержд\w*|одобр\w*|согласен|согласна|подтвержд\w*|принят\w*|аппрув\w*|approved?|"
    r"давай|делай|выполняй|реализуй|исполняй|поехали|начинай|приступ\w*|окей|\bок\b|\bgo\b|\bда\b)",
    re.IGNORECASE,
)
_REJECTION_RE = re.compile(
    r"(не\s+утвержд\w*|не\s+одобр\w*|не\s+согласен|не\s+согласна|не\s+подтвержд\w*|"
    r"не\s+давай|не\s+делай|не\s+выполняй|не\s+реализуй|не\s+начинай|"
    r"передела\w*|переработ\w*|правк\w*|доработ\w*|измени\s+план|стоп|отмен\w*)",
    re.IGNORECASE,
)


class ForbiddenTransition(Exception):
    """Raised when a stage transition is not allowed by the automaton."""


def _now() -> str:
    return datetime.now().isoformat()


# --- State persistence (per branch, in SQLite) ----------------------------------


def default_state() -> dict:
    """Fresh automaton state: no task, idle."""
    return {
        "stage": "idle",
        "step": STAGE_INFO["idle"]["step"],
        "history": [],
        "checks": [],
        "plan_presented_at": None,
        "finished_at": None,
        "updated_at": None,
    }


def load_state(branch_id: int = store.ROOT_BRANCH_ID) -> dict:
    """Load the branch's task state; a branch without one starts idle."""
    raw = store.load_task_state(branch_id)
    if not raw:
        return default_state()
    state = default_state()
    state.update(raw)
    if state["stage"] not in STATES:  # corrupted row — fail safe to idle
        return default_state()
    return state


def _save_state(state: dict, branch_id: int) -> None:
    state["updated_at"] = _now()
    store.save_task_state(state, branch_id)


# --- Transition checks ----------------------------------------------------------


def can_transition(stage: str, to_stage: str) -> tuple[bool, str]:
    """Pure table check: is the edge ``stage → to_stage`` in the automaton.

    Returns ``(ok, reason)``; ``reason`` explains the rejection and lists the
    allowed transitions. Guards are NOT run here (they need branch context).
    """
    if to_stage not in STATES:
        allowed_all = ", ".join(f"{s} → {t}" for s, ts in TRANSITIONS.items() for t in ts)
        return False, (f"неизвестное состояние '{to_stage}'. "
                       f"Допустимые состояния: {', '.join(STATES)}. Все рёбра автомата: {allowed_all}.")
    if stage not in STATES:
        return False, f"текущее состояние '{stage}' неизвестно автомату."
    if to_stage == stage:
        allowed = ", ".join(TRANSITIONS.get(stage, ())) or "нет"
        return False, f"задача уже на этапе {stage}; допустимые переходы: {allowed}."
    if to_stage not in TRANSITIONS.get(stage, ()):
        allowed = ", ".join(f"{stage} → {t}" for t in TRANSITIONS.get(stage, ())) or "нет"
        return False, (
            f"переход {stage} → {to_stage} запрещён конечным автоматом (этап нельзя перепрыгнуть). "
            f"Допустимые переходы из '{stage}': {allowed}."
        )
    return True, ""


def _guard_plan_approved(branch_id: int, state: dict) -> tuple[bool, str]:
    """Code guard for awaiting_approval → execution.

    Scans user messages stored AFTER the plan was presented
    (``plan_presented_at``); the plan is approved only if one of them matches
    an approval marker and none matches a rejection marker.
    """
    messages = store.load_messages(branch_id)
    since = int(state.get("plan_presented_at") or 0)
    for m in messages[since:]:
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if _REJECTION_RE.search(text):
            return False, ("пользователь отклонил план или попросил его переработать: "
                           "вернитесь в planning (task_transition(to_stage='planning')) и "
                           "переработайте план.")
        if _APPROVAL_RE.search(text):
            return True, ""
    return False, ("план ещё не утверждён пользователем — реализация до утверждённого "
                   "плана запрещена. Предъявите план и дождитесь явного одобрения "
                   "(например: «план утверждаю», «ок, выполняй»).")


def _guard_validation_done(branch_id: int, state: dict) -> tuple[bool, str]:
    """Code guard for validation → done: at least one recorded check."""
    if state.get("checks"):
        return True, ""
    return False, ("финал без валидации запрещён: на этапе validation не зафиксирована "
                   "ни одна проверка. Сначала вызовите task_check(item=…, ok=…, evidence=…) "
                   "для каждой проверки, затем повторите переход validation → done.")


def check_transition(branch_id: int, to_stage: str) -> tuple[bool, str]:
    """Full pre-check (table + guards) WITHOUT applying anything."""
    state = load_state(branch_id)
    ok, reason = can_transition(state["stage"], to_stage)
    if not ok:
        return False, reason
    guard_name = GUARDS.get((state["stage"], to_stage))
    if guard_name:
        guard = globals()[guard_name]
        return guard(branch_id, state)
    return True, ""


def transition(branch_id: int, to_stage: str,
               step: str | None = None, note: str | None = None) -> dict:
    """Validate the transition by code, then apply it. Raises
    :class:`ForbiddenTransition` on any illegal jump — the state is left
    untouched in that case."""
    state = load_state(branch_id)
    stage = state["stage"]

    ok, reason = can_transition(stage, to_stage)
    if not ok:
        raise ForbiddenTransition(reason)

    guard_name = GUARDS.get((stage, to_stage))
    if guard_name:
        guard = globals()[guard_name]
        ok, reason = guard(branch_id, state)
        if not ok:
            raise ForbiddenTransition(f"переход {stage} → {to_stage} отклонён: {reason}")

    state["stage"] = to_stage
    info = STAGE_INFO[to_stage]
    state["step"] = (step or "").strip() or info["step"]
    state["history"].append({
        "from": stage, "to": to_stage, "at": _now(), "note": (note or "").strip(),
    })
    if to_stage == "awaiting_approval":
        # remember where the plan enters the dialog: approval is only counted
        # from user messages after this point
        state["plan_presented_at"] = len(store.load_messages(branch_id))
    if to_stage == "done":
        state["finished_at"] = _now()
    _save_state(state, branch_id)
    if to_stage == "done":
        # Завершение задачи очищает память: рабочая память ветки стирается
        # (факты принадлежали задаче), ttl-факты удаляются из долговременной;
        # остальная долговременная память сохраняется. Очистка фиксируется
        # в журнале памяти (app/memory.on_task_finished).
        from . import memory as _memory  # local import: avoids a module cycle
        try:
            _memory.on_task_finished(branch_id, reason="задача завершена (этап done)")
        except Exception as exc:  # очистка не должна ронять переход
            print(f"[task_fsm] не удалось очистить память после done: {exc}")
    return state


def record_check(branch_id: int, item: str, ok: bool = True,
                 evidence: str | None = None) -> dict:
    """Record one validation check; only meaningful on the validation stage.

    This is what the ``validation → done`` guard counts — a final answer
    without at least one recorded check is rejected by code.
    """
    state = load_state(branch_id)
    if state["stage"] != "validation":
        raise ValueError(
            f"Проверка фиксируется только на этапе validation (сейчас: {state['stage']}). "
            f"Сначала переведите задачу в validation."
        )
    text = str(item or "").strip()
    if not text:
        raise ValueError("Пункт проверки (item) не может быть пустым.")
    state["checks"].append({
        "item": text, "ok": bool(ok),
        "evidence": (str(evidence).strip() if evidence else ""),
        "at": _now(),
    })
    _save_state(state, branch_id)
    return state


def reset(branch_id: int, note: str = "") -> dict:
    """Return to idle from ANY stage (explicit cancellation, not a stage
    transition). Used by the ``task_cancel`` tool and by dialog resets."""
    state = default_state()
    if note:
        state["history"].append({"from": "?", "to": "idle", "at": _now(), "note": note})
    _save_state(state, branch_id)
    return state


# --- Rendering -------------------------------------------------------------------


def summary(branch_id: int = store.ROOT_BRANCH_ID) -> dict:
    """Compact state view for the UI / API."""
    state = load_state(branch_id)
    stage = state["stage"]
    return {
        "stage": stage,
        "title": STAGE_TITLES[stage],
        "step": state["step"],
        "expected": STAGE_INFO[stage]["expected"],
        "allowed": list(TRANSITIONS.get(stage, ())),
        "checks": state.get("checks", []),
        "active": stage not in ("idle", "done"),
    }


def render_state_block(branch_id: int = store.ROOT_BRANCH_ID) -> str:
    """The system-message block describing the automaton's current position.

    Injected into every request so a paused task continues without
    re-explanations. This block only DESCRIBES the state — enforcement lives
    in :func:`transition` / :func:`record_check` (code runs before apply).
    """
    state = load_state(branch_id)
    stage = state["stage"]
    lines = [
        "ЖИЗНЕННЫЙ ЦИКЛ ЗАДАЧИ — конечный автомат; переходы проверяет КОД "
        "(запрещённые отклоняются с сообщением об ошибке).",
        f"Этапы: {' → '.join(STATES[1:])}; старт задачи: idle → planning.",
    ]
    if stage == "idle":
        lines.append(
            "Сейчас задач в работе нет (idle). Если запрос пользователя — многошаговая "
            "задача (что-то спланировать/сделать/проверить), начни её: "
            "task_transition(to_stage='planning', step=…, note='суть задачи'). "
            "Простой вопрос без задачи — просто отвечай, автомат не трогай."
        )
        return "\n".join(lines)

    info = STAGE_INFO[stage]
    lines.append(f"Текущий этап: {stage} ({info['step']}).")
    if stage != "done":
        lines.append(f"Ожидаемое действие: {info['expected']}.")
    if stage == "planning":
        lines.append(
            "ВАЖНО: сначала вызови task_transition(to_stage='awaiting_approval', "
            "note='план'), и только потом излагай план пользователю в ответе."
        )
    allowed = TRANSITIONS.get(stage, ())
    lines.append("Допустимые переходы: "
                 + (", ".join(f"{stage} → {t}" for t in allowed) or "нет")
                 + ".")
    if stage == "awaiting_approval":
        lines.append(
            "ВАЖНО: переход awaiting_approval → execution код пропустит только после "
            "явного одобрения плана пользователем в диалоге. Попроси подтвердить план; "
            "если пользователь просит правки — вернись в planning."
        )
    if stage == "validation":
        checks = state.get("checks", [])
        lines.append(f"Зафиксировано проверок: {len(checks)}.")
        if checks:
            for c in checks:
                mark = "✓" if c.get("ok") else "✗"
                lines.append(f"  {mark} {c['item']}" +
                             (f" — {c['evidence']}" if c.get("evidence") else ""))
        lines.append("Переход validation → done код пропустит только при "
                     "минимум одной зафиксированной проверке (task_check).")
    if stage == "done":
        lines.append("Задача завершена. Новый запрос-задача — начни заново: "
                     "task_transition(to_stage='planning').")
    lines.append(
        "Запрещено (код отклонит): перепрыгивать этапы, начинать реализацию до "
        "утверждённого плана, завершать задачу без валидации. Состояние меняется "
        "только инструментами task_transition / task_check; прочитать — task_status; "
        "отменить задачу по просьбе пользователя — task_cancel."
    )
    return "\n".join(lines)
