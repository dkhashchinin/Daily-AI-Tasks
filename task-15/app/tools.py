"""Tool functions for the agent. Model decides which to call and with what args."""
import ast
import datetime
import operator
import random

from . import store, task_fsm

# In-memory list of tasks the agent can search.
TASKS = [
    {"id": 1, "title": "Купить молоко", "done": False},
    {"id": 2, "title": "Записаться к врачу", "done": False},
    {"id": 3, "title": "Написать отчёт", "done": True},
    {"id": 4, "title": "Позвонить родителям", "done": False},
    {"id": 5, "title": "Починить лампочку", "done": False},
]

# Hardcoded exchange rates relative to 1 unit of the base currency.
RATES = {"USD": 92.5, "EUR": 100.0, "RUB": 1.0}


def get_current_time() -> str:
    """Return the current local date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '2+3*4'."""
    allowed = "0123456789+-*///%() _.,x"
    expr = str(expression).replace(",", ".").replace("x", "*")
    if any(c not in allowed for c in expr):
        return "Ошибка: разрешены только числа и операторы + - * / // % ** ( )"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "Ошибка: не удалось разобрать выражение"
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _eval(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        raise ValueError("unsupported node")

    try:
        result = _eval(tree)
    except Exception as exc:
        return f"Ошибка вычисления: {exc}"
    return f"{result}"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between USD, EUR and RUB at hardcoded rates."""
    src = str(from_currency).upper()
    dst = str(to_currency).upper()
    if src not in RATES or dst not in RATES:
        return (f"Ошибка: известные валюты — {', '.join(RATES)}")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return "Ошибка: amount должен быть числом"
    result = amount * RATES[src] / RATES[dst]
    precision = 2 if not (result > 0 and result < 0.01 and result != 0) else 6
    return f"{amount:.2f} {src} = {result:.{precision}f} {dst}"


def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll one or more dice with the given number of sides."""
    try:
        sides = int(sides)
        count = int(count)
    except (TypeError, ValueError):
        return "Ошибка: sides и count должны быть целыми числами"
    if sides < 2 or count < 1 or count > 100:
        return "Ошибка: sides >= 2 и 1 <= count <= 100"
    rolls = [random.randint(1, sides) for _ in range(count)]
    return f"Результат: {rolls} (сумма {sum(rolls)})"


def search_tasks(query: str) -> str:
    """Search the in-memory task list by keyword in the title."""
    q = str(query).lower()
    if not q:
        return "Ошибка: query не должен быть пустым"
    found = [t for t in TASKS if q in t["title"].lower()]
    if not found:
        return "Ничего не найдено по запросу: " + query
    lines = [f"{t['id']}. {t['title']} — {'выполнено' if t['done'] else 'не выполнено'}" for t in found]
    return "\n".join(lines)


# --- Task lifecycle FSM tools (transitions are enforced by code) ----------------


def task_transition(to_stage: str, step: str = "", note: str = "",
                    branch_id: int = store.ROOT_BRANCH_ID) -> str:
    """Move the task to another lifecycle stage. The FSM checks the transition
    in code BEFORE applying: skips and guard violations are rejected."""
    if not str(to_stage or "").strip():
        return "Ошибка: to_stage не указан."
    try:
        state = task_fsm.transition(branch_id, str(to_stage).strip(),
                                    step=str(step or ""), note=str(note or ""))
    except task_fsm.ForbiddenTransition as exc:
        return f"Отказано: {exc}"
    stage = state["stage"]
    return (f"Переход выполнен. Текущий этап: {stage} — {state['step']}. "
            f"Ожидаемое действие: {task_fsm.STAGE_INFO[stage]['expected']} "
            f"Допустимые переходы: {', '.join(task_fsm.TRANSITIONS[stage]) or 'нет'}.")


def task_check(item: str, ok: bool = True, evidence: str = "",
               branch_id: int = store.ROOT_BRANCH_ID) -> str:
    """Record one validation check on the validation stage (the
    validation → done guard requires at least one of these)."""
    try:
        state = task_fsm.record_check(branch_id, item, ok=bool(ok), evidence=str(evidence or ""))
    except ValueError as exc:
        return f"Отказано: {exc}"
    return (f"Проверка зафиксирована ({len(state['checks'])} всего). "
            f"Этап: {state['stage']}; допустимые переходы: "
            f"{', '.join(task_fsm.TRANSITIONS[state['stage']]) or 'нет'}. "
            f"Ожидаемое действие: {task_fsm.STAGE_INFO[state['stage']]['expected']}")


def task_status(branch_id: int = store.ROOT_BRANCH_ID) -> str:
    """Read the current task state: stage, step, expected action, allowed transitions."""
    return task_fsm.render_state_block(branch_id)


def task_cancel(note: str = "", branch_id: int = store.ROOT_BRANCH_ID) -> str:
    """Cancel the current task on explicit user request: any stage → idle."""
    state = task_fsm.load_state(branch_id)
    if state["stage"] == "idle":
        return "Активной задачи нет — отменять нечего (этап idle)."
    task_fsm.reset(branch_id, note=str(note or "задача отменена пользователем"))
    # Отмена задачи — тот же финал, что и done: рабочая память очищается,
    # ttl-факты долговременной памяти удаляются, остальная LTM сохраняется.
    from . import memory  # local import: avoids a module cycle
    removed = memory.on_task_finished(branch_id, reason="задача отменена пользователем")
    return (f"Задача отменена, автомат возвращён в idle. Рабочая память очищена "
            f"(удалено фактов: {removed}); долговременная память сохранена. "
            "Новая задача — через task_transition(to_stage='planning').")


# Schema sent to the model so it knows the tools and their arguments.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time. No arguments.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression. Example: '2+3*4', '(4-1)*5'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic expression to evaluate."}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between USD, EUR and RUB at hardcoded rates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert."},
                    "from_currency": {"type": "string", "description": "Source currency (USD, EUR or RUB)."},
                    "to_currency": {"type": "string", "description": "Target currency (USD, EUR or RUB)."},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Roll one or more dice. Defaults to a single 6-sided die.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sides": {"type": "integer", "description": "Number of sides per die (default 6)."},
                    "count": {"type": "integer", "description": "Number of dice to roll (default 1)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tasks",
            "description": "Search the in-memory task list by a keyword in the title.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keyword to search for."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_transition",
            "description": (
                "Перевести жизненный цикл текущей задачи в новый этап (конечный автомат). "
                "Этапы: planning → awaiting_approval → execution → validation → done; "
                "старт: planning (из idle). Переход ПРОВЕРЯЕТСЯ КОДОМ до применения: "
                "перепрыгнуть этап, начать реализацию без утверждённого плана или завершить "
                "задачу без валидации нельзя — инструмент вернёт отказ и список допустимых переходов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to_stage": {
                        "type": "string",
                        "enum": list(task_fsm.STATES),
                        "description": "Целевой этап автомата.",
                    },
                    "step": {
                        "type": "string",
                        "description": "Название текущего шага внутри целевого этапа (необязательно).",
                    },
                    "note": {
                        "type": "string",
                        "description": "Краткое пояснение перехода: план при awaiting_approval, итог при done и т.п.",
                    },
                },
                "required": ["to_stage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_check",
            "description": (
                "Зафиксировать результат одной проверки на этапе validation. "
                "Переход validation → done код пропустит только при минимум одной "
                "зафиксированной проверке."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "Что проверялось."},
                    "ok": {"type": "boolean", "description": "Прошла ли проверка."},
                    "evidence": {"type": "string", "description": "Чем подтверждается результат проверки."},
                },
                "required": ["item", "ok"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_status",
            "description": (
                "Прочитать текущее состояние задачи: этап, текущий шаг, ожидаемое действие, "
                "допустимые переходы."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_cancel",
            "description": (
                "Отменить текущую задачу по явной просьбе пользователя: любой этап → idle. "
                "Не переход автомата, а отмена жизненного цикла."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Причина отмены (необязательно)."},
                },
                "required": [],
            },
        },
    },
]

# Map schema name -> python function. Each tool returns a plain string.
TOOL_FUNCS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "convert_currency": convert_currency,
    "roll_dice": roll_dice,
    "search_tasks": search_tasks,
    "task_transition": task_transition,
    "task_check": task_check,
    "task_status": task_status,
    "task_cancel": task_cancel,
}

# Tools that operate on the calling branch's task state; run_tool injects it.
TASK_TOOLS = frozenset({"task_transition", "task_check", "task_status", "task_cancel"})


def run_tool(name: str, args: dict,
             branch_id: int = store.ROOT_BRANCH_ID) -> str:
    """Look up a tool by name, execute it and return the text result.

    Task lifecycle tools receive the calling branch's id so the automaton
    state is always the one of the dialog the model is answering in.
    """
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"Ошибка: неизвестный инструмент {name}"
    if not isinstance(args, dict):
        args = {}
    if name in TASK_TOOLS:
        args = {**args, "branch_id": branch_id}
    try:
        return str(func(**args))
    except TypeError as exc:
        return f"Ошибка аргументов: {exc}"
