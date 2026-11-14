"""Tool functions for the agent. Model decides which to call and with what args."""
import ast
import datetime
import operator
import random

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
]

# Map schema name -> python function. Each tool returns a plain string.
TOOL_FUNCS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "convert_currency": convert_currency,
    "roll_dice": roll_dice,
    "search_tasks": search_tasks,
}


def run_tool(name: str, args: dict) -> str:
    """Look up a tool by name, execute it and return the text result."""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"Ошибка: неизвестный инструмент {name}"
    if not isinstance(args, dict):
        args = {}
    try:
        return str(func(**args))
    except TypeError as exc:
        return f"Ошибка аргументов: {exc}"
