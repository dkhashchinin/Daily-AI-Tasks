"""Benchmark/generation tasks with correctness + format checks.

Each task exposes:
  - ``id``: stable identifier used by the web UI.
  - ``title``: short human-readable name shown in the dropdown.
  - ``prompt``: the question sent to the model (never contains the answer).
  - ``answer``: expected answer (int for numeric tasks, None for generation).
  - ``kind``: ``numeric`` (single-number answer) or ``generation`` (text/ideas/names).
  - ``format_desc``: short human-readable description of the expected format.
  - ``min_items``: for generation tasks, the minimum number of items to be valid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    prompt: str
    answer: int | None
    kind: str = "numeric"  # "numeric" | "generation"
    format_desc: str = ""
    min_items: int = 0

    def check(self, text: str) -> bool | None:
        """Return True if the answer is correct, False if wrong.

        For generation tasks there is no single correct answer, so None is returned
        (correctness is not applicable) — only ``check_format`` matters there.
        """
        if self.kind == "generation":
            return None
        if not text:
            return False
        numbers = re.findall(r"\d+", text)
        return str(self.answer) in numbers

    def check_format(self, text: str) -> bool:
        """Return True if the output matches the expected format."""
        if not text:
            return False
        if self.kind == "numeric":
            return bool(re.search(r"\d+", text))
        items = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return len(items) >= self.min_items


TASKS: list[Task] = [
    Task(
        id="cube",
        title="Окрашенный куб",
        prompt="Куб 3×3×3 окрасили со всех сторон и распилили на 27 маленьких кубиков. Сколько кубиков не окрашены ни с одной стороны? Ответ дай одним числом.",
        answer=1,
    ),
    Task(
        id="gen_names",
        title="Генерация вариантов названия",
        prompt="Придумай 5 коротких запоминающихся вариантов названия для сервиса доставки еды. Каждый вариант выведи отдельным пунктом с новой строки, без лишних пояснений.",
        answer=None,
        kind="generation",
        format_desc="не менее 5 пунктов, каждый с новой строки",
        min_items=5,
    ),
]


def by_id(task_id: str | None) -> Task | None:
    """Return the task with the given id, or None."""
    if not task_id:
        return None
    return next((t for t in TASKS if t.id == task_id), None)
