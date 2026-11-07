"""A small set of benchmark tasks with known answers for correctness checks.

Each task exposes:
  - ``id``: stable identifier used by the web UI.
  - ``title``: short human-readable name shown in the dropdown.
  - ``prompt``: the question sent to the model (never contains the answer).
  - ``answer``: the expected answer, used ONLY for grading (never sent to the model).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    prompt: str
    answer: int

    def check(self, text: str) -> bool:
        """Return True if the model's final answer matches the expected answer.

        The answer is a single number, so we extract all integers from the
        model's final text and look for the expected one.
        """
        if not text:
            return False
        numbers = re.findall(r"\d+", text)
        return str(self.answer) in numbers


TASKS: list[Task] = [
    Task(
        id="clock",
        title="Часы: сколько ударов?",
        prompt="Стенные часы отбивают 5 ударов за 4 секунды. За сколько секунд они отобьют 10 ударов? Ответ дай одним числом.",
        answer=9,
    ),
    Task(
        id="months28",
        title="Месяцы с 28 днями",
        prompt="Сколько месяцев в году содержат хотя бы 28 дней? Ответ дай одним числом.",
        answer=12,
    ),
    Task(
        id="sibs",
        title="Семь братьев",
        prompt="У семи братьев по одной сестре. Сколько всего детей в семье? Ответ дай одним числом.",
        answer=8,
    ),
    Task(
        id="chicken",
        title="Куры и кролики",
        prompt="Во дворе гуляют куры и кролики. Всего 35 голов и 94 ноги. Сколько кур? Ответ дай одним числом.",
        answer=23,
    ),
    Task(
        id="cube",
        title="Окрашенный куб",
        prompt="Куб 3×3×3 окрасили со всех сторон и распилили на 27 маленьких кубиков. Сколько кубиков не окрашены ни с одной стороны? Ответ дай одним числом.",
        answer=1,
    ),
]


def by_id(task_id: str | None) -> Task | None:
    """Return the task with the given id, or None."""
    if not task_id:
        return None
    return next((t for t in TASKS if t.id == task_id), None)
