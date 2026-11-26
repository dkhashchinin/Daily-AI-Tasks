"""Запуск планировщика (День 18): python scheduler.py [--once|--report|...].

Если в проекте есть .venv, скрипт автоматически перезапускается её питоном —
команда `python scheduler.py` работает независимо от того, какой python
лежит на PATH. Подробности — в app/scheduler.py.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _rerun_with_venv() -> None:
    """Перезапуск через venv-интерпретатор, если текущий — не он."""
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    try:
        same = Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        return
    if not same:
        os.execv(str(venv_python), [str(venv_python), "scheduler.py", *sys.argv[1:]])


if __name__ == "__main__":
    _rerun_with_venv()
    from app.scheduler import main

    main()
