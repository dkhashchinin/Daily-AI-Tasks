"""A/B experiment: "Эксперимент с хранением контекста".

The same scripted dialog (``app/experiment_script.json``) is played twice
through the real agent — first in "полная история" mode, then in
"summary + 10 последних" mode. After each dialog the detail questions from the
same file are asked in the same mode; the stored expected answers are used
only for grading. The report compares how many details were remembered and how
many tokens each strategy spent.

The experiment runs step by step (one agent turn per :func:`step` call) so the
web UI can show the dialog live. State is kept in memory; a page reload can
resume an unfinished run via :func:`snapshot`.
"""
import json
import re
from pathlib import Path

from . import store
from .agent import Agent

SCRIPT_PATH = Path(__file__).resolve().parent / "experiment_script.json"

RUNS = [
    {"mode": "full", "label": "полная история"},
    {"mode": "summary", "label": "summary + 10 последних"},
]

_state: dict = {"phase": "idle", "script": None, "runs": [], "report": None}


def _load_script() -> dict:
    if _state["script"] is None:
        with open(SCRIPT_PATH, encoding="utf-8") as fh:
            _state["script"] = json.load(fh)
    return _state["script"]


def _new_run(mode: str, label: str) -> dict:
    return {
        "mode": mode,
        "label": label,
        "transcript": [],
        "qa": [],
        "qa_history_tokens": [],
        "dialog_history_tokens": [],
        "api_prompt": 0,
        "api_completion": 0,
        "api_cost": 0.0,
        "summary_usage": {"prompt": 0, "completion": 0, "cost": 0.0, "calls": 0},
    }


def _norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    # уравниваем латинскую x, кириллическую х и знак умножения
    text = text.replace("\u00d7", "x").replace("\u0445", "x")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _grade(answer: str, fact_groups: list[list[str]]) -> tuple[int, int]:
    """Return (matched_groups, total_groups) for an agent answer."""
    hay = _norm(answer)
    matched = 0
    for variants in fact_groups:
        if any(_norm(v) in hay for v in variants):
            matched += 1
    return matched, len(fact_groups)


def _run() -> dict:
    return _state["runs"][-1]


def start() -> dict:
    """(Re)start the experiment: wipe the chat and begin run 0."""
    _load_script()
    store.clear_history()
    _state.update({
        "phase": "running",
        "stage": "dialog",
        "run": 0,
        "turn": 0,
        "runs": [_new_run(**RUNS[0])],
        "report": None,
        "error": None,
    })
    return snapshot()


def _accumulate_usage(run: dict, tokens: dict, context_info: dict) -> None:
    run["api_prompt"] += tokens.get("prompt") or 0
    run["api_completion"] += tokens.get("completion") or 0
    run["api_cost"] += tokens.get("cost") or 0.0
    su = (context_info or {}).get("summary_usage")
    if su:
        acc = run["summary_usage"]
        acc["prompt"] += su.get("prompt") or 0
        acc["completion"] += su.get("completion") or 0
        acc["cost"] += su.get("cost") or 0.0
        acc["calls"] += 1


def step() -> dict:
    """Perform exactly one agent turn and return progress for the UI."""
    if _state["phase"] != "running":
        return snapshot()

    script = _load_script()
    run = _run()
    agent = Agent()

    try:
        if _state["stage"] == "dialog":
            user_text = script["dialog"][_state["turn"]]
            result = agent.ask(user_text, run["mode"])
            tokens = result.get("tokens") or {}
            run["dialog_history_tokens"].append(tokens.get("history") or 0)
            run["transcript"].append({
                "user": user_text,
                "reply": result.get("reply"),
            })
            _accumulate_usage(run, tokens, result.get("context"))
            progress = {
                "user": user_text,
                "reply": result.get("reply"),
                "tokens": tokens,
                "context": result.get("context"),
            }
            _state["turn"] += 1
            if _state["turn"] >= len(script["dialog"]):
                _state["stage"] = "questions"
                _state["turn"] = 0

        elif _state["stage"] == "questions":
            q = script["questions"][_state["turn"]]
            result = agent.ask(q["question"], run["mode"])
            tokens = result.get("tokens") or {}
            matched, total = _grade(result.get("reply") or "", q["facts"])
            run["qa_history_tokens"].append(tokens.get("history") or 0)
            run["qa"].append({
                "question": q["question"],
                "expected": q["expected"],
                "answer": result.get("reply"),
                "facts_found": matched,
                "facts_total": total,
                "ok": matched == total,
            })
            _accumulate_usage(run, tokens, result.get("context"))
            progress = {
                "user": q["question"],
                "reply": result.get("reply"),
                "tokens": tokens,
                "context": result.get("context"),
                "ok": matched == total,
                "facts_found": matched,
                "facts_total": total,
            }
            _state["turn"] += 1
            if _state["turn"] >= len(script["questions"]):
                if _state["run"] + 1 < len(RUNS):
                    nxt = RUNS[_state["run"] + 1]
                    _state["run"] += 1
                    _state["stage"] = "dialog"
                    _state["turn"] = 0
                    _state["runs"].append(_new_run(**nxt))
                    store.clear_history()  # второй прогон — с чистого листа
                else:
                    _state["phase"] = "done"
                    _state["report"] = _build_report()
        else:  # pragma: no cover
            return snapshot()
    except Exception as exc:  # состояние не двигаем — шаг можно повторить
        _state["error"] = str(exc)
        snapshot_ = snapshot()
        snapshot_["error"] = str(exc)
        return snapshot_

    return {**snapshot(), "progress": progress}


def _build_report() -> dict:
    script = _state["script"]
    questions = script["questions"]
    details_total = sum(len(q["facts"]) for q in questions)
    runs_out = []
    for run in _state["runs"]:
        hist = run["qa_history_tokens"] or [0]
        dialog_hist = run["dialog_history_tokens"] or [0]
        runs_out.append({
            "mode": run["mode"],
            "label": run["label"],
            "remembered": sum(1 for q in run["qa"] if q["ok"]),
            "questions_total": len(questions),
            "details_found": sum(q["facts_found"] for q in run["qa"]),
            "details_total": details_total,
            "api": {
                "prompt": run["api_prompt"],
                "completion": run["api_completion"],
                "cost": round(run["api_cost"], 4),
            },
            "summary_usage": {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in run["summary_usage"].items()},
            "avg_context_tokens": round(sum(hist) / len(hist)),
            "avg_dialog_context_tokens": round(sum(dialog_hist) / len(dialog_hist)),
            "qa": run["qa"],
            "transcript": run["transcript"],
        })
    return {"title": script["title"], "runs": runs_out}


def snapshot() -> dict:
    """Current experiment state for UI rendering and resume."""
    script = _state.get("script")
    snap = {
        "phase": _state["phase"],
        "stage": _state.get("stage"),
        "run": _state.get("run", 0),
        "mode": _state["runs"][-1]["mode"] if _state["runs"] else None,
        "label": _state["runs"][-1]["label"] if _state["runs"] else None,
        "turn": _state.get("turn", 0),
        "dialog_total": len(script["dialog"]) if script else 0,
        "questions_total": len(script["questions"]) if script else 0,
        "report": _state.get("report"),
    }
    if _state["phase"] == "running" and script:
        total = (script["dialog"] if _state["stage"] == "dialog"
                 else script["questions"])
        snap["step_index"] = _state["turn"]
        snap["step_total"] = len(total)
    return snap


def running() -> bool:
    return _state["phase"] == "running"
