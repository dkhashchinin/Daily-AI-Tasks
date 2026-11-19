"""Profile experiment: «Эксперимент с профилями» (personalization check).

Two checks run through the real agent, step by step (one agent turn per
:func:`step` call so the web UI shows the dialog live):

1. **Противоположные профили** — three scripted profiles with opposite
   preferences (кратко/подробно, без кода/с кодом, формально/неформально)
   are activated one by one, and everyone is asked the SAME question. Each
   answer is auto-graded against the profile's expectations: how many
   preferences are actually observed.

2. **Устойчивость** — one profile («Кратко и формально») gets 10 questions in
   a row. Every answer is graded against the profile; the report fixes at
   which answer the assistant starts ignoring the preferences (if it ever
   does).

Profiles created by the experiment, the dialog and the working memory are
snapshotted at start and restored at the end.
"""
import json
import re
from pathlib import Path

from . import profile as profile_mod
from . import store
from .agent import Agent

SCRIPT_PATH = Path(__file__).resolve().parent / "profile_script.json"

# Auto-grading thresholds (words per answer).
BRIEF_MAX_WORDS = 120
DETAILED_MIN_WORDS = 150

# Informal-tone markers: 2nd person singular, casual words, laughter, emoji.
_INFORMAL_RES = [(re.compile(p), label) for p, label in (
    (r"\bты\b", "«ты»"), (r"\bтебе\b", "«тебе»"), (r"\bтебя\b", "«тебя»"),
    (r"\bтво(?:й|я|ё|е|и|ю)\b", "«твой»"), (r"\bтобой\b", "«тобой»"),
    (r"\bпривет\b", "«привет»"), (r"\bдавай\b", "«давай»"), (r"\bокей\b", "«окей»"),
    (r"\bого\b", "«ого»"), (r"\bвау\b", "«вау»"), (r"\bкрут[ооая]\b", "«круто»"),
    (r"\bклёво\b", "«клёво»"), (r"\bклассн", "«классно»"), (r"\bприкольно\b", "«прикольно»"),
    (r"\bкороче\b", "«короче»"), (r"\bшутк", "шутка"), (r"\bюмор", "юмор"),
    (r"\bпростыми словами\b", "«простыми словами»"), (r"\bпо-простому\b", "«по-простому»"),
    (r"\bпогнали\b", "«погнали»"), (r"\bсмотри\b", "«смотри»"),
    (r"\bразберём\b", "«разберём»"), (r"\bполучится\b", "«получится»"),
    (r"\bсупер\b", "«супер»"),
)]
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_CODE_RES = [re.compile(p) for p in (
    r"```",                                        # fenced code block
    r"^\s*(?:def |class |import |from \w+ import |print\()",  # python-ish lines
    r"\b(?:def|return|import)\b.*[\(\[]",          # keywords with brackets
    r"\w+\s*=\s*[\w\[\(].*[+\-*/<>=]",             # assignments / comparisons
)]

_state: dict = {"phase": "idle", "script": None, "report": None}


def _load_script() -> dict:
    if _state.get("script") is None:
        with open(SCRIPT_PATH, encoding="utf-8") as fh:
            _state["script"] = json.load(fh)
    return _state["script"]


def _new_state() -> dict:
    return {
        "phase": "running",
        "stage": "start",
        "profile_run": 0,
        "turn": 0,
        "snap": None,
        "profile_ids": [],
        "results": {"profiles": [], "stability": []},
        "report": None,
        "error": None,
    }


def _words(text: str) -> int:
    return len(re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text or ""))


def _code_found(text: str) -> bool:
    return any(r.search(text or "") for r in _CODE_RES)


def _informal_markers(text: str) -> list[str]:
    hits = [label for regex, label in _INFORMAL_RES if regex.search(text or "")]
    if _EMOJI_RE.search(text or ""):
        hits.append("эмодзи")
    return hits


def _grade_dim(dim: str, expect: str, answer: str) -> dict:
    """Grade one preference dimension; 'any' always passes."""
    words = _words(answer)
    if expect == "any":
        return {"expect": "any", "ok": True, "evidence": "не проверяется"}
    if dim == "style":
        if expect == "brief":
            ok = 0 < words <= BRIEF_MAX_WORDS
            return {"expect": "brief", "ok": ok,
                    "evidence": f"{words} слов (лимит ≤ {BRIEF_MAX_WORDS})"}
        ok = words >= DETAILED_MIN_WORDS or (
            words >= 100 and re.search(r"\n\s*(?:[-*•]|\d[.)])", answer or ""))
        return {"expect": "detailed", "ok": ok,
                "evidence": f"{words} слов (норма ≥ {DETAILED_MIN_WORDS})"}
    if dim == "format":
        code = _code_found(answer)
        if expect == "code":
            return {"expect": "code", "ok": code,
                    "evidence": "код найден" if code else "кода нет"}
        return {"expect": "no_code", "ok": not code,
                "evidence": "кода нет" if not code else "код найден"}
    if dim == "tone":
        markers = _informal_markers(answer)
        if expect == "informal":
            ok = bool(markers)
            return {"expect": "informal", "ok": ok,
                    "evidence": ("маркеры: " + ", ".join(markers[:4])) if ok
                    else "неформальных маркеров нет"}
        ok = not markers
        return {"expect": "formal", "ok": ok,
                "evidence": "формально" if ok else "маркеры: " + ", ".join(markers[:4])}
    return {"expect": expect, "ok": True, "evidence": "?"}


def _check_dims(expect: dict, answer: str) -> dict:
    dims = {d: _grade_dim(d, e, answer) for d, e in
            (("style", expect.get("style", "any")),
             ("format", expect.get("format", "any")),
             ("tone", expect.get("tone", "any")))}
    checked = [d for d, v in dims.items() if v["expect"] != "any"]
    return {
        "dims": dims,
        "prefs_total": len(checked),
        "prefs_met": sum(1 for d in checked if dims[d]["ok"]),
        "ok_all": all(dims[d]["ok"] for d in checked),
        "violations": [d for d in checked if not dims[d]["ok"]],
    }


def _snapshot() -> dict:
    return {
        "profiles": store.snapshot_profiles(),
        "messages": store.load_messages(),
        "facts": store.load_facts(),
    }


def _restore(snap: dict) -> None:
    store.save_messages(snap["messages"])
    store.save_facts(snap["facts"])
    store.restore_profiles(snap["profiles"])


def start() -> dict:
    """(Re)start the profile experiment: snapshot, wipe the chat, seed profiles."""
    script = _load_script()
    _state.clear()
    _state.update(_new_state())
    _state["snap"] = _snapshot()
    store.clear_dialog(store.ROOT_BRANCH_ID)
    store.clear_task_memory(store.ROOT_BRANCH_ID)
    ids = []
    for spec in script["profiles"]:
        pid = store.create_profile(spec["name"], spec.get("description", ""),
                                   spec.get("rules") or [])
        ids.append(pid)
    _state["profile_ids"] = ids
    store.set_active_profile(ids[0])
    _state["stage"] = "compare"
    _state["profile_run"] = 0
    _state["turn"] = 0
    return snapshot()


def step() -> dict:
    """Perform exactly one agent turn (or state transition) for the UI."""
    if _state["phase"] != "running":
        return snapshot()

    script = _load_script()
    stage = _state["stage"]
    agent = Agent()
    progress: dict | None = None

    try:
        if stage == "compare":
            spec = script["profiles"][_state["profile_run"]]
            res = agent.ask(script["shared_question"])
            grade = _check_dims(spec.get("expect") or {}, res.get("reply") or "")
            entry = {"name": spec["name"], "expect": spec.get("expect") or {},
                     "answer": res.get("reply"), "grade": grade}
            _state["results"]["profiles"].append(entry)
            progress = {
                "user": f"[профиль «{spec['name']}»] {script['shared_question']}",
                "reply": res.get("reply"),
                "tokens": res.get("tokens"),
                "context": res.get("context"),
                "grade": grade,
            }
            _state["profile_run"] += 1
            if _state["profile_run"] >= len(script["profiles"]):
                # переходим к проверке устойчивости: тот же первый профиль
                store.set_active_profile(
                    _state["profile_ids"][script.get("stability_profile_index", 0)])
                store.clear_dialog(store.ROOT_BRANCH_ID)
                store.clear_task_memory(store.ROOT_BRANCH_ID)
                _state["stage"] = "stability"
                _state["turn"] = 0
            else:
                store.set_active_profile(_state["profile_ids"][_state["profile_run"]])
                store.clear_dialog(store.ROOT_BRANCH_ID)
                store.clear_task_memory(store.ROOT_BRANCH_ID)

        elif stage == "stability":
            spec = script["profiles"][script.get("stability_profile_index", 0)]
            questions = script["stability_questions"]
            question = questions[_state["turn"]]
            res = agent.ask(question)
            grade = _check_dims(spec.get("expect") or {}, res.get("reply") or "")
            entry = {"n": _state["turn"] + 1, "question": question,
                     "answer": res.get("reply"), "grade": grade}
            _state["results"]["stability"].append(entry)
            progress = {
                "user": f"[ответ #{_state['turn'] + 1}] {question}",
                "reply": res.get("reply"),
                "tokens": res.get("tokens"),
                "context": res.get("context"),
                "grade": grade,
            }
            _state["turn"] += 1
            if _state["turn"] >= len(questions):
                _state["stage"] = "restore"

        elif stage == "restore":
            _restore(_state["snap"])
            _state["snap"] = None
            _state["phase"] = "done"
            _state["report"] = _build_report()
            return {**snapshot()}

        else:  # pragma: no cover
            return snapshot()
    except Exception as exc:  # состояние не двигаем — шаг можно повторить
        _state["error"] = str(exc)
        snap = snapshot()
        snap["error"] = str(exc)
        return snap

    return {**snapshot(), "progress": progress}


def _build_report() -> dict:
    script = _state["script"]
    profiles = _state["results"]["profiles"]
    stability = _state["results"]["stability"]

    prefs_total = sum(p["grade"]["prefs_total"] for p in profiles)
    prefs_met = sum(p["grade"]["prefs_met"] for p in profiles)
    profiles_ok = sum(1 for p in profiles if p["grade"]["ok_all"])

    viol = [s for s in stability if not s["grade"]["ok_all"]]
    first_violation = viol[0]["n"] if viol else None
    learned = []
    sid = _state["profile_ids"][script.get("stability_profile_index", 0)] \
        if _state["profile_ids"] else None
    if sid:
        learned = store.list_profile_rules(sid, source="dialog")

    return {
        "title": script["title"],
        "shared_question": script["shared_question"],
        "brief_limit": BRIEF_MAX_WORDS,
        "detailed_limit": DETAILED_MIN_WORDS,
        "profiles": profiles,
        "summary": {
            "profiles": len(profiles),
            "profiles_ok": profiles_ok,
            "prefs_total": prefs_total,
            "prefs_met": prefs_met,
        },
        "stability": {
            "profile": script["profiles"][script.get("stability_profile_index", 0)]["name"],
            "expect": script["profiles"][script.get("stability_profile_index", 0)].get("expect") or {},
            "qa": stability,
            "questions_total": len(stability),
            "compliant": sum(1 for s in stability if s["grade"]["ok_all"]),
            "violations_total": len(viol),
            "first_violation": first_violation,
            "learned_rules": learned,
        },
    }


def snapshot() -> dict:
    """Current experiment state for UI rendering and resume."""
    stage = _state.get("stage")
    script = _state.get("script")
    labels = {
        "start": "Готовлю профили",
        "compare": "Один и тот же вопрос разным профилям",
        "stability": "Устойчивость: 10 вопросов подряд",
        "restore": "Восстанавливаю состояние",
    }
    snap = {
        "phase": _state["phase"],
        "stage": stage,
        "stage_label": labels.get(stage, ""),
        "profile_run": _state.get("profile_run", 0),
        "turn": _state.get("turn", 0),
        "report": _state.get("report"),
    }
    if _state["phase"] == "running":
        if stage == "compare" and script:
            snap["step_index"] = _state.get("profile_run", 0)
            snap["step_total"] = len(script.get("profiles") or [])
        elif stage == "stability" and script:
            snap["step_index"] = _state.get("turn", 0)
            snap["step_total"] = len(script.get("stability_questions") or [])
        else:
            snap["step_index"] = 0
            snap["step_total"] = 1
    return snap


def running() -> bool:
    return _state["phase"] == "running"
