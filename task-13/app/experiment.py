"""Memory-layer experiment: «Эксперимент с памятью».

The scripted scenario (``app/experiment_script.json``) plays out in two
sessions through the real agent in the single memory mode:

1. **Диалог 1** — session one, its facts accumulate in working memory (WM);
2. **Перенос** — the session-one facts are promoted into long-term memory
   (LTM, ``memory.seed_from_experiment``) and the dialog is cleared: from now
   on task-one details survive only in LTM;
3. **Диалог 2** — a new task for a new client; its facts fill WM, its recent
   messages live in short-term memory (STM);
4. **Базовый прогон** — all questions are asked with every layer in place;
   per-layer token contributions (LTM / WM / STM) are measured;
5. **Очистка слоёв по очереди** — for each layer (WM → LTM → STM): snapshot
   all layers, clear the target one, re-ask every question, grade, restore.
   The report fixes exactly which questions the agent stops answering
   without each layer.

The experiment runs step by step (one agent turn per :func:`step` call) so the
web UI can show the dialog live; a page reload resumes via :func:`snapshot`.
"""
import json
import re
from pathlib import Path

from . import store
from .agent import Agent
from .memory import seed_from_experiment

SCRIPT_PATH = Path(__file__).resolve().parent / "experiment_script.json"

# Layers cleared one by one, in order, with their store-level actions.
LAYERS = [
    {
        "layer": "wm",
        "title": "Рабочая (данные текущей задачи)",
        "clear": lambda: store.clear_task_memory(store.ROOT_BRANCH_ID),
    },
    {
        "layer": "ltm",
        "title": "Долговременная (профиль, решения, знания)",
        "clear": lambda: store.clear_long_term(),
    },
    {
        "layer": "stm",
        "title": "Краткосрочная (текущий диалог)",
        "clear": lambda: store.clear_dialog(store.ROOT_BRANCH_ID),
    },
]

_state: dict = {"phase": "idle", "script": None, "report": None}


def _load_script() -> dict:
    if _state["script"] is None:
        with open(SCRIPT_PATH, encoding="utf-8") as fh:
            _state["script"] = json.load(fh)
    return _state["script"]


def _questions(script: dict) -> list[dict]:
    """All questions in asking order, each tagged with its memory layer."""
    out: list[dict] = []
    for key, layer in (("questions", "ltm"), ("questions2", "wm")):
        for q in script.get(key) or []:
            out.append({**q, "layer": q.get("layer", layer)})
    return out


def _new_result(label: str, layer: str | None = None) -> dict:
    return {
        "label": label,
        "layer": layer,
        "qa": [],
        "layer_tokens": {"ltm": [], "wm": [], "stm": []},
        "api": {"prompt": 0, "completion": 0, "cost": 0.0},
    }


def _snapshot_layers() -> dict:
    """Capture all three layers so a cleared one can be restored."""
    return {
        "messages": store.load_messages(),
        "facts": store.load_facts(),
        "ltm": store.list_long_term(),
    }


def _restore_layers(snap: dict) -> None:
    """Put all three layers back exactly as captured in ``snap``."""
    store.save_messages(snap["messages"])
    store.save_facts(snap["facts"])
    store.clear_long_term()
    for entry in snap["ltm"]:
        store.add_long_term(entry["category"], entry["key"], entry["value"],
                            source=entry["source"])


def start() -> dict:
    """(Re)start the experiment: wipe the chat and begin with dialog one."""
    _load_script()
    store.clear_history()
    _state.update({
        "phase": "running",
        "stage": "dialog1",
        "turn": 0,
        "layer_run": 0,
        "snap": None,
        "results": {"baseline": _new_result("все слои на месте"), "layers": []},
        "report": None,
        "error": None,
    })
    return snapshot()


def _avg(values: list[int]) -> int:
    return round(sum(values) / len(values)) if values else 0


def _accumulate(result: dict, tokens: dict, context_info: dict) -> None:
    api = result["api"]
    api["prompt"] += tokens.get("prompt") or 0
    api["completion"] += tokens.get("completion") or 0
    api["cost"] += tokens.get("cost") or 0.0
    ci = context_info or {}
    lt = result["layer_tokens"]
    if "ltm_tokens" in ci:
        lt["ltm"].append(ci.get("ltm_tokens") or 0)
        lt["wm"].append(ci.get("facts_tokens") or 0)
        lt["stm"].append(ci.get("stm_tokens") or 0)


def _ask_question(agent: Agent, result: dict, q: dict) -> dict:
    """Ask one graded question, record the outcome, return the UI progress."""
    res = agent.ask(q["question"])
    tokens = res.get("tokens") or {}
    matched, total = _grade(res.get("reply") or "", q["facts"])
    entry = {
        "question": q["question"],
        "expected": q["expected"],
        "layer": q["layer"],
        "answer": res.get("reply"),
        "facts_found": matched,
        "facts_total": total,
        "ok": matched == total,
    }
    result["qa"].append(entry)
    _accumulate(result, tokens, res.get("context"))
    return {
        "user": q["question"],
        "reply": res.get("reply"),
        "tokens": tokens,
        "context": res.get("context"),
        "ok": entry["ok"],
        "facts_found": matched,
        "facts_total": total,
        "layer": q["layer"],
    }


def _norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
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


def step() -> dict:
    """Perform exactly one agent turn (or state transition) for the UI."""
    if _state["phase"] != "running":
        return snapshot()

    script = _load_script()
    stage = _state["stage"]
    agent = Agent()
    branch = store.ROOT_BRANCH_ID
    progress: dict | None = None

    try:
        if stage in ("dialog1", "dialog2"):
            dialog = script["dialog"] if stage == "dialog1" else script["dialog2"]
            user_text = dialog[_state["turn"]]
            res = agent.ask(user_text, branch)
            _accumulate(_state["results"]["baseline"], res.get("tokens") or {},
                        res.get("context"))
            progress = {"user": user_text, "reply": res.get("reply"),
                        "tokens": res.get("tokens"), "context": res.get("context")}
            _state["turn"] += 1
            if _state["turn"] >= len(dialog):
                _state["stage"] = ("promote" if stage == "dialog1" else "baseline")
                _state["turn"] = 0

        elif stage == "promote":
            # Session-one facts become the basis of long-term memory, then the
            # dialog is cleared: task-one details survive only in LTM now.
            seed_from_experiment(store.load_facts(branch))
            store.clear_dialog(branch)
            progress = {"user": None, "reply": (
                "Факты задачи 1 перенесены в долговременную память, диалог очищен. "
                "Новая задача — диалог 2."), "tokens": {}, "context": None}
            _state["stage"] = "dialog2"
            _state["turn"] = 0

        elif stage == "baseline":
            questions = _questions(script)
            result = _state["results"]["baseline"]
            progress = _ask_question(agent, result, questions[_state["turn"]])
            _state["turn"] += 1
            if _state["turn"] >= len(questions):
                _state["results"]["layers"] = []
                _state["layer_run"] = 0
                _state["stage"] = "clear"
                _state["turn"] = 0

        elif stage == "clear":
            layer = LAYERS[_state["layer_run"]]
            _state["snap"] = _snapshot_layers()
            layer["clear"]()
            _state["results"]["layers"].append(
                _new_result(f"без слоя: {layer['title']}", layer["layer"]))
            progress = {"user": None, "reply": (
                f"Слой «{layer['title']}» очищен — задаю вопросы снова."),
                "tokens": {}, "context": None}
            _state["stage"] = "reask"
            _state["turn"] = 0

        elif stage == "reask":
            questions = _questions(script)
            result = _state["results"]["layers"][_state["layer_run"]]
            progress = _ask_question(agent, result, questions[_state["turn"]])
            _state["turn"] += 1
            if _state["turn"] >= len(questions):
                _state["stage"] = "restore"
                _state["turn"] = 0

        elif stage == "restore":
            _restore_layers(_state["snap"])
            _state["snap"] = None
            layer = LAYERS[_state["layer_run"]]
            progress = {"user": None, "reply": (
                f"Слой «{layer['title']}» восстановлен."), "tokens": {}, "context": None}
            _state["layer_run"] += 1
            if _state["layer_run"] >= len(LAYERS):
                _state["phase"] = "done"
                _state["report"] = _build_report()
            else:
                _state["stage"] = "clear"
                _state["turn"] = 0

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
    questions = _questions(script)
    baseline = _state["results"]["baseline"]
    details_total = sum(len(q["facts"]) for q in questions)

    base_qa = baseline["qa"]
    base_ok = {entry["question"]: entry["ok"] for entry in base_qa}
    layers_out = []
    for res in _state["results"]["layers"]:
        broken = [e["question"] for e in res["qa"]
                  if base_ok.get(e["question"], True) and not e["ok"]]
        layers_out.append({
            "layer": res["layer"],
            "label": res["label"],
            "remembered": sum(1 for e in res["qa"] if e["ok"]),
            "details_found": sum(e["facts_found"] for e in res["qa"]),
            "broken": broken,
            "api": res["api"],
            "qa": res["qa"],
        })

    lt = baseline["layer_tokens"]
    return {
        "title": script["title"],
        "questions_total": len(questions),
        "details_total": details_total,
        "baseline": {
            "label": "все слои на месте",
            "remembered": sum(1 for e in base_qa if e["ok"]),
            "details_found": sum(e["facts_found"] for e in base_qa),
            "layer_tokens": {
                "ltm": _avg(lt["ltm"]),
                "wm": _avg(lt["wm"]),
                "stm": _avg(lt["stm"]),
            },
            "api": baseline["api"],
            "qa": base_qa,
        },
        "layers": layers_out,
        # after the last restore every layer is back to its baseline state
        "wm_facts": store.load_facts(),
        "ltm_entries": store.list_long_term(),
    }


def snapshot() -> dict:
    """Current experiment state for UI rendering and resume."""
    script = _state.get("script")
    stage = _state.get("stage")
    layer = LAYERS[_state["layer_run"]] if _state.get("layer_run") is not None \
        and _state.get("phase") == "running" and stage in ("clear", "reask", "restore") \
        else None
    if stage == "dialog1":
        seq, label = script["dialog"] if script else [], "Диалог 1 · задача 1"
    elif stage == "promote":
        seq, label = [1], "Перенос фактов задачи 1 в долговременную память"
    elif stage == "dialog2":
        seq, label = script["dialog2"] if script else [], "Диалог 2 · новая задача"
    elif stage == "baseline":
        seq, label = _questions(script) if script else [], "Вопросы · все слои на месте"
    elif stage == "clear":
        seq, label = [1], f"Очищаю слой: {layer['title']}" if layer else ""
    elif stage == "reask":
        seq, label = (_questions(script) if script else []), \
            f"Вопросы без слоя: {layer['title']}" if layer else ""
    elif stage == "restore":
        seq, label = [1], f"Восстанавливаю слой: {layer['title']}" if layer else ""
    else:
        seq, label = [], ""
    snap = {
        "phase": _state["phase"],
        "stage": stage,
        "stage_label": label,
        "turn": _state.get("turn", 0),
        "layer_run": _state.get("layer_run", 0),
        "report": _state.get("report"),
    }
    if _state["phase"] == "running" and seq:
        snap["step_index"] = _state.get("turn", 0)
        snap["step_total"] = len(seq)
    return snap


def running() -> bool:
    return _state["phase"] == "running"
