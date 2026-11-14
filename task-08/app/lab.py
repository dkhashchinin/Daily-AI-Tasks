"""Token lab: how usage grows with dialog length, and what breaks at the limit.

The agent resends the whole conversation on every turn (see app/agent.py), so a
dialog of ``n`` turns that adds ``k`` tokens per turn costs roughly
``k * n^2 / 2`` input tokens: the price of a single message grows linearly, the
price of the whole dialog quadratically. This module runs that curve for real
and records where it breaks.

Scenarios are driven through the actual Flask routes with
``app.test_client()``, so what the lab measures is exactly what a user gets --
including the failure path. The model itself is swappable:

* **stub** -- :class:`StubModel`, which tokenizes with the same local counter
  the app uses and refuses anything that does not fit the window. No network,
  no cost: the whole pipeline and the failure mode can be demonstrated free.
* **real** -- the configured endpoint. ``LabRun.cap`` aborts the run before the
  accumulated cost can exceed it.

A run owns its own SQLite file and its own model override, both installed in
the worker thread's context (see ``store.use_db`` and ``llm.use_chat``), so the
live chat on the main thread is unaffected.
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from . import llm, store
from .tokens import count_messages, count_tokens

# --- price model -----------------------------------------------------------
# Published AITUNNEL prices for deepseek-v4.1-flash, RUB per 1M tokens. The
# calibration run is expected to replace these with measured values: the
# ``cost`` the provider reports has come out higher than this formula suggests.
PRICE_IN = 9.0 / 1_000_000
PRICE_OUT = 100.0 / 1_000_000

# Tool schemas and request scaffolding on top of the message text. Measured:
# two calls with 19- and 78-token user messages reported 1360 and 1664 prompt
# tokens in total, i.e. ~700 per call.
OVERHEAD_TOKENS = 700

# Claimed by the provider; the ``real`` mode is what actually measures it.
CONTEXT_LIMIT = 1_000_000

DEFAULT_SCENARIOS = ["baseline", "short", "long", "overflow"]


# --- filler text -----------------------------------------------------------
_BLOCK_WORDS = (
    "диалог контекст история запрос ответ токен модель стоимость предел лимит "
    "память окно обрезка сводка кэш префикс бюджет счётчик прогресс"
).split()
_BLOCK = " ".join(_BLOCK_WORDS)


def make_text(target_tokens: int) -> str:
    """Return filler text of approximately ``target_tokens`` tokens."""
    if target_tokens <= 0:
        return ""
    block_tokens = count_tokens(_BLOCK)
    reps, remainder = divmod(target_tokens, block_tokens)
    parts = [_BLOCK] * reps
    tail: list[str] = []
    used = 0
    i = 0
    while used < remainder and i < 500:
        word = _BLOCK_WORDS[i % len(_BLOCK_WORDS)]
        cost = count_tokens(" " + word)
        if used + cost > remainder:
            break
        tail.append(word)
        used += cost
        i += 1
    if tail:
        parts.append(" ".join(tail))
    return " ".join(parts).strip()


# --- offline model ---------------------------------------------------------
class ContextLimitError(RuntimeError):
    """Mimics the provider's 400 when a request does not fit the window."""


class _Function:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = _Function(name, arguments)

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class _Message:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message):
        self.message = message


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, cost: float, balance: float):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.cost = cost
        self.balance = balance


class _Response:
    def __init__(self, message: _Message, usage: _Usage):
        self.choices = [_Choice(message)]
        self.usage = usage


class StubModel:
    """Offline stand-in for the provider.

    Tokenizes whatever it is handed with the same local counter the app uses,
    charges the configured price and refuses requests that do not fit the
    simulated window -- so the failure path is exercised without spending.
    """

    def __init__(
        self,
        *,
        reply_tokens: int = 200,
        emit_tools: bool = False,
        limit: int = CONTEXT_LIMIT,
        overhead: int = OVERHEAD_TOKENS,
        price_in: float = PRICE_IN,
        price_out: float = PRICE_OUT,
        balance: float = 1000.0,
    ):
        self.reply_tokens = reply_tokens
        self.emit_tools = emit_tools
        self.limit = limit
        self.overhead = overhead
        self.price_in = price_in
        self.price_out = price_out
        self.balance = balance
        self.calls = 0

    def __call__(self, messages, *, model=None, temperature=None, tools=None,
                 tool_choice="auto"):
        prompt_tokens = count_messages(messages) + self.overhead
        if prompt_tokens + self.reply_tokens > self.limit:
            raise ContextLimitError(
                f"This model's maximum context length is {self.limit} tokens. "
                f"However, you requested {prompt_tokens} tokens "
                f"({prompt_tokens} in the messages). Please reduce the length of the messages."
            )

        self.calls += 1
        completion_tokens = self.reply_tokens
        cost = round(prompt_tokens * self.price_in + completion_tokens * self.price_out, 2)
        self.balance = round(self.balance - cost, 2)

        # The first call of a turn asks for a tool; the call after the tool
        # result is a final answer, which keeps the loop -- and the stored
        # history -- shaped like a real tool-using dialog.
        if self.emit_tools and messages[-1].get("role") == "user":
            call = _ToolCall("call_1", "calculate", '{"expression": "2+2"}')
            message = _Message("", tool_calls=[call])
        else:
            message = _Message(make_text(self.reply_tokens))

        return _Response(message, _Usage(prompt_tokens, completion_tokens, cost, self.balance))


# --- scenarios -------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    title: str
    turns: int
    user_tokens: int
    reply_tokens: int = 200
    use_tools: bool = False
    seed_tokens: int = 0
    note: str = ""
    optional: bool = False


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        "baseline", "Нулевой замер", turns=1, user_tokens=10, reply_tokens=50,
        note="пустая история и минимальное сообщение: сколько стоит сам запрос",
    ),
    "short": Scenario(
        "short", "Короткий диалог", turns=8, user_tokens=150, reply_tokens=150,
        note="8 ходов по ~150 токенов: рост есть, но суммарно копейки",
    ),
    "long": Scenario(
        "long", "Длинный диалог", turns=30, user_tokens=1500, reply_tokens=500,
        note="история растёт на ~2000 токенов за ход: видно квадратичную стоимость",
    ),
    "long_tools": Scenario(
        "long_tools", "Длинный с инструментами", turns=15, user_tokens=600,
        reply_tokens=400, use_tools=True, optional=True,
        note="каждый ход вызывает инструмент: в истории появляются ещё два сообщения",
    ),
    "overflow": Scenario(
        "overflow", "Переполнение лимита", turns=3, user_tokens=200, reply_tokens=100,
        seed_tokens=1_050_000,
        note="история засеяна выше лимита: смотрим, что именно ломается",
    ),
}


def seed_history(total_tokens: int, chunk_tokens: int = 10_000) -> int:
    """Write a synthetic conversation straight into SQLite (no API calls)."""
    messages = []
    remaining = total_tokens
    i = 0
    while remaining > 0:
        size = min(chunk_tokens, remaining)
        messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": make_text(size),
        })
        remaining -= size
        i += 1
    store.save_messages(messages)
    return len(messages)


# --- runner ----------------------------------------------------------------
ROW_FIELDS = [
    "scenario", "turn", "status", "steps", "tool_calls",
    "prompt_tokens", "completion_tokens", "cost", "cum_cost",
    "history_tokens", "request_tokens", "reply_tokens", "pred_first_call",
    "latency_s", "db_messages", "http_status", "error",
]


class LabRun:
    """One lab run: a list of scenarios executed in a background thread."""

    def __init__(
        self,
        *,
        mode: str = "stub",
        cap: float = 45.0,
        scenarios: list[str] | None = None,
        limit: int = CONTEXT_LIMIT,
        overhead: int = OVERHEAD_TOKENS,
        price_in: float = PRICE_IN,
        price_out: float = PRICE_OUT,
        db_dir: str | Path | None = None,
        overrides: dict[str, dict] | None = None,
    ):
        if mode not in ("stub", "real"):
            raise ValueError("mode must be 'stub' or 'real'")
        names = list(scenarios) if scenarios else list(DEFAULT_SCENARIOS)
        unknown = [n for n in names if n not in SCENARIOS]
        if unknown:
            raise ValueError("unknown scenarios: " + ", ".join(unknown))
        if not names:
            raise ValueError("выберите хотя бы один сценарий")
        if cap <= 0:
            raise ValueError("бюджет должен быть больше нуля")

        self.mode = mode
        self.cap = float(cap)
        self.scenario_names = names
        self.limit = int(limit)
        self.overhead = int(overhead)
        self.price_in = float(price_in)
        self.price_out = float(price_out)
        self.overrides = overrides or {}
        self.db_dir = Path(db_dir) if db_dir else Path(tempfile.mkdtemp(prefix="lab-"))

        self.state = "pending"
        self.rows: list[dict] = []
        self.summaries: list[dict] = []
        self.spent = 0.0
        self.error: str | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.progress = {"scenario": None, "title": "", "turn": 0, "turns": 0}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "LabRun":
        self.state = "running"
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._thread = threading.Thread(target=self._run, name="lab-run", daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self.state == "running"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "mode": self.mode,
                "cap": self.cap,
                "spent": round(self.spent, 4),
                "progress": dict(self.progress),
                "rows": [dict(r) for r in self.rows],
                "summaries": [dict(s) for s in self.summaries],
                "scenarios": [
                    {
                        "name": n,
                        "title": SCENARIOS[n].title,
                        "note": SCENARIOS[n].note,
                        "turns": self._scenario(n).turns,
                        "optional": SCENARIOS[n].optional,
                    }
                    for n in self.scenario_names
                ],
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "rev": len(self.rows),
                "stop_requested": self._stop.is_set(),
                "failure": self._failure_summary(),
            }

    # -- internals ----------------------------------------------------------
    def _scenario(self, name: str) -> Scenario:
        base = SCENARIOS[name]
        override = self.overrides.get(name) or {}
        return replace(base, **override) if override else base

    def _set_progress(self, scn: Scenario, turn: int) -> None:
        with self._lock:
            self.progress = {"scenario": scn.name, "title": scn.title,
                             "turn": turn, "turns": scn.turns}

    def _add_row(self, row: dict) -> None:
        with self._lock:
            self.rows.append(row)

    def _failure_summary(self) -> dict | None:
        errors = [r for r in self.rows if r["status"] == "error"]
        if not errors:
            return None
        first, last = errors[0], errors[-1]
        return {
            "scenario": first["scenario"],
            "attempts": len(errors),
            "http_status": first["http_status"],
            "error": first["error"],
            "db_before": first["db_messages"],
            "db_after": last["db_messages"],
            "history_unchanged": first["db_messages"] == last["db_messages"],
        }

    @staticmethod
    def _client():
        # Imported lazily: app.web imports this module to register the lab page.
        from . import web as web_module

        return web_module.app.test_client()

    def _run(self) -> None:
        try:
            for name in self.scenario_names:
                if self._stop.is_set():
                    break
                self._run_scenario(self._scenario(name))
            self.state = "stopped" if self._stop.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the page
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished_at = datetime.now().isoformat(timespec="seconds")
            llm.use_chat(None)
            store.use_db(store.DB_PATH)

    def _run_scenario(self, scn: Scenario) -> None:
        db_path = self.db_dir / f"{scn.name}.db"
        store.use_db(db_path)
        if db_path.exists():
            db_path.unlink()
        store.init_db()

        # Every scenario gets a model tuned to it: reply size and whether the
        # turn calls a tool both change how fast the history grows. Both
        # overrides live in this thread's context only.
        if self.mode == "stub":
            llm.use_chat(StubModel(reply_tokens=scn.reply_tokens, emit_tools=scn.use_tools,
                                   limit=self.limit, overhead=self.overhead,
                                   price_in=self.price_in, price_out=self.price_out))
        seeded_messages = seed_history(scn.seed_tokens) if scn.seed_tokens else 0

        client = self._client()
        cum_cost = 0.0
        first_prompt = peak_prompt = 0
        total_prompt = total_completion = 0
        turns_done = 0
        status = "ok"
        error_text: str | None = None
        steps = 2 if scn.use_tools else 1

        for turn in range(1, scn.turns + 1):
            if self._stop.is_set():
                status = "stopped"
                break
            self._set_progress(scn, turn)

            history = store.load_messages()
            history_tokens = count_messages(history)
            user_text = make_text(scn.user_tokens)
            request_tokens = count_tokens(user_text)
            pred_first_call = history_tokens + request_tokens + self.overhead

            projected = (pred_first_call * steps * self.price_in
                         + scn.reply_tokens * steps * self.price_out)
            if self.spent + cum_cost + projected > self.cap:
                self._add_row({
                    "scenario": scn.name, "turn": turn, "status": "aborted",
                    "steps": 0, "tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "cost": 0.0, "cum_cost": round(cum_cost, 4),
                    "history_tokens": history_tokens, "request_tokens": request_tokens,
                    "reply_tokens": 0, "pred_first_call": pred_first_call,
                    "latency_s": 0.0, "db_messages": len(history), "http_status": 0,
                    "error": f"бюджет {self.cap} ₽ был бы превышен",
                })
                status = "aborted"
                break

            started = time.perf_counter()
            try:
                response = client.post("/api/chat", json={"message": user_text})
                http_status = response.status_code
                payload = response.get_json() or {}
            except Exception as exc:  # noqa: BLE001
                http_status = 0
                payload = {"error": f"{type(exc).__name__}: {exc}"}
            latency = time.perf_counter() - started
            db_messages = len(store.load_messages())

            if payload.get("error"):
                error_text = error_text or payload["error"]
                self._add_row({
                    "scenario": scn.name, "turn": turn, "status": "error",
                    "steps": 0, "tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "cost": 0.0, "cum_cost": round(cum_cost, 4),
                    "history_tokens": history_tokens, "request_tokens": request_tokens,
                    "reply_tokens": 0, "pred_first_call": pred_first_call,
                    "latency_s": round(latency, 2), "db_messages": db_messages,
                    "http_status": http_status, "error": payload["error"],
                })
                continue

            tokens = payload.get("tokens") or {}
            cost = float(tokens.get("cost") or 0.0)
            cum_cost += cost
            self.spent += cost
            total_prompt += int(tokens.get("prompt") or 0)
            total_completion += int(tokens.get("completion") or 0)
            if first_prompt == 0:
                first_prompt = int(tokens.get("prompt") or 0)
            peak_prompt = max(peak_prompt, int(tokens.get("prompt") or 0))
            turns_done += 1

            self._add_row({
                "scenario": scn.name, "turn": turn, "status": "ok",
                "steps": int(payload.get("steps") or 0),
                "tool_calls": int(payload.get("tool_calls_used") or 0),
                "prompt_tokens": int(tokens.get("prompt") or 0),
                "completion_tokens": int(tokens.get("completion") or 0),
                "cost": round(cost, 4), "cum_cost": round(cum_cost, 4),
                "history_tokens": history_tokens, "request_tokens": request_tokens,
                "reply_tokens": int(tokens.get("response") or 0),
                "pred_first_call": pred_first_call,
                "latency_s": round(latency, 2), "db_messages": db_messages,
                "http_status": http_status, "error": "",
            })

        if status == "ok" and turns_done == 0:
            status = "failed"

        with self._lock:
            self.summaries.append({
                "scenario": scn.name,
                "title": scn.title,
                "note": scn.note,
                "turns_planned": scn.turns,
                "turns_done": turns_done,
                "seeded_messages": seeded_messages,
                "seeded_tokens": scn.seed_tokens,
                "use_tools": scn.use_tools,
                "status": status,
                "first_prompt": first_prompt,
                "peak_prompt": peak_prompt,
                "total_prompt": total_prompt,
                "total_completion": total_completion,
                "total_cost": round(cum_cost, 4),
                "error": error_text,
            })


# --- chart -----------------------------------------------------------------
CHART_W, CHART_H = 940, 300
PAD_L, PAD_R, PAD_T, PAD_B = 78, 24, 34, 46
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def _fmt_int(value) -> str:
    return f"{int(value):,}".replace(",", " ")


def _fmt_money(value) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _sx(v, vmin, vmax, px0, px1):
    if vmax == vmin:
        return px0
    return px0 + (v - vmin) * (px1 - px0) / (vmax - vmin)


def _sy(v, vmin, vmax, py0, py1):
    if vmax == vmin:
        return py1
    return py1 - (v - vmin) * (py1 - py0) / (vmax - vmin)


def _panel(title: str, ylabel: str, series: list[dict], x_max: int, y_offset: int,
           fmt_y=_fmt_int) -> str:
    x0, x1 = PAD_L, CHART_W - PAD_R
    y0, y1 = y_offset + PAD_T, y_offset + CHART_H - PAD_B
    y_max = max((max((p[1] for p in s["points"]), default=0) for s in series), default=0)
    y_max = y_max * 1.15 or 1

    parts = [f'<text x="{PAD_L}" y="{y_offset + 20}" class="ptitle">{title}</text>']
    for i in range(6):
        value = y_max * i / 5
        py = _sy(value, 0, y_max, y0, y1)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x1}" y2="{py:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 8}" y="{py + 4:.1f}" class="ytick">{fmt_y(value)}</text>')
    for i in range(7):
        turn = 1 + (x_max - 1) * i / 6
        px = _sx(turn, 1, max(x_max, 2), x0, x1)
        parts.append(f'<text x="{px:.1f}" y="{y1 + 20}" class="xtick">{turn:.0f}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 40}" class="axtitle">ход диалога</text>')
    parts.append(f'<text x="{x0 - 58}" y="{y0 - 10}" class="axtitle">{ylabel}</text>')

    for s in series:
        points = " ".join(
            f"{_sx(x, 1, max(x_max, 2), x0, x1):.1f},{_sy(y, 0, y_max, y0, y1):.1f}"
            for x, y in s["points"])
        dash = ' stroke-dasharray="7 5"' if s.get("dashed") else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{s["color"]}" '
                     f'stroke-width="2.2"{dash}/>')
        for x, y in s["points"]:
            parts.append(f'<circle cx="{_sx(x, 1, max(x_max, 2), x0, x1):.1f}" '
                         f'cy="{_sy(y, 0, y_max, y0, y1):.1f}" r="3" fill="{s["color"]}"/>')

    for idx, s in enumerate(series):
        lx = x0 + 14 + idx * 250
        ly = y0 + 16
        dash = ' stroke-dasharray="7 5"' if s.get("dashed") else ""
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" '
                     f'stroke="{s["color"]}" stroke-width="2.2"{dash}/>')
        parts.append(f'<text x="{lx + 33}" y="{ly + 4}" class="legend">{s["label"]}</text>')
    return "\n".join(parts)


def render_chart(rows: list[dict], *, title: str = "") -> str:
    """Render measured token/cost growth as a standalone SVG document."""
    ok: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") == "ok":
            ok.setdefault(row["scenario"], []).append(row)
    ok = {name: sorted(rs, key=lambda r: r["turn"]) for name, rs in ok.items()}

    height = CHART_H * 2 + 60
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{height}" '
        f'viewBox="0 0 {CHART_W} {height}" font-family="Segoe UI, system-ui, sans-serif">',
        "<style>"
        ".ptitle{font-size:15px;font-weight:600;fill:#0f172a}"
        ".grid{stroke:#e2e8f0;stroke-width:1}"
        ".ytick{font-size:11px;fill:#64748b;text-anchor:end}"
        ".xtick{font-size:11px;fill:#64748b;text-anchor:middle}"
        ".axtitle{font-size:11px;fill:#475569;text-anchor:middle}"
        ".legend{font-size:11px;fill:#334155}"
        ".empty{font-size:14px;fill:#94a3b8;text-anchor:middle}"
        "</style>",
        f'<rect width="{CHART_W}" height="{height}" fill="#ffffff"/>',
    ]
    if not ok:
        head.append(f'<text x="{CHART_W / 2}" y="{height / 2}" class="empty">'
                    "нет данных: запустите прогон</text>")
        head.append("</svg>")
        return "\n".join(head)

    order = [n for n in ("baseline", "short", "long", "long_tools") if n in ok]
    order += [n for n in ok if n not in order]

    series_prompt, series_cost = [], []
    for idx, name in enumerate(order):
        rs = ok[name]
        color = PALETTE[idx % len(PALETTE)]
        series_prompt.append({"label": name, "color": color,
                              "points": [(r["turn"], r["prompt_tokens"]) for r in rs]})
        series_cost.append({"label": name, "color": color,
                            "points": [(r["turn"], r["cum_cost"]) for r in rs]})
        last = rs[-1]
        n = last["turn"]
        if n >= 2 and last["cum_cost"] > 0:
            a = last["cum_cost"] / (n * n)
            series_cost.append({"label": f"{name}: a·n²", "color": color, "dashed": True,
                                "points": [(i, a * i * i) for i in range(1, n + 1)]})

    x_max = max(r["turn"] for rs in ok.values() for r in rs)
    head.append(_panel("Токены промпта на ход — линейный рост", "токены",
                       series_prompt, x_max, 0))
    head.append(_panel("Накопленная стоимость — квадратичный рост", "₽",
                       series_cost, x_max, CHART_H + 40, fmt_y=_fmt_money))
    head.append("</svg>")
    return "\n".join(head)
