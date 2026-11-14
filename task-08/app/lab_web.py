"""Web control panel for the token lab (mounted at /lab).

Lets the user pick scenarios, switch between the free stub and the real
endpoint, set a budget and watch token/cost growth -- and the failure at the
context limit -- as it happens. A run executes in a background thread, so the
page polls for progress instead of blocking on a multi-minute request.
"""

from __future__ import annotations

import threading

from flask import Blueprint, Response, jsonify, render_template_string, request

from .lab import (
    CONTEXT_LIMIT,
    DEFAULT_SCENARIOS,
    OVERHEAD_TOKENS,
    PRICE_IN,
    PRICE_OUT,
    SCENARIOS,
    LabRun,
    render_chart,
)

lab_bp = Blueprint("lab", __name__)

MAX_CAP = 200.0
_state_lock = threading.Lock()
_current: LabRun | None = None


def _idle_snapshot() -> dict:
    return {
        "state": "idle", "mode": "stub", "cap": 0.0, "spent": 0.0,
        "progress": {"scenario": None, "title": "", "turn": 0, "turns": 0},
        "rows": [], "summaries": [], "scenarios": [], "error": None,
        "started_at": None, "finished_at": None, "rev": 0,
        "stop_requested": False, "failure": None,
    }


@lab_bp.get("/", strict_slashes=False)
def page():
    cards = "".join(
        '<label class="chk" title="{note}">'
        '<input type="checkbox" value="{name}"{checked}> {title}</label>'.format(
            name=name, title=scn.title, note=scn.note,
            checked=" checked" if name in DEFAULT_SCENARIOS else "",
        )
        for name, scn in SCENARIOS.items()
    )
    return render_template_string(
        PAGE,
        scenario_cards=cards,
        price_in=f"{PRICE_IN * 1_000_000:.0f}",
        price_out=f"{PRICE_OUT * 1_000_000:.0f}",
        overhead=OVERHEAD_TOKENS,
        limit=CONTEXT_LIMIT,
        long_turns=SCENARIOS["long"].turns,
    )


@lab_bp.post("/api/run")
def api_run():
    global _current
    data = request.get_json(silent=True) or {}
    mode = data.get("mode") or "stub"
    scenarios = data.get("scenarios") or list(DEFAULT_SCENARIOS)

    try:
        cap = min(float(data.get("cap") or 45.0), MAX_CAP)
    except (TypeError, ValueError):
        return jsonify({"error": "Бюджет должен быть числом."}), 400

    overrides: dict[str, dict] = {}
    try:
        long_turns = int(data.get("long_turns") or 0)
    except (TypeError, ValueError):
        long_turns = 0
    if long_turns >= 2:
        overrides["long"] = {"turns": min(long_turns, 500)}

    try:
        limit = int(data.get("limit") or CONTEXT_LIMIT)
    except (TypeError, ValueError):
        limit = CONTEXT_LIMIT

    with _state_lock:
        if _current is not None and _current.running:
            return jsonify({"error": "Прогон уже идёт — остановите его или дождитесь конца."}), 409
        try:
            run = LabRun(mode=mode, cap=cap, scenarios=scenarios,
                         limit=max(limit, 1000), overrides=overrides)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        run.start()
        _current = run
        return jsonify(run.snapshot())


@lab_bp.get("/api/status")
def api_status():
    run = _current
    return jsonify(run.snapshot() if run is not None else _idle_snapshot())


@lab_bp.post("/api/stop")
def api_stop():
    run = _current
    if run is None or not run.running:
        return jsonify({"error": "Сейчас ничего не выполняется."}), 409
    run.stop()
    return jsonify({"ok": True})


@lab_bp.get("/api/chart.svg")
def api_chart():
    run = _current
    rows = run.snapshot()["rows"] if run is not None else []
    return Response(render_chart(rows), mimetype="image/svg+xml",
                    headers={"Cache-Control": "no-store"})


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лаборатория токенов</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: radial-gradient(1200px 600px at 20% -10%, #1e293b, #0b1120 60%);
         color:#e2e8f0; min-height:100vh; }
  .wrap { max-width:1000px; margin:0 auto; padding:0 20px 60px; }
  header { display:flex; align-items:center; gap:12px; padding:18px 0;
           border-bottom:1px solid rgba(255,255,255,.08); margin-bottom:20px; }
  header h1 { font-size:16px; margin:0; }
  header .logo { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
                 background:linear-gradient(135deg,#2563eb,#7c3aed); }
  header a.link { margin-left:auto; font-size:12px; color:#cbd5e1; text-decoration:none;
                  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
                  padding:6px 12px; border-radius:999px; }
  header a.link:hover { background:rgba(255,255,255,.12); }
  .badge { font-size:12px; color:#94a3b8; background:rgba(255,255,255,.06);
           border:1px solid rgba(255,255,255,.1); padding:6px 12px; border-radius:999px; }
  .card { background:rgba(15,23,42,.72); border:1px solid rgba(255,255,255,.08);
          border-radius:16px; padding:18px; margin-bottom:16px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:#94a3b8;
             margin:0 0 12px; font-weight:600; }
  .row { display:flex; gap:18px; flex-wrap:wrap; }
  .field { display:flex; flex-direction:column; gap:6px; margin-bottom:12px; }
  .field > label { font-size:12px; color:#94a3b8; }
  input[type=number] { width:150px; padding:9px 11px; border-radius:10px; font-size:14px;
        border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.05); color:#e2e8f0; }
  .seg { display:flex; gap:14px; }
  .seg label, .chk { font-size:13.5px; display:flex; align-items:center; gap:7px; cursor:pointer; }
  .scenarios { display:flex; flex-wrap:wrap; gap:10px 18px; }
  details summary { cursor:pointer; font-size:12.5px; color:#94a3b8; margin-bottom:10px; }
  .hint { font-size:12px; color:#64748b; margin:4px 0 0; }
  .actions { display:flex; align-items:center; gap:12px; margin-top:6px; }
  button { padding:10px 22px; border-radius:11px; border:none; cursor:pointer; font-size:14px;
           font-weight:600; color:#fff; background:linear-gradient(135deg,#2563eb,#4f46e5); }
  button.ghost { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12); color:#cbd5e1; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .state { font-size:12.5px; color:#94a3b8; }
  .bar { height:5px; border-radius:4px; background:rgba(255,255,255,.08); margin-top:14px; overflow:hidden; }
  .bar > div { height:100%; width:0; background:linear-gradient(90deg,#2563eb,#7c3aed); transition:width .3s; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th { text-align:left; color:#64748b; font-weight:500; padding:6px 8px; white-space:nowrap;
       border-bottom:1px solid rgba(255,255,255,.08); }
  td { padding:6px 8px; border-bottom:1px solid rgba(255,255,255,.04); white-space:nowrap; }
  td.err { color:#fca5a5; white-space:normal; }
  .scroll { max-height:340px; overflow:auto; }
  .scroll::-webkit-scrollbar { width:8px; }
  .scroll::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12); border-radius:8px; }
  #chart { width:100%; border-radius:12px; background:#fff; display:block; }
  .warn { font-size:12.5px; color:#fca5a5; background:rgba(239,68,68,.1);
          border-left:3px solid #ef4444; padding:10px 12px; border-radius:9px; margin-bottom:10px;
          white-space:pre-wrap; }
  .ok { color:#86efac; }
  .pill { font-size:11px; padding:2px 8px; border-radius:999px; background:rgba(255,255,255,.08); }
  .pill.ok { background:rgba(34,197,94,.15); color:#86efac; }
  .pill.err { background:rgba(239,68,68,.15); color:#fca5a5; }
  .pill.run { background:rgba(37,99,235,.2); color:#93c5fd; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <span class="logo"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"><path d="M3 20h18M6 20V10M11 20V4M16 20v-7M21 20V7"/></svg></span>
  <h1>Лаборатория токенов</h1>
  <span class="badge" id="spentBadge">потрачено 0 ₽</span>
  <a class="link" href="/">← к чату</a>
</header>

<section class="card">
  <h2>Параметры прогона</h2>
  <div class="row">
    <div class="field">
      <label>Режим</label>
      <div class="seg">
        <label><input type="radio" name="mode" value="stub" checked> Заглушка · 0 ₽</label>
        <label><input type="radio" name="mode" value="real"> Реальный API · тратит деньги</label>
      </div>
    </div>
    <div class="field">
      <label>Бюджет, ₽ (жёсткий предел)</label>
      <input id="cap" type="number" value="45" min="1" max="200" step="5">
    </div>
  </div>
  <div class="field">
    <label>Сценарии</label>
    <div class="scenarios" id="scenarios">{{ scenario_cards|safe }}</div>
  </div>
  <details>
    <summary>Дополнительно</summary>
    <div class="row">
      <div class="field">
        <label>Ходов в длинном диалоге</label>
        <input id="longTurns" type="number" value="{{ long_turns }}" min="2" max="500">
      </div>
      <div class="field">
        <label>Лимит контекста (для заглушки)</label>
        <input id="limit" type="number" value="{{ limit }}" min="1000" step="1000">
      </div>
    </div>
    <p class="hint">Цены AITUNNEL: вход {{ price_in }} ₽/1M, выход {{ price_out }} ₽/1M;
       накладные расходы ~{{ overhead }} токенов на каждый вызов модели.
       Заглушка считает токены тем же счётчиком, что и приложение, и обрывает запрос
       по лимиту — то есть весь путь, включая поломку, воспроизводится бесплатно.</p>
  </details>
  <div class="actions">
    <button id="startBtn">Запустить</button>
    <button id="stopBtn" class="ghost" disabled>Остановить</button>
    <span class="state" id="stateText"></span>
  </div>
  <div class="bar"><div id="bar"></div></div>
</section>

<section class="card" id="failureCard" hidden>
  <h2>Что ломается при переполнении</h2>
  <div class="warn" id="failureText"></div>
  <p class="hint" id="failureNote"></p>
</section>

<section class="card">
  <h2>Сводка по сценариям</h2>
  <table id="summaryTable"><tbody><tr><td class="hint">нет данных</td></tr></tbody></table>
</section>

<section class="card">
  <h2>Ходы</h2>
  <div class="scroll">
    <table id="rowsTable"><tbody><tr><td class="hint">нет данных</td></tr></tbody></table>
  </div>
</section>

<section class="card">
  <h2>Рост токенов и стоимости</h2>
  <img id="chart" alt="график роста" src="/lab/api/chart.svg">
  <p class="hint">Сверху — токены промпта на каждый ход (линейный рост). Снизу — накопленная
     стоимость (квадратичный рост). Пунктир — модель <code>a·n²</code>, построенная по последней
     точке: совпадение означает, что рост объясняется именно повторной отправкой всей истории.</p>
</section>
</div>

<script>
const $ = id => document.getElementById(id);
let timer = null, lastRev = -1, polling = false;

function esc(v){ return String(v ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function num(v){ return (v === 0 || v) ? Number(v).toLocaleString('ru-RU') : '—'; }
function money(v){ return (Number(v) || 0).toFixed(2) + ' ₽'; }

function config(){
  return {
    mode: document.querySelector('input[name=mode]:checked').value,
    cap: parseFloat($('cap').value || '45'),
    scenarios: [...document.querySelectorAll('#scenarios input:checked')].map(i => i.value),
    long_turns: parseInt($('longTurns').value || '0', 10),
    limit: parseInt($('limit').value || '1000000', 10),
  };
}

async function start(){
  const cfg = config();
  if (!cfg.scenarios.length) { alert('Выберите хотя бы один сценарий.'); return; }
  if (cfg.mode === 'real' && !confirm(
      'Реальный режим тратит деньги с баланса AITUNNEL.\nБюджет: ' + cfg.cap +
      ' ₽ — прогон остановится сам, если упрётся в него.\n\nЗапускаем?')) return;
  const r = await fetch('/lab/api/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(cfg)
  });
  const data = await r.json();
  if (data.error) { alert(data.error); return; }
  lastRev = -1;
  render(data);
  poll();
}

async function stop(){ await fetch('/lab/api/stop', {method: 'POST'}); poll(); }

async function poll(){
  if (polling) return;
  polling = true;
  clearTimeout(timer);
  try {
    const s = await (await fetch('/lab/api/status')).json();
    render(s);
    if (s.state === 'running') timer = setTimeout(poll, 700);
  } catch (e) {
    timer = setTimeout(poll, 1500);
  } finally {
    polling = false;
  }
}

function render(s){
  const running = s.state === 'running';
  $('stopBtn').disabled = !running;
  $('startBtn').disabled = running;
  $('spentBadge').textContent = 'потрачено ' + money(s.spent) +
    (s.cap ? ' из ' + money(s.cap) : '');

  const p = s.progress || {};
  if (running) {
    $('stateText').textContent = 'выполняется: ' + (p.title || p.scenario || '') +
      (p.turns ? ', ход ' + p.turn + ' из ' + p.turns : '') +
      (s.stop_requested ? ' · останавливаемся…' : '');
    $('bar').style.width = p.turns ? (100 * p.turn / p.turns) + '%' : '5%';
  } else {
    const names = {idle:'ожидание запуска', done:'готово', stopped:'остановлено',
                   error:'ошибка', pending:'подготовка'};
    $('stateText').textContent = names[s.state] || s.state;
    $('bar').style.width = s.state === 'done' ? '100%' : '0%';
  }

  const sums = s.summaries || [];
  if (sums.length) {
    let html = '<thead><tr><th>сценарий</th><th>ходов</th><th>промпт 1-го</th>' +
      '<th>пик промпта</th><th>вход всего</th><th>выход всего</th><th>стоимость</th>' +
      '<th>статус</th></tr></thead><tbody>';
    for (const t of sums) {
      const cls = t.status === 'ok' ? 'ok' : 'err';
      html += '<tr><td>' + esc(t.title || t.scenario) + '</td>' +
        '<td>' + t.turns_done + ' / ' + t.turns_planned + '</td>' +
        '<td>' + num(t.first_prompt) + '</td>' +
        '<td>' + num(t.peak_prompt) + '</td>' +
        '<td>' + num(t.total_prompt) + '</td>' +
        '<td>' + num(t.total_completion) + '</td>' +
        '<td>' + money(t.total_cost) + '</td>' +
        '<td><span class="pill ' + cls + '">' + esc(t.status) + '</span></td></tr>';
    }
    $('summaryTable').innerHTML = html + '</tbody>';
  }

  const rows = s.rows || [];
  if (rows.length) {
    let html = '<thead><tr><th>сценарий</th><th>ход</th><th>история</th><th>запрос</th>' +
      '<th>промпт</th><th>ответ</th><th>цена</th><th>накоплено</th><th>сек</th>' +
      '<th>статус</th></tr></thead><tbody>';
    for (const r of rows.slice(-160)) {
      const bad = r.status !== 'ok';
      html += '<tr><td>' + esc(r.scenario) + '</td><td>' + r.turn + '</td>' +
        '<td>' + num(r.history_tokens) + '</td><td>' + num(r.request_tokens) + '</td>' +
        '<td>' + num(r.prompt_tokens) + '</td><td>' + num(r.reply_tokens) + '</td>' +
        '<td>' + money(r.cost) + '</td><td>' + money(r.cum_cost) + '</td>' +
        '<td>' + (r.latency_s || 0).toFixed(2) + '</td>' +
        (bad ? '<td class="err">' + esc(r.status) + ': ' + esc(r.error) + '</td>'
             : '<td><span class="pill ok">ok</span></td>') + '</tr>';
    }
    $('rowsTable').innerHTML = html + '</tbody>';
  }

  const f = s.failure;
  if (f) {
    $('failureCard').hidden = false;
    $('failureText').textContent = f.error;
    $('failureNote').textContent =
      'Попыток: ' + f.attempts + ' · HTTP-статус ответа: ' + f.http_status +
      ' (ошибка отдаётся как успех) · сообщений в БД до: ' + f.db_before +
      ', после: ' + f.db_after +
      (f.history_unchanged
        ? '. История не изменилась: упавший ход не сохранился, а следующее сообщение снова ' +
          'отправит ту же переросшую историю и снова упадёт — диалог заклинило, лечится только «Сбросить».'
        : '.');
  } else {
    $('failureCard').hidden = true;
  }

  if (s.rev !== lastRev) {
    lastRev = s.rev;
    $('chart').src = '/lab/api/chart.svg?rev=' + s.rev;
  }
}

$('startBtn').addEventListener('click', start);
$('stopBtn').addEventListener('click', stop);
document.querySelectorAll('input[name=mode]').forEach(el =>
  el.addEventListener('change', () => {
    $('startBtn').textContent = config().mode === 'real' ? 'Запустить (реальный API)' : 'Запустить';
  }));
poll();
</script>
</body>
</html>"""
