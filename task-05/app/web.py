"""Flask web interface for chatting with the LLM (streaming)."""
import json

from flask import Flask, Response, jsonify, render_template_string, request

from .config import MODELS, MODEL_PRICES, PRICE_DEFAULT
from .llm import chat, stream_chat
from .metrics import last
from .tasks import TASKS, by_id

app = Flask(__name__)

HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>LLM Chat</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: radial-gradient(1200px 600px at 20% -10%, #1e293b, #0b1120 60%); color:#e2e8f0; height:100vh;
         display:flex; }
  .app { width:100%; height:100%; display:flex; }
  .sidebar { width:290px; flex:0 0 290px; padding:18px 18px 22px; display:flex; flex-direction:column; gap:14px;
             background:rgba(15,23,42,.55); border-right:1px solid rgba(255,255,255,.08); backdrop-filter:blur(6px);
             overflow-y:auto; }
  .sidebar h2 { font-size:11px; text-transform:uppercase; letter-spacing:.8px; color:#94a3b8; margin:0 0 2px; padding-bottom:10px;
                border-bottom:1px solid rgba(255,255,255,.08); }
  .main { flex:1; min-width:0; display:flex; flex-direction:column; }
  header { padding:16px 20px; display:flex; align-items:center; gap:12px;
           background:rgba(15,23,42,.5); border-bottom:1px solid rgba(255,255,255,.08); backdrop-filter:blur(6px); }
  header .logo { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
                 background:linear-gradient(135deg,#2563eb,#7c3aed); box-shadow:0 4px 14px rgba(37,99,235,.4); }
  header .logo svg { width:17px; height:17px; }
  header h1 { font-size:16px; margin:0; font-weight:700; letter-spacing:.2px; }
  header .model { margin-left:auto; font-size:12px; color:#94a3b8; background:rgba(255,255,255,.06);
                  border:1px solid rgba(255,255,255,.1); padding:5px 12px; border-radius:999px;
                  display:flex; align-items:center; gap:7px; }
  header .model::before { content:''; width:7px; height:7px; border-radius:50%; background:#34d399; box-shadow:0 0 8px #34d399; }
  #chat { flex:1; overflow-y:auto; padding:24px 20px; display:flex; flex-direction:column; gap:16px; scroll-behavior:smooth; }
  #chat::-webkit-scrollbar { width:8px; }
  #chat::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12); border-radius:8px; }
  #chat::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,.22); }
  .row { display:flex; animation:pop .25s ease both; }
  .row.user { justify-content:flex-end; }
  .row.assistant { justify-content:flex-start; }
  @keyframes pop { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
  .bubble { max-width:78%; padding:11px 15px; border-radius:16px; line-height:1.55;
            white-space:pre-wrap; word-wrap:break-word; font-size:14.5px; box-shadow:0 2px 10px rgba(0,0,0,.25); }
  .user .bubble { background:linear-gradient(135deg,#2563eb,#4f46e5); color:#fff; border-bottom-right-radius:5px; }
  .assistant .bubble { background:#0f172a; border:1px solid rgba(255,255,255,.07); border-bottom-left-radius:5px; }
  .meta { font-size:11px; color:#64748b; margin-top:8px; }
  .note { margin-top:10px; padding:9px 12px; border-radius:9px; font-size:12.5px; white-space:pre-wrap;
          border:1px solid rgba(255,255,255,.08); }
  .note.warn { background:rgba(251,191,36,.1); color:#fcd34d; border-left:3px solid #fbbf24; }
  .note.error { background:rgba(239,68,68,.1); color:#fca5a5; border-left:3px solid #ef4444; }
  .note.ok { background:rgba(52,211,153,.1); color:#6ee7b7; border-left:3px solid #34d399; }
  .badge { display:inline-block; margin-left:6px; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:700; }
  .badge.ok { color:#6ee7b7; background:rgba(52,211,153,.15); border:1px solid rgba(52,211,153,.4); }
  .badge.bad { color:#fca5a5; background:rgba(239,68,68,.15); border:1px solid rgba(239,68,68,.4); }
  .verdict-title { margin:14px 0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:#94a3b8; font-weight:700; }
  .note.verdict { margin-top:6px; background:rgba(255,255,255,.04); }
  .cap { font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:#93c5fd; font-weight:700;
         margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,.08); }
  .bubble.cmp { max-width:100%; }
  .expwrap { overflow-x:auto; }
  .cmp-table { width:100%; border-collapse:collapse; font-size:13px; margin-top:2px; border-spacing:0; }
  .cmp-table th, .cmp-table td { padding:8px 12px; text-align:left; border-bottom:1px solid rgba(255,255,255,.08); vertical-align:middle; }
  .cmp-table th { color:#94a3b8; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.4px; }
  .cmp-table th.grp { text-align:center; font-size:12px; text-transform:none; color:#cbd5e1; background:rgba(255,255,255,.03); }
  .cmp-table th.sub { font-size:10.5px; text-align:center; }
  .cmp-table th.sub.num, .cmp-table td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .cmp-table td.num { color:#6ee7b7; }
  .cmp-table tr:nth-child(even) td { background:rgba(255,255,255,.025); }
  .cmp-table tr:last-child td { border-bottom:none; }
  .cmp-table tr.totals td { border-top:2px solid rgba(255,255,255,.15); font-weight:700; }
  .cmp-table .badge { margin-left:5px; }
  .verd-sum { margin-top:12px; font-size:12.5px; color:#cbd5e1; }
  .verd-sum .line { padding:2px 0; }
  details.reasoning { margin-bottom:8px; font-size:12.5px; }
  details.reasoning summary { cursor:pointer; color:#fbbf24; font-weight:600; user-select:none; list-style:none; }
  details.reasoning summary::before { content:'\25B8 '; }
  details[open].reasoning summary::before { content:'\25BE '; }
  .reason-body { padding:9px 12px; background:rgba(251,191,36,.05); border-left:3px solid #fbbf24;
                 border-radius:6px; color:#fcd34d; margin-top:8px; white-space:pre-wrap; }
  .typing { display:inline-flex; gap:4px; align-items:center; color:#94a3b8; font-size:13px; }
  .typing .t-dot { width:7px; height:7px; border-radius:50%; background:#94a3b8; animation:blink 1.2s infinite; }
  .typing .t-dot:nth-child(2){ animation-delay:.2s; }
  .typing .t-dot:nth-child(3){ animation-delay:.4s; }
  @keyframes blink { 0%,80%,100%{ opacity:.2; transform:scale(1); } 40%{ opacity:1; transform:scale(1.15); } }
  .field { display:flex; flex-direction:column; gap:6px; }
  .field label { font-size:12px; color:#94a3b8; }
  .field input, .field select { width:100%; padding:9px 12px; border-radius:10px; font-size:13px;
               background:rgba(255,255,255,.04); color:#e2e8f0; border:1px solid rgba(255,255,255,.12); outline:none; transition:.15s; }
  .field input:focus, .field select:focus { border-color:#2563eb; background:rgba(255,255,255,.07); }
  .field select option { background:#0f172a; color:#e2e8f0; }
  .hint { font-size:11px; color:#94a3b8; padding:6px 9px; border-radius:8px; line-height:1.4;
          border:1px solid transparent; transition:.15s; }
  .hint.show { background:rgba(37,99,235,.12); color:#93c5fd; border-color:rgba(37,99,235,.35); }
  form { display:flex; flex-direction:column; gap:10px; padding:14px 20px 18px; background:rgba(15,23,42,.5);
         border-top:1px solid rgba(255,255,255,.08); backdrop-filter:blur(6px); }
  .toolrow { display:flex; gap:8px; }
  .toolbtn { padding:9px 14px; border-radius:11px; cursor:pointer; font-size:13px; font-weight:600;
             color:#cbd5e1; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12); transition:.15s; }
  .toolbtn:hover { background:rgba(255,255,255,.12); color:#e2e8f0; border-color:rgba(255,255,255,.25); }
  .compare-panel { display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; padding:12px 14px;
                   background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); border-radius:12px; }
  .compare-panel .field { flex:1; min-width:150px; }
  .cmp-run { padding:10px 18px; border-radius:11px; border:none; cursor:pointer; font-size:13.5px; font-weight:600;
             color:#fff; background:linear-gradient(135deg,#2563eb,#4f46e5); box-shadow:0 4px 14px rgba(37,99,235,.35); transition:.15s; }
  .cmp-run:hover { filter:brightness(1.08); }
  .cmp-run:disabled { opacity:.5; cursor:not-allowed; }
  .cmp-close { padding:10px 14px; border-radius:11px; border:none; cursor:pointer; font-size:13.5px; font-weight:600;
               color:#cbd5e1; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); transition:.15s; }
  .cmp-close:hover { background:rgba(255,255,255,.12); }
  .sendrow { display:flex; gap:10px; }
  #msg { flex:1; padding:13px 15px; border-radius:13px; border:1px solid rgba(255,255,255,.12);
         background:rgba(255,255,255,.05); color:#e2e8f0; font-size:14.5px; outline:none; transition:.15s; }
  #msg:focus { border-color:#2563eb; background:rgba(255,255,255,.07); }
  #send { padding:12px 24px; border-radius:13px; border:none; cursor:pointer; font-size:14.5px; font-weight:600;
          color:#fff; background:linear-gradient(135deg,#2563eb,#4f46e5); box-shadow:0 4px 14px rgba(37,99,235,.35);
          transition:.15s; }
  #send:hover { filter:brightness(1.08); }
  #send:active { transform:translateY(1px); }
  #send:disabled { opacity:.5; cursor:not-allowed; filter:none; transform:none; box-shadow:none; }
  #stop { padding:12px 20px; border-radius:13px; border:none; cursor:pointer; font-size:14.5px; font-weight:600;
          color:#fff; background:linear-gradient(135deg,#dc2626,#b91c1c); box-shadow:0 4px 14px rgba(220,38,38,.35); transition:.15s; }
  #stop:hover { filter:brightness(1.08); }
  #stop:active { transform:translateY(1px); }
</style>
</head>
<body>
<div class="app">
<aside class="sidebar">
  <h2>Параметры</h2>
  <div class="field">
    <label for="model">Модель</label>
    <select id="model">
      {% for m in models %}<option value="{{ m }}" {% if m == model %}selected{% endif %}>{{ m }}</option>{% endfor %}
    </select>
  </div>
  <div class="field">
    <label for="task">Задача</label>
    <select id="task">
      <option value="">Ручной ввод</option>
      {% for t in tasks_view %}<option value="{{ t.id }}">{{ t.title }}</option>{% endfor %}
    </select>
  </div>
  <div class="field">
    <label for="max_tokens">max_tokens</label>
    <input id="max_tokens" type="number" min="1" max="8192" placeholder="1024">
  </div>
  <div class="field">
    <label for="max_completion_tokens">max_completion_tokens</label>
    <input id="max_completion_tokens" type="number" min="1" max="8192" placeholder="не задано">
  </div>
  <div class="field">
    <label for="budget_tokens">budget_tokens</label>
    <input id="budget_tokens" type="number" min="1" placeholder="лимит рассуждений">
  </div>
  <div class="field">
    <label for="stop">stop_sequence</label>
    <input id="stop" placeholder="стоп-фразы">
  </div>
  <div class="field">
    <label for="response_format">формат ответа</label>
    <select id="response_format">
      <option value="">не задано</option>
      <option value="Обычный текст">Обычный текст</option>
      <option value="Markdown с заголовками и списками">Markdown</option>
      <option value="JSON объект без пояснений">JSON объект</option>
      <option value="JSON с полями answer и reason">JSON: answer/reason</option>
      <option value="Маркированный список">Маркированный список</option>
      <option value="CSV: первая строка — заголовки полей">CSV</option>
    </select>
  </div>
  <div class="field">
    <label for="reasoning_effort">reasoning_effort</label>
    <select id="reasoning_effort">
      <option value="">не задано</option>
      <option value="low">low</option>
      <option value="medium">medium</option>
      <option value="high">high</option>
    </select>
  </div>
</aside>
<div class="main">
<header>
  <span class="logo">
    <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>
    </svg>
  </span>
  <h1>Чат с LLM</h1>
  <span class="model" id="model-badge">{{ model }}</span>
</header>
<div id="chat">
  <div class="row assistant"><div class="bubble">Привет! Задай вопрос и при желании настрой параметры слева.</div></div>
</div>
<form id="chat-form">
  <div class="toolrow">
    <button type="button" class="toolbtn" id="btn-compare-toggle">Сравнение</button>
  </div>
  <div id="compare-panel" class="compare-panel" style="display:none">
    <div class="field">
      <label for="cmp-model1">Модель 1</label>
      <select id="cmp-model1">
        {% for m in models %}<option value="{{ m }}" {% if m == models[0] %}selected{% endif %}>{{ m }}</option>{% endfor %}
      </select>
    </div>
    <div class="field">
      <label for="cmp-model2">Модель 2</label>
      <select id="cmp-model2">
        {% for m in models %}<option value="{{ m }}" {% if models|length > 1 and m == models[1] %}selected{% endif %}>{{ m }}</option>{% endfor %}
      </select>
    </div>
    <div class="field">
      <label for="cmp-model3">Модель 3</label>
      <select id="cmp-model3">
        {% for m in models %}<option value="{{ m }}" {% if models|length > 2 and m == models[2] %}selected{% endif %}>{{ m }}</option>{% endfor %}
      </select>
    </div>
    <button type="button" class="cmp-run" id="btn-compare-run">Сравнить</button>
    <button type="button" class="cmp-close" id="btn-compare-close">Закрыть</button>
  </div>
  <div class="sendrow">
    <input id="msg" placeholder="Введите задачу..." autocomplete="off">
    <button id="send" type="submit">Отправить</button>
    <button id="stop" type="button" style="display:none">Стоп</button>
  </div>
</form>
</div>
</div>
<script>
const TASKS = {{ tasks_view|tojson }};
const chat = document.getElementById('chat');
const input = document.getElementById('msg');
const modelInput = document.getElementById('model');
const taskInput = document.getElementById('task');
const maxTokensInput = document.getElementById('max_tokens');
const maxCompInput = document.getElementById('max_completion_tokens');
const budgetInput = document.getElementById('budget_tokens');
const stopInput = document.getElementById('stop');
const formatInput = document.getElementById('response_format');
const effortInput = document.getElementById('reasoning_effort');
const sendBtn = document.getElementById('send');
const stopBtn = document.getElementById('stop');
const comparePanel = document.getElementById('compare-panel');
const cmpToggleBtn = document.getElementById('btn-compare-toggle');
const cmpRunBtn = document.getElementById('btn-compare-run');
const cmpCloseBtn = document.getElementById('btn-compare-close');
const cmpModel1 = document.getElementById('cmp-model1');
const cmpModel2 = document.getElementById('cmp-model2');
const cmpModel3 = document.getElementById('cmp-model3');
let busy = false;
let currentCtrl = null;

function setBusy(state){
  busy = state;
  sendBtn.disabled = state;
  stopBtn.style.display = state ? 'block' : 'none';
}

stopBtn.addEventListener('click', () => {
  if (currentCtrl && !currentCtrl.signal.aborted) currentCtrl.abort();
});

taskInput.addEventListener('change', () => {
  const id = taskInput.value;
  const task = TASKS.find(t => t.id === id);
  if (task) input.value = task.prompt;
});

modelInput.addEventListener('change', () => {
  document.getElementById('model-badge').textContent = modelInput.value;
});

function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function collectParams(){
  const body = {};
  const mdl = modelInput.value.trim();
  if (mdl) body.model = mdl;
  const tk = taskInput.value.trim();
  if (tk) body.task_id = tk;
  const mt = parseInt(maxTokensInput.value, 10);
  if (mt && mt > 0) body.max_tokens = mt;
  const mc = parseInt(maxCompInput.value, 10);
  if (mc && mc > 0) body.max_completion_tokens = mc;
  const bt = parseInt(budgetInput.value, 10);
  if (bt && bt > 0) body.budget_tokens = bt;
  const stops = stopInput.value.split(',').map(s => s.trim()).filter(Boolean);
  if (stops.length) body.stop = stops.length === 1 ? stops[0] : stops;
  const fmt = formatInput.value.trim();
  if (fmt) body.response_format = fmt;
  const eff = effortInput.value.trim();
  if (eff) body.reasoning_effort = eff;
  return body;
}

function scroll(){ chat.scrollTop = chat.scrollHeight; }

function addUser(text){
  const row = document.createElement('div'); row.className = 'row user';
  const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
  row.appendChild(b); chat.appendChild(row); scroll();
}

function addAssistant(label){
  const row = document.createElement('div'); row.className = 'row assistant';
  const b = document.createElement('div'); b.className = 'bubble';
  if (label){
    const cap = document.createElement('div'); cap.className = 'cap'; cap.textContent = label; b.appendChild(cap);
  }
  const ans = document.createElement('div'); ans.className = 'answer';
  const details = document.createElement('details'); details.className = 'reasoning'; details.style.display = 'none';
  const sum = document.createElement('summary'); sum.textContent = 'Размышления';
  const rb = document.createElement('div'); rb.className = 'reason-body';
  details.appendChild(sum); details.appendChild(rb);
  const meta = document.createElement('div'); meta.className = 'meta'; meta.style.display = 'none';
  const asst = { ans, details, rb, meta, bubble: b };
  b.appendChild(ans); b.appendChild(details); b.appendChild(meta);
  row.appendChild(b); chat.appendChild(row); scroll();
  return asst;
}

function showNote(asst, text, cls){
  const n = document.createElement('div');
  n.className = 'note ' + cls;
  n.textContent = text;
  asst.bubble.appendChild(n); scroll();
}

function warnText(r){
  if (r === 'length') return 'Ответ обрезан: достигнут лимит max_tokens.';
  if (r === 'content_filter') return 'Ответ остановлен фильтром контента.';
  return 'Ответ остановлен: ' + r;
}

function showTyping(){
  hideTyping();
  const row = document.createElement('div'); row.className = 'row assistant';
  const b = document.createElement('div'); b.className = 'bubble';
  b.innerHTML = '<span class="typing"><span class="t-dot"></span><span class="t-dot"></span><span class="t-dot"></span> Модель печатает…</span>';
  row.appendChild(b); row.id = 'typing'; chat.appendChild(row); scroll();
}
function hideTyping(){ const t = document.getElementById('typing'); if (t) t.remove(); }

function fmt(metrics){
  if (!metrics) return '';
  const u = metrics.usage || {};
  const lat = metrics.latency_ms == null ? '?' : metrics.latency_ms;
  const pt = u.prompt_tokens == null ? '?' : u.prompt_tokens;
  const ct = u.completion_tokens == null ? '?' : u.completion_tokens;
  const tt = u.total_tokens == null ? '?' : u.total_tokens;
  const mdl = metrics.model || '?';
  return lat + ' мс | токены ' + pt + '+' + ct + '=' + tt + ' | ' + mdl;
}

async function runStream(asst, payload){
  let gotContent = false;
  const res = { metrics: null, finish_reason: null, empty: false, stopped_by_stop: false, stopped: false, correct: null };
  const ctrl = new AbortController();
  currentCtrl = ctrl;
  try {
    const r = await fetch('/chat/stream', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload), signal: ctrl.signal });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream:true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const evt = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of evt.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const d = JSON.parse(line.slice(6));
          if (d.kind === 'reasoning') {
            asst.details.style.display = 'block';
            asst.rb.textContent += d.text; scroll();
          } else if (d.kind === 'content') {
            if (!gotContent) { gotContent = true; hideTyping(); }
            asst.ans.textContent += d.text; scroll();
          } else if (d.kind === 'error') {
            gotContent = true; hideTyping();
            showNote(asst, 'Ошибка: ' + d.message, 'error');
          } else if (d.kind === 'done') {
            res.metrics = d.metrics;
            res.finish_reason = d.finish_reason;
            res.empty = d.empty;
            res.stopped_by_stop = d.stopped_by_stop;
            res.correct = d.correct === undefined ? null : d.correct;
            if (d.metrics && d.metrics.model) {
              document.getElementById('model-badge').textContent = d.metrics.model;
            }
            if (d.stopped_by_stop) showNote(asst, 'Ответ остановлен стоп-последовательностью.', 'warn');
            else if (d.empty) showNote(asst, 'Модель не вернула ответ.', 'warn');
            else if (d.finish_reason && d.finish_reason !== 'stop') showNote(asst, warnText(d.finish_reason), 'warn');
          }
        }
      }
    }
  } catch (err) {
    hideTyping();
    if (err && err.name === 'AbortError') {
      res.stopped = true;
      showNote(asst, 'Остановлено пользователем.', 'warn');
    } else {
      showNote(asst, 'Ошибка запроса: ' + err, 'error');
    }
  } finally {
    currentCtrl = null;
  }
  return res;
}

cmpToggleBtn.addEventListener('click', () => {
  comparePanel.style.display = comparePanel.style.display === 'none' ? 'flex' : 'none';
});
cmpCloseBtn.addEventListener('click', () => { comparePanel.style.display = 'none'; });

function renderCompareModelsTable(asst, rows, verdict){
  const b = document.createElement('div'); b.className = 'bubble cmp';
  let html = '<table class="cmp-table">'
           + '<tr><th>Модель</th><th class="num">Верных</th><th class="num">Время, мс</th>'
           + '<th class="num">Токены</th><th class="num">Стоимость, ₽</th></tr>';
  for (const r of rows){
    const time = r.time_ms == null ? '—' : String(Math.round(r.time_ms));
    const tokens = r.tokens == null ? '—' : String(r.tokens);
    const cost = r.cost == null ? '—' : r.cost.toFixed(4);
    let mark = '';
    if (r.correct === r.total && r.total > 0) mark = ' <span class="badge ok">✓ все</span>';
    else if (r.correct === 0) mark = ' <span class="badge bad">✗ нет</span>';
    html += '<tr><td>' + esc(r.model) + '</td><td class="num">' + r.correct + '/' + r.total + mark + '</td>'
          + '<td class="num">' + time + '</td><td class="num">' + tokens + '</td><td class="num">' + cost + '</td></tr>';
  }
  html += '</table>';
  if (verdict && verdict.length){
    html += '<div class="verdict-title">Вывод</div>';
    verdict.forEach(line => { html += '<div class="note verdict">' + esc(line) + '</div>'; });
  }
  b.innerHTML = html;
  asst.bubble.appendChild(b);
  asst.ans.style.display = 'none';
  scroll();
}

async function runCompareModelsStream(asst, statusEl, models){
  const ctrl = new AbortController();
  currentCtrl = ctrl;
  try {
    const r = await fetch('/compare', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ models }), signal: ctrl.signal });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream:true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const evt = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of evt.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const d = JSON.parse(line.slice(6));
          if (d.kind === 'progress') { statusEl.textContent = d.text; scroll(); }
          else if (d.kind === 'result') { statusEl.remove(); renderCompareModelsTable(asst, d.rows, d.verdict); }
          else if (d.kind === 'error') { showNote(asst, 'Ошибка: ' + d.message, 'error'); }
        }
      }
    }
  } catch (err) {
    if (err && err.name === 'AbortError') { showNote(asst, 'Остановлено пользователем.', 'warn'); }
    else { showNote(asst, 'Ошибка запроса: ' + err, 'error'); }
  } finally {
    currentCtrl = null;
  }
}

async function runCompareModels(){
  if (busy) return;
  const asst = addAssistant('Сравнение моделей');
  const models = [cmpModel1.value, cmpModel2.value, cmpModel3.value].map(v => v.trim()).filter(Boolean);
  if (new Set(models).size !== models.length || models.length < 2){
    showNote(asst, 'Выберите 2–3 разные модели.', 'warn'); scroll(); return;
  }
  setBusy(true);
  const statusEl = document.createElement('div'); statusEl.className = 'meta';
  asst.bubble.appendChild(statusEl); scroll();
  try {
    await runCompareModelsStream(asst, statusEl, models);
  } catch (err) {
    showNote(asst, 'Ошибка: ' + err, 'error');
  } finally {
    setBusy(false); input.focus();
  }
}

cmpRunBtn.addEventListener('click', runCompareModels);

document.getElementById('chat-form').addEventListener('submit', async function (e) {
  e.preventDefault();
  if (busy) return;
  const m = input.value.trim();
  if (!m) return;
  input.value = '';
  addUser(m);
  setBusy(true); showTyping();
  const asst = addAssistant();
  const payload = Object.assign({ message: m }, collectParams());
  try {
    const res = await runStream(asst, payload);
    hideTyping();
    const s = fmt(res.metrics);
    if (s) { asst.meta.textContent = 'metrics: ' + s; asst.meta.style.display = 'block'; }
  } catch (err) {
    hideTyping();
    showNote(asst, 'Ошибка запроса: ' + err, 'error');
  } finally {
    setBusy(false); input.focus();
  }
});
</script>
</body>
</html>"""


@app.route("/")
def index():
    tasks_view = [{"id": t.id, "title": t.title, "prompt": t.prompt} for t in TASKS]
    response = app.make_response(
        render_template_string(HTML, model=MODELS[0], models=MODELS,
                               tasks_view=tasks_view)
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _load_params(data: dict) -> tuple:
    model = (data.get("model") or "").strip() or None
    max_tokens = data.get("max_tokens")
    stop = data.get("stop")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                max_tokens = None
        except (TypeError, ValueError):
            max_tokens = None
    reasoning_effort = (data.get("reasoning_effort") or "").strip() or None

    def _int_arg(key: str):
        val = data.get(key)
        if val is None:
            return None
        try:
            val = int(val)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    max_completion_tokens = _int_arg("max_completion_tokens")
    budget_tokens = _int_arg("budget_tokens")
    return model, max_tokens, stop, reasoning_effort, max_completion_tokens, budget_tokens


def _build_messages(data: dict) -> list[dict]:
    messages = []
    fmt = (data.get("response_format") or "").strip()
    if fmt:
        messages.append({"role": "system", "content": f"Формат ответа: {fmt}"})
    instr = (data.get("instruction") or "").strip()
    if instr:
        messages.append({"role": "system", "content": instr})
    messages.append({"role": "user", "content": data["message"]})
    return messages


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    model, max_tokens, stop, reasoning_effort, max_completion_tokens, budget_tokens = _load_params(data)
    try:
        reply = chat(_build_messages(data), model=model, max_tokens=max_tokens, stop=stop,
                     reasoning_effort=reasoning_effort,
                     max_completion_tokens=max_completion_tokens, budget_tokens=budget_tokens)
        return jsonify({"reply": reply, "metrics": last(), "error": None})
    except Exception as exc:
        return jsonify({"reply": None, "metrics": last(), "error": str(exc)})


@app.route("/chat/stream", methods=["POST"])
def chat_stream_endpoint():
    data = request.get_json()
    model, max_tokens, stop, reasoning_effort, max_completion_tokens, budget_tokens = _load_params(data)
    messages = _build_messages(data)
    task = by_id((data.get("task_id") or "").strip())

    stop_active = bool(stop)

    def generate():
        produced = False
        finish_reason = None
        answer_buf = []
        try:
            for kind, text in stream_chat(messages, model=model, max_tokens=max_tokens, stop=stop,
                                          reasoning_effort=reasoning_effort,
                                          max_completion_tokens=max_completion_tokens,
                                          budget_tokens=budget_tokens):
                if kind == "content":
                    answer_buf.append(text)
                    produced = True
                elif kind == "finish":
                    finish_reason = text
                    continue
                yield f"data: {json.dumps({'kind': kind, 'text': text}, ensure_ascii=False)}\n\n"
            stopped_by_stop = stop_active and finish_reason == "stop"
            answer = "".join(answer_buf)
            correct = task.check(answer) if task else None
            payload = {"kind": "done", "metrics": last(), "finish_reason": finish_reason,
                       "empty": not produced, "stopped_by_stop": stopped_by_stop,
                       "answer": answer, "correct": correct}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error = {"kind": "error", "message": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            finish = {"kind": "done", "metrics": None, "finish_reason": None,
                      "empty": not produced, "stopped_by_stop": False}
            yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-store"})


def _run_task(task, model: str) -> dict:
    """Run one benchmark task on a model and return its grading + metrics."""
    messages = [{"role": "user", "content": task.prompt}]
    try:
        reply = chat(messages, model=model)
    except Exception:
        return {"correct": False, "answer": "", "prompt_tokens": 0,
                "completion_tokens": 0, "time_ms": 0}
    m = last() or {}
    usage = m.get("usage") or {}
    latency = m.get("latency_ms")
    return {
        "correct": bool(task.check(reply)),
        "answer": reply,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "time_ms": latency or 0,
    }


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int):
    """Estimate cost in money units from token usage.

    Uses a model-specific price (``GPUSTACK_PRICES``) if available, otherwise the
    default price (``GPUSTACK_PRICE_DEFAULT``).
    """
    price = MODEL_PRICES.get(model) or PRICE_DEFAULT
    prompt_p, comp_p = price
    return (prompt_tokens / 1_000_000) * prompt_p + (completion_tokens / 1_000_000) * comp_p


@app.route("/compare", methods=["POST"])
def compare_endpoint():
    """Run all benchmark tasks on each of the chosen models and return an aggregated table."""
    data = request.get_json() or {}
    models = [m.strip() for m in data.get("models", []) if m.strip()][:3]

    def generate():
        rows: list[dict] = []
        total_units = len(models) * len(TASKS)
        done_units = 0
        try:
            for model in models:
                correct = 0
                prompt_tokens = 0
                completion_tokens = 0
                time_ms = 0.0
                details: list[dict] = []
                for task in TASKS:
                    done_units += 1
                    progress = {
                        "kind": "progress",
                        "text": f"({done_units}/{total_units}) {model} → {task.title}",
                    }
                    yield f"data: {json.dumps(progress, ensure_ascii=False)}\n\n"
                    r = _run_task(task, model)
                    correct += 1 if r["correct"] else 0
                    prompt_tokens += r["prompt_tokens"]
                    completion_tokens += r["completion_tokens"]
                    time_ms += r["time_ms"] or 0
                    details.append({"id": task.id, "title": task.title, "correct": r["correct"]})
                rows.append({
                    "model": model,
                    "correct": correct,
                    "total": len(TASKS),
                    "time_ms": round(time_ms, 1),
                    "tokens": prompt_tokens + completion_tokens,
                    "cost": _compute_cost(model, prompt_tokens, completion_tokens),
                    "details": details,
                })
            result = {"kind": "result", "rows": rows, "verdict": _model_verdict(rows)}
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error = {"kind": "error", "message": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-store"})


def _fmt_time(ms) -> str:
    return f"{ms / 1000:.1f} с" if ms is not None else "—"


def _model_verdict(rows: list[dict]) -> list[str]:
    """Build a short data-driven conclusion from the model comparison rows."""
    lines: list[str] = []
    if not rows:
        return lines
    total = rows[0]["total"]

    lines.append(
        "Верность (" + str(total) + " задач): "
        + ", ".join(f"{r['model']} — {r['correct']}/{total}" for r in rows) + "."
    )
    lines.append(
        "Скорость (время на весь прогон): "
        + ", ".join(f"{r['model']} — {_fmt_time(r['time_ms'])}" for r in rows) + "."
    )
    lines.append(
        "Ресурсоёмкость (токены на весь прогон): "
        + ", ".join(f"{r['model']} — {r['tokens']}" for r in rows) + "."
    )

    if len(rows) < 2:
        lines.append("Для вывода по «флагман vs маленькая» нужно минимум две модели.")
        return lines

    small = min(rows, key=lambda r: (r["cost"] if r["cost"] is not None else float("inf")))
    flagship = max(rows, key=lambda r: (r["cost"] if r["cost"] is not None else float("-inf")))

    if flagship["model"] == small["model"]:
        lines.append("Стоимость у всех моделей одинаковая — «флагман» и «маленькая» не выделяются.")
    else:
        cost_msg = f"Флагман {flagship['model']} дороже маленькой {small['model']}"
        if small["cost"]:
            cost_msg += f" в {flagship['cost'] / small['cost']:.1f} раз"
        else:
            cost_msg += " (маленькая модель бесплатна)"
        lines.append(cost_msg + ".")

        diff = flagship["correct"] - small["correct"]
        if diff > 0:
            lines.append(
                f"Точность: флагман решает верно на {diff} из {total} задач больше "
                f"({flagship['correct']}/{total} против {small['correct']}/{total})."
            )
        elif diff < 0:
            lines.append(
                f"Точность: маленькая модель решает верно на {-diff} из {total} задач больше флагмана."
            )
        else:
            lines.append(
                f"Точность: флагман и маленькая модель решили одинаково ({small['correct']}/{total})."
            )

        ok = [d["title"] for d in small.get("details", []) if d["correct"]]
        if len(ok) == total and total > 0:
            lines.append("Маленькая модель достаточна на всех задачах — флагман не нужен.")
        else:
            good = ", ".join(ok) if ok else "ни на одной"
            lines.append("Маленькая модель достаточна на задачах: " + good + ".")

    return lines


def main() -> None:
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
