"""Minimal Flask web interface for chatting with the LLM (streaming)."""
import ast
import json
import time

from flask import Flask, Response, jsonify, render_template_string, request

from .config import MODEL, MODELS
from .llm import chat, stream_chat
from .metrics import last
from .tasks import TASKS, by_id

app = Flask(__name__)

# Prompting techniques compared in the "Day 3" experiment (label -> extra instruction).
COMPARE_METHODS = [
    ("Прямой ответ", ""),
    ("Пошагово", "решай пошагово"),
    ("Промпт", "Сначала составь подробный промпт для решения этой задачи, а затем используй его для решения."),
    ("Эксперты", "Сформируй группу экспертов (аналитик, инженер, критик) и предоставь решение от каждого по отдельности."),
]

# Reasoning settings compared: (label, reasoning_effort value).
REASONING_SETTINGS = [("low", "low"), ("high", "high")]

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
  .check-btn { margin-top:10px; padding:6px 12px; border-radius:8px; font-size:12px; cursor:pointer;
               background:rgba(255,255,255,.05); color:#94a3b8; border:1px solid rgba(255,255,255,.14); transition:.15s; }
  .check-btn:hover { background:rgba(255,255,255,.12); color:#e2e8f0; border-color:rgba(255,255,255,.25); }
  .check-btn:disabled { opacity:.5; cursor:not-allowed; }
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
  .presets { display:flex; gap:8px; flex-wrap:wrap; }
  .preset { padding:9px 14px; border-radius:11px; cursor:pointer; font-size:13px; font-weight:600;
            color:#cbd5e1; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12); transition:.15s; }
  .preset:hover { background:rgba(255,255,255,.12); color:#e2e8f0; border-color:rgba(255,255,255,.25); }
  .preset.active { background:linear-gradient(135deg,#2563eb,#4f46e5); color:#fff; border-color:transparent;
                   box-shadow:0 4px 14px rgba(37,99,235,.35); }
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
  <div class="presets">
    <button type="button" class="preset" data-instr="">Прямой ответ</button>
    <button type="button" class="preset" data-instr="решай пошагово">Пошагово</button>
    <button type="button" class="preset" data-instr="Сначала составь подробный промпт для решения этой задачи, а затем используй его для решения.">Промпт</button>
    <button type="button" class="preset" data-instr="Сформируй группу экспертов (аналитик, инженер, критик) и предоставь решение от каждого по отдельности.">Эксперты</button>
    <button type="button" class="preset" data-compare="1">Сравнение</button>
    <button type="button" class="preset" data-experiment="1">Эксперимент</button>
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
const presetBtns = document.querySelectorAll('.preset');
let busy = false;
let currentCtrl = null;
let activeInstr = '';

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

const COMPARE = {{ compare_methods|tojson }};

function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

presetBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    if (busy) return;
    if (btn.dataset.experiment) { runExperiment(); return; }
    if (!input.value.trim()) return;
    if (btn.dataset.compare) { runCompare(); return; }
    activeInstr = btn.dataset.instr || '';
    presetBtns.forEach(b => b.classList.toggle('active', b === btn));
    document.getElementById('chat-form').requestSubmit();
  });
});

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
  const checkBtn = document.createElement('button');
  checkBtn.className = 'check-btn'; checkBtn.type = 'button'; checkBtn.textContent = 'Проверить парсинг';
  const asst = { ans, details, rb, meta, bubble: b, checkBtn, parseNote: null };
  checkBtn.addEventListener('click', () => parseAnswer(asst));
  b.appendChild(ans); b.appendChild(details); b.appendChild(meta); b.appendChild(checkBtn);
  row.appendChild(b); chat.appendChild(row); scroll();
  return asst;
}

function setParseNote(asst, text, cls){
  if (!asst.parseNote){
    asst.parseNote = document.createElement('div');
    asst.bubble.appendChild(asst.parseNote);
  }
  asst.parseNote.className = 'note ' + cls;
  asst.parseNote.textContent = text;
  scroll();
}

async function parseAnswer(asst){
  let text = asst.ans.textContent.trim();
  if (!text){ setParseNote(asst, 'Пустой ответ.', 'warn'); return; }
  const fence = text.match(/```[a-z]*\s*([\s\S]*?)```/);
  if (fence) text = fence[1].trim();
  const btn = asst.checkBtn;
  btn.disabled = true; btn.textContent = 'Проверяю...';
  try {
    const r = await fetch('/parse-check', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text }) });
    const d = await r.json();
    if (d.ok) {
      const good = d.results.filter(x => x.ok).map(x => x.method + ' (' + x.type + ')').join(', ');
      setParseNote(asst, 'Распарсится: ' + good, 'ok');
    } else {
      setParseNote(asst, d.reason || 'Не удалось распарсить ответ.', 'warn');
    }
  } catch (err) {
    setParseNote(asst, 'Ошибка проверки: ' + err, 'error');
  } finally {
    btn.disabled = false; btn.textContent = 'Проверить парсинг';
  }
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

function renderCompareTable(rows){
  const row = document.createElement('div'); row.className = 'row assistant';
  const b = document.createElement('div'); b.className = 'bubble cmp';
  let html = '<div class="cap">Сравнение</div><table class="cmp-table">'
           + '<tr><th>способ</th><th>reasoning_effort</th><th>токены</th><th>время</th><th>ответ</th></tr>';
  for (const r of rows){
    const tokens = r.tokens == null ? '—' : String(r.tokens);
    const time = r.time == null ? '—' : (r.time + ' мс');
    const ans = r.answer == null ? '' : String(r.answer);
    const shortAns = ans.length > 160 ? ans.slice(0, 150) + '…' : ans;
    const title = ans ? ' title="' + esc(ans.slice(0, 500)) + '"' : '';
    let mark = '';
    if (r.correct === true) mark = ' <span class="badge ok">✓ верно</span>';
    else if (r.correct === false) mark = ' <span class="badge bad">✗ неверно</span>';
    html += '<tr><td>' + esc(r.label) + '</td><td>' + esc(r.effort || '—') + '</td>'
          + '<td class="num">' + tokens + '</td><td class="num">' + time + '</td>'
          + '<td' + title + '>' + (shortAns ? esc(shortAns) : '—') + mark + '</td></tr>';
  }
  html += '</table>';
  b.innerHTML = html;
  row.appendChild(b); chat.appendChild(row); scroll();
}

async function runCompare(){
  if (busy) return;
  const m = input.value.trim();
  if (!m) return;
  input.value = '';
  addUser(m);
  const rows = [];
  const effort = (effortInput.value.trim() || '—');
  try {
    for (const item of COMPARE){
      setBusy(true); showTyping();
      const asst = addAssistant(item.label);
      const payload = Object.assign({ message: m, instruction: item.instr }, collectParams());
      const res = await runStream(asst, payload);
      hideTyping();
      const u = res.metrics && res.metrics.usage;
      rows.push({
        label: item.label,
        effort: effort,
        answer: asst.ans.textContent || '',
        correct: res.correct,
        tokens: u && u.total_tokens != null ? u.total_tokens : null,
        time: res.metrics && res.metrics.latency_ms != null ? res.metrics.latency_ms : null,
      });
      const s = fmt(res.metrics);
      if (s) { asst.meta.textContent = 'metrics: ' + s; asst.meta.style.display = 'block'; }
    }
    renderCompareTable(rows);
  } finally {
    hideTyping();
    setBusy(false); input.focus();
  }
}

async function runExperiment(){
  if (busy) return;
  setBusy(true); showTyping();
  const asst = addAssistant('Эксперимент');
  const payload = { model: modelInput.value.trim() };
  try {
    await runExperimentStream(asst, payload);
  } catch (err) {
    hideTyping();
    showNote(asst, 'Ошибка: ' + err, 'error');
  } finally {
    hideTyping();
    setBusy(false); input.focus();
  }
}

async function runExperimentStream(asst, payload){
  const ctrl = new AbortController();
  currentCtrl = ctrl;
  try {
    const r = await fetch('/experiment', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload), signal: ctrl.signal });
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
          if (d.kind === 'progress') { asst.ans.textContent = d.text; scroll(); }
          else if (d.kind === 'result') { asst.ans.textContent = ''; renderExperiment(asst, d.matrix, d.verdict); }
          else if (d.kind === 'error') { showNote(asst, 'Ошибка: ' + d.message, 'error'); }
        }
      }
    }
  } catch (err) {
    if (!(err && err.name === 'AbortError')) {
      hideTyping();
      showNote(asst, 'Ошибка запроса: ' + err, 'error');
    }
  } finally {
    currentCtrl = null;
  }
}

function renderExperiment(asst, rows, verdict){
  const b = document.createElement('div'); b.className = 'bubble cmp';
  const wrap = document.createElement('div'); wrap.className = 'expwrap';
  const cells = {}, settings = [], methods = [];
  const totals = {};
  rows.forEach(r => {
    if (!cells[r.method]) cells[r.method] = {};
    cells[r.method][r.setting] = r;
    if (!settings.includes(r.setting)) settings.push(r.setting);
    if (!methods.includes(r.method)) methods.push(r.method);
    totals[r.setting] = totals[r.setting] || { correct: 0, tasks: 0, tokens: 0, time: 0 };
    const t = totals[r.setting];
    t.correct += r.correct; t.tasks += r.total; t.tokens += r.tokens || 0; t.time += r.time;
  });

  let html = '<div class="cap">Эксперимент · способ × рассуждение</div>'
           + '<div class="verd-sum"><b>Рассуждение:</b> «low» = минимум, «high» = максимум '
           + '(параметр reasoning_effort). Столбцы ниже считаются для каждой настройки отдельно.</div>';
  html += '<table class="cmp-table">';
  html += '<tr><th rowspan="2">Способ</th>';
  settings.forEach(s => { html += '<th colspan="3" class="grp">рассуждение (effort): «' + esc(s) + '»</th>'; });
  html += '</tr>';
  html += '<tr>';
  settings.forEach(() => { html += '<th class="sub">верных</th><th class="sub num">токены</th><th class="sub num">время, с</th>'; });
  html += '</tr>';

  methods.forEach(m => {
    html += '<tr><td>' + esc(m) + '</td>';
    settings.forEach(s => {
      const r = cells[m][s];
      if (!r) { html += '<td class="num">—</td><td class="num">—</td><td class="num">—</td>'; return; }
      let mark = '';
      if (r.correct === r.total && r.total > 0) mark = ' <span class="badge ok" title="все задачи решены верно">✓</span>';
      else if (r.correct === 0) mark = ' <span class="badge bad" title="ни одной задачи решено верно">✗</span>';
      html += '<td class="num">' + r.correct + '/' + r.total + mark + '</td>'
            + '<td class="num">' + (r.tokens || 0) + '</td>'
            + '<td class="num">' + r.time.toFixed(1) + '</td>';
    });
    html += '</tr>';
  });

  html += '<tr class="totals"><td>итого</td>';
  settings.forEach(s => {
    const t = totals[s];
    const pct = t.tasks ? Math.round(100 * t.correct / t.tasks) : 0;
    html += '<td class="num">' + t.correct + '/' + t.tasks + ' (' + pct + '%)</td>'
          + '<td class="num">' + t.tokens + '</td>'
          + '<td class="num">' + t.time.toFixed(1) + '</td>';
  });
  html += '</tr>';
  html += '</table>';

  html += '<div class="verd-sum">';
  settings.forEach(s => {
    const t = totals[s];
    html += '<div class="line">«' + esc(s) + '»: верных <b>' + t.correct + '/' + t.tasks + '</b>, '
          + 'токены <b>' + t.tokens + '</b>, время <b>' + t.time.toFixed(1) + ' с</b> ('
          + (t.tasks ? (t.tokens / t.tasks).toFixed(0) : 0) + ' токенов/задачу)</div>';
  });
  html += '</div>';

  if (verdict && verdict.length) {
    html += '<div class="verdict-title">Вывод</div>';
    verdict.forEach(line => { html += '<div class="note verdict">' + esc(line) + '</div>'; });
  }
  wrap.innerHTML = html;
  b.appendChild(wrap);
  asst.bubble.appendChild(b);
  asst.ans.style.display = 'none';
  scroll();
}

document.getElementById('chat-form').addEventListener('submit', async function (e) {
  e.preventDefault();
  if (busy) return;
  const m = input.value.trim();
  if (!m) return;
  input.value = '';
  addUser(m);
  setBusy(true); showTyping();
  const asst = addAssistant();
  const payload = Object.assign({ message: m, instruction: activeInstr }, collectParams());
  activeInstr = '';
  presetBtns.forEach(b => b.classList.remove('active'));
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
    compare_methods = [{"label": l, "instr": i} for l, i in COMPARE_METHODS]
    response = app.make_response(
        render_template_string(HTML, model=MODELS[0], models=MODELS,
                               tasks_view=tasks_view, compare_methods=compare_methods)
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


def _run_case(task, instr: str, effort: str, model: str) -> dict:
    """Run one (task, method, reasoning) case and return grading + metrics."""
    messages = []
    if instr:
        messages.append({"role": "system", "content": instr})
    messages.append({"role": "user", "content": task.prompt})

    start = time.perf_counter()
    buf = []
    for kind, text in stream_chat(messages, model=model, reasoning_effort=effort):
        if kind == "content":
            buf.append(text)
    elapsed = time.perf_counter() - start

    answer = "".join(buf)
    m = last() or {}
    usage = m.get("usage") or {}
    tokens = usage.get("total_tokens")
    latency = m.get("latency_ms")
    time_s = latency / 1000.0 if latency is not None else elapsed
    return {
        "correct": bool(task.check(answer)),
        "answer": answer,
        "tokens": tokens,
        "time_s": time_s,
    }


def _verdict(rows: list[dict]) -> list[str]:
    """Build a short data-driven analysis from the aggregated comparison rows."""
    settings = [l for l, _ in REASONING_SETTINGS]
    by_method: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], {})[r["setting"]] = r

    lines = []
    direct = by_method.get(COMPARE_METHODS[0][0], {})
    for set_label in settings:
        base = direct.get(set_label)
        if not base:
            continue
        helpers = []
        for mlabel, _ in COMPARE_METHODS[1:]:
            c = by_method.get(mlabel, {}).get(set_label)
            if c and c["correct"] > base["correct"]:
                helpers.append(f"{mlabel} (+{c['correct'] - base['correct']})")
        if helpers:
            lines.append(f"На «{set_label}» доп. приёмы дают верность выше прямого ответа: " + ", ".join(helpers))
        else:
            lines.append(f"На «{set_label}» доп. приёмы не поднимают верность выше прямого ответа.")

    # Compare effort levels: if token totals barely differ, effort is likely not effective.
    totals = {}
    for set_label in settings:
        totals[set_label] = round(sum(
            by_method.get(m, {}).get(set_label, {}).get("tokens") or 0
            for m, _ in COMPARE_METHODS
        ))
    lines.append(
        f"Суммарные токены: {settings[0]} (минимум рассуждения) = {totals.get(settings[0], '—')}, "
        f"{settings[1]} (максимум рассуждения) = {totals.get(settings[1], '—')}."
    )
    if totals.get(settings[0]) and totals.get(settings[1]) and abs(totals[settings[0]] - totals[settings[1]]) < 0.15:
        lines.append("Разница между low и high по токенам минимальна — вероятно, reasoning_effort не управляет "
                     "глубиной рассуждения на этом эндпоинте (параметр игнорируется).")
    return lines


@app.route("/experiment", methods=["POST"])
def experiment_endpoint():
    """Run the full 'Day 3' experiment: 4 methods x 2 reasoning settings x all tasks."""
    data = request.get_json() or {}
    model = (data.get("model") or "").strip() or MODELS[0]

    def generate():
        rows: list[dict] = []
        total_units = len(REASONING_SETTINGS) * len(COMPARE_METHODS) * len(TASKS)
        done_units = 0
        try:
            for set_label, effort in REASONING_SETTINGS:
                for mlabel, instr in COMPARE_METHODS:
                    correct = 0
                    tokens = 0
                    ttime = 0.0
                    count = 0
                    for task in TASKS:
                        done_units += 1
                        progress = {
                            "kind": "progress",
                            "text": f"({done_units}/{total_units}) {mlabel} · reasoning_effort={set_label} → {task.title}",
                        }
                        yield f"data: {json.dumps(progress, ensure_ascii=False)}\n\n"
                        r = _run_case(task, instr, effort, model)
                        correct += 1 if r["correct"] else 0
                        tokens += r["tokens"] or 0
                        ttime += r["time_s"]
                        count += 1
                    rows.append({
                        "method": mlabel,
                        "setting": set_label,
                        "correct": correct,
                        "total": count,
                        "tokens": tokens,
                        "time": round(ttime, 1),
                    })
            result = {"kind": "result", "matrix": rows, "verdict": _verdict(rows)}
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error = {"kind": "error", "message": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-store"})


def _friendly_json(exc: Exception) -> str:
    msg = getattr(exc, "msg", None) or str(exc)
    mapping = [
        ("Expecting value", "ожидалось значение"),
        ("Expecting property name", "не хватает названия поля в двойных кавычках"),
        ("Expecting ',' delimiter", "не хватает запятой между элементами"),
        ("Expecting ':' delimiter", "не хватает двоеточия после названия поля"),
        ("Extra data", "есть лишние данные после завершения структуры"),
        ("Unterminated string", "незакрытая строка"),
    ]
    for key, friendly in mapping:
        if key in msg:
            return friendly
    return "некорректный JSON"


def _friendly_python(exc: Exception) -> str:
    msg = str(exc).lower()
    if "unexpected eof" in msg:
        return "не хватает закрывающей скобки или кавычки"
    if "unterminated string" in msg:
        return "незакрытая строка"
    if "malformed node" in msg or "invalid syntax" in msg:
        return "это не одиночный литерал Python (dict, list, число, строка)"
    return "некорректный Python-литерал"


@app.route("/parse-check", methods=["POST"])
def parse_check_endpoint():
    text = (request.get_json() or {}).get("text", "")
    results = []
    try:
        obj = json.loads(text)
        results.append({"method": "JSON", "ok": True, "type": type(obj).__name__})
    except Exception as exc:
        results.append({"method": "JSON", "ok": False, "detail": _friendly_json(exc)})
    try:
        obj = ast.literal_eval(text)
        results.append({"method": "Python", "ok": True, "type": type(obj).__name__})
    except Exception as exc:
        results.append({"method": "Python", "ok": False, "detail": _friendly_python(exc)})

    if any(r["ok"] for r in results):
        return jsonify({"ok": True, "results": results})

    reason = (f"Это не JSON и не Python-структура. "
              f"JSON: {results[0]['detail']}. Python: {results[1]['detail']}.")
    return jsonify({"ok": False, "results": results, "reason": reason})


def main() -> None:
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
