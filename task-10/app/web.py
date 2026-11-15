"""Minimal Flask web agent: user query -> LLM via API -> display answer."""
from flask import Flask, jsonify, render_template_string, request

from .agent import Agent
from . import experiment, store
from .config import MODELS, MODEL

app = Flask(__name__)

HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Простой агент</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: radial-gradient(1200px 600px at 20% -10%, #1e293b, #0b1120 60%);
         color:#e2e8f0; height:100vh; display:flex; }
  .main { width:100%; max-width:860px; margin:0 auto; display:flex; flex-direction:column; height:100%; }
  header { padding:18px 20px; display:flex; align-items:center; gap:12px; flex-wrap:wrap;
           border-bottom:1px solid rgba(255,255,255,.08); }
  header .logo { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
                 background:linear-gradient(135deg,#2563eb,#7c3aed); }
  header h1 { font-size:16px; margin:0; }
  header .ctx { margin-left:auto; display:flex; align-items:center; gap:7px; font-size:12px; color:#94a3b8; }
  header .model { font-size:12px; color:#94a3b8; background:rgba(255,255,255,.06);
                  border:1px solid rgba(255,255,255,.1); padding:5px 12px; border-radius:999px; }
  #contextMode, #branchSelect { font-size:12px; color:#e2e8f0; background:rgba(255,255,255,.06);
                  border:1px solid rgba(255,255,255,.1); padding:5px 8px; border-radius:999px;
                  cursor:pointer; outline:none; transition:.15s; }
  #contextMode:hover, #branchSelect:hover { border-color:#2563eb; }
  #contextMode option, #branchSelect option { background:#0f172a; color:#e2e8f0; }
  #clear, #checkpoint, #branchAdd, #branchDel { font-size:12px; color:#cbd5e1; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
           padding:5px 12px; border-radius:999px; cursor:pointer; transition:.15s; }
  #clear:hover { background:rgba(239,68,68,.15); color:#fca5a5; border-color:rgba(239,68,68,.4); }
  #checkpoint:hover, #branchAdd:hover { border-color:#2563eb; color:#bfdbfe; }
  #branchDel:hover { background:rgba(239,68,68,.15); color:#fca5a5; border-color:rgba(239,68,68,.4); }
  #branchDel:disabled { opacity:.4; cursor:not-allowed; }
  #experiment { font-size:12px; color:#bfdbfe; background:rgba(37,99,235,.15); border:1px solid rgba(37,99,235,.4);
                padding:5px 12px; border-radius:999px; cursor:pointer; transition:.15s; }
  #experiment:hover { background:rgba(37,99,235,.3); color:#fff; }
  #experiment:disabled { opacity:.5; cursor:not-allowed; }
  .exp-divider { text-align:center; font-size:11.5px; color:#93c5fd; background:rgba(37,99,235,.08);
                 border:1px dashed rgba(37,99,235,.35); border-radius:999px; padding:5px 12px; margin:2px auto; max-width:92%; }
  .exp-status { text-align:center; font-size:12px; color:#94a3b8; padding:6px; }
  .report { background:#0f172a; border:1px solid rgba(255,255,255,.09); border-radius:14px; padding:14px 16px; font-size:13px; }
  .report h3 { margin:0 0 8px; font-size:14px; }
  .report table { width:100%; border-collapse:collapse; margin:8px 0; }
  .report th, .report td { border-bottom:1px solid rgba(255,255,255,.07); padding:6px 8px; text-align:left; font-size:12.5px; }
  .report th { color:#94a3b8; font-weight:600; }
  .report .good { color:#4ade80; } .report .bad { color:#f87171; }
  .report .verdict { margin-top:8px; color:#e2e8f0; line-height:1.55; }
  .report details { margin-top:8px; }
  .report summary { cursor:pointer; color:#93c5fd; font-size:12.5px; }
  .report .qa-item { padding:7px 0; border-bottom:1px dashed rgba(255,255,255,.06); }
  .report .qa-q { font-weight:600; }
  .report .qa-answer { font-size:12.5px; margin-top:2px; color:#cbd5e1; white-space:pre-wrap; }
  .report .qa-expected { font-size:11.5px; color:#94a3b8; margin-top:2px; }
  .report .repeat { margin-top:10px; font-size:12px; color:#cbd5e1; background:rgba(255,255,255,.06);
                    border:1px solid rgba(255,255,255,.1); padding:5px 12px; border-radius:999px; cursor:pointer; }
  #chat { flex:1; overflow-y:auto; padding:24px 20px; display:flex; flex-direction:column; gap:16px; }
  #chat::-webkit-scrollbar { width:8px; }
  #chat::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12); border-radius:8px; }
  #chat::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,.22); }
  .row { display:flex; animation:pop .25s ease both; }
  .row.user { justify-content:flex-end; }
  .row.assistant { justify-content:flex-start; }
  @keyframes pop { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
  .bubble { max-width:80%; padding:11px 15px; border-radius:16px; line-height:1.55;
            white-space:pre-wrap; word-wrap:break-word; font-size:14.5px;
            box-shadow:0 2px 10px rgba(0,0,0,.25); }
  .user .bubble { background:linear-gradient(135deg,#2563eb,#4f46e5); color:#fff; border-bottom-right-radius:5px; }
  .assistant .bubble { background:#0f172a; border:1px solid rgba(255,255,255,.07); border-bottom-left-radius:5px; }
  .error { margin-top:10px; padding:9px 12px; border-radius:9px; font-size:12.5px;
           background:rgba(239,68,68,.1); color:#fca5a5; border-left:3px solid #ef4444; white-space:pre-wrap; }
  .typing { display:inline-flex; gap:4px; align-items:center; color:#94a3b8; font-size:13px; }
  .typing .t-dot { width:7px; height:7px; border-radius:50%; background:#94a3b8; animation:blink 1.2s infinite; }
  .typing .t-dot:nth-child(2){ animation-delay:.2s; }
  .typing .t-dot:nth-child(3){ animation-delay:.4s; }
  @keyframes blink { 0%,80%,100%{ opacity:.2; transform:scale(1); } 40%{ opacity:1; transform:scale(1.15); } }
  form { display:flex; gap:10px; padding:16px 20px 20px; border-top:1px solid rgba(255,255,255,.08); }
  #msg { flex:1; padding:13px 15px; border-radius:13px; border:1px solid rgba(255,255,255,.12);
         background:rgba(255,255,255,.05); color:#e2e8f0; font-size:14.5px; outline:none; transition:.15s; }
  #msg:focus { border-color:#2563eb; background:rgba(255,255,255,.07); }
  #send { padding:12px 24px; border-radius:13px; border:none; cursor:pointer; font-size:14.5px; font-weight:600;
          color:#fff; background:linear-gradient(135deg,#2563eb,#4f46e5);
          box-shadow:0 4px 14px rgba(37,99,235,.35); transition:.15s; }
  #send:hover { filter:brightness(1.08); }
  #send:disabled { opacity:.5; cursor:not-allowed; filter:none; }
</style>
</head>
<body>
<div class="main">
<header>
  <span class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/></svg></span>
  <h1>Простой агент</h1>
  <span class="ctx">
    <label for="contextMode" title="Что отправлять модели: сводку/факты + последние сообщения или только окно">Управление контекстом</label>
    <select id="contextMode">
      <option value="summary">summary + 10 последних</option>
      <option value="window">sliding window (10 последних)</option>
      <option value="facts">facts + 10 последних</option>
    </select>
  </span>
  <span class="ctx">
    <label for="branchSelect" title="Ветки диалога — независимые продолжения одного диалога">Ветка</label>
    <select id="branchSelect" title="Ветки диалога: выберите, в какой ветке продолжить диалог"></select>
  </span>
  <button id="checkpoint" type="button" title="Запомнить текущее место диалога — от этого checkpoint создаются ветки">&#9873; Checkpoint</button>
  <button id="branchAdd" type="button" title="Создать ветку от checkpoint текущей ветки (или от её конца, если checkpoint не сохранён) и переключиться в неё">+ ветка</button>
  <button id="branchDel" type="button" title="Удалить текущую ветку (основную удалить нельзя)">Удалить ветку</button>
  <button id="experiment" type="button" title="Прогнать один диалог в двух режимах и сравнить память и расход токенов">Эксперимент с хранением контекста</button>
  <button id="clear" type="button" title="Очистить историю">Сбросить</button>
  <span class="model">{{ model }}</span>
</header>
<div id="chat"></div>
<form id="form">
  <input id="msg" placeholder="Введите ваш запрос..." autocomplete="off">
  <button id="send" type="submit">Отправить</button>
</form>
</div>
<script>
const input = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const chatEl = document.getElementById('chat');
const modelBadge = document.querySelector('.model');
const modeSelect = document.getElementById('contextMode');
const branchSelect = document.getElementById('branchSelect');
const checkpointBtn = document.getElementById('checkpoint');
const branchAddBtn = document.getElementById('branchAdd');
const branchDelBtn = document.getElementById('branchDel');
let currentBranch = 1;
let rootBranchId = 1;

try {
  const saved = localStorage.getItem('contextMode');
  if (saved && [...modeSelect.options].some(o => o.value === saved)) modeSelect.value = saved;
} catch (e) {}
modeSelect.addEventListener('change', () => {
  try { localStorage.setItem('contextMode', modeSelect.value); } catch (e) {}
});

function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function loadBranches(keepCurrent){
  const data = await (await fetch('/api/branches')).json();
  const branches = data.branches || [];
  if (!branches.length) return;
  rootBranchId = branches[0].id;
  let saved = null;
  try { saved = parseInt(localStorage.getItem('branchId'), 10); } catch (e) {}
  const found = branches.find(b => b.id === saved);
  currentBranch = (keepCurrent && branches.some(b => b.id === currentBranch))
    ? currentBranch : (found ? found.id : rootBranchId);
  branchSelect.innerHTML = branches.map(b =>
    '<option value="' + b.id + '">' + esc(b.name) + ' · ' + b.count + ' сообщ.' +
    (b.checkpoint_len !== null && b.checkpoint_len !== undefined ? ' · \u2691' + b.checkpoint_len : '') +
    '</option>').join('');
  branchSelect.value = currentBranch;
  try { localStorage.setItem('branchId', currentBranch); } catch (e) {}
  branchDelBtn.disabled = currentBranch === rootBranchId || branches.length <= 1;
}

branchSelect.addEventListener('change', async () => {
  currentBranch = parseInt(branchSelect.value, 10);
  try { localStorage.setItem('branchId', currentBranch); } catch (e) {}
  chatEl.innerHTML = '';
  await loadHistory();
  await loadBranches(true);
});

checkpointBtn.addEventListener('click', async () => {
  try {
    const r = await (await fetch('/api/checkpoint', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ branch_id: currentBranch })
    })).json();
    if (r.error) { addRow('assistant', 'Ошибка: ' + r.error, true); return; }
    addRow('assistant', '\u2691 Checkpoint сохранён: ' + r.checkpoint_len +
      ' сообщ. в текущей ветке. Кнопка «+ ветка» создаст ветку от этой точки.');
    await loadBranches(true);
  } catch (err) { addRow('assistant', 'Ошибка запроса: ' + err, true); }
});

branchAddBtn.addEventListener('click', async () => {
  try {
    const s = await (await fetch('/api/branches')).json();
    const n = (s.branches || []).length + 1;
    const name = prompt('Название ветки. Она создастся от checkpoint текущей ветки ' +
                        '(или от её конца, если checkpoint не сохранён):', 'Ветка ' + n);
    if (name === null) return;
    const r = await (await fetch('/api/branch/create', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from_branch_id: currentBranch, name: name })
    })).json();
    if (r.error) { addRow('assistant', 'Ошибка: ' + r.error, true); return; }
    currentBranch = r.branch.id;
    try { localStorage.setItem('branchId', currentBranch); } catch (e) {}
    chatEl.innerHTML = '';
    await loadBranches(false);
    await loadHistory();
    addRow('assistant', 'Создана ветка «' + r.branch.name + '»: первые ' + r.branch.fork_len +
      ' сообщ. скопированы, дальше диалог идёт независимо. Переключайтесь списком «Ветка».');
  } catch (err) { addRow('assistant', 'Ошибка запроса: ' + err, true); }
});

branchDelBtn.addEventListener('click', async () => {
  if (!confirm('Удалить текущую ветку со всей её историей?')) return;
  try {
    const r = await (await fetch('/api/branch/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ branch_id: currentBranch })
    })).json();
    if (r.error) { addRow('assistant', 'Ошибка: ' + r.error, true); return; }
    currentBranch = rootBranchId;
    try { localStorage.setItem('branchId', currentBranch); } catch (e) {}
    chatEl.innerHTML = '';
    await loadBranches(false);
    await loadHistory();
  } catch (err) { addRow('assistant', 'Ошибка запроса: ' + err, true); }
});
function scroll(){ chatEl.scrollTop = chatEl.scrollHeight; }
function addRow(cls, text, isError){
  const row = document.createElement('div'); row.className = 'row ' + cls;
  const b = document.createElement('div'); b.className = 'bubble';
  if (cls === 'assistant' && isError) b.className = 'bubble error';
  b.textContent = text;
  row.appendChild(b); chatEl.appendChild(row); scroll();
  return b;
}
function showTyping(){
  hideTyping();
  const b = addRow('assistant', '');
  b.innerHTML = '<span class="typing"><span class="t-dot"></span><span class="t-dot"></span><span class="t-dot"></span> Агент думает…</span>';
  b.id = 'typing';
}
function hideTyping(){ const t = document.getElementById('typing'); if (t) t.remove(); }

function tokenMeta(t, contextInfo){
  if (!t) return null;
  let metaText = 'токены: запрос ' + t.request + ', история ' + t.history +
    ', ответ ' + t.response;
  if (typeof t.cost === 'number') metaText += ' · стоимость ' + t.cost.toFixed(4) + ' \u20bd';
  if (typeof t.balance === 'number') metaText += ' · баланс ' + t.balance.toFixed(2) + ' \u20bd';
  if (contextInfo && contextInfo.mode === 'summary') {
    metaText += ' · контекст: summary';
    if (contextInfo.summary_used) metaText += ' (сводка ' + contextInfo.summary_tokens + ' ток.)';
    metaText += ' + ' + contextInfo.as_is + ' сообщ. как есть';
  } else if (contextInfo && contextInfo.mode === 'window') {
    metaText += ' · контекст: окно ' + contextInfo.as_is + ' сообщ.';
  } else if (contextInfo && contextInfo.mode === 'facts') {
    metaText += ' · контекст: facts';
    if (contextInfo.facts_used) metaText += ' (' + contextInfo.facts_tokens + ' ток.)';
    metaText += ' + ' + contextInfo.as_is + ' сообщ. как есть';
  }
  return metaText;
}

function appendMeta(bubble, metaText){
  if (!metaText) return;
  const meta = document.createElement('div');
  meta.style.cssText = 'margin-top:8px;font-size:11px;color:#64748b;';
  meta.textContent = metaText;
  bubble.appendChild(meta);
}

async function run(){
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addRow('user', text);
  showTyping();
  sendBtn.disabled = true;
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, context_mode: modeSelect.value, branch_id: currentBranch })
    });
    const data = await r.json();
    hideTyping();
    if (data.error) {
      addRow('assistant', 'Ошибка: ' + data.error, true);
    } else {
      const bubble = addRow('assistant', data.reply || '(пустой ответ)');
      let metaText = null;
      if (data.tokens) {
        metaText = 'Инструментов вызвано: ' + data.tool_calls_used +
          ' · шагов: ' + data.steps + ' · ' + tokenMeta(data.tokens, data.context);
      } else if (data.tool_calls_used > 0) {
        metaText = 'Инструментов вызвано: ' + data.tool_calls_used + ' · шагов: ' + data.steps;
      }
      appendMeta(bubble, metaText);
      if (data.model) modelBadge.textContent = data.model;
    }
  } catch (err) {
    hideTyping();
    addRow('assistant', 'Ошибка запроса: ' + err, true);
  } finally {
    sendBtn.disabled = false;
    input.focus();
    scroll();
  }
}

async function loadHistory(){
  try {
    const r = await fetch('/api/history?branch_id=' + currentBranch);
    const data = await r.json();
    const messages = data.messages || [];
    if (!messages.length) {
      addRow('assistant', 'Привет! Отправьте запрос — агент передаст его в LLM и покажет ответ.');
      return;
    }
    let bubble = null;
    for (const m of messages) {
      if (m.role === 'user') addRow('user', m.content);
      else if (m.role === 'assistant') bubble = addRow('assistant', m.content || '(пустой ответ)');
    }
  } catch (err) {
    addRow('assistant', 'Не удалось загрузить историю: ' + err, true);
  }
}

document.getElementById('clear').addEventListener('click', async () => {
  if (!confirm('Очистить историю диалога, все ветки и чекпоинты?')) return;
  await fetch('/api/clear', { method: 'POST' });
  chatEl.innerHTML = '';
  await loadBranches(false);
  addRow('assistant', 'История очищена. Начинаем заново!');
});

const expBtn = document.getElementById('experiment');
const exp = { running: false, lastKey: null, statusEl: null };

function setExpUI(disabled){
  input.disabled = disabled;
  sendBtn.disabled = disabled;
  expBtn.disabled = disabled;
  checkpointBtn.disabled = disabled;
  branchAddBtn.disabled = disabled;
  branchDelBtn.disabled = disabled || currentBranch === rootBranchId;
}

function expStatus(text){
  if (!exp.statusEl) {
    exp.statusEl = document.createElement('div');
    exp.statusEl.className = 'exp-status';
    chatEl.appendChild(exp.statusEl);
  }
  exp.statusEl.textContent = text;
  scroll();
}
function clearExpStatus(){ if (exp.statusEl) { exp.statusEl.remove(); exp.statusEl = null; } }
function expDivider(text){
  const d = document.createElement('div');
  d.className = 'exp-divider';
  d.textContent = text;
  chatEl.appendChild(d);
  scroll();
}

function renderStep(data){
  const p = data.progress || {};
  const key = data.run + ':' + data.stage;
  if (exp.lastKey !== key) {
    if (data.stage === 'dialog') {
      expDivider('Прогон ' + (data.run + 1) + ' · режим «' + data.label + '»' +
                 (data.run > 0 ? ' · история очищена' : ''));
    } else {
      expDivider('Вопросы по деталям · режим «' + data.label + '»');
    }
    exp.lastKey = key;
  }
  addRow('user', p.user);
  const bubble = addRow('assistant', p.reply || '(пустой ответ)');
  let metaText = p.tokens ? tokenMeta(p.tokens, p.context) : null;
  if (p.ok !== undefined && p.ok !== null) {
    const verdict = p.ok ? 'верно' : 'неверно';
    metaText = (metaText ? metaText + ' · ' : '') +
      verdict + ' (деталей: ' + p.facts_found + '/' + p.facts_total + ')';
  }
  appendMeta(bubble, metaText);
  const stepNo = (data.step_index || 0) + 1;
  expStatus('Эксперимент · ' + (data.stage === 'dialog' ? 'диалог' : 'вопросы') +
    ' · шаг ' + stepNo + '/' + data.step_total + ' · «' + data.label + '»');
}

function num(n){ return (n || 0).toLocaleString('ru-RU'); }

function bestFlags(values, mode){
  const b = mode === 'min' ? Math.min.apply(null, values) : Math.max.apply(null, values);
  return values.map(v => v === b);
}

function renderReport(rep){
  const runs = rep.runs;
  const div = document.createElement('div');
  div.className = 'report';

  const totals = runs.map(r => r.api.prompt + r.api.completion);
  const qt = runs[0].questions_total;
  const dt = runs[0].details_total;
  const remBest = bestFlags(runs.map(r => r.remembered), 'max');
  const memBest = bestFlags(runs.map(r => r.details_found), 'max');
  const tokBest = bestFlags(totals, 'min');

  let html = '<h3>Эксперимент: ' + esc(rep.title) + '</h3>';
  html += '<table><tr><th>Стратегия</th><th>Вспомнено вопросов</th><th>Деталей найдено</th>' +
          '<th>Токены API (prompt + completion)</th><th>Стоимость</th>' +
          '<th>Средний контекст вопроса</th><th>Поддержка: код и доп. вызовы</th></tr>';
  runs.forEach((r, i) => {
    const eu = r.extra_usage || {};
    const mt = r.maintenance || {};
    html += '<tr>' +
      '<td><b>' + esc(r.label) + '</b></td>' +
      '<td class="' + (remBest[i] ? 'good' : '') + '">' + r.remembered + ' / ' + qt + '</td>' +
      '<td class="' + (memBest[i] ? 'good' : '') + '">' + r.details_found + ' / ' + dt + '</td>' +
      '<td class="' + (tokBest[i] ? 'good' : '') + '">' + num(totals[i]) +
        '<div style="font-size:11px;color:#94a3b8;">prompt ' + num(r.api.prompt) +
        ' / completion ' + num(r.api.completion) + '</div></td>' +
      '<td>' + (r.api.cost ? r.api.cost.toFixed(4) + ' \u20bd' : '&mdash;') + '</td>' +
      '<td>' + num(r.avg_context_tokens) + '</td>' +
      '<td>' + (mt.code_lines ? '&asymp;' + mt.code_lines + ' строк кода' : '&mdash;') +
        (mt.code ? '<div style="font-size:11px;color:#94a3b8;">' + esc(mt.code) + '</div>' : '') +
        '<div style="font-size:11px;color:#94a3b8;">' +
        (eu.calls ? eu.calls + ' доп. выз. · ' + num(eu.prompt + eu.completion) + ' ток. (замер)' : 'без доп. вызовов') +
        '</div>' +
        (mt.calls_note ? '<div style="font-size:11px;color:#94a3b8;">' + esc(mt.calls_note) + '</div>' : '') +
      '</td>' +
      '</tr>';
  });
  html += '</table>';
  html += '<div style="font-size:11px;color:#64748b;margin-top:4px;">Поддержка: строк кода — приближение по стратегическому коду в app/context.py; доп. вызовы — фактический замер прогона.</div>';

  html += '<div class="verdict" style="margin-top:10px;">' +
    '<b>Какая стратегия для какого сценария:</b>' +
    '<ul style="margin:6px 0 0 18px;padding:0;line-height:1.55;">' +
    '<li><b>summary + 10 последних</b> — длинные многоходовые диалоги (работа над задачей, поддержка, онбординг): ничего не теряется совсем, контекст ограничен; цена — вызов суммаризатора за ход при переполнении окна.</li>' +
    '<li><b>sliding window (10 последних)</b> — короткие или «одноразовые» диалоги, где старые детали не нужны: минимум кода и ноль доп. вызовов, но всё за пределами окна забывается навсегда.</li>' +
    '<li><b>facts + 10 последних</b> — диалоги с множеством конкретики (заявки, брони, договорённости с параметрами): ключевые факты живут до конца диалога в компактном key-value блоке; цена — вызов извлечения после каждой реплики и код парсинга JSON.</li>' +
    '</ul></div>';

  const memTop = Math.max.apply(null, runs.map(r => r.details_found));
  const memWinners = runs.filter(r => r.details_found === memTop);
  const tokTop = Math.min.apply(null, totals);
  const tokWinners = runs.filter((r, i) => totals[i] === tokTop);
  let verdict = '<div class="verdict">';
  verdict += memWinners.length === runs.length
    ? 'По деталям режимы равны (' + memTop + ' из ' + dt + '). '
    : 'Деталей лучше помнит режим «' + esc(memWinners[0].label) + '» (' + memTop + ' из ' + dt + '). ';
  verdict += tokWinners.length === runs.length
    ? 'Расход токенов одинаков.'
    : 'Дешевле по токенам: «' + esc(tokWinners[0].label) + '» (' + num(tokTop) +
      ' против ' + num(Math.max.apply(null, totals)) + ' ток. без учёта доп. вызовов).';
  verdict += '</div>';
  html += verdict;

  html += '<details><summary>Ответы на вопросы по деталям</summary>';
  rep.runs.forEach((run, i) => {
    html += '<div style="margin-top:8px;font-weight:600;color:#93c5fd;">Прогон ' + (i + 1) + ' · ' + esc(run.label) + '</div>';
    run.qa.forEach((q, j) => {
      html += '<div class="qa-item">' +
        '<div class="qa-q">' + (j + 1) + '. ' + esc(q.question) + '</div>' +
        '<div class="qa-answer">' + esc(q.answer || '(пустой ответ)') + '</div>' +
        '<div class="' + (q.ok ? 'good' : 'bad') + '" style="font-size:11.5px;margin-top:2px;">' +
        (q.ok ? 'верно' : 'неверно') + ' · деталей: ' + q.facts_found + '/' + q.facts_total + '</div>' +
        '<div class="qa-expected">Ожидалось: ' + esc(q.expected) + '</div>' +
        '</div>';
    });
  });
  html += '</details>';

  html += '<details><summary>Посмотреть историю диалога</summary>';
  rep.runs.forEach((run, i) => {
    html += '<div style="margin-top:8px;font-weight:600;color:#93c5fd;">Прогон ' + (i + 1) + ' · ' + esc(run.label) + '</div>';
    run.transcript.forEach((m) => {
      html += '<div style="margin:6px 0;font-size:12.5px;"><b style="color:#93c5fd;">Пользователь:</b> ' + esc(m.user) + '</div>' +
              '<div style="margin:0 0 4px 12px;font-size:12.5px;color:#cbd5e1;"><b>Агент:</b> ' + esc(m.reply || '(пустой ответ)') + '</div>';
    });
  });
  html += '</details>';

  html += '<button class="repeat" type="button">Повторить эксперимент</button>';
  div.innerHTML = html;
  chatEl.appendChild(div);
  div.querySelector('.repeat').addEventListener('click', () => expBtn.click());
  scroll();
}

async function experimentLoop(){
  exp.running = true;
  setExpUI(true);
  try {
    for (;;) {
      const r = await fetch('/api/experiment/step', { method: 'POST' });
      const data = await r.json();
      if (data.error && !data.progress) {
        clearExpStatus();
        addRow('assistant', 'Ошибка эксперимента: ' + data.error + ' — нажмите кнопку, чтобы продолжить.', true);
        expBtn.textContent = 'Продолжить эксперимент';
        break;
      }
      renderStep(data);
      if (data.phase === 'done') {
        clearExpStatus();
        exp.lastKey = null;
        renderReport(data.report);
        break;
      }
    }
  } catch (err) {
    clearExpStatus();
    addRow('assistant', 'Ошибка запроса: ' + err, true);
    expBtn.textContent = 'Продолжить эксперимент';
  } finally {
    exp.running = false;
    setExpUI(false);
    expBtn.textContent = 'Эксперимент с хранением контекста';
    input.focus();
  }
}

expBtn.addEventListener('click', async () => {
  if (exp.running) return;
  setExpUI(true);
  try {
    const s = await (await fetch('/api/experiment/state')).json();
    if (s.phase !== 'running') {
      if (!confirm('Эксперимент очистит текущую историю диалога, все ветки и чекпоинты ' +
                   'и прогонит один и тот же диалог ' +
                   'в трёх режимах — «summary + 10 последних», ' +
                   '«sliding window (10 последних)» и «facts + 10 последних», — затем задаст ' +
                   'вопросы по деталям (несколько минут). Запустить?')) {
        setExpUI(false);
        return;
      }
      await fetch('/api/experiment/start', { method: 'POST' });
      chatEl.innerHTML = '';
      exp.lastKey = null;
    } else {
      exp.lastKey = s.run + ':' + s.stage;
    }
  } catch (err) {
    setExpUI(false);
    return;
  }
  experimentLoop();
});

(async function initExperiment(){
  try {
    const s = await (await fetch('/api/experiment/state')).json();
    if (s.phase === 'running') {
      exp.lastKey = s.run + ':' + s.stage;
      experimentLoop();
    } else if (s.phase === 'done' && s.report) {
      renderReport(s.report);
    }
  } catch (e) {}
})();

loadBranches(false).then(loadHistory);

document.getElementById('form').addEventListener('submit', (e) => { e.preventDefault(); run(); });
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, model=MODELS[0])


def _branch_id(value) -> int:
    """Parse a branch id from a request; unknown values fall back to the root."""
    try:
        branch_id = int(value or store.ROOT_BRANCH_ID)
    except (TypeError, ValueError):
        return store.ROOT_BRANCH_ID
    return branch_id if store.get_branch(branch_id) else store.ROOT_BRANCH_ID


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": None, "error": "Запрос не может быть пустым."}), 400
    context_mode = data.get("context_mode")
    if context_mode not in ("summary", "window", "facts"):
        context_mode = "summary"
    branch_id = _branch_id(data.get("branch_id"))
    try:
        agent = Agent()
        result = agent.ask(message, context_mode, branch_id)
        return jsonify({
            "reply": result["reply"],
            "tool_calls_used": result["tool_calls_used"],
            "steps": result["steps"],
            "tokens": result.get("tokens"),
            "context": result.get("context"),
            "error": None,
            "model": MODEL,
        })
    except Exception as exc:
        return jsonify({"reply": None, "error": str(exc), "model": MODEL})


@app.route("/api/history")
def api_history():
    branch_id = _branch_id(request.args.get("branch_id"))
    messages = [{"role": m["role"], "content": m["content"]}
                for m in store.load_messages(branch_id)
                if m["role"] in ("user", "assistant") and m.get("content")]
    return jsonify({"messages": messages, "error": None})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    store.clear_history()
    return jsonify({"error": None})


@app.route("/api/branches")
def api_branches():
    return jsonify({"branches": store.list_branches(), "error": None})


@app.route("/api/checkpoint", methods=["POST"])
def api_checkpoint():
    """Save a checkpoint in a branch: the fork point for new branches."""
    data = request.get_json(silent=True) or {}
    branch_id = _branch_id(data.get("branch_id"))
    count = len(store.load_messages(branch_id))
    store.set_checkpoint(branch_id, count)
    return jsonify({"branch_id": branch_id, "checkpoint_len": count, "error": None})


@app.route("/api/branch/create", methods=["POST"])
def api_branch_create():
    """Fork a branch from the parent's checkpoint (or from its end)."""
    data = request.get_json(silent=True) or {}
    parent_id = _branch_id(data.get("from_branch_id"))
    name = (data.get("name") or "").strip() or None
    try:
        branch = store.create_branch(parent_id, name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"branch": branch, "branches": store.list_branches(), "error": None})


@app.route("/api/branch/delete", methods=["POST"])
def api_branch_delete():
    data = request.get_json(silent=True) or {}
    branch_id = _branch_id(data.get("branch_id"))
    try:
        store.delete_branch(branch_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"branches": store.list_branches(), "error": None})


@app.route("/api/experiment/start", methods=["POST"])
def api_experiment_start():
    return jsonify(experiment.start())


@app.route("/api/experiment/step", methods=["POST"])
def api_experiment_step():
    return jsonify(experiment.step())


@app.route("/api/experiment/state")
def api_experiment_state():
    return jsonify(experiment.snapshot())


def main() -> None:
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
