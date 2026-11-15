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
  #contextMode { font-size:12px; color:#e2e8f0; background:rgba(255,255,255,.06);
                 border:1px solid rgba(255,255,255,.1); padding:5px 8px; border-radius:999px;
                 cursor:pointer; outline:none; transition:.15s; }
  #contextMode:hover { border-color:#2563eb; }
  #contextMode option { background:#0f172a; color:#e2e8f0; }
  #clear { font-size:12px; color:#cbd5e1; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
           padding:5px 12px; border-radius:999px; cursor:pointer; transition:.15s; }
  #clear:hover { background:rgba(239,68,68,.15); color:#fca5a5; border-color:rgba(239,68,68,.4); }
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
    <label for="contextMode" title="Что отправлять модели: всю историю или сводку + последние сообщения">Управление контекстом</label>
    <select id="contextMode">
      <option value="full">полная история</option>
      <option value="summary">summary + 10 последних</option>
    </select>
  </span>
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

try {
  const saved = localStorage.getItem('contextMode');
  if (saved === 'full' || saved === 'summary') modeSelect.value = saved;
} catch (e) {}
modeSelect.addEventListener('change', () => {
  try { localStorage.setItem('contextMode', modeSelect.value); } catch (e) {}
});

function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
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
  } else if (contextInfo) {
    metaText += ' · контекст: полная история';
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
      body: JSON.stringify({ message: text, context_mode: modeSelect.value })
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
    const r = await fetch('/api/history');
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
  if (!confirm('Очистить историю диалога?')) return;
  await fetch('/api/clear', { method: 'POST' });
  chatEl.innerHTML = '';
  addRow('assistant', 'История очищена. Начинаем заново!');
});

const expBtn = document.getElementById('experiment');
const exp = { running: false, lastKey: null, statusEl: null };

function setExpUI(disabled){
  input.disabled = disabled;
  sendBtn.disabled = disabled;
  expBtn.disabled = disabled;
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
    if (exp.lastKey === null) expDivider('Прогон 1 · режим «' + data.label + '»');
    else if (data.stage === 'dialog') expDivider('Прогон 2 · режим «' + data.label + '» · история очищена');
    else expDivider('Вопросы по деталям · режим «' + data.label + '»');
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

function renderReport(rep){
  const [a, b] = rep.runs;
  const div = document.createElement('div');
  div.className = 'report';

  let html = '<h3>Эксперимент: ' + esc(rep.title) + '</h3>';
  html += '<table><tr><th>Метрика</th><th>' + esc(a.label) + '</th><th>' + esc(b.label) + '</th></tr>';
  html += '<tr><td>Вопросов вспомнено полностью</td><td class="' + (a.remembered >= b.remembered ? 'good' : '') + '">' +
          a.remembered + ' / ' + a.questions_total + '</td><td class="' + (b.remembered >= a.remembered ? 'good' : '') + '">' +
          b.remembered + ' / ' + b.questions_total + '</td></tr>';
  html += '<tr><td>Деталей найдено</td><td class="' + (a.details_found >= b.details_found ? 'good' : '') + '">' +
          a.details_found + ' / ' + a.details_total + '</td><td class="' + (b.details_found >= a.details_found ? 'good' : '') + '">' +
          b.details_found + ' / ' + b.details_total + '</td></tr>';
  const ta = a.api.prompt + a.api.completion;
  const tb = b.api.prompt + b.api.completion;
  html += '<tr><td>Prompt + completion (API)</td><td class="' + (ta <= tb ? 'good' : '') + '">' + num(ta) + '</td>' +
          '<td class="' + (tb <= ta ? 'good' : '') + '">' + num(tb) + '</td></tr>';
  html += '<tr><td>&mdash; prompt / completion</td><td>' + num(a.api.prompt) + ' / ' + num(a.api.completion) + '</td>' +
          '<td>' + num(b.api.prompt) + ' / ' + num(b.api.completion) + '</td></tr>';
  html += '<tr><td>Суммаризатор (доп. вызовы)</td><td>&mdash;</td>' +
          '<td>' + b.summary_usage.calls + ' выз. · ' + num(b.summary_usage.prompt + b.summary_usage.completion) + ' ток.</td></tr>';
  if (a.api.cost || b.api.cost) {
    html += '<tr><td>Стоимость</td><td>' + a.api.cost.toFixed(4) + ' \u20bd</td><td>' + b.api.cost.toFixed(4) + ' \u20bd</td></tr>';
  }
  html += '<tr><td>Средний контекст вопроса (локально)</td><td>' + num(a.avg_context_tokens) + '</td>' +
          '<td>' + num(b.avg_context_tokens) + '</td></tr>';
  html += '</table>';

  const memWinner = a.details_found > b.details_found ? a.label : (b.details_found > a.details_found ? b.label : null);
  const tokWinner = ta < tb ? a.label : (tb < ta ? b.label : null);
  let verdict = '<div class="verdict">';
  verdict += memWinner
    ? 'Деталей лучше помнит режим «' + esc(memWinner) + '» (' + Math.max(a.details_found, b.details_found) + ' из ' + a.details_total + '). '
    : 'По деталям режимы равны (' + a.details_found + ' из ' + a.details_total + '). ';
  verdict += tokWinner
    ? 'Дешевле по токенам: «' + esc(tokWinner) + '» (' + num(Math.min(ta, tb)) + ' против ' + num(Math.max(ta, tb)) + ' ток. без учёта суммаризатора).'
    : 'Расход токенов одинаков.';
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
      if (!confirm('Эксперимент очистит текущую историю диалога и прогонит один и тот же диалог ' +
                   'в режимах «полная история» и «summary + 10 последних», затем задаст вопросы ' +
                   'по деталям (пара минут). Запустить?')) {
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

loadHistory();

document.getElementById('form').addEventListener('submit', (e) => { e.preventDefault(); run(); });
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, model=MODELS[0])


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": None, "error": "Запрос не может быть пустым."}), 400
    context_mode = data.get("context_mode")
    if context_mode not in ("full", "summary"):
        context_mode = "full"
    try:
        agent = Agent()
        result = agent.ask(message, context_mode)
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
    messages = [{"role": m["role"], "content": m["content"]}
                for m in store.load_messages()
                if m["role"] in ("user", "assistant") and m.get("content")]
    return jsonify({"messages": messages, "error": None})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    store.clear_history()
    return jsonify({"error": None})


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
