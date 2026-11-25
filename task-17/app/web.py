"""Minimal Flask web agent: user query -> LLM via API -> display answer."""
from flask import Flask, jsonify, render_template_string, request

from .agent import Agent
from .config import MODELS, MODEL

app = Flask(__name__)

HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Подключение MCP</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: radial-gradient(1200px 600px at 20% -10%, #1e293b, #0b1120 60%);
         color:#e2e8f0; height:100vh; display:flex; }
  .main { width:100%; max-width:860px; margin:0 auto; display:flex; flex-direction:column; height:100%; }
  header { padding:18px 20px; display:flex; align-items:center; gap:12px;
           border-bottom:1px solid rgba(255,255,255,.08); }
  header .logo { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
                 background:linear-gradient(135deg,#2563eb,#7c3aed); }
  header h1 { font-size:16px; margin:0; }
  header .model { font-size:12px; color:#94a3b8; background:rgba(255,255,255,.06);
                  border:1px solid rgba(255,255,255,.1); padding:5px 12px; border-radius:999px; }
  .hdr-btn { display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px;
             font-size:12px; cursor:pointer; color:#e2e8f0; background:rgba(255,255,255,.06);
             border:1px solid rgba(255,255,255,.1); transition:.15s; font-family:inherit; }
  .hdr-btn:hover { background:rgba(255,255,255,.12); }
  #mcp-btn { margin-left:auto; }
  .dot { width:8px; height:8px; border-radius:50%; background:#64748b; display:inline-block; flex:none; }
  .dot.ok { background:#22c55e; box-shadow:0 0 6px rgba(34,197,94,.7); }
  .dot.err { background:#ef4444; box-shadow:0 0 6px rgba(239,68,68,.7); }
  .overlay { position:fixed; inset:0; background:rgba(2,6,23,.72); backdrop-filter:blur(2px);
             display:none; align-items:center; justify-content:center; z-index:50; padding:24px; }
  .overlay.open { display:flex; }
  .panel { width:100%; max-width:640px; max-height:80vh; overflow-y:auto; background:#0f172a;
           border:1px solid rgba(255,255,255,.1); border-radius:16px; padding:20px 22px;
           box-shadow:0 10px 40px rgba(0,0,0,.5); }
  .panel-head { display:flex; align-items:center; margin-bottom:6px; }
  .panel-head h2 { font-size:15px; margin:0; color:#e2e8f0; }
  .panel-close { margin-left:auto; background:none; border:none; color:#94a3b8; font-size:16px;
                 cursor:pointer; }
  .panel-close:hover { color:#e2e8f0; }
  #faq-panel details { border:1px solid rgba(255,255,255,.08); border-radius:10px; padding:10px 14px;
                       margin-top:10px; background:rgba(255,255,255,.03); }
  #faq-panel summary { cursor:pointer; font-size:13.5px; color:#e2e8f0; font-weight:600; }
  #faq-panel p { margin:10px 0 2px; font-size:13px; line-height:1.6; color:#cbd5e1; }
  #faq-panel code { background:rgba(255,255,255,.07); padding:1px 5px; border-radius:5px;
                    font-size:12px; color:#7dd3fc; }
  #mcp-body { margin-top:14px; display:flex; flex-direction:column; gap:12px; }
  .mcp-card { border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px 14px;
              background:rgba(255,255,255,.03); }
  .mcp-head { display:flex; align-items:center; gap:9px; }
  .mcp-title { font-size:13.5px; font-weight:600; color:#e2e8f0; }
  .mcp-title .ver { font-weight:400; color:#94a3b8; font-size:11.5px; }
  .mcp-state { margin-left:auto; font-size:11.5px; flex:none; }
  .mcp-state.ok { color:#4ade80; }
  .mcp-state.err { color:#f87171; }
  .mcp-state.off { color:#94a3b8; }
  .mcp-meta { margin-top:6px; font-size:11.5px; color:#94a3b8; line-height:1.5; word-break:break-all; }
  .mcp-card > details { margin-top:8px; }
  .mcp-card summary, .tool summary { cursor:pointer; font-size:12.5px; color:#cbd5e1; user-select:none; }
  .mcp-card summary:hover, .tool summary:hover { color:#e2e8f0; }
  .tool { margin-top:8px; padding:8px 10px; border:1px solid rgba(255,255,255,.07);
          border-radius:9px; background:rgba(11,17,32,.6); }
  .tool-name { font-size:12.5px; font-weight:600; color:#7dd3fc;
               font-family:ui-monospace, Consolas, monospace; }
  .tool-desc { margin-top:3px; font-size:12px; color:#cbd5e1; line-height:1.5; }
  .tool details { margin-top:6px; }
  .tool pre { margin:6px 0 0; padding:8px; max-height:220px; overflow:auto; background:#0b1120;
              border:1px solid rgba(255,255,255,.08); border-radius:8px; font-size:11px;
              color:#a5b4fc; white-space:pre-wrap; word-break:break-word; }
  .mcp-empty { margin:8px 0 0; font-size:12px; color:#94a3b8; font-style:italic; }
  .mcp-err { margin-top:6px; font-size:12px; color:#fca5a5; white-space:pre-wrap;
             word-break:break-word; }
  #mcp-refresh { margin-top:2px; align-self:flex-start; }
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
  .dbg { margin-top:8px; font-size:11.5px; color:#94a3b8; }
  .dbg summary { cursor:pointer; user-select:none; padding:5px 10px; border-radius:8px;
                 background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08);
                 display:inline-block; }
  .dbg summary:hover { background:rgba(255,255,255,.09); color:#e2e8f0; }
  .dbg pre { margin:8px 0 0; padding:10px; max-height:320px; overflow:auto;
             background:#0b1120; border:1px solid rgba(255,255,255,.08); border-radius:9px;
             font-size:11px; line-height:1.5; white-space:pre-wrap; word-wrap:break-word; color:#7dd3fc; }
  .dbg .trace { margin:6px 0 0; padding:8px 10px; background:#0b1120;
                border:1px solid rgba(255,255,255,.08); border-radius:9px;
                white-space:pre-wrap; word-wrap:break-word; color:#a5b4fc; }
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
  <h1>Подключение MCP</h1>
  <button id="mcp-btn" class="hdr-btn" type="button" title="MCP-серверы и их инструменты">
    <span class="dot" id="mcp-dot"></span>MCP
  </button>
  <button id="faq-btn" class="hdr-btn" type="button">Ответы на вопросы</button>
  <span class="model">{{ model }}</span>
</header>
<div id="chat">
  <div class="row assistant"><div class="bubble">Привет! Агент отвечает через LLM и вызывает инструменты: локальные (День 6) и из MCP-серверов — курсы ЦБ РФ и T-Invest API (Т-Банк, песочница).</div></div>
</div>
<form id="form">
  <input id="msg" placeholder="Введите ваш запрос..." autocomplete="off">
  <button id="send" type="submit">Отправить</button>
</form>
</div>

<div id="mcp-overlay" class="overlay">
  <div class="panel">
    <div class="panel-head">
      <h2>MCP-серверы и инструменты</h2>
      <button id="mcp-close" class="panel-close" type="button">✕</button>
    </div>
    <div id="mcp-body"><p class="mcp-empty">Загрузка…</p></div>
    <button id="mcp-refresh" class="hdr-btn" type="button">Обновить</button>
  </div>
</div>

<div id="faq-overlay" class="overlay">
  <div id="faq-panel" class="panel">
    <div class="panel-head">
      <h2>Ответы на вопросы</h2>
      <button id="faq-close" class="panel-close" type="button">✕</button>
    </div>
    <details>
      <summary>Как модель различает локальный инструмент и MCP-инструмент?</summary>
      <p>Никак — и не обязана. Для модели все инструменты выглядят одинаково: просто список схем
      (имя, описание, параметры) в формате OpenAI, переданный в поле <code>tools</code> запроса.
      Локальные схемы мы описали руками, MCP-схемы получили из <code>session.list_tools()</code>
      и сконвертировали в тот же формат; <code>get_all_schemas()</code> склеивает оба списка в один.
      Различие живёт только в нашем коде: когда модель просит вызвать инструмент,
      <code>run_tool(name, args)</code> ищет имя в локальном словаре <code>TOOL_FUNCS</code> —
      если нашёл, вызывает python-функцию, если нет, отправляет аргументы в MCP-сессию
      (<code>bridge.call_tool</code>).</p>
    </details>
    <details>
      <summary>Что именно в вашем коде изменилось при переходе на MCP?</summary>
      <p>Сам цикл агента (запрос → выбор инструмента моделью → результат → ответ) не изменился
      вообще. Изменилось окружение:</p>
      <p>• <b>новый файл <code>mcp_bridge.py</code></b> — держит постоянную сессию с MCP-сервером
      (subprocess по stdio в фоновом потоке), отдаёт его схемы в OpenAI-формате и исполняет вызовы
      <code>call_tool</code>, превращая ответ MCP в строку для модели;<br>
      • <code>tools.py</code> — вместо одного захардкоженного списка схем и словаря функций теперь
      <code>get_all_schemas()</code> (локальные + MCP) и маршрутизация в <code>run_tool</code>;<br>
      • <code>agent.py</code> — одна строка: в запрос передаётся <code>get_all_schemas()</code>
      вместо <code>TOOL_SCHEMAS</code>;<br>
      • <code>web.py</code> — MCP-сессия поднимается один раз при старте приложения.</p>
    </details>
    <details>
      <summary>Зачем тогда нужен MCP?</summary>
      <p>Чтобы не писать и не поддерживать код инструментов самому. MCP-сервер — готовый внешний
      компонент со своими тестами, кэшем и обновлениями: здесь это актуальные курсы ЦБ РФ вместо
      захардкоженных <code>RATES</code> из Дня 6. Протокол стандартный, поэтому тот же сервер без
      переделок подключается к любому MCP-клиенту (Claude, Cursor, свой агент), а наш агент может
      работать с любым другим MCP-сервером — достаточно поменять одну команду запуска в
      <code>mcp_bridge.py</code>. Получается интеграция «вбок»: источник данных развивается
      отдельно, агент лишь читает список инструментов по протоколу. Свой сервер
      (День 17) подключается так же: достаточно добавить команду запуска в
      <code>SERVER_REGISTRY</code> в <code>mcp_bridge.py</code>.</p>
    </details>
  </div>
</div>
<script>
const input = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const chatEl = document.getElementById('chat');
const modelBadge = document.querySelector('.model');

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

async function run(){
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addRow('user', text);
  showTyping();
  sendBtn.disabled = true;
  const bubble = addRow('assistant', '');
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await r.json();
    hideTyping();
    if (data.error) {
      bubble.className = 'bubble error';
      bubble.textContent = 'Ошибка: ' + data.error;
    } else {
      bubble.textContent = data.reply || '(пустой ответ)';
      if (data.tool_calls_used > 0) {
        const meta = document.createElement('div');
        meta.style.cssText = 'margin-top:8px;font-size:11px;color:#64748b;';
        meta.textContent = 'Инструментов вызвано: ' + data.tool_calls_used + ' · шагов: ' + data.steps;
        bubble.appendChild(meta);
      }
      if (data.tool_trace && data.tool_trace.length) {
        const d = document.createElement('details');
        d.className = 'dbg';
        const s = document.createElement('summary');
        s.textContent = 'Вызовы инструментов (' + data.tool_trace.length + ')';
        d.appendChild(s);
        data.tool_trace.forEach(t => {
          const p = document.createElement('div');
          p.className = 'trace';
          p.textContent = t.name + '(' + JSON.stringify(t.args) + ')\n→ ' + t.result;
          d.appendChild(p);
        });
        bubble.appendChild(d);
      }
      if (data.requests && data.requests.length) {
        data.requests.forEach((req, i) => {
          const d = document.createElement('details');
          d.className = 'dbg';
          const s = document.createElement('summary');
          s.textContent = 'Запрос #' + (i + 1) + ' к модели: ' + req.messages.length
            + ' сообщ. · ' + (req.tools ? req.tools.length : 0) + ' инструментов';
          const pre = document.createElement('pre');
          pre.textContent = JSON.stringify(req, null, 2);
          d.appendChild(s);
          d.appendChild(pre);
          bubble.appendChild(d);
        });
      }
      if (data.model) modelBadge.textContent = data.model;
    }
  } catch (err) {
    hideTyping();
    bubble.className = 'bubble error';
    bubble.textContent = 'Ошибка запроса: ' + err;
  } finally {
    sendBtn.disabled = false;
    input.focus();
    scroll();
  }
}

document.getElementById('form').addEventListener('submit', (e) => { e.preventDefault(); run(); });

const faqOverlay = document.getElementById('faq-overlay');
document.getElementById('faq-btn').addEventListener('click', () => faqOverlay.classList.add('open'));
document.getElementById('faq-close').addEventListener('click', () => faqOverlay.classList.remove('open'));
faqOverlay.addEventListener('click', (e) => { if (e.target === faqOverlay) faqOverlay.classList.remove('open'); });

const mcpOverlay = document.getElementById('mcp-overlay');
const mcpDot = document.getElementById('mcp-dot');
document.getElementById('mcp-btn').addEventListener('click', () => { mcpOverlay.classList.add('open'); loadMcp(); });
document.getElementById('mcp-close').addEventListener('click', () => mcpOverlay.classList.remove('open'));
document.getElementById('mcp-refresh').addEventListener('click', loadMcp);
mcpOverlay.addEventListener('click', (e) => { if (e.target === mcpOverlay) mcpOverlay.classList.remove('open'); });

function addToolList(tools, container){
  if (!tools || !tools.length){
    const p = document.createElement('p'); p.className = 'mcp-empty'; p.textContent = 'Нет инструментов';
    container.appendChild(p); return;
  }
  tools.forEach(t => {
    const d = document.createElement('div'); d.className = 'tool';
    const n = document.createElement('div'); n.className = 'tool-name'; n.textContent = t.name; d.appendChild(n);
    if (t.description){
      const de = document.createElement('div'); de.className = 'tool-desc'; de.textContent = t.description; d.appendChild(de);
    }
    const props = t.parameters && t.parameters.properties ? Object.keys(t.parameters.properties) : [];
    const pd = document.createElement('details');
    const ps = document.createElement('summary');
    ps.textContent = props.length ? 'Параметры (' + props.length + ')' : 'Параметры';
    pd.appendChild(ps);
    const pre = document.createElement('pre');
    pre.textContent = props.length ? JSON.stringify(t.parameters, null, 2) : '(нет аргументов)';
    pd.appendChild(pre);
    d.appendChild(pd);
    container.appendChild(d);
  });
}

function addServerCard(s){
  const card = document.createElement('div'); card.className = 'mcp-card';
  const head = document.createElement('div'); head.className = 'mcp-head';
  const dot = document.createElement('span'); dot.className = 'dot ' + (s.connected ? 'ok' : 'err');
  head.appendChild(dot);
  const title = document.createElement('div'); title.className = 'mcp-title';
  title.textContent = s.server_name || 'MCP-сервер';
  if (s.server_version){
    const v = document.createElement('span'); v.className = 'ver'; v.textContent = ' v' + s.server_version;
    title.appendChild(v);
  }
  head.appendChild(title);
  const state = document.createElement('div'); state.className = 'mcp-state ' + (s.connected ? 'ok' : 'err');
  state.textContent = s.connected ? 'подключено' : 'нет подключения';
  head.appendChild(state);
  card.appendChild(head);
  const meta = document.createElement('div'); meta.className = 'mcp-meta';
  const parts = ['транспорт: stdio'];
  if (s.protocol_version) parts.push('протокол ' + s.protocol_version);
  if (s.command) parts.push('команда: ' + s.command);
  meta.textContent = parts.join(' · ');
  card.appendChild(meta);
  if (s.error){
    const e = document.createElement('div'); e.className = 'mcp-err';
    e.textContent = 'Ошибка подключения: ' + s.error;
    card.appendChild(e);
  }
  const det = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = 'Инструменты (' + (s.tools ? s.tools.length : 0) + ')';
  det.appendChild(sum);
  addToolList(s.tools, det);
  card.appendChild(det);
  return card;
}

function addLocalCard(tools){
  const card = document.createElement('div'); card.className = 'mcp-card';
  const head = document.createElement('div'); head.className = 'mcp-head';
  const dot = document.createElement('span'); dot.className = 'dot';
  head.appendChild(dot);
  const title = document.createElement('div'); title.className = 'mcp-title';
  title.textContent = 'Локальные инструменты';
  head.appendChild(title);
  const state = document.createElement('div'); state.className = 'mcp-state off';
  state.textContent = 'встроены в агент';
  head.appendChild(state);
  card.appendChild(head);
  const meta = document.createElement('div'); meta.className = 'mcp-meta';
  meta.textContent = 'Python-функции (День 6), подключены напрямую, без MCP';
  card.appendChild(meta);
  const det = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = 'Инструменты (' + (tools ? tools.length : 0) + ')';
  det.appendChild(sum);
  addToolList(tools, det);
  card.appendChild(det);
  return card;
}

async function loadMcp(){
  const body = document.getElementById('mcp-body');
  body.innerHTML = '<p class="mcp-empty">Загрузка…</p>';
  try {
    const r = await fetch('/api/mcp');
    const data = await r.json();
    body.innerHTML = '';
    (data.servers || []).forEach(s => body.appendChild(addServerCard(s)));
    body.appendChild(addLocalCard(data.local_tools));
    const ok = (data.servers || []).some(s => s.connected);
    mcpDot.className = 'dot ' + (ok ? 'ok' : 'err');
  } catch(err){
    body.innerHTML = '<p class="mcp-err">Не удалось загрузить статус: ' + esc(err) + '</p>';
    mcpDot.className = 'dot err';
  }
}

async function refreshMcpDot(){
  try {
    const r = await fetch('/api/mcp');
    const data = await r.json();
    const ok = (data.servers || []).some(s => s.connected);
    const any = (data.servers || []).length > 0;
    mcpDot.className = 'dot ' + (ok ? 'ok' : (any ? 'err' : ''));
  } catch(e){ mcpDot.className = 'dot'; }
}

refreshMcpDot();

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
    try:
        agent = Agent()
        result = agent.ask(message)
        return jsonify({
            "reply": result["reply"],
            "tool_calls_used": result["tool_calls_used"],
            "steps": result["steps"],
            "requests": result.get("requests", []),
            "tool_trace": result.get("tool_trace", []),
            "error": None,
            "model": MODEL,
        })
    except Exception as exc:
        return jsonify({"reply": None, "error": str(exc), "model": MODEL})


@app.route("/api/mcp")
def api_mcp():
    """MCP servers (status + tools) and local tools for the UI panel."""
    from .mcp_bridge import get_bridge
    from .tools import TOOL_SCHEMAS

    local_tools = [
        {
            "name": s["function"]["name"],
            "description": s["function"]["description"],
            "parameters": s["function"]["parameters"],
        }
        for s in TOOL_SCHEMAS
    ]
    servers = get_bridge().get_statuses()
    return jsonify({"servers": servers, "local_tools": local_tools})


def main() -> None:
    from .mcp_bridge import close_bridge, get_bridge
    try:
        hub = get_bridge()
        print(f"MCP connected, tools: {', '.join(hub.tool_names)}")
    except Exception as exc:
        print(f"MCP unavailable, local tools only: {exc}")
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        close_bridge()


if __name__ == "__main__":
    main()
