"""Minimal Flask web agent: user query -> LLM via API -> display answer."""
from flask import Flask, jsonify, render_template_string, request

from .agent import Agent
from . import store
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
  header { padding:18px 20px; display:flex; align-items:center; gap:12px;
           border-bottom:1px solid rgba(255,255,255,.08); }
  header .logo { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
                 background:linear-gradient(135deg,#2563eb,#7c3aed); }
  header h1 { font-size:16px; margin:0; }
  header .model { margin-left:auto; font-size:12px; color:#94a3b8; background:rgba(255,255,255,.06);
                  border:1px solid rgba(255,255,255,.1); padding:5px 12px; border-radius:999px; }
  #clear { font-size:12px; color:#cbd5e1; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
           padding:5px 12px; border-radius:999px; cursor:pointer; transition:.15s; }
  #clear:hover { background:rgba(239,68,68,.15); color:#fca5a5; border-color:rgba(239,68,68,.4); }
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
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await r.json();
    hideTyping();
    if (data.error) {
      addRow('assistant', 'Ошибка: ' + data.error, true);
    } else {
      const bubble = addRow('assistant', data.reply || '(пустой ответ)');
      let metaText = null;
      if (data.tokens) {
        const t = data.tokens;
        metaText = 'Инструментов вызвано: ' + data.tool_calls_used +
          ' · шагов: ' + data.steps +
          ' · токены: запрос ' + t.request + ', история ' + t.history +
          ', ответ ' + t.response + ' (prompt ' + t.prompt + ', completion ' + t.completion + ')';
        if (typeof t.cost === 'number') metaText += ' · стоимость ' + t.cost.toFixed(4) + ' \u20bd';
        if (typeof t.balance === 'number') metaText += ' · баланс ' + t.balance.toFixed(2) + ' \u20bd';
      } else if (data.tool_calls_used > 0) {
        metaText = 'Инструментов вызвано: ' + data.tool_calls_used + ' · шагов: ' + data.steps;
      }
      if (metaText) {
        const meta = document.createElement('div');
        meta.style.cssText = 'margin-top:8px;font-size:11px;color:#64748b;';
        meta.textContent = metaText;
        bubble.appendChild(meta);
      }
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
    try:
        agent = Agent()
        result = agent.ask(message)
        return jsonify({
            "reply": result["reply"],
            "tool_calls_used": result["tool_calls_used"],
            "steps": result["steps"],
            "tokens": result.get("tokens"),
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


def main() -> None:
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
