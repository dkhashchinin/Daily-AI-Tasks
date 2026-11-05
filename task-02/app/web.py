"""Minimal Flask web interface for chatting with the LLM (streaming)."""
import ast
import json

from flask import Flask, Response, jsonify, render_template_string, request

from .config import MODEL
from .llm import chat, stream_chat
from .metrics import last

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
  .shell { margin:0 auto; width:100%; max-width:920px; height:100%; display:flex; flex-direction:column; }
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
  .check-btn { margin-top:10px; padding:6px 12px; border-radius:8px; font-size:12px; cursor:pointer;
               background:rgba(255,255,255,.05); color:#94a3b8; border:1px solid rgba(255,255,255,.14); transition:.15s; }
  .check-btn:hover { background:rgba(255,255,255,.12); color:#e2e8f0; border-color:rgba(255,255,255,.25); }
  .check-btn:disabled { opacity:.5; cursor:not-allowed; }
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
  .settings { display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:12px 20px 8px;
              background:rgba(15,23,42,.5); border-top:1px solid rgba(255,255,255,.08); backdrop-filter:blur(6px); }
  .settings label { font-size:12px; color:#94a3b8; white-space:nowrap; }
  .settings input, .settings select { flex:1; max-width:320px; padding:9px 12px; border-radius:10px; font-size:13px;
               background:rgba(255,255,255,.04); color:#e2e8f0; border:1px solid rgba(255,255,255,.12); outline:none; transition:.15s; }
  .settings input:focus, .settings select:focus { border-color:#2563eb; background:rgba(255,255,255,.07); }
  .settings select option { background:#0f172a; color:#e2e8f0; }
  .settings input.num { max-width:90px; flex:0 0 90px; }
  .settings input.wide, .settings select { flex:1 1 300px; max-width:none; }
  form { display:flex; gap:10px; padding:14px 20px 18px; background:rgba(15,23,42,.5);
         border-top:1px solid rgba(255,255,255,.08); backdrop-filter:blur(6px); }
  #msg { flex:1; padding:13px 15px; border-radius:13px; border:1px solid rgba(255,255,255,.12);
         background:rgba(255,255,255,.05); color:#e2e8f0; font-size:14.5px; outline:none; transition:.15s; }
  #msg:focus { border-color:#2563eb; background:rgba(255,255,255,.07); }
  #send { padding:12px 24px; border-radius:13px; border:none; cursor:pointer; font-size:14.5px; font-weight:600;
          color:#fff; background:linear-gradient(135deg,#2563eb,#4f46e5); box-shadow:0 4px 14px rgba(37,99,235,.35);
          transition:.15s; }
  #send:hover { filter:brightness(1.08); }
  #send:active { transform:translateY(1px); }
  #send:disabled { opacity:.5; cursor:not-allowed; filter:none; transform:none; box-shadow:none; }
</style>
</head>
<body>
<div class="shell">
<header>
  <span class="logo">
    <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>
    </svg>
  </span>
  <h1>Чат с LLM</h1>
  <span class="model">{{ model }}</span>
</header>
<div id="chat">
  <div class="row assistant"><div class="bubble">Привет! Задай вопрос и при желании настрой параметры выше.</div></div>
</div>
<div class="settings">
  <label for="max_tokens">max_tokens</label>
  <input id="max_tokens" class="num" type="number" min="1" max="8192" placeholder="1024">
  <label for="stop">stop_sequence</label>
  <input id="stop" placeholder="стоп-фразы">
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
<form id="chat-form">
  <input id="msg" placeholder="Введите сообщение..." autocomplete="off">
  <button id="send" type="submit">Отправить</button>
</form>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('msg');
const maxTokensInput = document.getElementById('max_tokens');
const stopInput = document.getElementById('stop');
const formatInput = document.getElementById('response_format');
const sendBtn = document.getElementById('send');
let busy = false;

function collectParams(){
  const mt = parseInt(maxTokensInput.value, 10);
  const body = {};
  if (mt && mt > 0) body.max_tokens = mt;
  const stops = stopInput.value.split(',').map(s => s.trim()).filter(Boolean);
  if (stops.length) body.stop = stops.length === 1 ? stops[0] : stops;
  const fmt = formatInput.value.trim();
  if (fmt) body.response_format = fmt;
  return body;
}

function scroll(){ chat.scrollTop = chat.scrollHeight; }

function addUser(text){
  const row = document.createElement('div'); row.className = 'row user';
  const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
  row.appendChild(b); chat.appendChild(row); scroll();
}

function addAssistant(){
  const row = document.createElement('div'); row.className = 'row assistant';
  const b = document.createElement('div'); b.className = 'bubble';
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

document.getElementById('chat-form').addEventListener('submit', async function (e) {
  e.preventDefault();
  if (busy) return;
  const m = input.value.trim();
  if (!m) return;
  input.value = '';
  addUser(m);
  busy = true; sendBtn.disabled = true; showTyping();
  const asst = addAssistant();
  let gotContent = false;
  const payload = Object.assign({ message: m }, collectParams());
  try {
    const r = await fetch('/chat/stream', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
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
            if (!gotContent) hideTyping();
            if (d.stopped_by_stop) showNote(asst, 'Ответ остановлен стоп-последовательностью.', 'warn');
            else if (d.empty) showNote(asst, 'Модель не вернула ответ.', 'warn');
            else if (d.finish_reason && d.finish_reason !== 'stop') showNote(asst, warnText(d.finish_reason), 'warn');
            const s = fmt(d.metrics);
            if (s) { asst.meta.textContent = 'metrics: ' + s; asst.meta.style.display = 'block'; }
          }
        }
      }
    }
  } catch (err) {
    hideTyping();
    showNote(asst, 'Ошибка запроса: ' + err, 'error');
  } finally {
    busy = false; sendBtn.disabled = false; input.focus();
  }
});
</script>
</body>
</html>"""


@app.route("/")
def index():
    response = app.make_response(render_template_string(HTML, model=MODEL))
    response.headers["Cache-Control"] = "no-store"
    return response


def _load_params(data: dict) -> tuple:
    max_tokens = data.get("max_tokens")
    stop = data.get("stop")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                max_tokens = None
        except (TypeError, ValueError):
            max_tokens = None
    return max_tokens, stop


def _build_messages(data: dict) -> list[dict]:
    messages = []
    fmt = (data.get("response_format") or "").strip()
    if fmt:
        messages.append({"role": "system", "content": f"Формат ответа: {fmt}"})
    messages.append({"role": "user", "content": data["message"]})
    return messages


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    max_tokens, stop = _load_params(data)
    try:
        reply = chat(_build_messages(data), max_tokens=max_tokens, stop=stop)
        return jsonify({"reply": reply, "metrics": last(), "error": None})
    except Exception as exc:
        return jsonify({"reply": None, "metrics": last(), "error": str(exc)})


@app.route("/chat/stream", methods=["POST"])
def chat_stream_endpoint():
    data = request.get_json()
    max_tokens, stop = _load_params(data)
    messages = _build_messages(data)

    stop_active = bool(stop)

    def generate():
        produced = False
        finish_reason = None
        try:
            for kind, text in stream_chat(messages, max_tokens=max_tokens, stop=stop):
                if kind == "content":
                    produced = True
                elif kind == "finish":
                    finish_reason = text
                    continue
                yield f"data: {json.dumps({'kind': kind, 'text': text}, ensure_ascii=False)}\n\n"
            stopped_by_stop = stop_active and finish_reason == "stop"
            payload = {"kind": "done", "metrics": last(), "finish_reason": finish_reason,
                       "empty": not produced, "stopped_by_stop": stopped_by_stop}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error = {"kind": "error", "message": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            finish = {"kind": "done", "metrics": None, "finish_reason": None,
                      "empty": not produced, "stopped_by_stop": False}
            yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"

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
