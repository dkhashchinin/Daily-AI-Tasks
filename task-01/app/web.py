"""Minimal Flask web interface for chatting with the LLM."""
from flask import Flask, jsonify, request, render_template_string

from .config import MODEL
from .llm import chat

app = Flask(__name__)

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>LLM Chat</title></head>
<body><h3>Чат с LLM ({{ model }})</h3>
<div id="log" style="white-space:pre-wrap"></div>
<input id="msg" placeholder="Введите сообщение" style="width:60%">
<button onclick="send()">Отправить</button>
<script>
async function send() {
  const m = document.getElementById('msg').value.trim();
  if (!m) return;
  document.getElementById('log').textContent += '\\nВы: ' + m;
  document.getElementById('msg').value = '';
  const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:m})});
  const d = await r.json();
  document.getElementById('log').textContent += '\\nLLM: ' + d.reply;
}
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(HTML, model=MODEL)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    reply = chat([{"role": "user", "content": data["message"]}])
    return jsonify({"reply": reply})


def main() -> None:
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
