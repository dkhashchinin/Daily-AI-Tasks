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
  #sched-refresh { margin-top:2px; align-self:flex-start; }
  .sched-controls { display:flex; flex-direction:column; gap:8px; margin-top:14px;
                    padding:12px 14px; border:1px solid rgba(255,255,255,.08); border-radius:12px;
                    background:rgba(255,255,255,.03); }
  .sched-controls .ctl { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .sched-controls .ctl > label { flex:0 0 130px; font-size:12.5px; color:#cbd5e1; font-weight:600; }
  .sched-controls .ctl > input { width:120px; padding:6px 9px; border-radius:8px; font-size:12.5px;
                          border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.05);
                          color:#e2e8f0; outline:none; font-family:inherit; }
  .sched-controls .ctl > input:focus { border-color:#2563eb; }
  .sched-controls .ctl > input:disabled, .sched-controls .ctl > .hint.dim { opacity:.45; }
  .sched-controls .hint { font-size:11px; color:#64748b; flex:1 1 auto; }
  .sched-controls .ctl.actions { margin-top:4px; }
  #sched-start { color:#4ade80; border-color:rgba(74,222,128,.4); }
  #sched-start:hover:not(:disabled) { background:rgba(74,222,128,.12); }
  #sched-stop { color:#f87171; border-color:rgba(248,113,113,.4); }
  #sched-stop:hover:not(:disabled) { background:rgba(248,113,113,.12); }
  #sched-start:disabled, #sched-stop:disabled { opacity:.4; cursor:not-allowed; }
  .rep-head { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .rep-ctl { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
  #report-depth { padding:6px 8px; border-radius:9px; background:#1e293b;
                  border:1px solid rgba(255,255,255,.12); color:#e2e8f0;
                  font-family:inherit; font-size:12px; cursor:pointer; }
  #report-depth option { background:#1e293b; color:#e2e8f0; }
  #report-build { color:#4ade80; border-color:rgba(74,222,128,.4); font-size:11.5px; padding:4px 10px; }
  #report-build:hover:not(:disabled) { background:rgba(74,222,128,.12); }
  #report-delete { color:#fbbf24; border-color:rgba(251,191,36,.4); font-size:11.5px; padding:4px 10px; }
  #report-delete:hover:not(:disabled) { background:rgba(251,191,36,.12); }
  #report-build:disabled, #report-delete:disabled { opacity:.4; cursor:not-allowed; }
  #reports-clear, #snaps-clear { color:#f87171; border-color:rgba(248,113,113,.4); font-size:11.5px; padding:4px 10px; }
  #reports-clear:hover:not(:disabled), #snaps-clear:hover:not(:disabled) { background:rgba(248,113,113,.12); }
  #reports-clear:disabled, #snaps-clear:disabled { opacity:.4; cursor:not-allowed; }
  .sched-status { display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }
  .stat { flex:1 1 150px; border:1px solid rgba(255,255,255,.08); border-radius:12px;
          padding:10px 12px; background:rgba(255,255,255,.03); }
  .stat .k { font-size:11px; color:#94a3b8; }
  .stat .v { margin-top:4px; font-size:13.5px; font-weight:600; color:#e2e8f0; word-break:break-word; }
  .stat .v.ok { color:#4ade80; }
  .stat .v.err { color:#f87171; }
  .sched-sec { margin-top:16px; }
  .sched-sec h3 { font-size:13px; color:#cbd5e1; margin:0 0 8px; }
  table.snaps { width:100%; border-collapse:collapse; font-size:12px; }
  table.snaps th, table.snaps td { padding:6px 8px; border-bottom:1px solid rgba(255,255,255,.06);
                                   text-align:left; font-family:ui-monospace, Consolas, monospace; }
  table.snaps th { color:#94a3b8; font-weight:600; }
  table.snaps td.num, table.snaps th.num { text-align:right; }
  #report-select { width:100%; margin-bottom:8px; padding:8px 30px 8px 10px; border-radius:9px;
                   background:#1e293b url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%2394a3b8' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E") no-repeat right 10px center;
                   border:1px solid rgba(255,255,255,.12);
                   color:#e2e8f0; font-family:inherit; font-size:12.5px; cursor:pointer;
                   appearance:none; -webkit-appearance:none; }
  #report-select:focus { border-color:#2563eb; outline:none; }
  /* Выпадающий список нативный: без явных цветов он белый со светлым текстом */
  #report-select option { background:#1e293b; color:#e2e8f0; }
  #report-view { margin:0; padding:10px; max-height:260px; overflow:auto; background:#0b1120;
                 border:1px solid rgba(255,255,255,.08); border-radius:9px; font-size:11.5px;
                 color:#a5b4fc; white-space:pre-wrap; word-break:break-word; }
  .sched-note { margin-top:12px; font-size:11.5px; color:#94a3b8; line-height:1.5; }
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
  <button id="sched-btn" class="hdr-btn" type="button" title="Планировщик фоновых задач">
    <span class="dot" id="sched-dot"></span>Планировщик
  </button>
  <button id="faq-btn" class="hdr-btn" type="button">Ответы на вопросы</button>
  <span class="model">{{ model }}</span>
</header>
<div id="chat">
  <div class="row assistant"><div class="bubble">Привет! Агент отвечает через LLM и вызывает инструменты: локальные, из MCP-серверов — курсы ЦБ РФ и T-Invest API (Т-Банк, песочница), плюс rates-store: получение данных, сохранение в SQLite и чтение накопленного. Кнопка «Планировщик» показывает фоновые запуски и отчёты за период.</div></div>
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

<div id="sched-overlay" class="overlay">
  <div class="panel">
    <div class="panel-head">
      <h2>Планировщик фоновых задач</h2>
      <button id="sched-close" class="panel-close" type="button">✕</button>
    </div>
    <div class="sched-controls">
      <div class="ctl">
        <label for="sched-currencies">Что собирать</label>
        <input id="sched-currencies" type="text" value="USD, EUR, CNY, GBP"
               placeholder="USD, EUR, JPY" title="Коды валют справочника ЦБ через запятую, до 8">
        <span class="hint">коды валют ЦБ через запятую, до 8 — например USD, EUR, JPY, KZT</span>
      </div>
      <div class="ctl">
        <label for="sched-interval">Сбор каждые, сек</label>
        <input id="sched-interval" type="number" value="5" min="1" step="1"
               title="Как часто планировщик вызывает fetch_rates и сохраняет снимок">
        <span class="hint">как часто ходить в ЦБ и сохранять снимок в SQLite</span>
      </div>
      <div class="ctl">
        <label for="sched-report">Отчёт каждые, сек</label>
        <input id="sched-report" type="number" value="15" min="1" step="1"
               title="Как часто пересобирать агрегированный отчёт из накопленных снимков">
        <span class="hint">как часто собирать сводку из уже накопленных снимков</span>
      </div>
      <div class="ctl">
        <label for="sched-hours">Глубина отчёта, ч</label>
        <input id="sched-hours" type="number" value="24" min="0" step="1"
               title="Сколько часов накопленных данных попадает в отчёт; 0 — вся история">
        <span class="hint">сколько часов назад смотреть; 0 — вся накопленная история</span>
      </div>
      <div class="ctl actions">
        <button id="sched-start" class="hdr-btn" type="button">Запустить</button>
        <button id="sched-stop" class="hdr-btn" type="button" disabled>Остановить</button>
        <span class="hint">первый сбор — сразу после запуска; параметры применяются на старте</span>
      </div>
    </div>
    <div id="sched-body"><p class="mcp-empty">Загрузка…</p></div>
    <button id="sched-refresh" class="hdr-btn" type="button">Обновить</button>
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
    <details>
      <summary>Как устроен планировщик и как он дёргает MCP-инструменты?</summary>
      <p>Это фоновый процесс-планировщик. Запустить его можно двумя способами: кнопкой
      «Запустить» в панели «Планировщик» (фоновый поток веб-приложения, тот же код) или
      в терминале — <code>python scheduler.py</code>. Перед запуском задаются параметры —
      что собирать (валюты ЦБ через запятую, до 8 — <code>SCHED_CURRENCIES</code>), период сбора
      (<code>SCHED_INTERVAL_SEC</code>), период отчёта (<code>SCHED_REPORT_INTERVAL_SEC</code>) и
      глубина отчёта в часах (<code>SCHED_REPORT_HOURS</code>, 0 — вся накопленная история).
      Внутри — цикл по расписанию: каждые
      <code>SCHED_INTERVAL_SEC</code> секунд он вызывает
      MCP-инструмент <code>fetch_rates</code> (получение данных), его результат передаёт в
      <code>save_snapshot</code> (сохранение: SQLite), а каждые <code>SCHED_REPORT_INTERVAL_SEC</code>
      секунд вызывает <code>read_snapshots</code> (чтение накопленного за период) и собирает
      агрегированный отчёт в <code>data/reports/</code>. Все три инструмента живут в своём
      MCP-сервере <code>mcp_rates_store</code> — планировщик общается с ним по stdio через тот же
      <code>McpBridge</code>, что и агент, то есть «дёргает инструменты именно через MCP».
      Каждый цикл пишет журнал (<code>data/scheduler.log</code>) и статус для этой панели
      (<code>data/scheduler_status.json</code>). Режим <code>--once</code> позволяет запускать
      сбор внешним планировщиком ОС (cron / Task Scheduler) — расписание тогда обеспечивает ОС.</p>
    </details>
    <details>
      <summary>Как убедиться, что данные накапливаются, а отчёт считается по всем запускам?</summary>
      <p>Данные хранятся в SQLite (<code>data/rates.db</code>): один запуск планировщика = одна строка,
      поэтому накопление переживает и перезапуски процесса, и перезагрузку машины. Отчёт строится
      только из строк, которые вернул <code>read_snapshots</code> за период: в нём видно
      количество снимков, первую и последнюю точку, min/max/avg по каждой валюте — то есть
      агрегат по всем запускам, а не только по последнему. Быстрая проверка:
      <code>python scheduler.py --interval 3 --report-interval 7 --max-loops 5</code> — пять запусков
      подряд без участия человека, промежуточный отчёт по уже накопленным снимкам и финальный отчёт;
      <code>python scheduler.py --report --hours 0</code> — отчёт по всей накопленной истории
      (то же самое — кнопкой «Собрать отчёт» в этой панели: планировщик может быть даже
      остановлен, отчёт строится из уже накопленных снимков с выбранной глубиной).
      Панель «Планировщик» показывает те же данные из браузера.</p>
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
  meta.textContent = 'Python-функции, подключены напрямую, без MCP';
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

const schedOverlay = document.getElementById('sched-overlay');
const schedDot = document.getElementById('sched-dot');
// Какой отчёт выбран в списке и что уже скачано — чтобы автообновление
// панели (раз в 5 с) не сбрасывало просмотр на самый свежий отчёт.
let schedSelectedReport = null;
const reportCache = new Map();
document.getElementById('sched-btn').addEventListener('click', () => { schedOverlay.classList.add('open'); loadSched(); });
document.getElementById('sched-close').addEventListener('click', () => schedOverlay.classList.remove('open'));
document.getElementById('sched-refresh').addEventListener('click', loadSched);
schedOverlay.addEventListener('click', (e) => { if (e.target === schedOverlay) schedOverlay.classList.remove('open'); });

document.getElementById('sched-start').addEventListener('click', async () => {
  const body = {
    currencies: document.getElementById('sched-currencies').value,
    interval: parseFloat(document.getElementById('sched-interval').value) || 5,
    report_interval: parseFloat(document.getElementById('sched-report').value) || 15,
    hours: parseFloat(document.getElementById('sched-hours').value) || 0
  };
  await fetch('/api/scheduler/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  loadSched();
});
document.getElementById('sched-stop').addEventListener('click', async () => {
  await fetch('/api/scheduler/stop', { method: 'POST' });
  loadSched();
});

function el(tag, cls, text){
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function renderSched(data){
  const body = document.getElementById('sched-body');
  body.innerHTML = '';
  const st = data.status || {};

  const status = el('div', 'sched-status');
  const schedState = data.running ? 'работает' : (st.updated_at ? 'остановлен' : 'не запускался');
  const schedStateCls = data.running ? 'ok' : (st.updated_at ? 'err' : '');
  const cards = [
    ['Состояние', schedState, schedStateCls],
    ['Что собираем', st.currencies || '—'],
    ['Запусков сбора', st.collects_total !== undefined ? st.collects_total : '—'],
    ['Ошибок', st.errors_total !== undefined ? st.errors_total : '—', st.errors_total ? 'err' : ''],
    ['Снимков в БД', data.snapshots_total !== undefined ? data.snapshots_total : '—'],
    ['Интервал сбора', st.collect_interval_sec ? st.collect_interval_sec + ' c' : '—'],
    ['Интервал отчёта', st.report_interval_sec ? st.report_interval_sec + ' c' : '—'],
    ['Глубина отчёта', st.report_hours !== undefined
      ? (st.report_hours ? st.report_hours + ' ч' : 'вся история') : '—'],
    ['Последний сбор', st.last_collect_at || '—'],
    ['Следующий сбор', st.next_collect_at || '—'],
    ['Последний отчёт', st.last_report_at || '—'],
  ];
  cards.forEach(([k, v, cls]) => {
    const card = el('div', 'stat');
    card.appendChild(el('div', 'k', k));
    card.appendChild(el('div', 'v' + (cls ? ' ' + cls : ''), String(v)));
    status.appendChild(card);
  });
  if (st.last_collect_summary) {
    const card = el('div', 'stat');
    card.style.flex = '1 1 100%';
    card.appendChild(el('div', 'k', 'Результат последнего сбора'));
    card.appendChild(el('div', 'v', st.last_collect_summary));
    status.appendChild(card);
  }
  body.appendChild(status);

  const snapsSec = el('div', 'sched-sec');
  const snapHead = el('div', 'rep-head');
  snapHead.appendChild(el('h3', null, 'Последние снимки (SQLite через MCP save_snapshot)'));
  const snaps = data.snapshots || [];
  const snapClear = el('button', 'hdr-btn');
  snapClear.id = 'snaps-clear';
  snapClear.type = 'button';
  snapClear.textContent = 'Очистить';
  snapClear.title = 'Удалить все снимки из SQLite, обнулить счётчики и журнал. Нумерация начнётся с #1. Отчёты останутся.';
  snapClear.disabled = data.running || !snaps.length;
  snapClear.addEventListener('click', async () => {
    const total = data.snapshots_total !== undefined ? data.snapshots_total : snaps.length;
    if (!confirm('Удалить все снимки (' + total + ')? Счётчики и журнал обнулятся, нумерация начнётся с #1. Отчёты останутся.')) return;
    snapClear.disabled = true;
    try {
      const r = await fetch('/api/scheduler/snapshots/clear', { method: 'POST' });
      const res = await r.json();
      if (!r.ok) alert(res.error || ('Ошибка: ' + r.status));
      loadSched();
    } catch(err){
      snapClear.disabled = false;
      alert('Не удалось очистить снимки: ' + err);
    }
  });
  snapHead.appendChild(snapClear);
  snapsSec.appendChild(snapHead);
  if (!snaps.length){
    snapsSec.appendChild(el('p', 'mcp-empty', 'База пуста — запустите планировщик кнопкой выше или в терминале (python scheduler.py).'));
  } else {
    const table = el('table', 'snaps');
    // Колонки — все валюты, встречающиеся в снимках (id и ts — фикс).
    const keys = [];
    snaps.forEach(r => Object.keys(r).forEach(k => {
      if (k !== 'id' && k !== 'ts' && keys.indexOf(k) === -1) keys.push(k);
    }));
    const thead = el('tr');
    thead.appendChild(el('th', null, '#'));
    thead.appendChild(el('th', null, 'Время'));
    keys.forEach(k => thead.appendChild(el('th', 'num', k.toUpperCase())));
    table.appendChild(thead);
    snaps.forEach(r => {
      const tr = el('tr');
      tr.appendChild(el('td', null, String(r.id)));
      tr.appendChild(el('td', null, r.ts));
      keys.forEach(k => {
        const v = r[k];
        tr.appendChild(el('td', 'num', v === null || v === undefined ? '—' : String(v)));
      });
      table.appendChild(tr);
    });
    snapsSec.appendChild(table);
  }
  body.appendChild(snapsSec);

  const repSec = el('div', 'sched-sec');
  const repHead = el('div', 'rep-head');
  repHead.appendChild(el('h3', null, 'Отчёты за период (read_snapshots -> агрегат)'));
  const reports = data.reports || [];
  const clearBtn = el('button', 'hdr-btn');
  clearBtn.id = 'reports-clear';
  clearBtn.type = 'button';
  clearBtn.textContent = 'Очистить';
  clearBtn.title = 'Удалить все файлы отчётов (data/reports/). Снимки в SQLite останутся.';
  clearBtn.disabled = !reports.length;
  clearBtn.addEventListener('click', async () => {
    if (!confirm('Удалить все отчёты (' + reports.length + ')? Снимки в SQLite останутся — отчёты можно пересобрать (--report --hours 0).')) return;
    clearBtn.disabled = true;
    try {
      await fetch('/api/scheduler/reports/clear', { method: 'POST' });
      schedSelectedReport = null;
      reportCache.clear();
      loadSched();
    } catch(err){
      clearBtn.disabled = false;
      alert('Не удалось очистить отчёты: ' + err);
    }
  });
  repHead.appendChild(clearBtn);
  repSec.appendChild(repHead);

  // Управление отчётом: собрать сейчас с выбранной глубиной / удалить выбранный.
  const repCtl = el('div', 'rep-ctl');
  const depth = el('select');
  depth.id = 'report-depth';
  depth.title = 'Какой период накопленных данных попадёт в отчёт';
  [['24', '24 ч'], ['1', '1 ч'], ['6', '6 ч'], ['168', '7 дней'], ['0', 'вся история']]
    .forEach(([val, label]) => {
      const opt = el('option', null, label); opt.value = val;
      depth.appendChild(opt);
    });
  repCtl.appendChild(depth);
  const buildBtn = el('button', 'hdr-btn');
  buildBtn.id = 'report-build';
  buildBtn.type = 'button';
  buildBtn.textContent = 'Собрать отчёт';
  buildBtn.title = 'Собрать отчёт прямо сейчас из накопленных снимков (планировщик может быть остановлен)';
  buildBtn.addEventListener('click', async () => {
    buildBtn.disabled = true;
    try {
      const r = await fetch('/api/scheduler/report/build', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hours: parseFloat(depth.value) })
      });
      const res = await r.json();
      if (res.name){
        schedSelectedReport = res.name;
        reportCache.set(res.name, res.content);
        loadSched();
      } else {
        alert(res.error || ('Ошибка: ' + r.status));
      }
    } catch(err){
      alert('Не удалось собрать отчёт: ' + err);
    } finally {
      buildBtn.disabled = false;
    }
  });
  repCtl.appendChild(buildBtn);
  const delBtn = el('button', 'hdr-btn');
  delBtn.id = 'report-delete';
  delBtn.type = 'button';
  delBtn.textContent = 'Удалить выбранный';
  delBtn.title = 'Удалить отчёт, выбранный в списке ниже (и его json-версию)';
  delBtn.disabled = !reports.length;
  delBtn.addEventListener('click', async () => {
    if (!schedSelectedReport || !confirm('Удалить отчёт ' + schedSelectedReport + '?')) return;
    delBtn.disabled = true;
    try {
      const r = await fetch('/api/scheduler/report/delete', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: schedSelectedReport })
      });
      const res = await r.json();
      if (!r.ok) alert(res.error || ('Ошибка: ' + r.status));
      reportCache.delete(schedSelectedReport);
      schedSelectedReport = null;
      loadSched();
    } catch(err){
      delBtn.disabled = false;
      alert('Не удалось удалить отчёт: ' + err);
    }
  });
  repCtl.appendChild(delBtn);
  repSec.appendChild(repCtl);

  if (!reports.length){
    repSec.appendChild(el('p', 'mcp-empty', 'Отчётов пока нет — планировщик собирает их по расписанию (data/reports/).'));
  } else {
    const sel = el('select'); sel.id = 'report-select';
    reports.forEach(name => {
      const opt = el('option', null, name); opt.value = name;
      sel.appendChild(opt);
    });
    const pre = el('pre'); pre.id = 'report-view';
    // Панель перерисовывается каждые 5 с: сохраняем выбранный отчёт,
    // иначе селект сбрасывается на самый свежий.
    const keep = (schedSelectedReport && reports.indexOf(schedSelectedReport) !== -1)
      ? schedSelectedReport : reports[0];
    schedSelectedReport = keep;
    sel.value = keep;
    sel.addEventListener('change', () => {
      schedSelectedReport = sel.value;
      loadReport(sel.value, pre);
    });
    repSec.appendChild(sel);
    repSec.appendChild(pre);
    const cached = reportCache.get(keep);
    if (cached !== undefined) pre.textContent = cached;
    else loadReport(keep, pre);
  }
  body.appendChild(repSec);

  const note = el('p', 'sched-note',
    'Планировщик дёргает MCP-инструменты rates-store: fetch_rates -> save_snapshot по расписанию ' +
    'сбора, а по расписанию отчёта читает read_snapshots и кладёт агрегат в data/reports/. ' +
    'Запустить можно кнопкой выше (фоновый поток этого процесса) или в терминале ' +
    '(python scheduler.py, режим --once — для cron/Task Scheduler). ' +
    'Эта панель читает накопленное: SQLite (read-only) и файлы отчётов.');
  body.appendChild(note);

  schedDot.className = 'dot ' + (data.running ? 'ok' : (st.updated_at ? 'err' : ''));

  const startBtn = document.getElementById('sched-start');
  const stopBtn = document.getElementById('sched-stop');
  if (startBtn) startBtn.disabled = data.running;
  if (stopBtn) stopBtn.disabled = !data.running;
  ['sched-currencies', 'sched-interval', 'sched-report', 'sched-hours'].forEach(id => {
    const inp = document.getElementById(id);
    if (inp) inp.disabled = data.running;
  });
}

async function loadReport(name, pre){
  // Отчёты неизменяемые (имя с timestamp), поэтому содержимое кэшируем:
  // перерисовка панели не дёргает сервер повторно.
  if (reportCache.has(name)){
    pre.textContent = reportCache.get(name);
    return;
  }
  try {
    const r = await fetch('/api/scheduler/report?name=' + encodeURIComponent(name));
    const data = await r.json();
    if (r.ok && data.content) reportCache.set(name, data.content);
    pre.textContent = data.content || ('не удалось прочитать отчёт: ' + (data.error || '?'));
  } catch(err){
    pre.textContent = 'Ошибка запроса: ' + err;
  }
}

async function loadSched(){
  const body = document.getElementById('sched-body');
  try {
    const r = await fetch('/api/scheduler');
    const data = await r.json();
    renderSched(data);
  } catch(err){
    body.innerHTML = '';
    body.appendChild(el('p', 'mcp-err', 'Не удалось загрузить состояние планировщика: ' + err));
    schedDot.className = 'dot err';
  }
}

async function refreshSchedDot(){
  try {
    const r = await fetch('/api/scheduler');
    const data = await r.json();
    schedDot.className = 'dot ' + (data.running ? 'ok' : (data.status && data.status.updated_at ? 'err' : ''));
  } catch(e){ schedDot.className = 'dot'; }
}

refreshSchedDot();
setInterval(() => { if (schedOverlay.classList.contains('open')) loadSched(); }, 5000);

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


@app.route("/api/scheduler")
def api_scheduler():
    """Состояние планировщика (День 18): статус, снимки, отчёты."""
    from .scheduler import scheduler_snapshot

    return jsonify(scheduler_snapshot())


@app.route("/api/scheduler/start", methods=["POST"])
def api_scheduler_start():
    """Запустить планировщик в фоновом потоке этого процесса."""
    import re as _re

    from .scheduler import scheduler_snapshot, start_background

    data = request.get_json(silent=True) or {}

    def _num(value):
        try:
            value = float(value)
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None

    def _num_or_zero(value):
        try:
            value = float(value)
            return value if value >= 0 else None
        except (TypeError, ValueError):
            return None

    currencies = data.get("currencies")
    currencies = (
        str(currencies).strip()
        if currencies and _re.fullmatch(r"[A-Za-z,\s;]{1,64}", str(currencies))
        else None
    )
    _, started = start_background(
        interval=_num(data.get("interval")),
        report_interval=_num(data.get("report_interval")),
        hours=_num_or_zero(data.get("hours")),
        currencies=currencies,
    )
    snapshot = scheduler_snapshot()
    snapshot["started"] = started
    return jsonify(snapshot)


@app.route("/api/scheduler/stop", methods=["POST"])
def api_scheduler_stop():
    """Мягко остановить планировщик, запущенный из веб-панели."""
    from .scheduler import scheduler_snapshot, stop_background

    _, stopped = stop_background()
    snapshot = scheduler_snapshot()
    snapshot["stopped"] = stopped
    return jsonify(snapshot)


@app.route("/api/scheduler/report/build", methods=["POST"])
def api_scheduler_report_build():
    """Собрать отчёт прямо сейчас из накопленных снимков (по запросу).

    Те же чистые функции, что и у планировщика (reports.aggregate +
    format_report_md), но данные читаются из SQLite напрямую — работает
    и когда планировщик остановлен, и когда запущен из терминала.
    """
    import json as _json
    from datetime import datetime as _dt

    from mcp_rates_store import store

    from .reports import aggregate, format_report_md
    from .scheduler import ROOT, get_background

    data = request.get_json(silent=True) or {}
    try:
        hours = float(data.get("hours", 0))
    except (TypeError, ValueError):
        hours = 0.0
    hours = max(hours, 0.0)  # 0 = вся накопленная история

    rows = store.read_snapshots(hours)
    report = aggregate(rows, hours)
    stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"report_{stamp}.md"
    json_path = reports_dir / f"report_{stamp}.json"
    md_path.write_text(format_report_md(report), encoding="utf-8")
    json_path.write_text(
        _json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    name = md_path.name
    now_iso = _dt.now().astimezone().isoformat(timespec="seconds")
    # Чтобы панель показывала «последний отчёт» честно, обновляем статус:
    # и файл, и живой экземпляр фонового планировщика (если он есть).
    background = get_background()
    if background is not None:
        background.last_report_at = now_iso
        background.last_report_file = name
    status_path = ROOT / "data" / "scheduler_status.json"
    if status_path.exists():
        try:
            st = _json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = None
        if isinstance(st, dict):
            st["last_report_at"] = now_iso
            st["last_report_file"] = name
            try:
                status_path.write_text(
                    _json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass

    return jsonify({"name": name, "content": md_path.read_text(encoding="utf-8")})


@app.route("/api/scheduler/report/delete", methods=["POST"])
def api_scheduler_report_delete():
    """Удалить выбранный отчёт и его json-версию."""
    import json as _json
    import re

    from .scheduler import ROOT, scheduler_snapshot

    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "")
    if not re.fullmatch(r"report_[\w.-]+\.md", name):
        return jsonify({"error": "некорректное имя отчёта"}), 400
    md_path = ROOT / "data" / "reports" / name
    if not md_path.is_file():
        return jsonify({"error": "отчёт не найден"}), 404
    md_path.unlink()
    json_path = md_path.with_suffix(".json")
    if json_path.is_file():
        json_path.unlink()

    # Если удалили «последний отчёт», статус указывает в пустоту — чиним.
    remaining = sorted(p.name for p in (ROOT / "data" / "reports").glob("report_*.md"))
    newest = remaining[-1] if remaining else None
    status_path = ROOT / "data" / "scheduler_status.json"
    if status_path.exists():
        try:
            st = _json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = None
        if isinstance(st, dict) and st.get("last_report_file") == name:
            st["last_report_file"] = newest
            try:
                status_path.write_text(
                    _json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass

    snapshot = scheduler_snapshot()
    snapshot["deleted"] = name
    return jsonify(snapshot)


@app.route("/api/scheduler/reports/clear", methods=["POST"])
def api_scheduler_reports_clear():
    """Удалить все файлы отчётов (report_*.md|.json) из data/reports/.

    Сами снимки в SQLite не трогаем — отчёты можно пересобрать заново
    (python scheduler.py --report --hours 0 или кнопкой «Отчёт сейчас»).
    """
    import re

    from .scheduler import ROOT

    reports_dir = ROOT / "data" / "reports"
    removed = 0
    if reports_dir.is_dir():
        for path in reports_dir.iterdir():
            if path.is_file() and re.fullmatch(r"report_[\w.-]+\.(?:md|json)", path.name):
                path.unlink()
                removed += 1
    return jsonify({"removed": removed})


@app.route("/api/scheduler/snapshots/clear", methods=["POST"])
def api_scheduler_snapshots_clear():
    """Очистить историю запусков: все снимки в SQLite + счётчики + журнал.

    Нумерация снимков начинается заново с #1. Отчёты (data/reports/)
    не трогаются. Работает только при остановленном планировщике.
    """
    import json as _json
    from datetime import datetime as _dt

    from mcp_rates_store import store

    from .scheduler import ROOT, scheduler_snapshot

    before = scheduler_snapshot()
    if before["running"]:
        return jsonify({"error": "сначала остановите планировщик"}), 409

    removed = store.clear_all()

    # Обнуляем счётчики запусков в статус-файле (отчётные поля не трогаем).
    status_path = ROOT / "data" / "scheduler_status.json"
    if status_path.exists():
        try:
            data = _json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            data.update({
                "collects_total": 0,
                "errors_total": 0,
                "last_collect_at": None,
                "last_collect_summary": None,
                "next_collect_at": None,
            })
            try:
                status_path.write_text(
                    _json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass

    # Журнал запусков тоже часть истории — начинаем с чистого листа.
    log_path = ROOT / "data" / "scheduler.log"
    try:
        log_path.write_text(
            f"[{_dt.now():%Y-%m-%d %H:%M:%S}] журнал очищен из веб-панели "
            f"(удалено снимков: {removed})\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    snapshot = scheduler_snapshot()
    snapshot["removed"] = removed
    return jsonify(snapshot)


@app.route("/api/scheduler/report")
def api_scheduler_report():
    """Текст выбранного отчёта планировщика (только файлы из data/reports/)."""
    import re
    from pathlib import Path

    from flask import abort

    from .scheduler import ROOT

    name = request.args.get("name", "")
    if not re.fullmatch(r"report_[\w.-]+\.md", name):
        abort(400)
    path = ROOT / "data" / "reports" / name
    if not path.is_file():
        abort(404)
    return jsonify({"name": name, "content": path.read_text(encoding="utf-8")})


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
