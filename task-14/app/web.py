"""Minimal Flask web agent: user query -> LLM via API -> display answer."""
import threading

from flask import Flask, jsonify, render_template_string, request

from .agent import Agent, CancelToken
from . import experiment, invariants, memory, profile, profile_experiment, store, task_fsm
from .config import MODELS, MODEL

app = Flask(__name__)

# One in-flight /api/chat run per branch: branch_id -> CancelToken. The stop
# button cancels the token; a second run for the same branch is rejected until
# the first one winds down, so concurrent runs can't corrupt the history.
_running_lock = threading.Lock()
_running: dict[int, CancelToken] = {}

HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Простой агент</title>
<style>
  :root {
    --bg: #070b16;
    --panel: rgba(148, 163, 184, .055);
    --panel-border: rgba(148, 163, 184, .16);
    --txt: #e2e8f0;
    --txt-dim: #94a3b8;
    --txt-faint: #64748b;
    --accent: #3b82f6;
    --accent-2: #8b5cf6;
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background:
      radial-gradient(900px 500px at 12% -8%, rgba(59, 130, 246, .16), transparent 60%),
      radial-gradient(800px 520px at 105% 108%, rgba(139, 92, 246, .13), transparent 60%),
      var(--bg);
    color: var(--txt);
    overflow: hidden;
  }
  button { font-family: inherit; }

  /* ---------- shell: left sidebar | chat | right sidebar ---------- */
  .shell { display: grid; grid-template-columns: 290px minmax(0, 1fr) 312px; height: 100vh; }
  .side { padding: 16px 14px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
  .side-left  { border-right: 1px solid rgba(148,163,184,.1); }
  .side-right { border-left:  1px solid rgba(148,163,184,.1); }
  .side::-webkit-scrollbar, #chat::-webkit-scrollbar, #memoryPanel::-webkit-scrollbar { width: 6px; }
  .side::-webkit-scrollbar-thumb, #chat::-webkit-scrollbar-thumb, #memoryPanel::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,.2); border-radius: 6px;
  }

  .brand { display: flex; align-items: center; gap: 11px; padding: 2px 4px 6px; }
  .logo {
    width: 36px; height: 36px; border-radius: 11px; flex: none;
    display: grid; place-items: center;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 4px 16px rgba(59,130,246,.35);
  }
  .brand h1 { font-size: 15px; margin: 0; letter-spacing: .2px; }
  .brand-sub { font-size: 11px; color: var(--txt-faint); }

  .card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: var(--radius);
    padding: 13px 14px;
  }
  .card-title {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.1px; color: var(--txt-dim); margin-bottom: 10px;
  }
  .hint { font-size: 11.5px; color: var(--txt-faint); line-height: 1.5; margin: 7px 0 0; }

  /* buttons in sidebars */
  .sbtn {
    display: flex; align-items: center; justify-content: center; gap: 7px;
    width: 100%; padding: 8px 12px; margin-top: 8px;
    font-size: 12.5px; font-weight: 500; color: #cbd5e1;
    background: rgba(255,255,255,.05); border: 1px solid var(--panel-border);
    border-radius: 10px; cursor: pointer; transition: .15s;
  }
  .sbtn:hover { border-color: rgba(59,130,246,.55); color: #dbeafe; background: rgba(59,130,246,.1); }
  .sbtn:disabled { opacity: .38; cursor: not-allowed; }
  .sbtn.primary {
    color: #fff; background: linear-gradient(135deg, rgba(59,130,246,.85), rgba(79,70,229,.85));
    border-color: transparent; box-shadow: 0 3px 12px rgba(59,130,246,.25);
  }
  .sbtn.primary:hover { filter: brightness(1.12); }
  .sbtn.danger:hover { background: rgba(239,68,68,.14); border-color: rgba(239,68,68,.45); color: #fecaca; }

  /* selects */
  .side select {
    width: 100%; font-size: 12.5px; color: var(--txt);
    background: rgba(15,23,42,.6); border: 1px solid var(--panel-border);
    padding: 8px 10px; border-radius: 10px; cursor: pointer; outline: none; transition: .15s;
  }
  .side select:hover, .side select:focus { border-color: rgba(59,130,246,.55); }
  .side select option, .mem-add select option { background: #0f172a; color: var(--txt); }

  /* branch list */
  .branch-list { display: flex; flex-direction: column; gap: 4px; }
  .branch-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; border-radius: 10px; cursor: pointer;
    border: 1px solid transparent; transition: .15s;
  }
  .branch-item:hover { background: rgba(255,255,255,.05); }
  .branch-item.active { background: rgba(59,130,246,.14); border-color: rgba(59,130,246,.4); }
  .branch-item .b-icon {
    width: 22px; height: 22px; border-radius: 7px; flex: none;
    display: grid; place-items: center; font-size: 12px; color: #93c5fd;
    background: rgba(148,163,184,.1);
  }
  .branch-item.active .b-icon { background: rgba(59,130,246,.3); color: #dbeafe; }
  .branch-item .b-name {
    font-size: 12.5px; font-weight: 500; color: #e2e8f0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .branch-item .b-meta { margin-left: auto; flex: none; font-size: 10.5px; color: var(--txt-faint); }

  /* ---------- center ---------- */
  .center { display: flex; flex-direction: column; height: 100vh; min-width: 0; }
  .topbar {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 18px; min-height: 54px;
    border-bottom: 1px solid rgba(148,163,184,.1);
  }
  .topbar h2 { font-size: 14px; font-weight: 600; margin: 0; color: #cbd5e1; }
  .spacer { flex: 1; }
  .chip {
    font-size: 11.5px; color: #bfdbfe; max-width: 260px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    background: rgba(59,130,246,.12); border: 1px solid rgba(59,130,246,.35);
    padding: 4px 11px; border-radius: 999px;
  }
  .chip.subtle { color: var(--txt-dim); background: rgba(255,255,255,.05); border-color: var(--panel-border); }
  /* статус задачи длиннее остальных чипов — два уровня: этап + шаг */
  #taskChip {
    max-width: 480px; white-space: normal; line-height: 1.3;
    padding: 3px 12px; flex: 0 0 auto; text-align: left;
  }
  #taskChip .task-stage { font-weight: 600; }
  #taskChip .task-step { display: block; font-size: 10.5px; color: #94a3b8; }
  .icon-btn {
    display: none; width: 34px; height: 34px; flex: none;
    align-items: center; justify-content: center; font-size: 16px;
    color: #cbd5e1; background: rgba(255,255,255,.05);
    border: 1px solid var(--panel-border); border-radius: 10px; cursor: pointer;
  }
  /* experiment visuals */
  .exp-divider {
    text-align: center; font-size: 11.5px; color: #93c5fd;
    background: rgba(59,130,246,.08); border: 1px dashed rgba(59,130,246,.35);
    border-radius: 999px; padding: 5px 14px; margin: 2px auto; max-width: 92%;
  }
  .exp-status { text-align: center; font-size: 12px; color: var(--txt-dim); padding: 6px; }
  .report {
    background: rgba(15,23,42,.85); border: 1px solid rgba(148,163,184,.14);
    border-radius: var(--radius); padding: 14px 16px; font-size: 13px;
    max-width: 880px; margin: 0 auto; width: 100%;
  }
  .report h3 { margin: 0 0 8px; font-size: 14px; }
  .report table { width: 100%; border-collapse: collapse; margin: 8px 0; }
  .report th, .report td { border-bottom: 1px solid rgba(148,163,184,.12); padding: 6px 8px; text-align: left; font-size: 12.5px; }
  .report th { color: var(--txt-dim); font-weight: 600; }
  .report .good { color: #4ade80; } .report .bad { color: #f87171; }
  .report .verdict { margin-top: 8px; color: var(--txt); line-height: 1.55; }
  .report details { margin-top: 8px; }
  .report summary { cursor: pointer; color: #93c5fd; font-size: 12.5px; }
  .report .qa-item { padding: 7px 0; border-bottom: 1px dashed rgba(148,163,184,.12); }
  .report .qa-q { font-weight: 600; }
  .report .qa-answer { font-size: 12.5px; margin-top: 2px; color: #cbd5e1; white-space: pre-wrap; }
  .report .qa-expected { font-size: 11.5px; color: var(--txt-dim); margin-top: 2px; }
  .report .repeat {
    margin-top: 10px; font-size: 12px; color: #cbd5e1; background: rgba(255,255,255,.06);
    border: 1px solid var(--panel-border); padding: 5px 12px; border-radius: 999px; cursor: pointer;
  }
  #chat {
    flex: 1; overflow-y: auto; padding: 26px 24px;
    display: flex; flex-direction: column; gap: 18px;
  }
  #chat > .row { width: 100%; max-width: 880px; margin: 0 auto; }

  .row { display: flex; gap: 10px; align-items: flex-start; animation: pop .25s ease both; }
  .row.user { justify-content: flex-end; }
  @keyframes pop { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

  /* avatars */
  .row.assistant::before, .row.user::after {
    content: ''; width: 30px; height: 30px; flex: none; border-radius: 10px;
    background-size: 16px 16px; background-position: center; background-repeat: no-repeat;
    margin-top: 2px;
  }
  .row.assistant::before {
    background-color: #101a33; border: 1px solid rgba(59,130,246,.35);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2393c5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='8' width='16' height='12' rx='3'/%3E%3Cpath d='M12 8V4'/%3E%3Ccircle cx='9' cy='14' r='1'/%3E%3Ccircle cx='15' cy='14' r='1'/%3E%3C/svg%3E");
  }
  .row.user::after {
    background-color: rgba(59,130,246,.25); border: 1px solid rgba(59,130,246,.45);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23dbeafe' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4 21c0-4 3.6-6 8-6s8 2 8 6'/%3E%3C/svg%3E");
  }

  .bubble {
    max-width: 76%; padding: 12px 16px; border-radius: 16px; line-height: 1.6;
    white-space: pre-wrap; word-wrap: break-word; font-size: 14.5px;
    box-shadow: 0 2px 12px rgba(0,0,0,.25);
  }
  .user .bubble {
    background: linear-gradient(135deg, #2563eb, #4f46e5); color: #fff;
    border-bottom-right-radius: 6px;
  }
  .assistant .bubble {
    background: rgba(15,23,42,.75); border: 1px solid rgba(148,163,184,.14);
    border-bottom-left-radius: 6px;
  }
  .bubble .meta { margin-top: 9px; font-size: 11px; color: var(--txt-faint); }

  .error {
    padding: 10px 14px; border-radius: 12px; font-size: 12.5px;
    background: rgba(239,68,68,.1); color: #fca5a5; border: 1px solid rgba(239,68,68,.3);
    white-space: pre-wrap;
  }

  .typing { display: inline-flex; gap: 4px; align-items: center; color: var(--txt-dim); font-size: 13px; }
  .typing .t-dot { width: 7px; height: 7px; border-radius: 50%; background: #94a3b8; animation: blink 1.2s infinite; }
  .typing .t-dot:nth-child(2){ animation-delay: .2s; }
  .typing .t-dot:nth-child(3){ animation-delay: .4s; }
  @keyframes blink { 0%,80%,100%{ opacity: .2; transform: scale(1); } 40%{ opacity: 1; transform: scale(1.15); } }

  /* composer */
  #form { padding: 14px 24px 20px; }
  .composer {
    display: flex; gap: 10px; max-width: 880px; margin: 0 auto;
    background: rgba(15,23,42,.7); border: 1px solid rgba(148,163,184,.18);
    border-radius: 18px; padding: 8px; transition: .15s;
  }
  .composer:focus-within { border-color: rgba(59,130,246,.6); box-shadow: 0 0 0 3px rgba(59,130,246,.12); }
  #msg {
    flex: 1; padding: 10px 12px; border: none; background: transparent;
    color: var(--txt); font-size: 14.5px; outline: none; font-family: inherit;
  }
  #msg::placeholder { color: var(--txt-faint); }
  #send {
    width: 42px; height: 42px; flex: none; border: none; border-radius: 13px; cursor: pointer;
    display: grid; place-items: center;
    background: linear-gradient(135deg, var(--accent), #4f46e5);
    box-shadow: 0 4px 14px rgba(59,130,246,.35); transition: .15s;
  }
  #send:hover { filter: brightness(1.12); }
  #send:disabled { opacity: .5; cursor: not-allowed; filter: none; }
  #send.stopping {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    box-shadow: 0 4px 14px rgba(239,68,68,.4);
    animation: pulse-stop 1.2s ease infinite;
  }
  #send.stopping:hover { filter: brightness(1.1); }
  @keyframes pulse-stop {
    0%, 100% { box-shadow: 0 4px 14px rgba(239,68,68,.4); }
    50% { box-shadow: 0 4px 22px rgba(239,68,68,.65); }
  }

  /* ---------- memory modal ---------- */
  .modal-backdrop {
    display: none; position: fixed; inset: 0; z-index: 50;
    background: rgba(2,6,17,.6); backdrop-filter: blur(6px);
    align-items: center; justify-content: center; padding: 24px;
  }
  .modal-backdrop.open { display: flex; }
  #memoryPanel {
    width: min(960px, 100%); max-height: 86vh; overflow-y: auto;
    padding: 18px 20px; font-size: 13px;
    background: #0d1526; border: 1px solid rgba(148,163,184,.2);
    border-radius: 18px; box-shadow: 0 24px 80px rgba(0,0,0,.5);
    animation: zoom .2s ease both;
  }
  @keyframes zoom { from { opacity: 0; transform: scale(.96); } to { opacity: 1; transform: none; } }
  .mem-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; font-size: 15px; }
  .mem-close { cursor: pointer; color: var(--txt-dim); font-size: 17px; padding: 4px; }
  .mem-close:hover { color: var(--txt); }
  .mem-rules {
    background: rgba(255,255,255,.03); border: 1px dashed rgba(148,163,184,.2);
    border-radius: 10px; padding: 10px 12px; margin-bottom: 13px; font-size: 12.5px; color: #cbd5e1;
  }
  .mem-rules .rule { margin: 5px 0; line-height: 1.5; }
  .mem-rules b { color: #93c5fd; }
  .mem-eq { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .mem-layer {
    background: rgba(255,255,255,.03); border: 1px solid rgba(148,163,184,.12);
    border-radius: 12px; padding: 11px 12px;
  }
  .mem-layer-title { font-weight: 600; color: #bfdbfe; margin-bottom: 7px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .mem-layer-title button {
    font-size: 11px; color: #cbd5e1; background: rgba(255,255,255,.06);
    border: 1px solid var(--panel-border); padding: 2px 8px; border-radius: 999px; cursor: pointer;
  }
  .mem-layer-title button:hover { border-color: rgba(59,130,246,.55); color: #bfdbfe; }
  .mem-layer-title select {
    font-size: 11px; color: #cbd5e1; background: rgba(255,255,255,.06);
    border: 1px solid var(--panel-border); padding: 2px 6px; border-radius: 999px; outline: none; cursor: pointer;
  }
  .mem-layer-title select option { background: #0d1526; color: #e2e8f0; }
  .mem-empty { color: #475569; font-size: 12px; font-style: italic; }
  .mem-item {
    display: flex; align-items: flex-start; gap: 6px; padding: 4px 0;
    font-size: 12.5px; color: #cbd5e1; line-height: 1.4;
    border-top: 1px dashed rgba(148,163,184,.1);
  }
  .mem-item:first-child { border-top: none; }
  .mem-item input[type="checkbox"] { accent-color: #2563eb; margin: 2px 0 0; flex: none; cursor: pointer; }
  .mem-val { flex: 1; min-width: 0; word-break: break-word; }
  .mem-item .k { color: #93c5fd; }
  .mem-item .del { margin-left: auto; cursor: pointer; color: var(--txt-faint); flex: none; }
  .mem-item .del:hover { color: #f87171; }
  .mem-item .cat {
    font-size: 10.5px; color: var(--txt-faint); background: rgba(255,255,255,.06);
    padding: 1px 6px; border-radius: 999px; flex: none;
  }
  .mem-add { display: flex; gap: 6px; margin-top: 9px; }
  .mem-add select, .mem-add input {
    font-size: 12px; color: var(--txt); background: rgba(255,255,255,.06);
    border: 1px solid var(--panel-border); padding: 5px 7px; border-radius: 8px; outline: none;
  }
  .mem-add input { flex: 1; min-width: 0; }
  .mem-add select { flex: none; }
  .mem-add button {
    font-size: 13px; color: #bfdbfe; background: rgba(59,130,246,.15);
    border: 1px solid rgba(59,130,246,.4); padding: 5px 11px; border-radius: 8px; cursor: pointer; flex: none;
  }
  .mem-add button:hover { background: rgba(59,130,246,.3); color: #fff; }

  /* ---------- profile modal ---------- */
  #profilePanel {
    width: min(960px, 100%); max-height: 86vh; overflow-y: auto;
    padding: 18px 20px; font-size: 13px;
    background: #0d1526; border: 1px solid rgba(148,163,184,.2);
    border-radius: 18px; box-shadow: 0 24px 80px rgba(0,0,0,.5);
    animation: zoom .2s ease both;
  }
  .prof-list { display: flex; flex-direction: column; gap: 10px; }
  .prof-card {
    background: rgba(255,255,255,.03); border: 1px solid rgba(148,163,184,.12);
    border-radius: 12px; padding: 12px 14px;
  }
  .prof-card.active { border-color: rgba(59,130,246,.5); background: rgba(59,130,246,.08); }
  .prof-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .prof-name { font-weight: 600; color: #bfdbfe; font-size: 13.5px; }
  .prof-active-badge {
    font-size: 10.5px; color: #dbeafe; background: rgba(59,130,246,.25);
    border: 1px solid rgba(59,130,246,.45); padding: 1px 8px; border-radius: 999px;
  }
  .prof-desc { font-size: 12px; color: var(--txt-dim); margin: 4px 0 6px; }
  .prof-actions { margin-left: auto; display: flex; gap: 6px; }
  .prof-actions button {
    font-size: 11px; color: #cbd5e1; background: rgba(255,255,255,.06);
    border: 1px solid var(--panel-border); padding: 3px 10px; border-radius: 999px; cursor: pointer;
  }
  .prof-actions button:hover { border-color: rgba(59,130,246,.55); color: #bfdbfe; }
  .prof-actions button.danger:hover { border-color: rgba(239,68,68,.5); color: #fecaca; }
  .prof-rule {
    display: flex; gap: 6px; align-items: flex-start; font-size: 12.5px; color: #cbd5e1;
    padding: 3px 0; border-top: 1px dashed rgba(148,163,184,.1); line-height: 1.4;
  }
  .prof-rule .kind {
    font-size: 10.5px; color: #93c5fd; background: rgba(59,130,246,.1);
    border: 1px solid rgba(59,130,246,.3); padding: 0 6px; border-radius: 999px; flex: none; margin-top: 1px;
  }
  .prof-rule .src {
    font-size: 10px; color: #a5b4fc; background: rgba(139,92,246,.12);
    border: 1px solid rgba(139,92,246,.3); padding: 0 6px; border-radius: 999px; flex: none; margin-top: 1px;
  }
  .prof-rule .del { margin-left: auto; cursor: pointer; color: var(--txt-faint); flex: none; }
  .prof-rule .del:hover { color: #f87171; }
  .prof-form { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-top: 12px; }
  .prof-form label { font-size: 11px; color: var(--txt-dim); display: block; margin-bottom: 3px; }
  .prof-form input, .prof-form select, .prof-form textarea {
    width: 100%; font-size: 12.5px; color: var(--txt);
    background: rgba(15,23,42,.6); border: 1px solid var(--panel-border);
    padding: 7px 9px; border-radius: 9px; outline: none; font-family: inherit;
  }
  .prof-form textarea { resize: vertical; min-height: 52px; }
  .prof-form input:focus, .prof-form select:focus, .prof-form textarea:focus {
    border-color: rgba(59,130,246,.55);
  }
  .prof-form .full { grid-column: 1 / -1; }
  .prof-form .save-row { grid-column: 1 / -1; display: flex; gap: 8px; align-items: center; }
  .prof-form .save-row button {
    font-size: 12.5px; color: #fff; background: linear-gradient(135deg, rgba(59,130,246,.85), rgba(79,70,229,.85));
    border: none; padding: 7px 16px; border-radius: 9px; cursor: pointer;
  }
  .prof-form .save-row button.ghost {
    color: #cbd5e1; background: rgba(255,255,255,.06); border: 1px solid var(--panel-border);
  }
  .prof-learned-note {
    margin-top: 8px; font-size: 11.5px; color: #a5b4fc;
    background: rgba(139,92,246,.08); border: 1px dashed rgba(139,92,246,.3);
    border-radius: 9px; padding: 7px 10px; line-height: 1.5;
  }

  /* ---------- invariants modal ---------- */
  #invariantPanel {
    width: min(960px, 100%); max-height: 86vh; overflow-y: auto;
    padding: 18px 20px; font-size: 13px;
    background: #0d1526; border: 1px solid rgba(148,163,184,.2);
    border-radius: 18px; box-shadow: 0 24px 80px rgba(0,0,0,.5);
    animation: zoom .2s ease both;
  }
  .inv-note {
    margin: 8px 0 12px; font-size: 11.5px; color: #fde68a;
    background: rgba(245,158,11,.08); border: 1px dashed rgba(245,158,11,.35);
    border-radius: 9px; padding: 8px 10px; line-height: 1.5;
  }
  .inv-cat {
    font-size: 10.5px; color: #93c5fd; background: rgba(59,130,246,.1);
    border: 1px solid rgba(59,130,246,.3); padding: 1px 8px; border-radius: 999px; flex: none;
  }
  .inv-code { font-size: 11px; color: #a5b4fc; font-weight: 600; }
  .inv-rule { font-size: 12.5px; color: #cbd5e1; margin: 5px 0 4px; line-height: 1.45; }
  .inv-patterns { display: flex; flex-wrap: wrap; gap: 4px; }
  .inv-patterns span {
    font-size: 10px; color: #86efac; background: rgba(34,197,94,.08);
    border: 1px solid rgba(34,197,94,.25); padding: 0 6px; border-radius: 999px;
    font-family: Consolas, monospace;
  }
  .inv-log { margin-top: 14px; }
  .inv-log table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  .inv-log th, .inv-log td { text-align: left; padding: 4px 6px; border-bottom: 1px solid rgba(148,163,184,.1); vertical-align: top; }
  .inv-log th { color: var(--txt-faint); font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: .6px; }
  .inv-log td { color: #cbd5e1; }
  .inv-log .src { color: #a5b4fc; white-space: nowrap; }

  /* mobile scrim + drawers */
  .scrim {
    display: none; position: fixed; inset: 0; z-index: 40;
    background: rgba(2,6,17,.55); backdrop-filter: blur(3px);
  }
  @media (max-width: 1180px) {
    .shell { grid-template-columns: 1fr; }
    .side {
      position: fixed; top: 0; bottom: 0; z-index: 45; width: 300px;
      background: #0b1222; transition: transform .25s ease;
    }
    .side-left { left: 0; transform: translateX(-105%); box-shadow: 20px 0 60px rgba(0,0,0,.45); }
    .side-right { right: 0; transform: translateX(105%); box-shadow: -20px 0 60px rgba(0,0,0,.45); }
    body.drawer-left .side-left, body.drawer-right .side-right { transform: none; }
    body.drawer-left .scrim, body.drawer-right .scrim { display: block; }
    .icon-btn { display: flex; }
    .bubble { max-width: 86%; }
  }
  @media (max-width: 640px){ .mem-eq { grid-template-columns: 1fr; } .chip { max-width: 150px; } #taskChip { max-width: 260px; } }
</style>
</head>
<body>
<div class="scrim" id="scrim"></div>
<div class="shell">
  <aside class="side side-left" id="sideLeft">
    <div class="brand">
      <span class="logo"><svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/></svg></span>
      <div><h1>AI-агент</h1><span class="brand-sub">диалог с памятью и ветками</span></div>
    </div>
    <section class="card">
      <div class="card-title">Ветки диалога</div>
      <div id="branchList" class="branch-list"></div>
      <button id="checkpoint" class="sbtn" type="button" title="Запомнить текущее место диалога — от этого checkpoint создаются ветки">&#9873; Checkpoint</button>
      <button id="branchAdd" class="sbtn primary" type="button" title="Создать ветку от checkpoint текущей ветки (или от её конца, если checkpoint не сохранён) и переключиться в неё">+ Новая ветка</button>
      <button id="branchDel" class="sbtn danger" type="button" title="Удалить текущую ветку (основную удалить нельзя)">Удалить ветку</button>
    </section>
    <section class="card">
      <div class="card-title">Профиль пользователя</div>
      <p class="hint">Предпочтения (стиль, формат, тон, ограничения) уходят в каждый запрос первым system-сообщением и пополняются из диалога.</p>
      <select id="profileSelect" title="Активный профиль — его предпочтения применяются к каждому ответу"></select>
      <button id="profilesBtn" class="sbtn" type="button">Редактировать профили</button>
    </section>
    <section class="card">
      <div class="card-title">Память</div>
      <p class="hint">Три слоя: краткосрочная (диалог), рабочая (текущая задача) и долговременная (профиль, решения, знания).</p>
      <button id="memoryBtn" class="sbtn" type="button">Открыть память</button>
    </section>
    <section class="card">
      <div class="card-title">Инварианты</div>
      <p class="hint">Правила, которые ассистент не имеет права нарушать: архитектура, техрешения, стек, бизнес-правила. Хранятся отдельно от диалога; каждый ответ проверяется кодом до выдачи. Изменяются только явной командой.</p>
      <button id="invariantsBtn" class="sbtn" type="button">Управление инвариантами</button>
    </section>
  </aside>

  <main class="center">
    <header class="topbar">
      <button class="icon-btn" id="toggleLeft" type="button" title="Ветки">&#9776;</button>
      <h2>Диалог</h2>
      <span class="chip" id="modelChip" title="Текущая модель"></span>
      <span class="chip subtle" id="profileChip" title="Активный профиль: его предпочтения добавляются в каждый запрос">Профиль: —</span>
      <span class="chip subtle" id="invChip" title="Инварианты проекта: обязательные правила (архитектура, стек, техрешения, бизнес). Каждый ответ проверяется кодом до выдачи; изменяются только явной командой">Инварианты: —</span>
      <span class="chip subtle" title="Контекст собирается из профиля и трёх слоёв памяти: Профиль + LTM + WM + STM">Профиль + LTM + WM + STM</span>
      <span class="chip subtle" id="taskChip" title="Жизненный цикл задачи — конечный автомат: idle → planning → awaiting_approval → execution → validation → done. Переходы проверяются кодом">Задача: —</span>
      <span class="spacer"></span>
      <button class="icon-btn" id="toggleRight" type="button" title="Настройки">&#9881;</button>
    </header>
    <div id="chat"></div>
    <form id="form">
      <div class="composer">
        <input id="msg" placeholder="Спросите что-нибудь..." autocomplete="off">
        <button id="send" type="submit" title="Отправить"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg></button>
      </div>
    </form>
  </main>

  <aside class="side side-right" id="sideRight">
    <section class="card">
      <div class="card-title">Модель</div>
      <select id="modelSelect">
        {% for m in models %}
        <option value="{{ m }}" {% if m == model %}selected{% endif %}>{{ m }}</option>
        {% endfor %}
      </select>
      <p class="hint">Модель, которая обрабатывает запросы. Список задаётся в .env (GPUSTACK_MODELS).</p>
    </section>
    <section class="card">
      <div class="card-title">Эксперимент с профилями</div>
      <p class="hint">Три профиля с противоположными предпочтениями отвечают на один и тот же вопрос — проверка авто-грейдером, соблюдены ли краткость/подробность, код и тон. Затем 10 вопросов подряд в одном профиле: на каком ответе предпочтения начинают игнорироваться.</p>
      <button id="profileExperiment" class="sbtn primary" type="button">Запустить эксперимент с профилями</button>
    </section>
    <section class="card">
      <div class="card-title">Эксперимент с памятью</div>
      <p class="hint">Прогнать два диалога, ответить на вопросы со всеми слоями памяти, затем очистить каждый слой по очереди и посмотреть, на какие вопросы агент перестаёт отвечать.</p>
      <button id="experiment" class="sbtn primary" type="button">Запустить эксперимент</button>
    </section>
    <section class="card">
      <div class="card-title">Сброс</div>
      <p class="hint">Очищает историю, ветки, чекпоинты и рабочую память. Долговременная память сохранится.</p>
      <button id="clear" class="sbtn danger" type="button">Сбросить диалог</button>
    </section>
  </aside>
</div>
<div class="modal-backdrop" id="memBackdrop">
<div id="memoryPanel">
  <div class="mem-head">
    <b>Память — три слоя</b>
    <span class="mem-close" id="memClose" title="Закрыть">&#10005;</span>
  </div>
  <div id="memRules" class="mem-rules"></div>
  <div class="mem-eq">
    <div class="mem-layer">
      <div class="mem-layer-title">Краткосрочная (текущий диалог)
        <button id="memStmClear" type="button" title="Очистить диалог текущей ветки: сообщения, сводку и checkpoint. Рабочая и долговременная память сохранятся">очистить</button>
      </div>
      <div id="memStm"></div>
    </div>
    <div class="mem-layer">
      <div class="mem-layer-title">Рабочая (текущая задача)
        <select id="memPromoteCat" title="Категория переноса: «авто» определяет её по смыслу факта">
          <option value="auto">категория: авто</option>
          <option value="profile">Профиль</option>
          <option value="decision">Решения</option>
          <option value="knowledge">Знания</option>
        </select>
        <button id="memPromote" type="button" title="Перенести отмеченные галочками факты в долговременную память">&#8594; в долговременную</button>
        <button id="memWmClear" type="button" title="Очистить рабочую память текущей ветки (все факты задачи). Долговременная память сохранится">очистить</button>
      </div>
      <div id="memWm"></div>
    </div>
    <div class="mem-layer">
      <div class="mem-layer-title">Долговременная (профиль, решения, знания)
        <button id="memClear" type="button" title="Очистить всю долговременную память">очистить</button>
      </div>
      <div id="memLtm"></div>
      <div class="mem-add">
        <select id="memCat">
          <option value="profile">Профиль</option>
          <option value="decision">Решения</option>
          <option value="knowledge">Знания</option>
        </select>
        <input id="memKey" placeholder="ключ" autocomplete="off">
        <input id="memVal" placeholder="значение" autocomplete="off">
        <button id="memAdd" type="button">+</button>
      </div>
    </div>
    </div>
  </div>
</div>
<div class="modal-backdrop" id="profBackdrop">
<div id="profilePanel">
  <div class="mem-head">
    <b>Профили пользователя — персонализация</b>
    <span class="mem-close" id="profClose" title="Закрыть">&#10005;</span>
  </div>
  <div class="mem-rules">
    <div class="rule"><b>Правило попадания в промпт:</b> активный профиль уходит в каждый запрос первым
      system-сообщением и задаёт, КАК отвечать (стиль, формат, тон, ограничения). Ручные правила
      задаёт пользователь; правила с меткой «из диалога» ассистент выводит сам из ваших просьб
      («отвечай короче», «не надо кода») — их можно удалить.</div>
  </div>
  <div id="profList" class="prof-list"></div>
  <div class="prof-learned-note" id="profLearnedNote" style="display:none;"></div>
  <div class="prof-form" id="profForm">
    <div><label>Название</label><input id="profName" placeholder="Например: Кратко и формально" autocomplete="off"></div>
    <div><label>Кто пользователь (необязательно)</label><input id="profDesc" placeholder="Например: руководитель, ценит время" autocomplete="off"></div>
    <div><label>Стиль</label>
      <select id="profStyle">
        <option value="">— не задано —</option>
        <option value="brief">Кратко (2–4 предложения)</option>
        <option value="detailed">Подробно (развёрнуто, с примерами)</option>
      </select>
    </div>
    <div><label>Код</label>
      <select id="profCode">
        <option value="">— не задано —</option>
        <option value="code">С примерами кода</option>
        <option value="no_code">Без кода</option>
      </select>
    </div>
    <div><label>Тон</label>
      <select id="profTone">
        <option value="">— не задано —</option>
        <option value="formal">Формальный («Вы», по-деловому)</option>
        <option value="informal">Неформальный («ты», дружелюбно)</option>
      </select>
    </div>
    <div><label>Доп. ограничения — по одному на строку</label>
      <textarea id="profExtra" placeholder="Например:&#10;Отвечай на русском языке&#10;Не используй эмодзи"></textarea>
    </div>
    <div class="save-row">
      <button id="profSave" type="button">Создать профиль</button>
      <button id="profCancel" type="button" class="ghost" style="display:none;">Отменить редактирование</button>
      <span id="profFormHint" style="font-size:11.5px;color:#64748b;"></span>
    </div>
  </div>
</div>
</div>
<div class="modal-backdrop" id="invBackdrop">
<div id="invariantPanel">
  <div class="mem-head">
    <b>Инварианты — правила, которые нельзя нарушать</b>
    <span class="mem-close" id="invClose" title="Закрыть">&#10005;</span>
  </div>
  <div class="inv-note">
    Инварианты (архитектура, техрешения, стек, бизнес-правила) хранятся <b>отдельно от диалога</b>
    и уходят в каждый запрос первым system-сообщением. Каждый ответ ассистента <b>проверяется кодом до
    выдачи</b>: нарушающий ответ перезапрашивается или заменяется отказом. Изменить инвариант можно
    <b>только явной командой</b> — здесь или сообщением <code>/invariant …</code> в чате; просьбы в обычном
    диалоге ничего не меняют, каждое изменение фиксируется в журнале (что на что заменено).
  </div>
  <div id="invList" class="prof-list"></div>
  <div class="prof-form" id="invForm">
    <div><label>Категория</label>
      <select id="invCat">
        <option value="architecture">Архитектура</option>
        <option value="decision">Техническое решение</option>
        <option value="stack">Ограничение стека</option>
        <option value="business">Бизнес-правило</option>
      </select>
    </div>
    <div><label>Название</label><input id="invTitle" placeholder="Например: Секреты — только в .env" autocomplete="off"></div>
    <div class="full"><label>Формулировка правила</label>
      <textarea id="invRule" placeholder="Что именно запрещено / обязательно"></textarea>
    </div>
    <div class="full"><label>Шаблоны проверки ответа (регулярные выражения через «;» — по ним код ловит нарушение)</label>
      <input id="invPatterns" placeholder="например: пароль в коде; aws_key; hardcode" autocomplete="off">
    </div>
    <div class="save-row">
      <button id="invSave" type="button">Добавить инвариант</button>
      <button id="invCancel" type="button" class="ghost" style="display:none;">Отменить редактирование</button>
      <span id="invFormHint" style="font-size:11.5px;color:#64748b;"></span>
    </div>
  </div>
  <div class="inv-log" id="invLog"></div>
</div>
</div>
<script>
const input = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const chatEl = document.getElementById('chat');
const modelSelect = document.getElementById('modelSelect');
const branchList = document.getElementById('branchList');
const checkpointBtn = document.getElementById('checkpoint');
const branchAddBtn = document.getElementById('branchAdd');
const branchDelBtn = document.getElementById('branchDel');
const modelChip = document.getElementById('modelChip');
const memBackdrop = document.getElementById('memBackdrop');
const scrim = document.getElementById('scrim');
const clearBtn = document.getElementById('clear');
let currentBranch = 1;
let rootBranchId = 1;

function syncModel(){ modelChip.textContent = modelSelect.value; }

/* ---------- task lifecycle chip ---------- */
const taskChip = document.getElementById('taskChip');
const STAGE_LABELS = {
  idle: 'задачи нет',
  planning: 'планирование',
  awaiting_approval: 'ожидание утверждения плана',
  execution: 'реализация',
  validation: 'валидация',
  done: 'завершено',
};
function syncTaskChip(task){
  if (!task) return;
  const label = STAGE_LABELS[task.stage] || task.stage;
  taskChip.innerHTML = '<span class="task-stage">Задача: ' + esc(label) + '</span>' +
    (task.active && task.step ? '<span class="task-step">шаг: ' + esc(task.step) + '</span>' : '');
  taskChip.style.borderColor = task.active ? 'rgba(139,92,246,.5)' : '';
  taskChip.style.color = task.active ? '#ddd6fe' : '';
}
async function loadTaskState(){
  try {
    const data = await (await fetch('/api/task/state?branch_id=' + currentBranch)).json();
    syncTaskChip(data.task);
  } catch (e) {}
}

try {
  const savedModel = localStorage.getItem('model');
  if (savedModel && [...modelSelect.options].some(o => o.value === savedModel)) modelSelect.value = savedModel;
} catch (e) {}
syncModel();
modelSelect.addEventListener('change', () => {
  try { localStorage.setItem('model', modelSelect.value); } catch (e) {}
  syncModel();
});

/* ---------- profiles ---------- */
const profileSelect = document.getElementById('profileSelect');
const profileChip = document.getElementById('profileChip');
const profilesBtn = document.getElementById('profilesBtn');
const profBackdrop = document.getElementById('profBackdrop');
const profList = document.getElementById('profList');
const profName = document.getElementById('profName');
const profDesc = document.getElementById('profDesc');
const profStyle = document.getElementById('profStyle');
const profCode = document.getElementById('profCode');
const profTone = document.getElementById('profTone');
const profExtra = document.getElementById('profExtra');
const profSaveBtn = document.getElementById('profSave');
const profCancelBtn = document.getElementById('profCancel');
const profFormHint = document.getElementById('profFormHint');
const KIND_LABELS = { style: 'Стиль', format: 'Формат', tone: 'Тон', constraint: 'Ограничение' };
let editingProfileId = null;

function syncProfileChip(profiles){
  const active = (profiles || []).find(p => p.is_active);
  profileChip.textContent = 'Профиль: ' + (active ? active.name : '—');
}

async function loadProfiles(){
  try {
    const data = await (await fetch('/api/profiles')).json();
    const profiles = data.profiles || [];
    profileSelect.innerHTML = profiles.map(p =>
      '<option value="' + p.id + '"' + (p.is_active ? ' selected' : '') + '>' +
      esc(p.name) + (p.is_active ? ' (активный)' : '') + '</option>').join('');
    syncProfileChip(profiles);
    return profiles;
  } catch (e) { return []; }
}

profileSelect.addEventListener('change', async () => {
  try {
    await fetch('/api/profiles/activate', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile_id: parseInt(profileSelect.value, 10) }) });
    await loadProfiles();
  } catch (err) {}
});

function setProfOpen(v){
  profBackdrop.classList.toggle('open', v);
  if (v) { resetProfForm(); loadProfiles(); renderProfList(); }
}
profilesBtn.addEventListener('click', () => setProfOpen(true));
document.getElementById('profClose').addEventListener('click', () => setProfOpen(false));
profBackdrop.addEventListener('click', (e) => { if (e.target === profBackdrop) setProfOpen(false); });

function resetProfForm(){
  editingProfileId = null;
  profName.value = ''; profDesc.value = '';
  profStyle.value = ''; profCode.value = ''; profTone.value = ''; profExtra.value = '';
  profSaveBtn.textContent = 'Создать профиль';
  profCancelBtn.style.display = 'none';
  profFormHint.textContent = '';
}

profCancelBtn.addEventListener('click', resetProfForm);

function fillProfForm(p){
  editingProfileId = p.id;
  profName.value = p.name || '';
  profDesc.value = p.description || '';
  const byKind = {};
  (p.rules || []).filter(r => r.source === 'manual').forEach(r => { byKind[r.kind] = r.text; });
  const style = byKind.style || '', format = byKind.format || '', tone = byKind.tone || '';
  profStyle.value = /подробно/i.test(style) ? 'detailed' : (/кратко/i.test(style) ? 'brief' : '');
  profCode.value = /не приводи код|без кода/i.test(format) ? 'no_code' : (/код/i.test(format) ? 'code' : '');
  profTone.value = /неформально|на «ты»/i.test(tone) ? 'informal' :
                   (/формально|по-деловому/i.test(tone) ? 'formal' : '');
  profExtra.value = (p.rules || []).filter(r => r.source === 'manual' && r.kind === 'constraint')
    .map(r => r.text).join('\n');
  profSaveBtn.textContent = 'Сохранить изменения';
  profCancelBtn.style.display = '';
  profFormHint.textContent = 'Редактирование профиля #' + p.id +
    ' (правила «из диалога» останутся)';
}

profSaveBtn.addEventListener('click', async () => {
  const name = profName.value.trim();
  if (!name) { profFormHint.textContent = 'Укажите название профиля.'; return; }
  const body = {
    name: name,
    description: profDesc.value.trim(),
    style: profStyle.value || null,
    code: profCode.value || null,
    tone: profTone.value || null,
    constraints: profExtra.value.split('\n').map(s => s.trim()).filter(Boolean),
  };
  if (editingProfileId) body.profile_id = editingProfileId;
  try {
    const r = await (await fetch('/api/profiles/save', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
    if (r.error) { profFormHint.textContent = 'Ошибка: ' + r.error; return; }
    resetProfForm();
    await loadProfiles();
    renderProfList();
  } catch (err) { profFormHint.textContent = 'Ошибка запроса: ' + err; }
});

async function renderProfList(){
  try {
    const data = await (await fetch('/api/profiles/full')).json();
    const profiles = data.profiles || [];
    profList.innerHTML = profiles.map(p => {
      const rules = p.rules || [];
      const rulesHtml = rules.length ? rules.map(r =>
        '<div class="prof-rule"><span class="kind">' + esc(KIND_LABELS[r.kind] || r.kind) + '</span>' +
        (r.source === 'dialog' ? '<span class="src" title="Выведено ассистентом из диалога">из диалога</span>' : '') +
        '<span style="flex:1;min-width:0;word-break:break-word;">' + esc(r.text) + '</span>' +
        '<span class="del" data-rule="' + r.id + '" title="Удалить правило">&#10005;</span></div>'
      ).join('') : '<div class="mem-empty">нет правил</div>';
      return '<div class="prof-card' + (p.is_active ? ' active' : '') + '">' +
        '<div class="prof-head"><span class="prof-name">' + esc(p.name) + '</span>' +
        (p.is_active ? '<span class="prof-active-badge">активный</span>' : '') +
        '<div class="prof-actions">' +
        (p.is_active ? '' : '<button data-act="activate" data-id="' + p.id + '">Сделать активным</button>') +
        '<button data-act="edit" data-id="' + p.id + '">Изменить</button>' +
        '<button data-act="delete" data-id="' + p.id + '" class="danger">Удалить</button>' +
        '</div></div>' +
        (p.description ? '<div class="prof-desc">' + esc(p.description) + '</div>' : '') +
        rulesHtml + '</div>';
    }).join('');
    profList.querySelectorAll('[data-act]').forEach(el => el.addEventListener('click', async () => {
      const id = parseInt(el.dataset.id, 10);
      const act = el.dataset.act;
      if (act === 'activate') {
        await fetch('/api/profiles/activate', { method: 'POST',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ profile_id: id }) });
        await loadProfiles(); renderProfList();
      } else if (act === 'edit') {
        const p = profiles.find(x => x.id === id);
        if (p) fillProfForm(p);
      } else if (act === 'delete') {
        if (!confirm('Удалить профиль со всеми его правилами?')) return;
        await fetch('/api/profiles/delete', { method: 'POST',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ profile_id: id }) });
        await loadProfiles(); renderProfList();
      }
    }));
    profList.querySelectorAll('[data-rule]').forEach(el => el.addEventListener('click', async () => {
      await fetch('/api/profiles/rule/delete', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rule_id: parseInt(el.dataset.rule, 10) }) });
      await loadProfiles(); renderProfList();
    }));
  } catch (err) { /* ignore */ }
}

/* ---------- invariants ---------- */
const invariantsBtn = document.getElementById('invariantsBtn');
const invBackdrop = document.getElementById('invBackdrop');
const invList = document.getElementById('invList');
const invCat = document.getElementById('invCat');
const invTitle = document.getElementById('invTitle');
const invRule = document.getElementById('invRule');
const invPatterns = document.getElementById('invPatterns');
const invSaveBtn = document.getElementById('invSave');
const invCancelBtn = document.getElementById('invCancel');
const invFormHint = document.getElementById('invFormHint');
const invChip = document.getElementById('invChip');
const INV_CAT_LABELS = {
  architecture: 'Архитектура', decision: 'Техрешение',
  stack: 'Стек', business: 'Бизнес-правило',
};
let editingInvariantId = null;

function syncInvChip(items){
  invChip.textContent = 'Инварианты: ' + (items ? items.length : '—');
}

async function loadInvariants(){
  try {
    const data = await (await fetch('/api/invariants')).json();
    const items = data.invariants || [];
    syncInvChip(items);
    renderInvList(items);
    renderInvLog(data.log || []);
    return items;
  } catch (e) { return []; }
}

function renderInvList(items){
  invList.innerHTML = items.length ? items.map(inv =>
    '<div class="prof-card">' +
      '<div class="prof-head">' +
        '<span class="inv-code">' + esc(inv.code) + '</span>' +
        '<span class="inv-cat">' + esc(INV_CAT_LABELS[inv.category] || inv.category) + '</span>' +
        '<span class="prof-name">' + esc(inv.title) + '</span>' +
        '<div class="prof-actions">' +
          '<button data-act="edit" data-id="' + inv.id + '">Изменить</button>' +
          '<button data-act="delete" data-id="' + inv.id + '" class="danger">Удалить</button>' +
        '</div>' +
      '</div>' +
      '<div class="inv-rule">' + esc(inv.rule) + '</div>' +
      ((inv.patterns || []).length ?
        '<div class="inv-patterns">' + inv.patterns.map(p => '<span>' + esc(p) + '</span>').join('') + '</div>' : '') +
    '</div>'
  ).join('') : '<div class="mem-empty">Инвариантов нет — все удалены явными командами.</div>';
  invList.querySelectorAll('[data-act]').forEach(el => el.addEventListener('click', async () => {
    const id = parseInt(el.dataset.id, 10);
    if (el.dataset.act === 'edit') {
      const items2 = await loadInvariants();
      const inv = (items2 || []).find(x => x.id === id);
      if (inv) fillInvForm(inv);
    } else {
      if (!confirm('Удалить инвариант? Удаление зафиксируется в журнале.')) return;
      const r = await (await fetch('/api/invariants/remove', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }) })).json();
      if (r.error) { alert('Ошибка: ' + r.error); return; }
      await loadInvariants();
    }
  }));
}

function renderInvLog(entries){
  const box = document.getElementById('invLog');
  if (!entries.length) {
    box.innerHTML = '<div class="card-title">Журнал изменений</div><div class="mem-empty">Изменений ещё не было: ' +
      'инварианты заданы сидом при первом запуске.</div>';
    return;
  }
  box.innerHTML = '<div class="card-title">Журнал изменений (что на что заменено)</div><table>' +
    '<tr><th>когда</th><th>действие</th><th>что изменилось</th><th>источник</th></tr>' +
    entries.map(e => {
      const t = (e.ts || '').replace('T', ' ').slice(0, 19);
      return '<tr><td>' + esc(t) + '</td><td>' + esc(e.action) + '</td><td>' + esc(e.detail || '') +
        '</td><td class="src">' + esc(e.source || '') + '</td></tr>';
    }).join('') + '</table>';
}

function resetInvForm(){
  editingInvariantId = null;
  invTitle.value = ''; invRule.value = ''; invPatterns.value = '';
  invSaveBtn.textContent = 'Добавить инвариант';
  invCancelBtn.style.display = 'none';
  invFormHint.textContent = '';
}

function fillInvForm(inv){
  editingInvariantId = inv.id;
  invCat.value = inv.category;
  invTitle.value = inv.title || '';
  invRule.value = inv.rule || '';
  invPatterns.value = (inv.patterns || []).join('; ');
  invSaveBtn.textContent = 'Сохранить изменения';
  invCancelBtn.style.display = '';
  invFormHint.textContent = 'Редактирование ' + inv.code + ' — изменение попадёт в журнал.';
}

invCancelBtn.addEventListener('click', resetInvForm);

invSaveBtn.addEventListener('click', async () => {
  const body = {
    category: invCat.value,
    title: invTitle.value.trim(),
    rule: invRule.value.trim(),
    patterns: invPatterns.value.split(';').map(s => s.trim()).filter(Boolean),
  };
  if (!body.title) { invFormHint.textContent = 'Укажите название инварианта.'; return; }
  if (!body.rule) { invFormHint.textContent = 'Укажите формулировку правила.'; return; }
  const url = editingInvariantId ? '/api/invariants/change' : '/api/invariants/add';
  if (editingInvariantId) body.id = editingInvariantId;
  try {
    const r = await (await fetch(url, { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
    if (r.error) { invFormHint.textContent = 'Ошибка: ' + r.error; return; }
    resetInvForm();
    await loadInvariants();
  } catch (err) { invFormHint.textContent = 'Ошибка запроса: ' + err; }
});

function setInvOpen(v){
  invBackdrop.classList.toggle('open', v);
  if (v) { resetInvForm(); loadInvariants(); }
}
invariantsBtn.addEventListener('click', () => setInvOpen(true));
document.getElementById('invClose').addEventListener('click', () => setInvOpen(false));
invBackdrop.addEventListener('click', (e) => { if (e.target === invBackdrop) setInvOpen(false); });

function closeDrawers(){ document.body.classList.remove('drawer-left', 'drawer-right'); }
document.getElementById('toggleLeft').addEventListener('click', () => {
  document.body.classList.toggle('drawer-left');
  document.body.classList.remove('drawer-right');
});
document.getElementById('toggleRight').addEventListener('click', () => {
  document.body.classList.toggle('drawer-right');
  document.body.classList.remove('drawer-left');
});
scrim.addEventListener('click', closeDrawers);
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (profBackdrop.classList.contains('open')) setProfOpen(false);
  else if (memBackdrop.classList.contains('open')) setMemOpen(false);
  else if (invBackdrop.classList.contains('open')) setInvOpen(false);
  else closeDrawers();
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
  branchList.innerHTML = branches.map(b =>
    '<div class="branch-item' + (b.id === currentBranch ? ' active' : '') + '" data-id="' + b.id + '">' +
    '<span class="b-icon">' + (b.id === rootBranchId ? '\u2302' : '\u2442') + '</span>' +
    '<span class="b-name">' + esc(b.name) + '</span>' +
    '<span class="b-meta">' + b.count + ' сообщ.' +
    (b.checkpoint_len !== null && b.checkpoint_len !== undefined ? ' \u2691' + b.checkpoint_len : '') +
    '</span></div>').join('');
  branchList.querySelectorAll('.branch-item').forEach(el =>
    el.addEventListener('click', () => switchBranch(parseInt(el.dataset.id, 10))));
  try { localStorage.setItem('branchId', currentBranch); } catch (e) {}
  branchDelBtn.disabled = currentBranch === rootBranchId || branches.length <= 1;
}

async function switchBranch(id){
  currentBranch = id;
  try { localStorage.setItem('branchId', currentBranch); } catch (e) {}
  chatEl.innerHTML = '';
  await loadHistory();
  await loadBranches(true);
  loadTaskState();
  closeDrawers();
}

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
    loadTaskState();
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
    loadTaskState();
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
  if (contextInfo && contextInfo.mode === 'memory') {
    metaText += ' · слои: инварианты';
    if (contextInfo.invariants_used) metaText += ' ' + (contextInfo.invariants_tokens || 0) + ' ток.';
    metaText += ' · профиль';
    if (contextInfo.profile_used) metaText += ' ' + (contextInfo.profile_tokens || 0) + ' ток.';
    metaText += ' · память';
    if (contextInfo.ltm_used) metaText += ' LTM ' + contextInfo.ltm_tokens + ' ток.';
    if (contextInfo.facts_used) metaText += ' · WM ' + contextInfo.facts_tokens + ' ток.';
    metaText += ' · STM ' + (contextInfo.stm_tokens || 0) + ' ток. (' + contextInfo.as_is + ' сообщ.)';
  }
  return metaText;
}

function appendMeta(bubble, metaText){
  if (!metaText) return;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = metaText;
  bubble.appendChild(meta);
}

/* ---------- send / stop button ---------- */
const SEND_ICON = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>';
const STOP_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="#fff"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
let busy = false;
let abortController = null;

function setBusy(on){
  busy = on;
  sendBtn.classList.toggle('stopping', on);
  sendBtn.innerHTML = on ? STOP_ICON : SEND_ICON;
  sendBtn.title = on ? 'Остановить агента' : 'Отправить';
}

async function doStop(){
  if (!busy) return;
  if (abortController) abortController.abort();
  try {
    await fetch('/api/chat/cancel', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ branch_id: currentBranch }) });
  } catch (e) {}
  hideTyping();
  addRow('assistant', '⏹ Остановлено пользователем. Агент завершает текущий шаг, частичный прогресс сохраняется в истории.');
  setBusy(false);
  input.focus();
  scroll();
  /* сервер дописывает частичный результат в историю — подтянуть её чуть позже */
  setTimeout(async () => {
    try {
      chatEl.innerHTML = '';
      await loadHistory();
      const d = await (await fetch('/api/task/state?branch_id=' + currentBranch)).json();
      syncTaskChip(d.task);
    } catch (e) {}
  }, 3000);
}

sendBtn.addEventListener('click', (e) => {
  if (busy) { e.preventDefault(); doStop(); }
});

async function run(){
  const text = input.value.trim();
  if (!text || busy) return;
  input.value = '';
  addRow('user', text);
  showTyping();
  setBusy(true);
  abortController = new AbortController();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text,
                             model: modelSelect.value, branch_id: currentBranch }),
      signal: abortController.signal,
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
      const pu = data.profile_updates;
      if (pu && pu.profile && (pu.added || []).length) {
        metaText = (metaText ? metaText + ' · ' : '') +
          'профиль пополнен из диалога: +' + pu.added.length + ' прав. (' +
          pu.added.map(r => r.text).join('; ') + ')';
      }
      const task = data.task;
      if (task && task.active) {
        metaText = (metaText ? metaText + ' · ' : '') +
          'этап: ' + (STAGE_LABELS[task.stage] || task.stage) + ' (' + task.step + ')';
      }
      if (data.fsm_rejections > 0) {
        metaText = (metaText ? metaText + ' · ' : '') +
          'автомат отклонил переходов: ' + data.fsm_rejections;
      }
      const inv = data.invariants;
      if (inv && inv.checked) {
        if (inv.blocked) {
          metaText = (metaText ? metaText + ' · ' : '') +
            'инварианты: ответ заменён отказом кода (' + (inv.violations || []).join(', ') +
            '); переписано попыток: ' + inv.rewrites;
        } else if (inv.rewrites > 0) {
          metaText = (metaText ? metaText + ' · ' : '') +
            'инварианты: нарушавший ответ перезапрошен (' + (inv.violations || []).join(', ') + ')';
        } else {
          metaText = (metaText ? metaText + ' · ' : '') +
            'инварианты: проверено кодом (' + inv.count + ' прав.)';
        }
      }
      if (data.stopped) {
        metaText = (metaText ? metaText + ' · ' : '') + '⏹ остановлено пользователем';
      }
      appendMeta(bubble, metaText);
      syncTaskChip(task);
      if (data.command) loadInvariants();  // чип инвариантов после /invariant-команды
    }
  } catch (err) {
    hideTyping();
    if (!(err && err.name === 'AbortError')) {  // AbortError обработан doStop
      addRow('assistant', 'Ошибка запроса: ' + err, true);
    }
  } finally {
    abortController = null;
    setBusy(false);
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
  if (!confirm('Очистить историю диалога, ветки, чекпоинты и рабочую память текущей задачи? Долговременная память (профиль, решения, знания) сохранится.')) return;
  await fetch('/api/clear', { method: 'POST' });
  chatEl.innerHTML = '';
  await loadBranches(false);
  addRow('assistant', 'История очищена. Долговременная память сохранена. Начинаем заново!');
  loadMemory();
  syncTaskChip({ stage: 'idle', active: false });
});

const memoryPanel = document.getElementById('memoryPanel');
const memoryBtn = document.getElementById('memoryBtn');
let memOpen = false;
function setMemOpen(v){
  memOpen = v;
  memBackdrop.classList.toggle('open', v);
  if (v) loadMemory();
}
memoryBtn.addEventListener('click', () => setMemOpen(!memOpen));
document.getElementById('memClose').addEventListener('click', () => setMemOpen(false));
memBackdrop.addEventListener('click', (e) => { if (e.target === memBackdrop) setMemOpen(false); });

async function loadMemory(){
  try {
    const data = await (await fetch('/api/memory?branch_id=' + currentBranch)).json();
    const rules = data.rules || [];
    document.getElementById('memRules').innerHTML = rules.map(r =>
      '<div class="rule"><b>' + esc(r.title) + '</b> — ' + esc(r.rule) +
      '<br><span style="color:#64748b;">хранение: ' + esc(r.storage) + '</span></div>'
    ).join('');

    const stm = data.short_term;
    document.getElementById('memStm').innerHTML =
      '<div class="mem-item ' + (stm.messages ? '' : 'mem-empty') + '">' +
      (stm.messages ? 'Веток: ' + stm.branches + ' · сообщений в ветке: ' + stm.messages : 'пусто') + '</div>';

    const wm = data.working || {};
    const wmKeys = Object.keys(wm);
    document.getElementById('memWm').innerHTML = wmKeys.length
      ? wmKeys.map(k => '<div class="mem-item">' +
          '<input type="checkbox" class="mem-wm-check" data-key="' + esc(k) + '" title="Отметить для переноса в долговременную память">' +
          '<span class="mem-val"><span class="k">' + esc(k) + '</span>: ' + esc(wm[k]) + '</span>' +
          '<span class="del" data-key="' + esc(k) + '" title="Удалить факт из рабочей памяти">&#10005;</span></div>').join('')
      : '<div class="mem-empty">пусто — факты появятся после сообщений</div>';
    document.querySelectorAll('#memWm .del').forEach(el => el.addEventListener('click', async () => {
      await fetch('/api/memory/fact/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ branch_id: currentBranch, key: el.dataset.key }) });
      loadMemory();
    }));

    const ltm = data.long_term || [];
    const catLabel = {profile:'Профиль', decision:'Решения', knowledge:'Знания'};
    document.getElementById('memLtm').innerHTML = ltm.length
      ? ltm.map(e => '<div class="mem-item"><span class="cat">' + esc(catLabel[e.category] || e.category) + '</span>' +
        '<span class="k">' + esc(e.key) + '</span>: ' + esc(e.value) +
        '<span class="del" data-id="' + e.id + '" title="Удалить">&#10005;</span></div>').join('')
      : '<div class="mem-empty">пусто</div>';
    document.querySelectorAll('#memLtm .del').forEach(el => el.addEventListener('click', async () => {
      await fetch('/api/memory/ltm/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: parseInt(el.dataset.id, 10) }) });
      loadMemory();
    }));
  } catch (err) { /* ignore */ }
}

document.getElementById('memPromote').addEventListener('click', async () => {
  const keys = Array.from(document.querySelectorAll('.mem-wm-check:checked')).map(el => el.dataset.key);
  if (!keys.length) {
    addRow('assistant', 'Отметьте галочками факты в панели «Память», которые перенести в долговременную.', true);
    return;
  }
  const cat = document.getElementById('memPromoteCat').value;
  const body = { branch_id: currentBranch, keys: keys };
  if (cat !== 'auto') body.category = cat;
  const r = await fetch('/api/memory/promote', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) });
  const d = await r.json();
  if (d.error) { addRow('assistant', 'Ошибка: ' + d.error, true); return; }
  addRow('assistant', 'В долговременную память перенесено фактов: ' + d.moved + '.');
  loadMemory();
});

document.getElementById('memStmClear').addEventListener('click', async () => {
  if (!confirm('Очистить диалог текущей ветки (сообщения, сводку и checkpoint)? Рабочая и долговременная память сохранятся.')) return;
  await fetch('/api/memory/clear/stm', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ branch_id: currentBranch }) });
  chatEl.innerHTML = '';
  await loadHistory();
  await loadBranches(true);
  loadMemory();
  syncTaskChip({ stage: 'idle', active: false });
});

document.getElementById('memWmClear').addEventListener('click', async () => {
  if (!confirm('Очистить рабочую память текущей задачи (все факты ветки)? Долговременная память сохранится.')) return;
  await fetch('/api/memory/clear/wm', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ branch_id: currentBranch }) });
  loadMemory();
});

document.getElementById('memClear').addEventListener('click', async () => {
  if (!confirm('Очистить всю долговременную память (профиль, решения, знания)?')) return;
  await fetch('/api/memory/ltm/clear', { method: 'POST' });
  loadMemory();
});

document.getElementById('memAdd').addEventListener('click', async () => {
  const key = document.getElementById('memKey').value.trim();
  const val = document.getElementById('memVal').value.trim();
  if (!key || !val) return;
  const category = document.getElementById('memCat').value;
  const r = await fetch('/api/memory/ltm/add', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ category, key, value: val }) });
  const d = await r.json();
  if (d.error) { addRow('assistant', 'Ошибка: ' + d.error, true); return; }
  document.getElementById('memKey').value = '';
  document.getElementById('memVal').value = '';
  loadMemory();
});


const expBtn = document.getElementById('experiment');
const profExpBtn = document.getElementById('profileExperiment');
const exp = { running: false, lastKey: null, statusEl: null };

function setExpUI(disabled){
  input.disabled = disabled;
  sendBtn.disabled = disabled;
  expBtn.disabled = disabled;
  profExpBtn.disabled = disabled;
  checkpointBtn.disabled = disabled;
  branchAddBtn.disabled = disabled;
  branchDelBtn.disabled = disabled || currentBranch === rootBranchId;
  modelSelect.disabled = disabled;
  profileSelect.disabled = disabled;
  profilesBtn.disabled = disabled;
  memoryBtn.disabled = disabled;
  clearBtn.disabled = disabled;
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
  const key = data.stage + ':' + (data.layer_run || 0);
  if (exp.lastKey !== key) {
    expDivider(data.stage_label);
    exp.lastKey = key;
  }
  if (p.user) addRow('user', p.user);
  if (p.reply || p.ok !== undefined) {
    const bubble = addRow('assistant', p.reply || '(пустой ответ)');
    let metaText = p.tokens && p.tokens.request !== undefined
      ? tokenMeta(p.tokens, p.context) : null;
    if (p.ok !== undefined && p.ok !== null) {
      const verdict = p.ok ? 'верно' : 'неверно';
      metaText = (metaText ? metaText + ' · ' : '') +
        verdict + ' (деталей: ' + p.facts_found + '/' + p.facts_total + ')';
    }
    appendMeta(bubble, metaText);
  }
  const stepNo = (data.step_index || 0) + 1;
  expStatus('Эксперимент · ' + data.stage_label +
    (data.step_total ? ' · шаг ' + stepNo + '/' + data.step_total : ''));
}

function num(n){ return (n || 0).toLocaleString('ru-RU'); }

const LAYER_LABELS = {
  ltm: 'долговременная (LTM)',
  wm: 'рабочая (WM)',
  stm: 'краткосрочная (STM)'
};

function renderReport(rep){
  const rows = [rep.baseline].concat(rep.layers);
  const div = document.createElement('div');
  div.className = 'report';

  const qt = rep.questions_total;
  const dt = rep.details_total;
  const totals = rows.map(r => r.api.prompt + r.api.completion);
  const bestRem = Math.max.apply(null, rows.map(r => r.remembered));
  const minTok = Math.min.apply(null, totals);

  let html = '<h3>Эксперимент с памятью: ' + esc(rep.title) + '</h3>';
  html += '<p style="font-size:12.5px;color:#94a3b8;">Базовый прогон — все слои на месте. ' +
          'Затем каждый слой очищается по очереди (остальные восстанавливаются), и те же вопросы задаются заново.</p>';
  html += '<table><tr><th>Прогон</th><th>Верно вопросов</th><th>Деталей найдено</th>' +
          '<th>Токены API (prompt + completion)</th><th>Стоимость</th><th>Перестал отвечать на</th></tr>';
  rows.forEach((r, i) => {
    html += '<tr>' +
      '<td><b>' + esc(r.label) + '</b></td>' +
      '<td class="' + (r.remembered === bestRem ? 'good' : '') + '">' + r.remembered + ' / ' + qt + '</td>' +
      '<td>' + r.details_found + ' / ' + dt + '</td>' +
      '<td class="' + (totals[i] === minTok ? 'good' : '') + '">' + num(totals[i]) +
        '<div style="font-size:11px;color:#94a3b8;">prompt ' + num(r.api.prompt) +
        ' / completion ' + num(r.api.completion) + '</div></td>' +
      '<td>' + (r.api.cost ? r.api.cost.toFixed(4) + ' \u20bd' : '&mdash;') + '</td>' +
      '<td style="font-size:12px;">' + (r.broken && r.broken.length
        ? '<ul style="margin:0;padding-left:16px;">' +
          r.broken.map(q => '<li>' + esc(q) + '</li>').join('') + '</ul>'
        : (r === rep.baseline ? '&mdash;' : 'ничего — все вопросы по-прежнему верны')) + '</td>' +
      '</tr>';
  });
  html += '</table>';

  const ltk = rep.baseline.layer_tokens;
  html += '<div class="verdict" style="margin-top:10px;">' +
    '<b>Токены слоёв в базовом прогоне</b> (в среднем на вопрос): ' +
    'LTM — ' + num(ltk.ltm) + ' ток., рабочая (WM) — ' + num(ltk.wm) + ' ток., ' +
    'краткосрочная (STM) — ' + num(ltk.stm) + ' ток.</div>';

  html += '<div class="verdict">';
  rep.layers.forEach(l => {
    html += '<div style="margin-top:4px;"><b>' + esc(l.label) + ':</b> ' +
      (l.broken.length
        ? 'агент перестал отвечать на ' + l.broken.length + ' из ' + qt + ' вопросов'
        : 'все вопросы по-прежнему верны') + '</div>';
  });
  html += '</div>';

  html += '<details><summary>Матрица ответов по вопросам</summary>';
  html += '<table><tr><th>Вопрос</th><th>Ожидаемый слой</th>';
  rows.forEach(r => html += '<th>' + esc(r.label) + '</th>');
  html += '</tr>';
  rep.baseline.qa.forEach((q, j) => {
    html += '<tr><td style="font-size:12px;">' + esc(q.question) + '</td>' +
      '<td style="font-size:11px;color:#94a3b8;">' + esc(LAYER_LABELS[q.layer] || q.layer) + '</td>';
    rows.forEach(r => {
      const e = r.qa[j];
      html += e
        ? '<td class="' + (e.ok ? 'good' : 'bad') + '" style="font-size:12px;">' +
          (e.ok ? '✓' : '✗') + ' ' + e.facts_found + '/' + e.facts_total + '</td>'
        : '<td>—</td>';
    });
    html += '</tr>';
  });
  html += '</table></details>';

  html += '<details><summary>Ответы на вопросы (базовый прогон)</summary>';
  rep.baseline.qa.forEach((q, j) => {
    html += '<div class="qa-item">' +
      '<div class="qa-q">' + (j + 1) + '. ' + esc(q.question) +
      ' <span style="font-size:11px;color:#64748b;">(' + esc(LAYER_LABELS[q.layer] || q.layer) + ')</span></div>' +
      '<div class="qa-answer">' + esc(q.answer || '(пустой ответ)') + '</div>' +
      '<div class="' + (q.ok ? 'good' : 'bad') + '" style="font-size:11.5px;margin-top:2px;">' +
      (q.ok ? 'верно' : 'неверно') + ' · деталей: ' + q.facts_found + '/' + q.facts_total + '</div>' +
      '<div class="qa-expected">Ожидалось: ' + esc(q.expected) + '</div>' +
      '</div>';
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
    expBtn.textContent = 'Запустить эксперимент';
    loadTaskState();
    input.focus();
  }
}

expBtn.addEventListener('click', async () => {
  if (exp.running) return;
  setExpUI(true);
  try {
    const s = await (await fetch('/api/experiment/state')).json();
    if (s.phase !== 'running') {
      if (!confirm('Эксперимент очистит текущую историю диалога, все ветки и чекпоинты, ' +
                   'прогонит два диалога (факты первого переносятся в долговременную память), ' +
                   'ответит на вопросы со всеми слоями памяти, а затем очистит каждый слой ' +
                   'по очереди, чтобы показать, на какие вопросы агент перестаёт отвечать ' +
                   '(несколько минут). Запустить?')) {
        setExpUI(false);
        return;
      }
      await fetch('/api/experiment/start', { method: 'POST' });
      chatEl.innerHTML = '';
      exp.lastKey = null;
    } else {
      exp.lastKey = s.stage + ':' + (s.layer_run || 0);
    }
  } catch (err) {
    setExpUI(false);
    return;
  }
  experimentLoop();
});

/* ---------- profile experiment ---------- */
const profExp = { running: false, lastKey: null };

function gradeBadges(grade){
  const dims = (grade && grade.dims) || {};
  const label = { style: 'стиль', format: 'код', tone: 'тон' };
  return Object.keys(label).map(d => {
    const v = dims[d];
    if (!v) return '';
    if (v.expect === 'any') return '<span style="color:#64748b;">' + label[d] + ': любой</span>';
    return '<span class="' + (v.ok ? 'good' : 'bad') + '">' + label[d] + ': ' +
      (v.ok ? '✓' : '✗') + ' (' + esc(v.evidence) + ')</span>';
  }).filter(Boolean).join(' · ');
}

function renderProfStep(data){
  const p = data.progress || {};
  const key = data.stage + ':' + (data.profile_run || 0) + ':' + (data.turn || 0);
  if (profExp.lastKey !== key) {
    expDivider(data.stage_label);
    profExp.lastKey = key;
  }
  if (p.user) addRow('user', p.user);
  if (p.reply || p.grade) {
    const bubble = addRow('assistant', p.reply || '(пустой ответ)');
    let metaText = p.tokens && p.tokens.request !== undefined
      ? tokenMeta(p.tokens, p.context) : null;
    if (p.grade) {
      const verdict = p.grade.ok_all ? 'предпочтения соблюдены' : 'нарушения';
      const g = document.createElement('div');
      g.className = 'meta';
      g.innerHTML = verdict + ' (' + p.grade.prefs_met + '/' + p.grade.prefs_total + '): ' +
        gradeBadges(p.grade);
      if (metaText) {
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = metaText;
        bubble.appendChild(meta);
      }
      bubble.appendChild(g);
    } else if (metaText) {
      appendMeta(bubble, metaText);
    }
  }
  const stepNo = (data.step_index || 0) + 1;
  expStatus('Эксперимент с профилями · ' + data.stage_label +
    (data.step_total ? ' · шаг ' + stepNo + '/' + data.step_total : ''));
}

function renderProfReport(rep){
  const div = document.createElement('div');
  div.className = 'report';
  const s = rep.summary;
  const st = rep.stability;
  const DIM_LABEL = { style: 'Стиль', format: 'Код', tone: 'Тон' };

  let html = '<h3>Эксперимент с профилями: ' + esc(rep.title) + '</h3>';
  html += '<p style="font-size:12.5px;color:#94a3b8;">Один и тот же вопрос — «' +
    esc(rep.shared_question) + '» — задан трём профилям с противоположными предпочтениями. ' +
    'Ответы проверены авто-грейдером: краткость/подробность (по числу слов), наличие кода, тон.</p>';

  html += '<table><tr><th>Профиль</th><th>Стиль</th><th>Код</th><th>Тон</th>' +
          '<th>Соблюдено</th></tr>';
  rep.profiles.forEach(p => {
    const g = p.grade;
    const dimCell = (d) => {
      const v = g.dims[d];
      if (!v || v.expect === 'any') return '<span style="color:#64748b;">—</span>';
      return '<span class="' + (v.ok ? 'good' : 'bad') + '">' + (v.ok ? '✓' : '✗') +
        ' <span style="font-size:11px;">' + esc(v.evidence) + '</span></span>';
    };
    html += '<tr><td><b>' + esc(p.name) + '</b></td>' +
      '<td>' + dimCell('style') + '</td><td>' + dimCell('format') + '</td><td>' + dimCell('tone') + '</td>' +
      '<td class="' + (g.ok_all ? 'good' : 'bad') + '">' + g.prefs_met + ' / ' + g.prefs_total + '</td></tr>';
  });
  html += '</table>';

  html += '<div class="verdict"><b>Итог:</b> предпочтений соблюдено ' + s.prefs_met +
    ' из ' + s.prefs_total + '; профилей, соблюдших все свои предпочтения, — ' +
    s.profiles_ok + ' из ' + s.profiles + '.</div>';

  html += '<div class="verdict" style="margin-top:12px;"><b>Устойчивость: 10 вопросов подряд в профиле «' +
    esc(st.profile) + '»</b></div>';
  html += '<table><tr><th>#</th><th>Вопрос</th><th>Результат</th></tr>';
  st.qa.forEach(q => {
    html += '<tr><td>' + q.n + '</td>' +
      '<td style="font-size:12px;">' + esc(q.question) + '</td>' +
      '<td class="' + (q.grade.ok_all ? 'good' : 'bad') + '" style="font-size:12px;">' +
      (q.grade.ok_all ? '✓ соблюдено (' + q.grade.prefs_met + '/' + q.grade.prefs_total + ')'
                      : '✗ ' + q.grade.violations.map(v => DIM_LABEL[v] || v).join(', ') +
                        ' (' + gradeBadges(q.grade) + ')') + '</td></tr>';
  });
  html += '</table>';
  html += '<div class="verdict"><b>Вывод:</b> ' +
    (st.first_violation === null
      ? 'предпочтения соблюдены во всех ' + st.questions_total + ' ответах подряд — ассистент не начал их игнорировать.'
      : 'первое нарушение — ответ #' + st.first_violation + '; соблюдено ' + st.compliant +
        ' из ' + st.questions_total + ' ответов.') + '</div>';
  if ((st.learned_rules || []).length) {
    html += '<div class="verdict" style="margin-top:6px;">За время теста профиль пополнился из диалога: ' +
      st.learned_rules.map(r => '«' + esc(r.text) + '»').join(', ') + '.</div>';
  }

  html += '<details><summary>Ответы профилей на общий вопрос</summary>';
  rep.profiles.forEach(p => {
    html += '<div class="qa-item"><div class="qa-q">' + esc(p.name) + '</div>' +
      '<div class="qa-answer">' + esc(p.answer || '(пустой ответ)') + '</div></div>';
  });
  html += '</details>';

  html += '<details><summary>Ответы на 10 вопросов (проверка устойчивости)</summary>';
  st.qa.forEach(q => {
    html += '<div class="qa-item"><div class="qa-q">' + q.n + '. ' + esc(q.question) + '</div>' +
      '<div class="qa-answer">' + esc(q.answer || '(пустой ответ)') + '</div>' +
      '<div class="' + (q.grade.ok_all ? 'good' : 'bad') + '" style="font-size:11.5px;margin-top:2px;">' +
      (q.grade.ok_all ? 'соблюдено' : 'нарушено') + ' · ' + esc(gradeBadges(q.grade).replace(/<[^>]+>/g, ' ')) +
      '</div></div>';
  });
  html += '</details>';

  html += '<button class="repeat" type="button">Повторить эксперимент с профилями</button>';
  div.innerHTML = html;
  chatEl.appendChild(div);
  div.querySelector('.repeat').addEventListener('click', () => profExpBtn.click());
  scroll();
}

async function profileExperimentLoop(){
  profExp.running = true;
  setExpUI(true);
  try {
    for (;;) {
      const r = await fetch('/api/profile_experiment/step', { method: 'POST' });
      const data = await r.json();
      if (data.error && !data.progress) {
        clearExpStatus();
        addRow('assistant', 'Ошибка эксперимента: ' + data.error + ' — нажмите кнопку, чтобы продолжить.', true);
        profExpBtn.textContent = 'Продолжить эксперимент с профилями';
        break;
      }
      renderProfStep(data);
      if (data.phase === 'done') {
        clearExpStatus();
        profExp.lastKey = null;
        renderProfReport(data.report);
        break;
      }
    }
  } catch (err) {
    clearExpStatus();
    addRow('assistant', 'Ошибка запроса: ' + err, true);
    profExpBtn.textContent = 'Продолжить эксперимент с профилями';
  } finally {
    profExp.running = false;
    setExpUI(false);
    profExpBtn.textContent = 'Запустить эксперимент с профилями';
    loadTaskState();
    input.focus();
  }
}

profExpBtn.addEventListener('click', async () => {
  if (profExp.running) return;
  setExpUI(true);
  try {
    const s = await (await fetch('/api/profile_experiment/state')).json();
    if (s.phase !== 'running') {
      if (!confirm('Эксперимент временно создаст три профиля с противоположными предпочтениями, ' +
                   'задаст им один и тот же вопрос, проверит ответы авто-грейдером, затем прогонит ' +
                   '10 вопросов подряд в одном профиле и покажет, на каком ответе предпочтения ' +
                   'начинают игнорироваться. Текущий диалог будет очищен и восстановлен после. ' +
                   'Запустить?')) {
        setExpUI(false);
        return;
      }
      await fetch('/api/profile_experiment/start', { method: 'POST' });
      chatEl.innerHTML = '';
      profExp.lastKey = null;
    } else {
      profExp.lastKey = s.stage + ':' + (s.profile_run || 0) + ':' + (s.turn || 0);
    }
  } catch (err) {
    setExpUI(false);
    return;
  }
  profileExperimentLoop();
});

(async function initProfileExperiment(){
  try {
    const s = await (await fetch('/api/profile_experiment/state')).json();
    if (s.phase === 'running') {
      profExp.lastKey = s.stage + ':' + (s.profile_run || 0) + ':' + (s.turn || 0);
      profileExperimentLoop();
    } else if (s.phase === 'done' && s.report) {
      renderProfReport(s.report);
    }
  } catch (e) {}
})();

loadProfiles();
loadInvariants();
loadBranches(false).then(() => { loadHistory(); loadTaskState(); });

document.getElementById('form').addEventListener('submit', (e) => { e.preventDefault(); run(); });
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, models=MODELS, model=MODEL)


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
    branch_id = _branch_id(data.get("branch_id"))
    model = data.get("model")
    if model not in MODELS:
        model = MODEL
    # Явный командный канал: «/invariant …» перехватывается ДО LLM и не
    # попадает в диалог. Поэтому просьбы в обычном диалоге («удали INV-2»)
    # не могут изменить инварианты — у диалога просто нет такого пути.
    handled, cmd_reply = invariants.run_command(message)
    if handled:
        return jsonify({
            "reply": cmd_reply,
            "tool_calls_used": 0,
            "steps": 0,
            "tokens": None,
            "context": None,
            "profile_updates": None,
            "invariants": None,
            "task": task_fsm.summary(branch_id),
            "fsm_rejections": 0,
            "stopped": False,
            "command": True,
            "error": None,
            "model": model,
        })
    with _running_lock:
        if branch_id in _running:
            return jsonify({
                "reply": None,
                "error": "Предыдущий запрос ещё обрабатывается — нажмите «Стоп» и подождите пару секунд.",
                "model": model,
            }), 409
        token = CancelToken()
        _running[branch_id] = token
    try:
        agent = Agent(model=model)
        result = agent.ask(message, branch_id, cancel=token)
        return jsonify({
            "reply": result["reply"],
            "tool_calls_used": result["tool_calls_used"],
            "steps": result["steps"],
            "tokens": result.get("tokens"),
            "context": result.get("context"),
            "profile_updates": result.get("profile_updates"),
            "invariants": result.get("invariants"),
            "task": result.get("task"),
            "fsm_rejections": result.get("fsm_rejections", 0),
            "stopped": result.get("stopped", False),
            "error": None,
            "model": model,
        })
    except Exception as exc:
        return jsonify({"reply": None, "error": str(exc), "model": model})
    finally:
        with _running_lock:
            if _running.get(branch_id) is token:
                del _running[branch_id]


@app.route("/api/chat/cancel", methods=["POST"])
def api_chat_cancel():
    """Stop button: cancel the agent run for the branch (cooperative — the
    loop stops before the next LLM call and saves the partial progress)."""
    data = request.get_json(silent=True) or {}
    branch_id = _branch_id(data.get("branch_id"))
    with _running_lock:
        token = _running.get(branch_id)
    if token is None:
        return jsonify({"cancelled": False, "error": None})
    token.cancel()
    return jsonify({"cancelled": True, "error": None})


@app.route("/api/task/state")
def api_task_state():
    """The branch's task lifecycle automaton state (stage/step/expected action)."""
    branch_id = _branch_id(request.args.get("branch_id"))
    return jsonify({"task": task_fsm.summary(branch_id), "error": None})


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


@app.route("/api/memory")
def api_memory():
    """The three memory layers for the UI: short-term / working / long-term."""
    branch_id = _branch_id(request.args.get("branch_id"))
    message_count = len(store.load_messages(branch_id))
    branches = store.list_branches()
    working = store.load_facts(branch_id)
    return jsonify({
        "short_term": {
            "branches": len(branches),
            "messages": message_count,
        },
        "working": working,
        "long_term": store.list_long_term(),
        "rules": memory.PROMPT_RULES,
        "error": None,
    })


@app.route("/api/memory/promote", methods=["POST"])
def api_memory_promote():
    """Explicitly move chosen working-memory facts into long-term memory.

    ``keys`` picks the facts to promote (empty/absent = nothing); ``category``
    forces one LTM category, "auto" (default) guesses it per fact.
    """
    data = request.get_json(silent=True) or {}
    branch_id = _branch_id(data.get("branch_id"))
    keys = data.get("keys")
    if not isinstance(keys, list):
        keys = []
    keys = [str(k) for k in keys]
    category = data.get("category")
    if category not in ("profile", "decision", "knowledge"):
        category = None  # auto
    moved = memory.promote_working(branch_id, keys=keys, category=category)
    return jsonify({"moved": moved, "error": None})


@app.route("/api/memory/fact/delete", methods=["POST"])
def api_memory_fact_delete():
    """Delete a single fact from the branch's working memory."""
    data = request.get_json(silent=True) or {}
    branch_id = _branch_id(data.get("branch_id"))
    key = str(data.get("key") or "").strip()
    if not key:
        return jsonify({"error": "Не указан ключ факта."}), 400
    store.delete_fact(branch_id, key)
    return jsonify({"error": None})


@app.route("/api/memory/clear/stm", methods=["POST"])
def api_memory_clear_stm():
    """Clear the dialog layer of one branch (messages, legacy state, checkpoint)."""
    data = request.get_json(silent=True) or {}
    branch_id = _branch_id(data.get("branch_id"))
    store.clear_dialog(branch_id)
    return jsonify({"error": None})


@app.route("/api/memory/clear/wm", methods=["POST"])
def api_memory_clear_wm():
    """Clear the working layer of one branch (all current-task facts)."""
    data = request.get_json(silent=True) or {}
    branch_id = _branch_id(data.get("branch_id"))
    store.clear_task_memory(branch_id)
    return jsonify({"error": None})


@app.route("/api/memory/ltm/add", methods=["POST"])
def api_memory_ltm_add():
    data = request.get_json(silent=True) or {}
    try:
        entry_id = memory.save_long_term(
            data.get("category") or "knowledge", data.get("key"), data.get("value"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"id": entry_id, "error": None})


@app.route("/api/memory/ltm/delete", methods=["POST"])
def api_memory_ltm_delete():
    data = request.get_json(silent=True) or {}
    try:
        store.delete_long_term(int(data.get("id")))
    except (TypeError, ValueError):
        return jsonify({"error": "Некорректный id."}), 400
    return jsonify({"error": None})


@app.route("/api/memory/ltm/clear", methods=["POST"])
def api_memory_ltm_clear():
    store.clear_long_term()
    return jsonify({"error": None})


@app.route("/api/profiles")
def api_profiles():
    """Profile list for the sidebar select (name, active flag, rule count)."""
    return jsonify({"profiles": store.list_profiles(), "error": None})


@app.route("/api/profiles/full")
def api_profiles_full():
    """Every profile with its rules split by kind — for the profile modal."""
    out = []
    for p in store.list_profiles():
        summary = profile.summary(p["id"])
        if summary:
            out.append(summary)
    return jsonify({"profiles": out, "error": None})


@app.route("/api/profiles/activate", methods=["POST"])
def api_profiles_activate():
    data = request.get_json(silent=True) or {}
    try:
        profile_id = int(data.get("profile_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "Некорректный id профиля."}), 400
    if not store.get_profile(profile_id):
        return jsonify({"error": "Профиль не найден."}), 404
    store.set_active_profile(profile_id)
    return jsonify({"active": profile_id, "error": None})


@app.route("/api/profiles/save", methods=["POST"])
def api_profiles_save():
    """Create or update a profile from the editor fields.

    ``style``/``code``/``tone`` are option keys or null; ``constraints`` are
    free-form rule lines. Manual rules are replaced, dialog-learned are kept.
    """
    data = request.get_json(silent=True) or {}
    constraints = data.get("constraints")
    if not isinstance(constraints, list):
        constraints = []
    try:
        profile_id = profile.save_profile(
            name=data.get("name") or "",
            description=data.get("description") or "",
            style=data.get("style"),
            code=data.get("code"),
            tone=data.get("tone"),
            constraints=[str(c) for c in constraints],
            profile_id=data.get("profile_id"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"id": profile_id, "error": None})


@app.route("/api/profiles/delete", methods=["POST"])
def api_profiles_delete():
    data = request.get_json(silent=True) or {}
    try:
        profile_id = int(data.get("profile_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "Некорректный id профиля."}), 400
    store.delete_profile(profile_id)
    return jsonify({"error": None})


@app.route("/api/profiles/rule/delete", methods=["POST"])
def api_profiles_rule_delete():
    data = request.get_json(silent=True) or {}
    try:
        store.delete_profile_rule(int(data.get("rule_id")))
    except (TypeError, ValueError):
        return jsonify({"error": "Некорректный id правила."}), 400
    return jsonify({"error": None})


@app.route("/api/profile_experiment/start", methods=["POST"])
def api_profile_experiment_start():
    return jsonify(profile_experiment.start())


# --- Invariants: management panel (explicit command channel #2) -----------------
# The DIALOG has no path into the invariants tables: only these routes (UI) and
# the /invariant chat commands (intercepted in api_chat before the LLM) mutate
# them, and every change is logged as "what was replaced with what".


@app.route("/api/invariants")
def api_invariants():
    """All invariants plus the change log, for the management panel."""
    return jsonify({
        "invariants": invariants.list_all(),
        "log": invariants.change_log(limit=50),
        "error": None,
    })


@app.route("/api/invariants/add", methods=["POST"])
def api_invariants_add():
    data = request.get_json(silent=True) or {}
    try:
        inv = invariants.add_invariant(
            category=data.get("category") or "",
            title=data.get("title") or "",
            rule=data.get("rule") or "",
            patterns=data.get("patterns") or [],
            source="command:ui",
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"invariant": inv, "error": None})


@app.route("/api/invariants/change", methods=["POST"])
def api_invariants_change():
    data = request.get_json(silent=True) or {}
    try:
        inv = invariants.change_invariant(
            data.get("id"),
            category=data.get("category"),
            title=data.get("title"),
            rule=data.get("rule"),
            patterns=data.get("patterns"),
            source="command:ui",
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"invariant": inv, "error": None})


@app.route("/api/invariants/remove", methods=["POST"])
def api_invariants_remove():
    data = request.get_json(silent=True) or {}
    try:
        inv = invariants.remove_invariant(data.get("id"), source="command:ui")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"removed": inv.get("code"), "error": None})


@app.route("/api/profile_experiment/step", methods=["POST"])
def api_profile_experiment_step():
    return jsonify(profile_experiment.step())


@app.route("/api/profile_experiment/state")
def api_profile_experiment_state():
    return jsonify(profile_experiment.snapshot())


def main() -> None:
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
