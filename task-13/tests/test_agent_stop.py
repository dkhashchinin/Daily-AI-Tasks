"""Tests for the stop button: cooperative cancellation of Agent.ask.

The LLM and auxiliary calls are faked, so no network is used. Run:

    python -m tests.test_agent_stop
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app import store  # noqa: E402
from app import agent as agent_mod  # noqa: E402
from app import context as context_mod  # noqa: E402
from app import profile as profile_mod  # noqa: E402

store.DB_PATH = Path(tempfile.mkdtemp(prefix="stop-test-")) / "test.db"
store.init_db()

# no tiktoken / no network in tests
agent_mod.count_tokens = lambda s: len(s or "")
agent_mod.count_messages = lambda msgs: sum(len(m.get("content") or "") for m in msgs)
context_mod.update_facts = lambda user_text, branch_id=1: ({"f": "v"}, None)
profile_mod.update_from_dialog = lambda *a, **k: None


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]
        self.usage = None


class _Func:
    def __init__(self, name, arguments="{}"):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, call_id, name, arguments="{}"):
        self.id = call_id
        self.function = _Func(name, arguments)

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


def test_cancel_before_start():
    """Стоп до начала: ни одного вызова LLM, история корректна."""
    token = agent_mod.CancelToken()
    token.cancel()
    calls = {"n": 0}

    def fake_chat(*a, **k):
        calls["n"] += 1
        return _Resp(_Msg(content="не должен вызваться"))

    orig = agent_mod.chat
    agent_mod.chat = fake_chat
    try:
        result = agent_mod.Agent().ask("тест", branch_id=301, cancel=token)
    finally:
        agent_mod.chat = orig

    assert result["stopped"] is True
    assert calls["n"] == 0
    assert "остановлено" in result["reply"]
    messages = store.load_messages(301)
    assert messages[-1]["role"] == "assistant"
    assert not any(m["role"] == "assistant" and m.get("tool_calls") and
                   not any(t["role"] == "tool" for t in messages[i + 1:])
                   for i, m in enumerate(messages)), "висячие tool_calls в истории"


def test_cancel_mid_run():
    """Стоп во время раунда с инструментом: цикл прерывается до второго
    вызова LLM, частичный прогресс сохранён."""
    token = agent_mod.CancelToken()
    llm_calls = {"n": 0}

    def fake_chat(*a, **k):
        llm_calls["n"] += 1
        if llm_calls["n"] == 1:
            return _Resp(_Msg(content=None,
                              tool_calls=[_TC("t1", "get_current_time")]))
        raise AssertionError("LLM не должен вызываться после отмены")

    def fake_run_tool(name, args, branch_id=store.ROOT_BRANCH_ID):
        token.cancel()  # пользователь нажал «Стоп» во время работы инструмента
        return "2026-11-19 12:00:00"

    orig_chat, orig_tool = agent_mod.chat, agent_mod.run_tool
    agent_mod.chat, agent_mod.run_tool = fake_chat, fake_run_tool
    try:
        result = agent_mod.Agent().ask("который час?", branch_id=302, cancel=token)
    finally:
        agent_mod.chat, agent_mod.run_tool = orig_chat, orig_tool

    assert result["stopped"] is True
    assert llm_calls["n"] == 1
    messages = store.load_messages(302)
    roles = [m["role"] for m in messages]
    assert "tool" in roles, "результат инструмента потерялся"
    assert messages[-1]["role"] == "assistant"
    # каждый assistant с tool_calls сопровождён ответом tool — API не сломается
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            assert any(t.get("tool_call_id") == m["tool_calls"][0]["id"]
                       for t in messages[i + 1:]), "tool result не найден"


def test_normal_run_without_cancel():
    """Без отмены поведение прежнее: ответ модели, stopped=False."""
    llm_calls = {"n": 0}

    def fake_chat(*a, **k):
        llm_calls["n"] += 1
        if llm_calls["n"] == 1:
            return _Resp(_Msg(content=None,
                              tool_calls=[_TC("t1", "calculate", '{"expression": "2+2"}')]))
        return _Resp(_Msg(content="Сумма 4."))

    orig = agent_mod.chat
    agent_mod.chat = fake_chat
    try:
        result = agent_mod.Agent().ask("сколько 2+2?", branch_id=303)
    finally:
        agent_mod.chat = orig

    assert result["stopped"] is False
    assert result["reply"] == "Сумма 4."
    assert llm_calls["n"] == 2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ok  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} тестов прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
