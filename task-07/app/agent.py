"""Agent: encapsulates the request -> tool call -> result -> response loop.

The full conversation (user messages, assistant replies, tool calls and their
results) is kept in persistent SQLite storage, so restarting the agent does not
lose context — the dialog continues exactly where it left off.
"""
import json

from . import store
from .llm import chat
from .tools import TOOL_SCHEMAS, run_tool

MAX_STEPS = 8  # safety cap so the agent never loops forever


class Agent:
    """A tiny agent that can use tools to answer a user query.

    The loop keeps running until the model stops requesting tools:
      1. send messages + tool schemas to the LLM
      2. if the model asked for a tool -> execute it, append result, repeat
      3. otherwise the model produced a final answer -> stop
    """

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.model = model
        self.temperature = temperature

    def ask(self, user_text: str) -> dict:
        """Answer a user message, possibly using tools. Returns a result dict.

        The existing conversation is loaded from persistent storage, the new
        user message and every tool call/result are appended, and the whole
        history is saved back at the end.
        """
        self.messages = store.load_messages()
        self.messages.append({"role": "user", "content": user_text})
        tool_calls_used = 0

        for _ in range(MAX_STEPS):
            response = chat(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            message = response.choices[0].message

            # No tool calls -> the model gave a final answer -> done.
            if not message.tool_calls:
                self.messages.append({"role": "assistant", "content": message.content or ""})
                store.save_messages(self.messages)
                text = (message.content or "").strip()
                return {
                    "reply": text,
                    "tool_calls_used": tool_calls_used,
                    "steps": tool_calls_used + 1,
                }

            # The model asked for one or more tools. Execute each and feed results back.
            tool_calls_used += 1
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [tc.model_dump() for tc in message.tool_calls],
            })
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Safety net: the cap was reached without a final answer.
        self.messages.append({
            "role": "assistant",
            "content": "Агент перестал отвечать (превышен лимит шагов). Попробуйте переформулировать запрос.",
        })
        store.save_messages(self.messages)
        return {
            "reply": "Агент перестал отвечать (превышен лимит шагов). Попробуйте переформулировать запрос.",
            "tool_calls_used": tool_calls_used,
            "steps": MAX_STEPS,
        }
