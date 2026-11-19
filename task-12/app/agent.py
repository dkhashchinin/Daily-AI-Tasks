"""Agent: encapsulates the request -> tool call -> result -> response loop.

The full conversation (user messages, assistant replies, tool calls and their
results) is kept in persistent SQLite storage, so restarting the agent does not
lose context — the dialog continues exactly where it left off.

Every request carries the active user profile (``app/profile.py``) as the
first system message; after each reply the profile updater may learn new
standing preferences from the dialog.
"""
import json

from . import profile, store
from .config import MODEL, model_price
from .context import build_context
from .llm import chat
from .tokens import count_messages, count_tokens
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

    def ask(self, user_text: str,
            branch_id: int = store.ROOT_BRANCH_ID) -> dict:
        """Answer a user message, possibly using tools. Returns a result dict.

        The context sent to the LLM is assembled from the active user profile
        plus the three memory layers (see ``app/profile.py``, ``app/context.py``,
        ``app/memory.py``): profile first, then long-term memory, then the
        current task's facts, then the recent dialog messages. The stored
        history always keeps every message: only the request is trimmed, the
        DB stays complete. After the reply the profile may be updated with
        preferences learned from the dialog (``profile_updates``).
        """
        self.messages, context_info = build_context(user_text, branch_id)
        active_model = self.model or MODEL
        price_in, price_out = model_price(active_model)
        history_tokens = count_messages(self.messages)
        full_history = store.load_messages(branch_id)
        base_len = len(self.messages)  # everything past this is new this turn
        self.messages.append({"role": "user", "content": user_text})
        request_tokens = count_tokens(user_text)
        tool_calls_used = 0  # individual tool executions, not rounds
        steps = 0            # LLM round-trips
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        balance = None

        for _ in range(MAX_STEPS):
            response = chat(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            steps += 1
            message = response.choices[0].message
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_delta = getattr(usage, "prompt_tokens", 0) or 0
                completion_delta = getattr(usage, "completion_tokens", 0) or 0
                prompt_tokens += prompt_delta
                completion_tokens += completion_delta
                # AITUNNEL extends the standard usage object with billing fields;
                # other providers (GPUStack) do not, so price locally as a fallback.
                api_cost = float(getattr(usage, "cost", 0) or 0)
                cost += api_cost if api_cost else (
                    prompt_delta * price_in / 1_000_000
                    + completion_delta * price_out / 1_000_000
                )
                if getattr(usage, "balance", None) is not None:
                    balance = usage.balance
            # No tool calls -> the model gave a final answer -> done.
            if not message.tool_calls:
                self.messages.append({"role": "assistant", "content": message.content or ""})
                store.save_messages(full_history + self.messages[base_len:], branch_id)
                text = (message.content or "").strip()
                profile_updates = self._learn_preferences(user_text, text)
                return {
                    "reply": text,
                    "tool_calls_used": tool_calls_used,
                    "steps": steps,
                    "context": context_info,
                    "profile_updates": profile_updates,
                    "tokens": {
                        "request": request_tokens,
                        "history": history_tokens,
                        "response": count_tokens(text),
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                        "cost": cost,
                        "balance": balance,
                    },
                }

            # The model asked for one or more tools. Execute each and feed results back.
            tool_calls_used += len(message.tool_calls)
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
        store.save_messages(full_history + self.messages[base_len:], branch_id)
        return {
            "reply": "Агент перестал отвечать (превышен лимит шагов). Попробуйте переформулировать запрос.",
            "tool_calls_used": tool_calls_used,
            "steps": steps,
            "context": context_info,
            "profile_updates": None,
            "tokens": {
                "request": request_tokens,
                "history": history_tokens,
                "response": count_tokens(self.messages[-1]["content"]),
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
                "cost": cost,
                "balance": balance,
            },
        }

    @staticmethod
    def _learn_preferences(user_text: str, reply: str) -> dict | None:
        """Let the profile updater merge standing preferences from the dialog.

        A failure here must never break the answer — the profile just stays
        as it was.
        """
        try:
            return profile.update_from_dialog(user_text, reply)
        except Exception as exc:
            print(f"[agent] не удалось пополнить профиль: {exc}")
            return None
