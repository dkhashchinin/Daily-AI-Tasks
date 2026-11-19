"""Agent: encapsulates the request -> tool call -> result -> response loop.

The full conversation (user messages, assistant replies, tool calls and their
results) is kept in persistent SQLite storage, so restarting the agent does not
lose context — the dialog continues exactly where it left off.

Every request carries the project invariants (``app/invariants.py``) as the
first system message; the FINAL answer is additionally checked by CODE
against the invariants before it is issued (see ``_invariant_gate``): a
violating answer is rewritten or replaced by a code-generated refusal.
Every request also carries the active user profile (``app/profile.py``);
after each reply the profile updater may learn new standing preferences
from the dialog.
"""
import json
import threading

from . import invariants, profile, store, task_fsm
from .config import MODEL, model_price
from .context import build_context
from .llm import chat
from .tokens import count_messages, count_tokens
from .tools import TOOL_SCHEMAS, run_tool

MAX_STEPS = 12  # safety cap so the agent never loops forever; the task FSM
                # lifecycle (transition -> work -> validation -> done) needs
                # several tool rounds per turn, so the cap is generous

INVARIANT_RETRIES = 2  # rewrite attempts before the code refusal takes over


class CancelToken:
    """Cooperative cancellation for a running :meth:`Agent.ask`.

    The web layer creates one token per in-flight request; the stop button
    sets it, and the agent loop checks it before every LLM round — so the
    "thinking" stops at the next round boundary and the partial progress is
    saved (the current LLM call itself finishes, it cannot be interrupted).
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


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
            branch_id: int = store.ROOT_BRANCH_ID,
            cancel: CancelToken | None = None) -> dict:
        """Answer a user message, possibly using tools. Returns a result dict.

        The context sent to the LLM is assembled from the project invariants
        plus the active user profile and the three memory layers (see
        ``app/invariants.py``, ``app/profile.py``, ``app/context.py``,
        ``app/memory.py``): invariants first, then profile, then long-term
        memory, then the current task's facts, then the recent dialog
        messages. The task lifecycle FSM block (``app/task_fsm.py``) goes
        right after them: it re-injects the current stage/step/expected
        action, so a paused task continues without re-explanations.
        Before the reply is issued, it is checked BY CODE against the
        invariants (``_invariant_gate``): a violating answer is rewritten or
        replaced by a code-generated refusal. The stored history always keeps
        every message: only the request is trimmed, the DB stays complete.
        After the reply the profile may be updated with preferences learned
        from the dialog (``profile_updates``).
        """
        self.messages, context_info = build_context(user_text, branch_id)
        # The automaton's position (stage / step / expected action / allowed
        # transitions) is injected every request; enforcement is in code
        # (task_fsm.transition), not in this prompt.
        fsm_block = task_fsm.render_state_block(branch_id)
        if fsm_block:
            self.messages.append({"role": "system", "content": fsm_block})
        active_model = self.model or MODEL
        price_in, price_out = model_price(active_model)
        history_tokens = count_messages(self.messages)
        full_history = store.load_messages(branch_id)
        base_len = len(self.messages)  # everything past this is new this turn
        self.messages.append({"role": "user", "content": user_text})
        # The user message is persisted immediately (and again after every
        # tool round): the FSM guards read the dialog from the DB mid-turn,
        # so an approval given in THIS message must already be visible there.
        store.save_messages(full_history + self.messages[base_len:], branch_id)
        request_tokens = count_tokens(user_text)
        tool_calls_used = 0  # individual tool executions, not rounds
        steps = 0            # LLM round-trips
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        balance = None
        fsm_rejections = 0   # transitions refused by the automaton this turn
        stopped = False      # прервано кнопкой «Стоп»

        for _ in range(MAX_STEPS):
            if cancel is not None and cancel.cancelled:
                stopped = True
                break
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
                text = (message.content or "").strip()
                # Гейт инвариантов: проверка КОДОМ до выдачи ответа пользователю
                # и до записи в диалог. Нарушающий ответ перезапрашивается
                # (INVARIANT_RETRIES раз), затем заменяется отказом, сгенерированным
                # кодом — нарушающий текст пользователю не попадает никогда.
                text, inv_info = self._invariant_gate(
                    text, user_text, price_in=price_in, price_out=price_out)
                extra = inv_info.pop("extra_usage", None)
                if extra:
                    prompt_tokens += extra.get("prompt", 0)
                    completion_tokens += extra.get("completion", 0)
                    cost += extra.get("cost", 0.0)
                store.save_messages(full_history + self.messages[base_len:], branch_id)
                profile_updates = self._learn_preferences(user_text, text)
                return {
                    "reply": text,
                    "tool_calls_used": tool_calls_used,
                    "steps": steps,
                    "context": context_info,
                    "profile_updates": profile_updates,
                    "invariants": inv_info,
                    "task": task_fsm.summary(branch_id),
                    "fsm_rejections": fsm_rejections,
                    "stopped": False,
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
                result = run_tool(name, args, branch_id=branch_id)
                if name in ("task_transition", "task_check") and result.startswith("Отказано"):
                    fsm_rejections += 1
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            store.save_messages(full_history + self.messages[base_len:], branch_id)

        # Safety net: the cap was reached (or the user pressed the stop
        # button) without a final answer. Either way the partial progress is
        # saved and the message sequence stays valid for the next turn.
        if stopped:
            reply = ("⏹ Мышление остановлено пользователем. Всё, что агент успел "
                     "сделать, сохранено в истории; продолжить можно следующим сообщением.")
        else:
            reply = "Агент перестал отвечать (превышен лимит шагов). Попробуйте переформулировать запрос."
        self.messages.append({"role": "assistant", "content": reply})
        store.save_messages(full_history + self.messages[base_len:], branch_id)
        return {
            "reply": reply,
            "tool_calls_used": tool_calls_used,
            "steps": steps,
            "context": context_info,
            "profile_updates": None,
            "invariants": None,  # финального ответа не было — гейт не запускался
            "task": task_fsm.summary(branch_id),
            "fsm_rejections": fsm_rejections,
            "stopped": stopped,
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

    def _invariant_gate(self, text: str, user_text: str,
                        price_in: float = 0.0, price_out: float = 0.0
                        ) -> tuple[str, dict]:
        """Code check of the final answer against the invariants BEFORE issuing.

        The answer text is scanned by :func:`invariants.check_answer` (regex
        patterns of every invariant). On a violation the answer is NOT issued:
        first the model is asked to rewrite (up to ``INVARIANT_RETRIES`` times,
        in a throwaway message list — the dialog never sees the violating
        text), and if it keeps violating, a code-generated refusal
        (:func:`invariants.refusal_text`) replaces the answer. Returns
        ``(final_text, info)``; ``self.messages[-1]`` is updated to the final
        text, so only the compliant answer is stored.
        """
        def info(rewrites: int, blocked: bool, violations: list[dict],
                 extra: dict | None) -> dict:
            data = {
                "checked": True,
                "count": len(invariants.list_all()),
                "rewrites": rewrites,
                "blocked": blocked,
                "violations": sorted({v["code"] for v in violations}),
            }
            if extra:
                data["extra_usage"] = extra
            return data

        violations = invariants.check_answer(text)
        if not violations:
            return text, info(0, False, [], None)

        rewrites = 0
        extra_usage = {"prompt": 0, "completion": 0, "cost": 0.0}
        current = violations
        for _ in range(INVARIANT_RETRIES):
            rewrites += 1
            fixed, usage = self._ask_rewrite(user_text, current,
                                             price_in=price_in, price_out=price_out)
            if usage:
                extra_usage["prompt"] += usage.get("prompt", 0)
                extra_usage["completion"] += usage.get("completion", 0)
                extra_usage["cost"] += usage.get("cost", 0.0)
            if fixed:
                new_violations = invariants.check_answer(fixed)
                if not new_violations:
                    self.messages[-1]["content"] = fixed
                    return fixed, info(rewrites, False, current, extra_usage)
                current = new_violations
        refusal = invariants.refusal_text(current)
        self.messages[-1]["content"] = refusal
        return refusal, info(rewrites, True, current, extra_usage)

    def _ask_rewrite(self, user_text: str, violations: list[dict],
                     price_in: float = 0.0, price_out: float = 0.0
                     ) -> tuple[str | None, dict | None]:
        """One rewrite round after a violation was caught by the gate.

        Runs in a throwaway copy of the message list (nothing is persisted);
        returns ``(text, usage)`` — ``text=None`` when the model failed to
        produce a plain-text answer this round.
        """
        codes = "; ".join(f"{v['code']} «{v['title']}»" for v in violations)
        instruction = (
            "КОНТРОЛЬ ИНВАРИАНТОВ (автоматическая проверка кодом): твой предыдущий "
            "ответ ОТКЛОНЁН — в нём есть предложения, нарушающие инварианты проекта: "
            f"{codes}.\n"
            "Напиши ответ заново по правилам:\n"
            "1) откажись выполнять конфликтующую часть запроса: назови код и название "
            "инварианта и одной фразой объясни конфликт;\n"
            "2) предложи решение, которое инварианты не нарушают (если возможно);\n"
            "3) НЕ упоминай запрещённое по названию — иначе проверка снова отклонит "
            "ответ;\n"
            "4) ответь обычным текстом, без вызова инструментов."
        )
        retry_messages = self.messages + [
            {"role": "user", "content": instruction},
        ]
        try:
            response = chat(retry_messages, model=self.model,
                            temperature=self.temperature)
        except Exception as exc:
            print(f"[agent] перезапрос после нарушения инвариантов не удался: {exc}")
            return None, None
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            return None, None  # модель снова требует инструменты — попытка не удалась
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
            "completion": getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
        }
        api_cost = float(getattr(usage, "cost", 0) or 0) if usage else 0.0
        usage_dict["cost"] = api_cost if api_cost else (
            usage_dict["prompt"] * price_in / 1_000_000
            + usage_dict["completion"] * price_out / 1_000_000
        )
        return (message.content or "").strip(), usage_dict

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
