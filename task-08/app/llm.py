"""LLM client wrapper around the OpenAI-compatible HTTP API."""
from contextvars import ContextVar
from typing import Callable

from openai import OpenAI

from .config import MODEL, BASE_URL, require_key

# The token lab exercises the whole application -- routes, storage, error
# handling -- against a simulated model. The override lives in a context
# variable, so it only affects the thread that installed it.
_CHAT_OVERRIDE: ContextVar[Callable | None] = ContextVar("chat_override", default=None)


def use_chat(fn: Callable | None) -> None:
    """Route :func:`chat` through ``fn`` for this context (thread) only."""
    _CHAT_OVERRIDE.set(fn)


def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=require_key())


def chat(messages: list[dict], *, model: str | None = None,
         temperature: float | None = None,
         tools: list[dict] | None = None,
         tool_choice: str = "auto"):
    """Send a chat request and return the raw response object.

    When ``tools`` are provided the model may return tool calls instead of
    text, so the caller inspects ``response.choices[0].message``.
    """
    override = _CHAT_OVERRIDE.get()
    if override is not None:
        return override(messages, model=model, temperature=temperature,
                        tools=tools, tool_choice=tool_choice)

    model = model or MODEL
    client = get_client()
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)
