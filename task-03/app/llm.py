"""LLM client wrapper around the OpenAI-compatible API."""
import time

from openai import OpenAI

from .config import PARAMS, MODEL, get_client_config
from .metrics import record


def get_client() -> OpenAI:
    return OpenAI(**get_client_config())


def build_params(max_tokens: int | None = None, stop: str | list[str] | None = None,
                 reasoning_effort: str | None = None,
                 max_completion_tokens: int | None = None,
                 budget_tokens: int | None = None) -> dict:
    """Merge per-request overrides into the base PARAMS."""
    params = dict(PARAMS)
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if stop:
        params["stop"] = stop
    if reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    if max_completion_tokens is not None:
        params.pop("max_tokens", None)
        params["max_completion_tokens"] = max_completion_tokens
    if budget_tokens is not None:
        params["extra_body"] = {"budget_tokens": budget_tokens}
    return params


def chat(messages: list[dict], *, max_tokens: int | None = None,
         stop: str | list[str] | None = None,
         reasoning_effort: str | None = None,
         max_completion_tokens: int | None = None,
         budget_tokens: int | None = None) -> str:
    """Send a chat request and return the assistant's reply."""
    client = get_client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        **build_params(max_tokens, stop, reasoning_effort,
                       max_completion_tokens, budget_tokens),
        model=MODEL,
        messages=messages,
    )
    elapsed = time.perf_counter() - start

    usage = None
    if getattr(response, "usage", None):
        u = response.usage
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    record(
        response_model=MODEL,
        usage=usage,
        elapsed=elapsed,
        message_count=len(messages),
    )
    return response.choices[0].message.content


def stream_chat(messages: list[dict], *, max_tokens: int | None = None,
                stop: str | list[str] | None = None,
                reasoning_effort: str | None = None,
                max_completion_tokens: int | None = None,
                budget_tokens: int | None = None):
    """Stream a chat completion.

    Yields ``(kind, text)`` pairs where ``kind`` is either ``"reasoning"``
    (the model's chain-of-thought) or ``"content"`` (the final answer).
    Metrics are recorded once the stream finishes.
    """
    client = get_client()
    start = time.perf_counter()
    stream = client.chat.completions.create(
        **build_params(max_tokens, stop, reasoning_effort,
                       max_completion_tokens, budget_tokens),
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    usage = None
    finish_reason = None
    for chunk in stream:
        if getattr(chunk, "usage", None):
            u = chunk.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
        delta = choice.delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("reasoning", reasoning)
        content = getattr(delta, "content", None)
        if content:
            yield ("content", content)

    record(
        response_model=MODEL,
        usage=usage,
        elapsed=time.perf_counter() - start,
        message_count=len(messages),
    )
    yield ("finish", finish_reason)
