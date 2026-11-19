"""LLM client wrapper around the OpenAI-compatible HTTP API."""
from openai import OpenAI

from .config import BASE_URL, MODEL, model_base_url, model_key, require_key

# One OpenAI client per (base_url, api_key); provider endpoints differ per model.
_CLIENTS: dict[tuple[str, str], OpenAI] = {}


def get_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    key = api_key or require_key()
    url = base_url or BASE_URL
    cache_key = (url, key)
    client = _CLIENTS.get(cache_key)
    if client is None:
        client = OpenAI(base_url=url, api_key=key)
        _CLIENTS[cache_key] = client
    return client


def chat(messages: list[dict], *, model: str | None = None,
         temperature: float | None = None,
         tools: list[dict] | None = None,
         tool_choice: str = "auto"):
    """Send a chat request and return the raw response object.

    When ``tools`` are provided the model may return tool calls instead of
    text, so the caller inspects ``response.choices[0].message``.
    """
    model = model or MODEL
    client = get_client(model_key(model), model_base_url(model))
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)
