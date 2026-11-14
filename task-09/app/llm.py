"""LLM client wrapper around the OpenAI-compatible HTTP API."""
from openai import OpenAI

from .config import MODEL, BASE_URL, require_key


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
    model = model or MODEL
    client = get_client()
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)
