"""LLM client wrapper around the OpenAI-compatible HTTP API."""
import json

from openai import OpenAI

from .config import BASE_URL, MODEL, model_base_url, model_key, require_key

# One OpenAI client per (base_url, api_key); provider endpoints differ per model.
_CLIENTS: dict[tuple[str, str], OpenAI] = {}


def parse_json_object(raw: str | None) -> dict | None:
    """Parse an LLM reply into a JSON object (tolerates ``` fences and prose)."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
