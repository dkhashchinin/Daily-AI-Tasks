"""LLM client wrapper around the OpenAI-compatible API."""
from openai import OpenAI

from .config import PARAMS, MODEL, get_client_config


def get_client() -> OpenAI:
    return OpenAI(**get_client_config())


def chat(messages: list[dict]) -> str:
    """Send a chat request and return the assistant's reply."""
    client = get_client()
    response = client.chat.completions.create(
        **PARAMS,
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
