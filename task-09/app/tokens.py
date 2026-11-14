"""Local token counting for the agent.

Counts tokens without making a network call, so we can report how many tokens
the current request, the whole conversation history and the model response
each consume. Uses ``tiktoken`` for accuracy when available and falls back to a
lightweight estimator otherwise.
"""
import math
import re

try:
    import tiktoken

    def _loaded_encoder():
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    _ENCODER = _loaded_encoder()
except Exception:
    _ENCODER = None

# Rough regex for "runs" of non-CJK characters (words + attached punctuation).
_WORD_RE = re.compile(r"[^\s\u4e00-\u9fff]+")


def count_tokens(text: str | None) -> int:
    """Return an approximate number of tokens in ``text``."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass

    # Estimator fallback: CJK chars are ~1 token each; other scripts group into
    # word runs, each contributing roughly a quarter of its length (min 1).
    tokens = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    for run in _WORD_RE.findall(text):
        tokens += max(1, math.ceil(len(run) / 4))
    return tokens


def count_messages(messages: list[dict]) -> int:
    """Return the total number of tokens across a list of OpenAI messages.

    Counts message content and tool-call arguments, plus a small constant for
    the per-message formatting overhead.
    """
    total = 0
    for message in messages:
        total += count_tokens(message.get("content"))
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += count_tokens(fn.get("name")) + count_tokens(fn.get("arguments"))
        total += 4  # role / structure overhead
    return total
