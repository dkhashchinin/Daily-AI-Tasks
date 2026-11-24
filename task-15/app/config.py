"""Agent configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GPUSTACK_API_KEY")
BASE_URL = os.getenv("GPUSTACK_BASE_URL", "https://gpustack.data.lmru.tech/v1")
MODEL = os.getenv("GPUSTACK_MODEL", "deepseek-v4.1-flash")

_TMP = os.getenv("GPUSTACK_MODELS", "")
MODELS = [m.strip() for m in _TMP.split(",") if m.strip()]
if not MODELS:
    MODELS = [MODEL]


def _parse_kv(raw: str) -> dict[str, str]:
    """Parse "model=value,model2=value2" into a dict (values may contain ':' etc.)."""
    result = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            key, value = key.strip(), value.strip()
            if key and value:
                result[key] = value
    return result


# Per-model overrides: some models live on other providers and need their own
# key / endpoint. Format: "model=value,model2=value2".
_MODEL_KEYS = _parse_kv(os.getenv("GPUSTACK_KEYS", ""))
_MODEL_URLS = _parse_kv(os.getenv("GPUSTACK_BASE_URLS", ""))

# Prices per 1M tokens: "model=in;out" (RUB). Falls back to GPUSTACK_PRICE_DEFAULT.
_MODEL_PRICES = {k: v for k, v in _parse_kv(os.getenv("GPUSTACK_PRICES", "")).items()}
_default_price = os.getenv("GPUSTACK_PRICE_DEFAULT", "10;30")


def model_key(model: str) -> str:
    """API key for ``model`` (per-model override, otherwise the default key)."""
    return _MODEL_KEYS.get(model, API_KEY)


def model_base_url(model: str) -> str:
    """Endpoint for ``model`` (per-model override, otherwise the default URL)."""
    return _MODEL_URLS.get(model, BASE_URL)


def model_price(model: str) -> tuple[float, float]:
    """Return (input, output) price per 1M tokens for ``model``."""
    raw = _MODEL_PRICES.get(model, _default_price)
    try:
        in_p, _, out_p = raw.partition(";")
        return float(in_p), float(out_p or in_p)
    except ValueError:
        return 10.0, 30.0


def require_key(model: str | None = None) -> str:
    """Return the API key for ``model``, raising a helpful error if it is missing."""
    key = model_key(model) if model else API_KEY
    if not key:
        raise RuntimeError(
            "GPUSTACK_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return key
