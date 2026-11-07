"""Application configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GPUSTACK_API_KEY")
BASE_URL = os.getenv("GPUSTACK_BASE_URL", "https://gpustack.data.lmru.tech/v1")
MODEL = os.getenv("GPUSTACK_MODEL", "deepseek-v4.1-flash")

_MODELS_RAW = os.getenv("GPUSTACK_MODELS", "")
MODELS = [m.strip() for m in _MODELS_RAW.split(",") if m.strip()]
if not MODELS:
    MODELS = [MODEL]

def _parse_pairs(raw: str) -> dict[str, str]:
    """Parse "key=value,key2=value2" into a dict."""
    out: dict[str, str] = {}
    for _pair in raw.split(","):
        if "=" in _pair:
            _m, _, _v = _pair.partition("=")
            if _m.strip() and _v.strip():
                out[_m.strip()] = _v.strip()
    return out


# Optional per-model API keys, format: "model1=key1,model2=key2"
MODEL_KEYS: dict[str, str] = _parse_pairs(os.getenv("GPUSTACK_KEYS", ""))

# Optional per-model base URLs, format: "model1=url1,model2=url2"
MODEL_BASE_URLS: dict[str, str] = _parse_pairs(os.getenv("GPUSTACK_BASE_URLS", ""))


def _parse_prices(raw: str) -> dict[str, tuple[float, float]]:
    """Parse per-model token prices.

    Format: ``model=prompt_price;completion_price,model2=prompt2;comp2``.
    Prices are for 1M tokens.
    """
    out: dict[str, tuple[float, float]] = {}
    for _pair in raw.split(","):
        if "=" not in _pair:
            continue
        _m, _, _v = _pair.partition("=")
        _name = _m.strip()
        if not _name:
            continue
        parts = _v.split(";")
        if len(parts) != 2:
            continue
        try:
            prompt_p = float(parts[0].strip())
            comp_p = float(parts[1].strip())
        except ValueError:
            continue
        out[_name] = (prompt_p, comp_p)
    return out


# Optional per-model token prices per 1M tokens, e.g. "deepseek-v4.1-flash=0.1;0.3"
MODEL_PRICES: dict[str, tuple[float, float]] = _parse_prices(os.getenv("GPUSTACK_PRICES", ""))


def _parse_default_price(raw: str) -> tuple[float, float]:
    """Parse a default price "prompt;completion" (per 1M tokens), else fall back."""
    parts = raw.split(";")
    if len(parts) == 2:
        try:
            return (float(parts[0].strip()), float(parts[1].strip()))
        except ValueError:
            pass
    return (10.0, 30.0)


# Default token price (per 1M tokens, rubles) used for models without a specific price.
PRICE_DEFAULT: tuple[float, float] = _parse_default_price(os.getenv("GPUSTACK_PRICE_DEFAULT", ""))


def get_client_config(model: str | None = None) -> dict:
    """Return OpenAI client init options, raising a helpful error if key is missing.

    Uses a model-specific key (``GPUSTACK_KEYS``) if available, else the default key.
    Uses a model-specific base URL (``GPUSTACK_BASE_URLS``), else the default URL.
    """
    key = MODEL_KEYS.get(model) if model else None
    key = key or API_KEY
    if not key:
        raise RuntimeError(
            "GPUSTACK_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    base = MODEL_BASE_URLS.get(model) if model else None
    base = base or BASE_URL
    return {"base_url": base, "api_key": key}


PARAMS = {
    "temperature": 1,
    "top_p": 1,
    "max_tokens": 4096,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
