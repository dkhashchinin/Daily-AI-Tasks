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

# Optional per-model API keys, format: "model1=key1,model2=key2"
_KEYS_RAW = os.getenv("GPUSTACK_KEYS", "")
MODEL_KEYS: dict[str, str] = {}
for _pair in _KEYS_RAW.split(","):
    if "=" in _pair:
        _m, _, _k = _pair.partition("=")
        if _m.strip() and _k.strip():
            MODEL_KEYS[_m.strip()] = _k.strip()


def get_client_config(model: str | None = None) -> dict:
    """Return OpenAI client init options, raising a helpful error if key is missing.

    Uses a model-specific key (``GPUSTACK_KEYS``) if available, else the default key.
    """
    key = MODEL_KEYS.get(model) if model else None
    key = key or API_KEY
    if not key:
        raise RuntimeError(
            "GPUSTACK_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return {"base_url": BASE_URL, "api_key": key}


PARAMS = {
    "temperature": 1,
    "top_p": 1,
    "max_tokens": 4096,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
