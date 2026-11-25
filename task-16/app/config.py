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


def require_key() -> str:
    """Return the API key, raising a helpful error if it is missing."""
    if not API_KEY:
        raise RuntimeError(
            "GPUSTACK_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return API_KEY
