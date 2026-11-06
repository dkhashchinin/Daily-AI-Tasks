"""Application configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GPUSTACK_API_KEY")
BASE_URL = os.getenv("GPUSTACK_BASE_URL", "https://gpustack.data.lmru.tech/v1")
MODEL = os.getenv("GPUSTACK_MODEL", "deepseek-v4.1-flash")


def get_client_config() -> dict:
    """Return OpenAI client init options, raising a helpful error if key is missing."""
    if not API_KEY:
        raise RuntimeError(
            "GPUSTACK_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return {"base_url": BASE_URL, "api_key": API_KEY}


PARAMS = {
    "temperature": 1,
    "top_p": 1,
    "max_tokens": 1024,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
