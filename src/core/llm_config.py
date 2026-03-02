"""LLM configuration and initialization for Gemini 2.0 Flash."""
import os
import time
import logging
from functools import wraps
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions
from dotenv import load_dotenv
import yaml

load_dotenv()
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_llm(temperature: Optional[float] = None) -> ChatGoogleGenerativeAI:
    """Create and return a configured Gemini 2.0 Flash LLM instance."""
    config = load_config()
    llm_config = config.get("llm", {})

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    return ChatGoogleGenerativeAI(
        model=llm_config.get("model", "gemini-2.0-flash"),
        temperature=temperature if temperature is not None else llm_config.get("temperature", 0.7),
        max_output_tokens=llm_config.get("max_tokens", 2048),
        google_api_key=api_key,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return HuggingFace embeddings for RAG (local model, no API key needed)."""
    config = load_config()
    embedding_model = config.get("rag", {}).get("embedding_model", "all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_rpm: int = 55):
        self.max_rpm = max_rpm
        self.min_interval = 60.0 / max_rpm
        self.last_call_time = 0.0

    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()


_rate_limiter = RateLimiter()


def rate_limited_call(func):
    """Decorator to apply rate limiting to LLM calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _rate_limiter.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper
