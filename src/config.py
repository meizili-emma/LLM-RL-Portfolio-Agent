"""
Project-level configuration.

This file intentionally reads secrets and runtime settings from environment
variables. Do not hard-code API keys or private paths here.

For local setup, copy `.env_sample` to `.env` and export/load the variables
in your shell before running data-download or LLM pipelines.
"""

from __future__ import annotations

import os


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


# =========================
# Financial Modeling Prep
# =========================

FMP_API_KEY = os.getenv("FMP_API_KEY", "")

BASE_URL_V3 = os.getenv(
    "FMP_BASE_URL_V3",
    "https://financialmodelingprep.com/api/v3/",
)

BASE_URL_V4 = os.getenv(
    "FMP_BASE_URL_V4",
    "https://financialmodelingprep.com/api/v4/",
)

BASE_URL_STABLE = os.getenv(
    "FMP_BASE_URL_STABLE",
    "https://financialmodelingprep.com/stable/",
)


# =========================
# Optional financial data APIs
# =========================

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


# =========================
# LLM backends
# =========================

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# =========================
# Runtime controls
# =========================

DEFAULT_TIMEOUT = _get_int("DEFAULT_TIMEOUT", 30)
FORCE_REDOWNLOAD = _get_bool("FORCE_REDOWNLOAD", False)
