"""Kevin configuration - loaded from environment variables."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"   # Anthropic SDK + ANTHROPIC_API_KEY
    OLLAMA    = "ollama"      # Local Ollama REST (http://localhost:11434)
    CLAUDE    = "claude"      # `claude -p "..."` subprocess
    NONE      = "none"        # Regex-only, no LLM fallback


def _parse_llm_provider() -> LLMProvider:
    val = os.getenv("KEVIN_LLM", "none").lower()
    try:
        return LLMProvider(val)
    except ValueError:
        valid = ", ".join(p.value for p in LLMProvider)
        raise ValueError(
            f"Invalid KEVIN_LLM={val!r}. Valid choices: {valid}"
        ) from None


class Config:
    # EDGAR
    EDGAR_UA: str = os.getenv(
        "KEVIN_EDGAR_UA",
        "kevin-fin-engine research@fin-engine.local"
    )
    EDGAR_RATE:  float = float(os.getenv("KEVIN_EDGAR_RATE", "8"))  # req/s
    # str() ensures os.getenv gets a proper string default, not a Path object
    EDGAR_CACHE: Path  = Path(os.getenv(
        "KEVIN_CACHE_DIR", str(Path.home() / ".kevin" / "cache")
    ))

    # LLM
    LLM_PROVIDER: LLMProvider = _parse_llm_provider()
    LLM_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("KEVIN_LLM_THRESHOLD", "40")   # score below this -> LLM fallback
    )
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL:   str        = os.getenv("KEVIN_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    OLLAMA_HOST:  str = os.getenv("KEVIN_OLLAMA_HOST",  "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("KEVIN_OLLAMA_MODEL", "llama3.1")

    # Pipeline
    DEFAULT_LOOKBACK_DAYS: int = int(os.getenv("KEVIN_LOOKBACK_DAYS", "30"))
    MAX_CONCURRENT_FILINGS: int = int(os.getenv("KEVIN_CONCURRENCY", "5"))


cfg = Config()
