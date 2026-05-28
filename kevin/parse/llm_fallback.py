"""
Multi-provider LLM fallback for when regex confidence is below threshold.

Supports:
  - anthropic   → Anthropic Python SDK (claude-haiku-4-5-20251001 by default)
  - ollama      → Local Ollama REST API (llama3.1 by default)
  - claude      → `claude -p "..."` subprocess (Claude Code CLI)
  - none        → no-op (returns None)

The LLM is given a small, focused snippet (~2000 chars) and asked to return
one JSON object with EPS and revenue. Hallucination risk is low because:
  1. We send the actual filing text, not a description
  2. We ask for a number that exists in the document
  3. We validate the response before accepting it

Kevin never sends full filings to the LLM — only the relevant snippet.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from kevin.config import LLMProvider, cfg
from kevin.models import EPSResult, EPSType

log = logging.getLogger(__name__)


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM = (
    "You are a financial data extraction assistant. "
    "Extract structured data from SEC filing excerpts. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

_USER_TEMPLATE = """Extract EPS and revenue from this SEC 8-K filing excerpt.

FILING EXCERPT:
{snippet}

Return EXACTLY this JSON structure (use null for fields you cannot find):
{{
  "eps_basic": <number or null>,
  "eps_diluted": <number or null>,
  "eps_adjusted": <number or null>,
  "revenue_mm": <revenue in USD millions, number or null>,
  "guidance_midpoint": <next-quarter EPS midpoint, number or null>
}}

Rules:
- EPS values must be per-share (typically 0.01 to 50 range)
- Revenue must be in USD millions
- Prefer GAAP over non-GAAP where both exist
- Use negative numbers for losses in parentheses e.g. (1.23) → -1.23
"""


def _build_prompt(html_snippet: str) -> str:
    # Strip HTML tags for the LLM — it doesn't need them
    clean = re.sub(r"<[^>]+>", " ", html_snippet)
    clean = re.sub(r"\s{2,}", " ", clean).strip()[:2500]
    return _USER_TEMPLATE.format(snippet=clean)


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON, returning empty dict on failure."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.M)
    cleaned = re.sub(r"\s*```$",         "", cleaned,      flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON: %s", raw[:200])
        return {}


def _validate(data: dict, key: str, lo: float = -500, hi: float = 5000) -> float | None:
    v = data.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if lo <= f <= hi else None
    except (TypeError, ValueError):
        return None


# ── Provider base class ───────────────────────────────────────────────────────

class LLMClient(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """Return raw completion text."""

    async def extract(self, html_snippet: str) -> dict[str, Any]:
        """Return parsed extraction dict or empty dict on failure."""
        try:
            raw  = await self.complete(_SYSTEM, _build_prompt(html_snippet))
            return _parse_response(raw)
        except Exception as exc:
            log.error("LLM extraction failed: %s", exc)
            return {}


# ── Anthropic provider ────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)

    async def complete(self, system: str, user: str) -> str:
        msg = await self._client.messages.create(
            model=cfg.ANTHROPIC_MODEL,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


# ── Ollama provider ───────────────────────────────────────────────────────────

class OllamaClient(LLMClient):
    """Creates a fresh httpx client per call — avoids resource leak from a stored client."""

    async def complete(self, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient(base_url=cfg.OLLAMA_HOST, timeout=60) as http:
            resp = await http.post(
                "/api/chat",
                json={
                    "model": cfg.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]


# ── Claude CLI provider ───────────────────────────────────────────────────────

class ClaudeCliClient(LLMClient):
    """Runs `claude -p "<prompt>"` as a subprocess."""

    async def complete(self, system: str, user: str) -> str:
        combined = f"{system}\n\n{user}"
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["claude", "-p", combined],
                capture_output=True, text=True, timeout=60,
            )
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI error: {result.stderr[:200]}")
        return result.stdout


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client() -> LLMClient | None:
    """Return the configured LLM client, or None if provider is 'none'."""
    p = cfg.LLM_PROVIDER
    if p == LLMProvider.ANTHROPIC:
        if not cfg.ANTHROPIC_API_KEY:
            log.warning("KEVIN_LLM=anthropic but ANTHROPIC_API_KEY not set — skipping LLM")
            return None
        return AnthropicClient()
    if p == LLMProvider.OLLAMA:
        return OllamaClient()
    if p == LLMProvider.CLAUDE:
        return ClaudeCliClient()
    return None   # LLMProvider.NONE


# ── Public helper ─────────────────────────────────────────────────────────────

async def llm_extract_eps(html_snippet: str, client: LLMClient) -> EPSResult | None:
    """
    Ask the LLM to extract EPS from a filing snippet.
    Returns an EPSResult tagged as source='llm', or None on failure.
    """
    data = await client.extract(html_snippet)
    if not data:
        return None

    # Prefer basic > diluted > adjusted
    for key, eps_type in [
        ("eps_basic",    EPSType.BASIC),
        ("eps_diluted",  EPSType.DILUTED),
        ("eps_adjusted", EPSType.ADJUSTED),
    ]:
        val = _validate(data, key, lo=-500, hi=500)
        if val is not None:
            return EPSResult(
                value=val,
                eps_type=eps_type,
                confidence=60.0,   # LLM fallback has fixed moderate confidence
                source="llm",
                label=None,
                snippet=None,
            )
    return None
