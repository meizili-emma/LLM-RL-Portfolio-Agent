from __future__ import annotations

import json
import re
import time
import os 
import threading
import collections
from pathlib import Path
from typing import Any, Dict, List, Type
import time 
import pandas as pd

from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import SystemMessage, HumanMessage

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore



class RobustBaseModel(BaseModel):
    """
    Base model that:
    - ignores unknown / extra keys from the LLM,
    - is friendly to both Pydantic v1 and v2 calling styles.
    """
    model_config = {"extra": "ignore"}


class Tier1SignalPack(RobustBaseModel):
    """
    Unified Tier-1 output signals consumed by:
      - RL observation builder (signal, confidence)
      - reward shaping builder (risk_score, optionally confidence)
      - senior analyst (rationale as interpretability bridge)

    Interpretation:
      - signal: signed directional impact (positive bullish, negative bearish)
      - risk_score: unsigned downside risk suitable for reward penalty
      - confidence: reliability of signal/risk estimate
    """

    signal: float = Field(
        0.0,
        description="Signed directional signal in [-10, 10]. -10 strongly bearish, +10 strongly bullish.",
    )
    risk_score: float = Field(
        0.0,
        description="Downside risk in [0, 10]. Used as reward penalty factor.",
    )
    confidence: float = Field(
        0.5,
        description="Confidence in [0, 1]. Can be used as feature/weight if enabled.",
    )
    rationale: str = Field(
        "",
        description="2–5 concise sentences referencing specific items from the reduced analysis.",
    )

    @field_validator("signal", mode="before")
    def _clamp_signal(cls, v):
        try:
            x = float(v)
        except Exception:
            return 0.0
        if x != x:
            return 0.0
        return max(-10.0, min(10.0, x))

    @field_validator("risk_score", mode="before")
    def _clamp_risk(cls, v):
        try:
            x = float(v)
        except Exception:
            return 0.0
        if x != x:
            return 0.0
        return max(0.0, min(10.0, x))

    @field_validator("confidence", mode="before")
    def _clamp_conf(cls, v):
        try:
            x = float(v)
        except Exception:
            return 0.5
        if x != x:
            return 0.5
        return max(0.0, min(1.0, x))


def _iso(ts) -> str:
    return pd.to_datetime(ts, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_json_object(text: str) -> str:
    """
    Best-effort: extract the first top-level JSON object from a text blob.
    This is only used as a fallback when structured parsing fails.
    """
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _log_llm_failure(
    *,
    model_cfg: Dict[str, Any],
    schema: Type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    raw_text: str | None,
    json_str: str | None,
    error: Exception | None,
) -> None:
    """
    Append a single JSON line describing a failed structured call.

    - Uses a base directory `failure_log_dir` from model_cfg if present,
      otherwise defaults to `data/raw/compression_failures`.
    - One file per schema class, e.g. `ECChunkMap_failures.jsonl`,
      `SECReduceFinal_failures.jsonl`.
    """
    try:
        base_dir = Path(model_cfg.get("failure_log_dir", "data/raw/compression_failures"))
        base_dir.mkdir(parents=True, exist_ok=True)

        schema_name = schema.__name__
        log_path = base_dir / f"{schema_name}_failures.jsonl"

        def _truncate(s: str | None, limit: int = 4000) -> str | None:
            if s is None:
                return None
            s = str(s)
            return s if len(s) <= limit else s[:limit] + "... [truncated]"

        payload = {
            "schema": schema_name,
            "error": repr(error) if error is not None else None,
            "raw_text": _truncate(raw_text, 8000),
            "json_str": _truncate(json_str, 4000),
            "system_prompt": _truncate(system_prompt, 2000),
            "user_prompt": _truncate(user_prompt, 2000),
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never break the main pipeline; swallow all logging errors.
        pass


# =========================
#   Cross-process Azure rate limiting
# =========================

class _CrossProcessRateLimiter:
    """
    Simple cross-process rate limiter: max_calls per 60-second window,
    backed by a small JSON file and (on Unix) an fcntl lock.

    This is approximate but good enough to keep multiple processes from
    hammering the same Azure deployment beyond a configured RPM.
    """

    def __init__(self, max_calls_per_min: int, lockfile_path: Path):
        self.max_calls = max_calls_per_min
        self.window = 60.0
        self.lockfile_path = lockfile_path
        # Ensure directory exists
        self.lockfile_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_timestamps(self, f) -> List[float]:
        f.seek(0)
        data = f.read().strip()
        if not data:
            return []
        try:
            arr = json.loads(data)
            if not isinstance(arr, list):
                return []
            ts_list: List[float] = []
            for v in arr:
                try:
                    x = float(v)
                    ts_list.append(x)
                except Exception:
                    continue
            return ts_list
        except Exception:
            # Corrupt file; reset
            return []

    def _store_timestamps(self, f, timestamps: List[float]) -> None:
        f.seek(0)
        f.truncate()
        f.write(json.dumps(timestamps))

    def acquire(self) -> None:
        """
        Block until a call slot is available for this deployment.
        """
        while True:
            now = time.time()
            sleep_for = 0.0

            # Open file in a+ mode so it is created if missing
            with self.lockfile_path.open("a+") as f:
                # Acquire an exclusive lock if fcntl is available (Unix)
                if fcntl is not None:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX)
                    except OSError:
                        # If locking fails, we still proceed best-effort
                        pass

                timestamps = self._load_timestamps(f)

                # Drop timestamps outside the window
                cutoff = now - self.window
                timestamps = [t for t in timestamps if t >= cutoff]

                if len(timestamps) < self.max_calls:
                    # We have capacity; record this call and return
                    timestamps.append(now)
                    self._store_timestamps(f, timestamps)

                    if fcntl is not None:
                        try:
                            fcntl.flock(f, fcntl.LOCK_UN)
                        except OSError:
                            pass
                    return

                # Otherwise, need to wait until oldest timestamp falls out of window
                if timestamps:
                    oldest = min(timestamps)
                    sleep_for = self.window - (now - oldest)

                if fcntl is not None:
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except OSError:
                        pass

            if sleep_for <= 0:
                sleep_for = 0.01
            time.sleep(sleep_for)


def _get_azure_rate_limiter(model_cfg: Dict[str, Any]) -> _CrossProcessRateLimiter:
    """
    Build a cross-process limiter keyed by deployment.

    Configuration precedence for RPM:
      1) model_cfg["max_requests_per_min"]
      2) env AZURE_RPM_LIMIT
      3) default 40
    """
    max_rpm = int(
        model_cfg.get(
            "max_requests_per_min",
            os.getenv("AZURE_RPM_LIMIT", "40"),
        )
    )

    deployment = model_cfg.get("azure_deployment", "default")
    base_dir = Path(os.getenv("AZURE_RATELIMIT_DIR", "/tmp/azure_llm_ratelimit"))
    lockfile_path = base_dir / f"{deployment}_rpm.json"

    return _CrossProcessRateLimiter(max_calls_per_min=max_rpm, lockfile_path=lockfile_path)


class RateLimitedAzureChatOpenAI:
    """
    Thin wrapper around AzureChatOpenAI that enforces a cross-process
    requests-per-minute limit.

    It forwards most attributes to the underlying client; only invoke /
    batch / ainvoke / abatch are explicitly rate-limited.
    """

    def __init__(self, inner_llm, rate_limiter: _CrossProcessRateLimiter):
        self._inner_llm = inner_llm
        self._rate_limiter = rate_limiter

    # --- Core sync interfaces ---

    def invoke(self, *args, **kwargs):
        self._rate_limiter.acquire()
        return self._inner_llm.invoke(*args, **kwargs)

    def batch(self, *args, **kwargs):
        # Treat one batch as a single request for rate limiting.
        self._rate_limiter.acquire()
        return self._inner_llm.batch(*args, **kwargs)

    # --- Async variants (best-effort, still block with time.sleep) ---

    async def ainvoke(self, *args, **kwargs):
        self._rate_limiter.acquire()
        return await self._inner_llm.ainvoke(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        self._rate_limiter.acquire()
        return await self._inner_llm.abatch(*args, **kwargs)

    # --- Delegate everything else to the inner client ---

    def __getattr__(self, item):
        return getattr(self._inner_llm, item)


def _make_llm(model_cfg: Dict[str, Any]):
    """
    Create an LLM client according to model_cfg.

    Supports:
      - backend: "azure"  -> AzureChatOpenAI
      - backend: "ollama" -> ChatOllama

    model_cfg is expected to be cfg["model"] from the YAML config.
    """
    backend = model_cfg.get("backend", "ollama").lower()

    if backend == "azure":
        # Required keys: azure_deployment, azure_api_base, azure_api_version
        from langchain_openai import AzureChatOpenAI
        os.environ["AZURE_OPENAI_API_KEY"] = model_cfg['azure_api_key']
        os.environ["AZURE_OPENAI_ENDPOINT"] = model_cfg['azure_endpoint']
        time.sleep(1)  
        base_llm = AzureChatOpenAI(
            azure_deployment=model_cfg["azure_deployment"],
            api_version=model_cfg["azure_api_version"],
            temperature=float(model_cfg.get("temperature", 0.0)),
        )
        rate_limiter = _get_azure_rate_limiter(model_cfg)
        return RateLimitedAzureChatOpenAI(inner_llm=base_llm, rate_limiter=rate_limiter)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_cfg["name"],
            base_url=model_cfg.get("base_url", "http://localhost:11434"),
            temperature=float(model_cfg.get("temperature", 0.0)),
        )


def _structured_call(
    model_cfg: Dict[str, Any],
    schema: Type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    retries: int,
) -> BaseModel:
    """
    Invoke LLM with system + user prompt and parse into the given schema using Pydantic.

    This uses a system+user message pair and retries a few times, logging
    failures but not crashing the entire pipeline.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = _make_llm(model_cfg)
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    last_err: Exception | None = None
    last_raw_text: str | None = None
    last_json_str: str | None = None

    for _ in range(max(1, int(retries) + 1)):
        try:
            raw = llm.invoke(msgs)
            raw_text = getattr(raw, "content", str(raw))
            json_str = _extract_json_object(raw_text)
            last_raw_text = raw_text
            last_json_str = json_str
            data = json.loads(json_str)
            return schema.model_validate(data)
        except Exception as e:
            last_err = e
            # On retry, reinforce JSON-only rule
            msgs[-1] = HumanMessage(
                content=(
                    user_prompt
                    + "\n\nIMPORTANT: Return ONLY a single JSON object that matches "
                    "the schema. No explanation, no markdown, no extra text."
                )
            )

    _log_llm_failure(
        model_cfg=model_cfg,
        schema=schema,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_text=last_raw_text,
        json_str=last_json_str,
        error=last_err,
    )

    raise ValueError(f"Structured call failed after retries. Last error: {last_err}")


def _simple_char_chunks(text: str, max_len: int, overlap: int, max_chunks: int) -> List[str]:
    """
    Sentence-aware chunker with approximate character budget and overlap.

    - Splits on sentence boundaries using a simple regex.
    - Packs sentences greedily into chunks up to ~max_len characters.
    - Adds overlap by carrying forward tail sentences from the previous chunk
      whose total length is <= overlap.
    - Respects max_chunks as an upper bound.

    This is deliberately simple and we control chunk size by config to
    fit 4o-mini's long context (e.g. 60–80k chars per chunk).
    """
    text = (text or "").strip()
    if not text:
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Simple sentence split
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text[:max_len]] if text else []

    chunks: List[str] = []
    curr: List[str] = []
    curr_len = 0
    prev_tail: List[str] = []

    def finalize_chunk():
        nonlocal chunks, curr, curr_len, prev_tail
        if not curr:
            return
        if prev_tail:
            chunk_sentences = prev_tail + curr
        else:
            chunk_sentences = curr
        chunk_text = " ".join(chunk_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)
        # compute new tail from curr
        tail: List[str] = []
        total = 0
        for s in reversed(curr):
            s_len = len(s) + 1
            if total + s_len > overlap:
                break
            tail.insert(0, s)
            total += s_len
        prev_tail = tail
        curr = []
        curr_len = 0

    for s in sentences:
        s_len = len(s) + (1 if curr else 0)
        if curr_len + s_len <= max_len:
            curr.append(s)
            curr_len += s_len
        else:
            finalize_chunk()
            if len(chunks) >= max_chunks:
                break
            curr = [s]
            curr_len = len(s)

    if curr and len(chunks) < max_chunks:
        finalize_chunk()

    if not chunks and text:
        return [text[:max_len]]
    return chunks[:max_chunks]