from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.analysis.paths import get_run_paths


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Simple JSONL reader returning a list of Python dicts."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_meta(run_dir: str | Path) -> Dict[str, Any]:
    """
    Load meta.json for a run.

    Expected keys (at minimum):
      - config
      - created_at
      - split_plan (added by run.py)
    """
    paths = get_run_paths(run_dir)
    meta_path: Path = paths["meta"]  # type: ignore[assignment]
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under run_dir: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_train_steps(run_dir: str | Path) -> pd.DataFrame:
    """
    Load and flatten train_steps.jsonl (PortfolioLoggingCallback output).

    Each line is expected to include nested dicts:
      - meta
      - reward_components
      - trading

    We flatten to columns with prefixes:
      meta.xxx, reward.xxx, trading.xxx
    """
    paths = get_run_paths(run_dir)
    path: Path = paths["train_steps"]  # type: ignore[assignment]
    records = _read_jsonl(path)

    flat_rows: List[Dict[str, Any]] = []
    for rec in records:
        row: Dict[str, Any] = {}

        meta = rec.get("meta", {}) or {}
        for k, v in meta.items():
            row[f"meta.{k}"] = v

        reward = rec.get("reward_components", {}) or {}
        for k, v in reward.items():
            row[f"reward.{k}"] = v

        trading = rec.get("trading", {}) or {}
        for k, v in trading.items():
            row[f"trading.{k}"] = v

        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)

    # Parse dates if present
    if "meta.date" in df.columns:
        df["meta.date"] = pd.to_datetime(df["meta.date"], errors="coerce")

    # Convenience alias if present
    if "reward.total_reward" in df.columns and "reward" not in df.columns:
        df["reward"] = pd.to_numeric(df["reward.total_reward"], errors="coerce")

    return df


def load_train_episodes(run_dir: str | Path) -> pd.DataFrame:
    """
    Load train_episodes.jsonl (episode_summary records).
    We keep scalar fields as columns and leave list-like histories as Python objects.
    """
    paths = get_run_paths(run_dir)
    path: Path = paths["train_episodes"]  # type: ignore[assignment]
    records = _read_jsonl(path)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # best-effort numeric coercion for common fields
    for col in [
        "episode_id",
        "final_portfolio_value",
        "sharpe_ratio",
        "max_drawdown",
        "episode_steps",
        "trade_count",
        "total_trade_cost",
        "avg_trade_cost_per_step",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # dates (if any)
    for col in ["start_date", "end_date", "terminal_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def load_test_steps(run_dir: str | Path) -> pd.DataFrame:
    """
    Load and flatten test_steps.jsonl (run_test_episodes output).

    Each line is expected to include:
      - top-level primitives: episode_id, env_id, step_idx, day_idx, date, close_ts_utc, reward
      - nested dicts: reward_components, trading

    We flatten nested dicts to prefixed columns.
    """
    paths = get_run_paths(run_dir)
    path: Path = paths["test_steps"]  # type: ignore[assignment]
    records = _read_jsonl(path)

    flat_rows: List[Dict[str, Any]] = []
    for rec in records:
        row: Dict[str, Any] = {}

        # Top-level primitive fields (do NOT assume all exist)
        for key in ["episode_id", "env_id", "step_idx", "day_idx", "date", "close_ts_utc", "reward"]:
            if key in rec:
                row[key] = rec[key]

        reward = rec.get("reward_components", {}) or {}
        for k, v in reward.items():
            row[f"reward.{k}"] = v

        trading = rec.get("trading", {}) or {}
        for k, v in trading.items():
            row[f"trading.{k}"] = v

        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "reward" in df.columns:
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce")

    return df


def load_test_episodes(run_dir: str | Path) -> pd.DataFrame:
    """
    Load test_episodes.jsonl (episode_summary records for test episodes).
    """
    paths = get_run_paths(run_dir)
    path: Path = paths["test_episodes"]  # type: ignore[assignment]
    records = _read_jsonl(path)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    for col in [
        "episode_id",
        "final_portfolio_value",
        "sharpe_ratio",
        "max_drawdown",
        "episode_steps",
        "trade_count",
        "total_trade_cost",
        "avg_trade_cost_per_step",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["start_date", "end_date", "terminal_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def load_monitor(run_dir: str | Path, phase: str) -> pd.DataFrame:
    """
    Load and concatenate Monitor CSVs for a given phase ("train" or "test").

    Files are globbed (multiple envs):
      train/StockPortfolioEnv_env*.monitor.csv
      test/StockPortfolioEnv_env*.monitor.csv

    If none exist, return an empty DataFrame.
    """
    paths = get_run_paths(run_dir)
    if phase == "train":
        files: List[Path] = paths["train_monitor_files"]  # type: ignore[assignment]
    elif phase == "test":
        files = paths["test_monitor_files"]  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown phase: {phase!r}. Use 'train' or 'test'.")

    dfs: List[pd.DataFrame] = []
    for f in files:
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            # monitor files sometimes have comment header; try python engine
            df = pd.read_csv(f, comment="#", engine="python")
        df["monitor_file"] = f.name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_train_progress(run_dir: str | Path) -> pd.DataFrame:
    """
    Load SB3 training progress CSV.
    We prefer sb3_train_progress.csv if it exists; otherwise fall back to progress.csv.
    """
    paths = get_run_paths(run_dir)
    path: Path = paths["train_progress"]  # type: ignore[assignment]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
