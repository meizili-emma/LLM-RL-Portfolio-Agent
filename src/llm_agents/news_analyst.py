from __future__ import annotations

import pandas as pd 
from typing import Any, Dict, List, Tuple, Type
from pathlib import Path
import json
import yaml
from tqdm import tqdm

import argparse

from src.llm_agents.utils import (
    _iso,
    _structured_call,
)
from src.llm_agents.news_schema import (
    NewsWeeklyReduce,
    NEWS_SYSTEM_PROMPT,
    news_reduce_user_prompt,
)


# =========================
#   News selection helpers
# =========================

DEFAULT_TIER_WEIGHT = {"A": 1.0, "B": 0.8, "C": 0.6}


def _compute_news_score_for_window(
    row: pd.Series,
    window_end: pd.Timestamp,
    *,
    tier_weight_map: Dict[str, float] = DEFAULT_TIER_WEIGHT,
    length_ref: int = 256,
    half_life_hours: float = 72.0,
    w_tier: float = 1.0,
    w_len: float = 0.5,
    w_rec: float = 1.0,
) -> float:
    """
    Compute a scalar news_score for a single article within a given weekly window.

    score = w_tier * f(source_tier) + w_len * f(text_length) + w_rec * f(recency)

    - source_tier: A/B/C (weak prior on quality)
    - text length: discount very short snippets, cap over-long texts
    - recency: half-life decay from window_end backward
    """
    # Tier weight (weak prior)
    tier = str(row.get("source_tier") or "C")
    tier_w = tier_weight_map.get(tier, tier_weight_map.get("C", 0.6))

    # Length normalization
    txt = str(row.get("text") or "")
    text_len = len(txt)
    length_norm = min(text_len / float(length_ref), 2.0)  # cap at 2x

    # Recency (half-life decay from window_end backwards)
    ts = pd.to_datetime(row.get("published_at_utc"))
    if pd.isna(ts):
        recency_weight = 0.0
    else:
        age_sec = max((window_end - ts).total_seconds(), 0.0)
        half_life_sec = max(half_life_hours * 3600.0, 1.0)
        recency_weight = 0.5 ** (age_sec / half_life_sec)

    score = (
        w_tier * tier_w
        + w_len * length_norm
        + w_rec * recency_weight
    )
    return float(score)


def select_weekly_news_context(
    ticker: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    all_news_df: pd.DataFrame,
    *,
    max_articles_per_day: int = 2,
    max_articles_total: int = 40,
    max_total_chars: int = 8000,
    max_chars_per_article: int = 500,
    min_text_chars: int = 40,
    half_life_hours: float = 72.0,
    tier_weight_map: Dict[str, float] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Two-step selection of news for a given (ticker, weekly window):

    Step 1 (coverage):
      - For each calendar day with news in the window,
        pick up to `max_articles_per_day` highest-scoring articles.

    Step 2 (global fill):
      - From remaining unselected articles in the week, pick the
        highest-scoring ones until hitting `max_articles_total` or
        `max_total_chars`.

    Returns:
      context_str: formatted "[ARTICLE] ..." blocks for LLM (may be empty).
      stats: simple stats for debugging / optional RL features.
    """
    if tier_weight_map is None:
        tier_weight_map = DEFAULT_TIER_WEIGHT

    df = all_news_df.copy()

    # Filter by ticker and time window
    window_start = pd.to_datetime(window_start, utc=True, errors="coerce")
    window_end = pd.to_datetime(window_end, utc=True, errors="coerce")
    df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], utc=True, errors="coerce")
    mask = (
        (df["ticker"] == ticker)
        & (df["published_at_utc"] >= window_start)
        & (df["published_at_utc"] <= window_end)
    )
    df = df.loc[mask].copy()

    num_candidates = int(len(df))
    if num_candidates == 0:
        stats = {
            "num_candidates": 0,
            "num_selected": 0,
            "count_tier_A": 0,
            "count_tier_B": 0,
            "count_tier_C": 0,
            "num_days_with_news": 0,
            "avg_source_score": float("nan"),
            "max_source_score": float("nan"),
        }
        return "", stats

    # Drop clearly unusable text
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() >= min_text_chars].copy()
    if df.empty:
        stats = {
            "num_candidates": num_candidates,
            "num_selected": 0,
            "count_tier_A": 0,
            "count_tier_B": 0,
            "count_tier_C": 0,
            "num_days_with_news": 0,
            "avg_source_score": float("nan"),
            "max_source_score": float("nan"),
        }
        return "", stats

    # Compute per-article scores
    df["news_score"] = df.apply(
        _compute_news_score_for_window,
        axis=1,
        window_end=window_end,
        tier_weight_map=tier_weight_map,
        half_life_hours=half_life_hours,
    )

    # Basic stats
    counts_by_tier = df["source_tier"].fillna("C").value_counts()
    count_A = int(counts_by_tier.get("A", 0))
    count_B = int(counts_by_tier.get("B", 0))
    count_C = int(counts_by_tier.get("C", 0))

    if "source_score" in df.columns:
        avg_src = float(df["source_score"].mean())
        max_src = float(df["source_score"].max())
    else:
        avg_src = float("nan")
        max_src = float("nan")

    # Per-day grouping
    df["day"] = df["published_at_utc"].dt.normalize()

    selected_indices: List[int] = []
    char_budget = 0

    # Step 1: per-day coverage
    for day, day_group in df.groupby("day"):
        if char_budget >= max_total_chars or len(selected_indices) >= max_articles_total:
            break

        day_sorted = day_group.sort_values("news_score", ascending=False)
        day_selected = 0

        for idx, row in day_sorted.iterrows():
            if day_selected >= max_articles_per_day:
                break
            if len(selected_indices) >= max_articles_total:
                break

            text = str(row["text"])
            proposed_len = min(len(text), max_chars_per_article)
            if char_budget + proposed_len > max_total_chars and char_budget > 0:
                continue

            selected_indices.append(idx)
            char_budget += proposed_len
            day_selected += 1

    # Step 2: global fill
    if char_budget < max_total_chars and len(selected_indices) < max_articles_total:
        remaining = df.drop(index=selected_indices)

        if not remaining.empty:
            remaining_sorted = remaining.sort_values(
                ["news_score", "published_at_utc"],
                ascending=[False, False],  # high score, then newest first
            )
            for idx, row in remaining_sorted.iterrows():
                if len(selected_indices) >= max_articles_total:
                    break

                text = str(row["text"])
                proposed_len = min(len(text), max_chars_per_article)
                if char_budget + proposed_len > max_total_chars and char_budget > 0:
                    continue

                selected_indices.append(idx)
                char_budget += proposed_len

                if char_budget >= max_total_chars:
                    break

    num_selected = len(selected_indices)
    if num_selected == 0:
        stats = {
            "num_candidates": num_candidates,
            "num_selected": 0,
            "count_tier_A": count_A,
            "count_tier_B": count_B,
            "count_tier_C": count_C,
            "num_days_with_news": int(df["day"].nunique()),
            "avg_source_score": avg_src,
            "max_source_score": max_src,
        }
        return "", stats

    # Order selected by time (oldest -> newest) for the LLM
    sel_df = df.loc[selected_indices].sort_values("published_at_utc")

    context_parts: List[str] = []
    for _, row in sel_df.iterrows():
        txt = str(row["text"])
        truncated = txt[:max_chars_per_article]

        part_lines = [
            "[ARTICLE]",
            f"published_at_utc: {row['published_at_utc'].isoformat()}",
            f"text: {truncated}",
        ]
        if "source" in row:
            part_lines.insert(2, f"source: {row.get('source', '')}")
        context_parts.append("\n".join(part_lines))
    context_str = "\n---\n".join(context_parts)

    stats = {
        "num_candidates": num_candidates,
        "num_selected": num_selected,
        "count_tier_A": count_A,
        "count_tier_B": count_B,
        "count_tier_C": count_C,
        "num_days_with_news": int(df["day"].nunique()),
        "avg_source_score": avg_src,
        "max_source_score": max_src,
    }
    return context_str, stats


def build_weekly_news_bundles_from_files(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Step 1: Given raw news + calendar, build weekly news bundles:

      (ticker, week_decision_date, window_start, window_end, news_context, stats...)

    This step is deterministic and cheap (no LLM), so we can safely overwrite.
    - If an existing bundles parquet exists, we load it and SKIP any
      (ticker, week_decision_date) pairs already present.
    - We flush results to disk incrementally at the end of each week_decision_date,
      so progress is saved and runs can be resumed.
    """
    news_cfg = cfg["news"]

    raw_news_path = Path(news_cfg["raw_news_path"])
    calendar_path = Path(news_cfg["calendar_path"])
    bundles_out_path = Path(news_cfg["bundles_out_path"])

    news_df = pd.read_parquet(raw_news_path)
    cal_df = pd.read_parquet(calendar_path)

    canonical_tickers = cfg["portfolio"]["canonical_tickers"]
    if not canonical_tickers:
        raise ValueError("cfg['portfolio']['canonical_tickers'] is empty. Please set it in YAML.")
    canonical_tickers = [str(t).strip() for t in canonical_tickers if str(t).strip()]

    # Ensure proper dtypes
    news_df = news_df.copy()
    news_df["published_at_utc"] = pd.to_datetime(
        news_df["published_at_utc"], utc=True, errors="coerce"
    )

    cal = cal_df.copy()
    cal["week_decision_date"] = pd.to_datetime(cal["week_decision_date"])
    cal["prev_close_utc"] = pd.to_datetime(cal["prev_close_utc"], utc=True)
    cal["curr_close_utc"] = pd.to_datetime(cal["curr_close_utc"], utc=True)

    tier_weight = news_cfg.get("tier_weight", {})

    if bundles_out_path.exists():
        existing_df = pd.read_parquet(bundles_out_path)
        # Normalise types to be safe
        if "week_decision_date" in existing_df.columns:
            existing_df["week_decision_date"] = pd.to_datetime(existing_df["week_decision_date"])
        existing_df["ticker"] = existing_df["ticker"].astype(str)

        done_pairs = set(
            zip(
                existing_df["ticker"].tolist(),
                existing_df["week_decision_date"].tolist(),
            )
        )
    else:
        existing_df = None
        done_pairs: set[tuple[str, pd.Timestamp]] = set()

    new_rows: List[Dict[str, Any]] = []

    for _, r in tqdm(cal.iterrows(), total=len(cal), desc="Building news bundles"):
        
        wdate = r["week_decision_date"]
        window_start = r["prev_close_utc"]
        window_end = r["curr_close_utc"]

        for ticker in canonical_tickers:
            key = (ticker, wdate)
            if key in done_pairs:
                # Already built previously; skip
                continue

            context_str, stats = select_weekly_news_context(
                ticker=ticker,
                window_start=window_start,
                window_end=window_end,
                all_news_df=news_df,
                max_articles_per_day=int(news_cfg["max_articles_per_day"]),
                max_articles_total=int(news_cfg["max_articles_total"]),
                max_total_chars=int(news_cfg["max_total_chars"]),
                max_chars_per_article=int(news_cfg["max_chars_per_article"]),
                min_text_chars=int(news_cfg["min_text_chars"]),
                half_life_hours=float(news_cfg["half_life_hours"]),
                tier_weight_map=tier_weight,
            )

            row_out: Dict[str, Any] = {
                "ticker": ticker,
                "week_decision_date": wdate,
                "window_start": window_start,
                "window_end": window_end,
                "news_context": context_str,
            }
            row_out.update(stats)
            new_rows.append(row_out)

        # --- Flush after finishing this week_decision_date ---
        if new_rows:
            new_df = pd.DataFrame(new_rows)

            if existing_df is not None:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined = new_df

            # Optional: keep a stable ordering
            combined = combined.sort_values(["week_decision_date", "ticker"]).reset_index(drop=True)

            bundles_out_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(bundles_out_path, index=False)

            # Update in-memory state for resume logic within this run
            existing_df = combined
            for rr in new_rows:
                done_pairs.add((rr["ticker"], rr["week_decision_date"]))
            new_rows = []

    # At the end, existing_df holds the full bundles table (old + new)
    if existing_df is None:
        # No rows produced at all; return an empty DataFrame with expected columns
        cols = [
            "ticker",
            "week_decision_date",
            "window_start",
            "window_end",
            "news_context",
        ]
        return pd.DataFrame(columns=cols)

    return existing_df


def _analyze_single_news_row(
    row: pd.Series,
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the news LLM analyst for a single (ticker, week) bundle row.

    Short-circuits to NO_MEANINGFUL_NEWS when context is empty.
    """
    ticker = str(row["ticker"])
    wdate = row["week_decision_date"]
    window_start = row["window_start"]
    window_end = row["window_end"]
    ctx = row.get("news_context", "") or ""

    # If no context, avoid calling the LLM and return the "no news" default.
    if not ctx.strip():
    # IMPORTANT: Must match NewsRLSignals exactly: signal / risk_score / confidence / rationale
        rl_default = {
            "signal": 0.0,
            "risk_score": 0.0,
            "confidence": 0.0,
            "rationale": "NO_MEANINGFUL_NEWS",
            }
        out = NewsWeeklyReduce(
            summary_text="",
            key_events=[],
            rl=rl_default,
            )
    else:
        user_prompt = news_reduce_user_prompt(
            ticker=ticker,
            week_decision_date=str(wdate.date()),
            window_start_utc=_iso(window_start),
            window_end_utc=_iso(window_end),
            news_context=ctx,
        )
        retries = int(model_cfg.get("max_retries", 2))
        out = _structured_call(
            model_cfg=model_cfg,
            schema=NewsWeeklyReduce,
            system_prompt=NEWS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            retries=retries,
        )

    # Flatten for parquet output
    rl = out.rl
    return {
        "ticker": ticker,
        "week_decision_date": wdate,
        "window_start": window_start,
        "window_end": window_end,
        "summary_text": out.summary_text,
        # RL features / risk factors
        "news_rl_signal": rl.signal,
        "news_rl_risk_score": rl.risk_score,
        "news_rl_confidence": rl.confidence,
        "news_rl_rationale": rl.rationale,
        # Optional: key_events as JSON for senior agents
        "key_events_json": json.dumps([e.model_dump() for e in out.key_events]),
    }


def run_news_analyst(cfg: Dict[str, Any]) -> None:
    """
    Full news pipeline:

      1) Build weekly bundles from raw news + calendar (deterministic, overwrites).
      2) Run LLM analyst on each (ticker, week) row with resume-on-interruption.
    """
    news_cfg = cfg["news"]
    bundles_path = Path(news_cfg["bundles_out_path"])
    out_path = Path(news_cfg["out_path"])

    # Step 1: bundles (always rebuilt; cheap and deterministic)
    if bundles_path.exists():
        bundles_df = pd.read_parquet(bundles_path)
    else:
        bundles_df = build_weekly_news_bundles_from_files(cfg)

    # Step 2: LLM analysis with resume
    if out_path.exists():
        out_df = pd.read_parquet(out_path)
    else:
        out_df = pd.DataFrame(
            columns=[
                "ticker",
                "week_decision_date",
                "window_start",
                "window_end",
                "summary_text",
                "news_rl_signal",
                "news_rl_risk_score",
                "news_rl_confidence",
                "news_rl_rationale",
                "key_events_json",
            ]
        )

    if out_df.empty:
        processed_keys: set[Tuple[str, str]] = set()
    else:
        processed_keys = set(
            zip(
                out_df["ticker"].astype(str),
                out_df["week_decision_date"].astype(str),
            )
        )

    todo_rows: List[pd.Series] = []
    for _, r in bundles_df.iterrows():
        tk = str(r["ticker"])
        wd = str(r["week_decision_date"])
        if (tk, wd) not in processed_keys:
            todo_rows.append(r)

    if not todo_rows:
        print("News analyst: nothing to do (all ticker-week rows already processed).")
        return

    print(f"News analyst: {len(todo_rows)} ticker-week rows to process.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("News out_path:", out_path.resolve())

    for r in tqdm(todo_rows, desc="News analysis"):
        try:
            out_row = _analyze_single_news_row(
                row=r,
                model_cfg=cfg["model"],
            )
        except Exception as e:
            print(
                f"[NEWS] Skipping row (ticker={r.get('ticker','')}, "
                f"week_decision_date={r.get('week_decision_date','')}) "
                f"due to error: {e}"
            )
            continue

        out_df = pd.concat(
            [out_df, pd.DataFrame([out_row])],
            ignore_index=True,
        )
        out_df.to_parquet(out_path, index=False)

    print(f"News analyst: wrote {len(out_df)} total ticker-week rows to {out_path}")


def _load_cfg(path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="data/raw/config/news_analyst.yaml",
        help="Path to compression config YAML.",
    )
    args = ap.parse_args()
    cfg = _load_cfg(Path(args.config))
    run_news_analyst(cfg)
    

if __name__ == "__main__":
    main()