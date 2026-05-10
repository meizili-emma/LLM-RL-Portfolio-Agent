from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Literal, Dict

import pandas as pd


def score_sources_terciles(
    news_parquet_path: str | Path,
    out_path: Optional[str | Path] = None,
    *,
    source_col: str = "source",
    tier_col: str = "source_tier",
    score_col: str = "source_score",
    top_label: Literal["A","T1"] = "A",
    mid_label: Literal["B","T2"] = "B",
    bot_label: Literal["C","T3"] = "C",
    top_score: int = 3,
    mid_score: int = 2,
    bot_score: int = 1,
    treat_null_source_as: Optional[Literal["A","B","C"]] = "C",
) -> Dict[str, object]:
    """
    Compute global source tiers (terciles) from the full news corpus and assign per-row scores.

    Ranking rule (deterministic):
      1) sort sources by article count descending,
      2) break ties by source name ascending,
      3) split into three equal-sized buckets by index (top/mid/bottom terciles).

    Args:
        news_parquet_path: Input parquet with at least a 'source' column.
        out_path: Where to write the updated parquet. If None, overwrite input.
        source_col: Column name for source.
        tier_col: Output column for tier label.
        score_col: Output column for numeric score.
        top/mid/bot_label: Labels for tiers.
        top/mid/bot_score: Numeric scores for tiers (3/2/1 by default).
        treat_null_source_as: How to score rows where source is null (default "C").
                              If None, leaves tier/score as NA for null sources.

    Returns:
        A dict summary with keys:
          - 'n_rows', 'n_sources', 'cut_indices', 'tier_sizes',
            'head_mapping' (first few sources), 'path_written'
    """
    news_parquet_path = Path(news_parquet_path)
    if out_path is None:
        out_path = news_parquet_path
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(news_parquet_path)

    if source_col not in df.columns:
        raise ValueError(f"Missing '{source_col}' in {news_parquet_path}")

    # Build counts per source (ignore nulls for tiering)
    src_counts = (
        df.dropna(subset=[source_col])
          .groupby(source_col, dropna=False, as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )

    if src_counts.empty:
        # Write through with NA tier/score if nothing to score
        df[tier_col] = pd.Series([pd.NA] * len(df), dtype="object")
        df[score_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df.to_parquet(out_path, index=False)
        return {
            "n_rows": int(len(df)),
            "n_sources": 0,
            "cut_indices": (0, 0),
            "tier_sizes": {"A": 0, "B": 0, "C": 0},
            "head_mapping": [],
            "path_written": str(out_path),
        }

    # Deterministic ordering: (-count, source asc)
    src_counts = src_counts.sort_values(["count", source_col], ascending=[False, True]).reset_index(drop=True)

    n_src = len(src_counts)
    # Compute 1/3 and 2/3 cut positions (ceil to keep top bucket at least as large when not divisible)
    cut1 = math.ceil(n_src / 3)
    cut2 = math.ceil(2 * n_src / 3)

    # Assign tiers by positional index
    tiers = pd.Series([None] * n_src, dtype="object")
    tiers.iloc[:cut1] = top_label
    tiers.iloc[cut1:cut2] = mid_label
    tiers.iloc[cut2:] = bot_label
    src_counts["tier"] = tiers

    # Numeric scores
    score_map = {top_label:float(top_score), mid_label: float(mid_score), bot_label: float(bot_score)}
    src_counts["score"] = src_counts["tier"].map(score_map)

    # Build mapping dicts
    tier_map = dict(zip(src_counts[source_col].astype(str), src_counts["tier"]))
    score_map_full = dict(zip(src_counts[source_col].astype(str), src_counts["score"]))

    # Apply to full dataframe (including rows with null sources)
    # Convert sources to string for consistent mapping; keep NA separately
    src_as_str = df[source_col].astype("string")
    df[tier_col] = src_as_str.map(tier_map)
    df[score_col] = src_as_str.map(score_map_full)

    if treat_null_source_as is not None:
        null_mask = df[source_col].isna()
        if null_mask.any():
            df.loc[null_mask, tier_col] = treat_null_source_as
            df.loc[null_mask, score_col] = float(score_map[treat_null_source_as])

    # Persist
    df.to_parquet(out_path, index=False)

    # Summaries
    tier_sizes = {
        str(top_label): int((src_counts["tier"] == top_label).sum()),
        str(mid_label): int((src_counts["tier"] == mid_label).sum()),
        str(bot_label): int((src_counts["tier"] == bot_label).sum()),
    }
    head_map = src_counts.head(8).to_dict(orient="records")

    return {
        "n_rows": int(len(df)),
        "n_sources": int(n_src),
        "cut_indices": (cut1, cut2),
        "tier_sizes": tier_sizes,
        "head_mapping": head_map,
        "path_written": str(out_path),
    }
