
import pandas as pd
from pathlib import Path
import logging 
from tqdm import tqdm 
from src.llm_agents.news_analyst import NewsAnalyst


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_news_sources_whitelist(news_df: pd.DataFrame, threshold: float = 0.15) -> list:
    """
    Get a whitelist of news sources based on their frequency in the dataset.
    
    Args:
        news_df (pd.DataFrame): DataFrame containing news data with a 'source' column.
        threshold (float): Top percentage of news sources to include in the whitelist.
        
    Returns:
        list: List of news sources that meet the frequency threshold.
    """
    news_sources_count = news_df['source'].value_counts()
    threshold = int(len(news_sources_count) * threshold)
    return news_sources_count.nlargest(threshold).index.tolist()


def get_weekly_news_context(
    ticker: str,
    week_end_date: str,
    all_news_df: pd.DataFrame,
    whitelist: list[str],
    max_articles: int = 100,
    max_length: int = 512
) -> str:
    """
    Selects, filters, and formats news for a given stock and week into
    a single string for LLM analysis.
    """
    week_end = pd.to_datetime(week_end_date)
    week_start = week_end - pd.DateOffset(days=6)
    all_news_df['date'] = pd.to_datetime(all_news_df['date'])
    weekly_news = all_news_df[
        (all_news_df['ticker'] == ticker) &
        (all_news_df['date'] >= week_start) &
        (all_news_df['date'] <= week_end)
    ].copy()
    
    if weekly_news.empty:
        return ""
    if len(weekly_news) > max_articles:
        whitelisted_news = weekly_news[weekly_news['source'].isin(whitelist)]
        if len(whitelisted_news) >= max_articles:
            selected_news = whitelisted_news.nlargest(max_articles, 'date')
        else:
            other_news = weekly_news[~weekly_news['source'].isin(whitelist)]
            needed = max_articles - len(whitelisted_news)
            selected_news = pd.concat([whitelisted_news, other_news.nlargest(needed, 'date')])
    else:
        selected_news = weekly_news

    context_parts = []
    for _, row in selected_news.iterrows():
        content = (str(row['headline']) + " " + str(row['content'] or "")).strip()
        truncated_content = content[:max_length]
        context_parts.append(f"Source: {row['source']} | Content: {truncated_content}")
    return "\n---\n".join(context_parts)


def generate_batch_news_analysis(
    start_date: str,
    end_date: str,
    portfolio_df: pd.DataFrame,
    news_df: pd.DataFrame,
    bull_analyst: NewsAnalyst,
    bear_analyst: NewsAnalyst,
    output_filepath: Path,
    rl_features_path: Path,
    news_whitelist: list
):
    """
    Orchestrates the generation of bull/bear LLM analysis for all tickers
    over a date range, with robust checkpointing and two final output files.
    """
    # --- Checkpointing based on the full analysis file ---
    completed_tasks = set()
    if output_filepath.exists() > 0:
        results_df = pd.read_parquet(output_filepath)
        completed_tasks = set(zip(results_df['ticker'], results_df['date']))
        logging.info(f"Loaded {len(completed_tasks)} already completed tasks.")
    else:
        results_df = pd.DataFrame()
    # --- Main Loop ---
    weekly_date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    tickers = portfolio_df['ticker'].unique().tolist()
    tasks_to_run = [
        (ticker, week_date.strftime('%Y-%-m-%d'))
        for week_date in weekly_date_range
        for ticker in tickers
        if (ticker, week_date.strftime('%Y-%-m-%d')) not in completed_tasks
    ]
    new_results = []
    for ticker, week_str in tqdm(tasks_to_run, desc="Generating Bull/Bear analysis"):
        company_profile = portfolio_df[portfolio_df['ticker'] == ticker].iloc[0]
        news_context = get_weekly_news_context(ticker, week_str, news_df, news_whitelist)
        bull_result = bull_analyst.analyze(
            ticker=ticker, description=company_profile['description'], news_text=news_context,
            start_date=(pd.to_datetime(week_str) - pd.DateOffset(days=6)).strftime('%Y-%m-%d'), end_date=week_str
        )
        bear_result = bear_analyst.analyze(
            ticker=ticker, description=company_profile['description'], news_text=news_context,
            start_date=(pd.to_datetime(week_str) - pd.DateOffset(days=6)).strftime('%Y-%m-%d'), end_date=week_str
        )
        flat_result = {
            'date': week_str,
            'ticker': ticker,
            'bull_directional_impact': bull_result.directional_impact,
            'bull_significance': bull_result.significance,
            'bull_justification': bull_result.justification,
            'bull_mechanism_channel': bull_result.mechanism.channel,
            'bull_mechanism_magnitude': bull_result.mechanism.magnitude,
            'bull_mechanism_confidence': bull_result.mechanism.confidence,
            'bear_directional_impact': bear_result.directional_impact,
            'bear_significance': bear_result.significance,
            'bear_justification': bear_result.justification,
            'bear_mechanism_channel': bull_result.mechanism.channel,
            'bear_mechanism_magnitude': bull_result.mechanism.magnitude,
            'bear_mechanism_confidence': bull_result.mechanism.confidence,
        }
        new_results.append(flat_result)
        if len(new_results) > 0 and len(new_results) % 20 == 0:
            temp_df = pd.DataFrame(new_results)
            combined_df = pd.concat([results_df, temp_df], ignore_index=True)
            combined_df.to_parquet(output_filepath)
            logging.info(f"Saved intermediate progress with {len(combined_df)} total results.")
    
    if new_results:
        temp_df = pd.DataFrame(new_results)
        final_df = pd.concat([results_df, temp_df], ignore_index=True)
        final_df.to_parquet(output_filepath)
        logging.info(f"✅ Final LLM news features saved to {output_filepath}")
        rl_feature_columns = [
            'date', 'ticker', 
            'bull_directional_impact', 'bull_significance',
            'bear_directional_impact', 'bear_significance'
            ]
        final_rl_df = final_df[rl_feature_columns]
        final_rl_df.to_parquet(rl_features_path)
        logging.info(f"✅ RL news feature file saved to {rl_features_path}")
    else:
        logging.info("No new analysis generated. Files are up to date.")

# src/indexer/source_scoring.py
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


def inspect_news_parquet(
    parquet_path: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    """
    Load a news parquet and report:
      - unique tickers (sorted),
      - smallest (min) date,
      - biggest (max) date.

    Assumes each row is a news item with at least [date_col, ticker_col].
    """
    df = pd.read_parquet(parquet_path, columns=[date_col, ticker_col])

    # Clean & normalize
    df = df.dropna(subset=[date_col, ticker_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=[date_col])
    df[ticker_col] = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    tickers = sorted(df[ticker_col].unique().tolist())
    tmin = df[date_col].min()
    tmax = df[date_col].max()
    return tickers, tmin, tmax


def weekly_news_counts(
    parquet_path: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Return a wide DataFrame of weekly news counts:
      - Index: each Friday (week ending Friday → week runs Sat..Fri)
      - Columns: tickers (sorted)
      - Values: number of news items within that Sat..Fri window

    Notes:
    - Uses 'W-FRI' resampling, which defines weekly bins ending on Friday,
      i.e., each bin covers Saturday→Friday inclusive.
    - Produces explicit 0s for weeks/tickers with no news.
    """
    df = pd.read_parquet(parquet_path, columns=[date_col, ticker_col])
    df = df.dropna(subset=[date_col, ticker_col]).copy()

    # Normalize fields
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=[date_col])
    df[ticker_col] = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
    ).astype("category")

    # Group by ticker, resample to weeks ending on Friday
    df = df.set_index(date_col)
    grouped = (
        df
        .groupby(ticker_col)
        .resample("W-FRI")      # Sat..Fri bins labeled by the Friday
        .size()
        .unstack(0)             # columns = tickers
    )

    # Ensure full Friday index and full ticker columns, fill with 0
    full_fridays = pd.date_range(
        start=df.index.min().floor("D"),
        end=df.index.max().ceil("D"),
        freq="W-FRI",
    )
    all_tickers = sorted(df[ticker_col].cat.categories.tolist())

    out = (
        grouped
        .reindex(full_fridays)
        .reindex(columns=all_tickers)
        .fillna(0)
        .astype("int64")
    )
    out.index.name = "week_ending"   # each row is the Friday for that Sat..Fri week
    out.columns.name = "tickers"
    return out
