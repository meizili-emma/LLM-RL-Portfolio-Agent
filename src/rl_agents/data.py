from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


def prepare_env_dataframe(
    raw_df: pd.DataFrame,
    canonical_tickers: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare df_env for StockPortfolioEnv:

    - Filter to canonical_tickers.
    - Ensure 'date', 'ticker', 'close' exist.
    - Sort by ['date', 'ticker'].
    - Assign integer day_idx per date (0..T-1) and use as index,
      so for each day t, df_env.loc[t] has N rows (one per ticker).
    - Return (df_env, missing_report).

    missing_report: dataframe listing dates where tickers are missing/extra.
    """
    df = raw_df.copy()
    if "date" not in df.columns or "ticker" not in df.columns or "close" not in df.columns:
        raise KeyError("raw_df must contain at least ['date', 'ticker', 'close'] columns.")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].isin(canonical_tickers)].copy()
    df.sort_values(["date", "ticker"], inplace=True)

    # detect missing tickers per date
    group = df.groupby("date")["ticker"].agg(set).reset_index()
    expected = set(canonical_tickers)
    problems = []
    for _, row in group.iterrows():
        present = row["ticker"]
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            problems.append(
                {
                    "date": row["date"],
                    "missing_tickers": missing,
                    "extra_tickers": extra,
                }
            )
    missing_report = pd.DataFrame(problems)

    # keep only dates with full coverage (strict)
    if not missing_report.empty:
        bad_dates = set(missing_report["date"])
        df = df[~df["date"].isin(bad_dates)].copy()

    df.sort_values(["date", "ticker"], inplace=True)

    # assign day_idx 0..T-1 for each unique date
    df["day_idx"] = df.groupby("date").ngroup()
    df.set_index("day_idx", inplace=True)

    # final sort by (day_idx, ticker)
    df.sort_index(inplace=True)
    return df, missing_report


def plan_rl_train_test_split(
    df_env: pd.DataFrame,
    first_trade_date: str,
    last_trade_date: str,
    terminal_date: str,
    train_ratio: float,
    norm_window_size: int,
    trading_state_history_len: int,
) -> Dict[str, Any]:
    """
    Plan train/test windows in terms of df_env index (day_idx).

    We assume:
      - df_env index is 'day_idx' (0..T-1),
      - df_env has a 'date' column (datetime),
      - each day_idx corresponds to exactly len(canonical_tickers) rows.

    We:
      - identify action days [first_trade_date, last_trade_date],
      - split action days into train/test by train_ratio,
      - extend each slice backwards by 'warmup_len' days,
      - extend slice end by +1 day to provide a terminal price for last action.
    """
    dates = pd.to_datetime(df_env["date"].drop_duplicates().sort_values().reset_index(drop=True))
    all_days = len(dates)

    ftd = pd.to_datetime(first_trade_date)
    ltd = pd.to_datetime(last_trade_date)
    ttd = pd.to_datetime(terminal_date)

    # Map dates to day indices
    try:
        first_trade_idx = int(dates.index[dates >= ftd][0])
    except IndexError:
        raise ValueError(f"first_trade_date={first_trade_date} not found in df_env date range.")
    try:
        last_trade_idx = int(dates.index[dates <= ltd][-1])
    except IndexError:
        raise ValueError(f"last_trade_date={last_trade_date} not found in df_env date range.")
    # terminal index: last date <= terminal_date
    try:
        terminal_idx = int(dates.index[dates <= ttd][-1])
    except IndexError:
        terminal_idx = last_trade_idx  # fallback

    if last_trade_idx <= first_trade_idx:
        raise ValueError("last_trade_date must be strictly after first_trade_date.")

    # action-day range
    action_indices = np.arange(first_trade_idx, last_trade_idx + 1, dtype=int)
    num_action_days = len(action_indices)
    if num_action_days < 2:
        raise ValueError("Not enough action days between first_trade_date and last_trade_date.")

    # train/test split on action days
    train_action_days = max(1, int(round(num_action_days * float(train_ratio))))
    if train_action_days >= num_action_days:
        train_action_days = num_action_days - 1

    train_last_action_idx = int(first_trade_idx + train_action_days - 1)
    test_first_action_idx = int(train_last_action_idx + 1)
    test_last_action_idx = last_trade_idx

    warmup_len = int(norm_window_size + trading_state_history_len - 2)
    #  This must match StockPortfolioEnv.reset()'s warmup_len formula.
    # If you change it in the env, update this function as well.

    # train slice
    train_episode_start_idx = max(0, first_trade_idx - warmup_len)
    # include one extra terminal day for final log return
    train_episode_end_idx = min(train_last_action_idx + 1, terminal_idx)

    # test slice
    test_episode_start_idx = max(0, test_first_action_idx - warmup_len)
    test_episode_end_idx = min(test_last_action_idx + 1, terminal_idx)

    split_plan: Dict[str, Any] = {
        "meta": {
            "first_trade_idx": int(first_trade_idx),
            "last_trade_idx": int(last_trade_idx),
            "terminal_idx": int(terminal_idx),
            "warmup_len": int(warmup_len),
            "num_action_days": int(num_action_days),
            "train_ratio": float(train_ratio),
        },
        "train": {
            "episode_start_idx": int(train_episode_start_idx),
            "episode_end_idx": int(train_episode_end_idx),
            "first_trade_idx": int(first_trade_idx),
            "last_trade_idx": int(train_last_action_idx),
        },
        "test": {
            "episode_start_idx": int(test_episode_start_idx),
            "episode_end_idx": int(test_episode_end_idx),
            "first_trade_idx": int(test_first_action_idx),
            "last_trade_idx": int(test_last_action_idx),
        },
        "all_dates": {
            "dates": dates.tolist(),
            "all_days": int(all_days),
        },
    }
    return split_plan


def plan_rl_train_test_split_with_forward_rolling_val(
    df_env: pd.DataFrame,
    first_trade_date: str,
    last_trade_date: str,
    terminal_date: str,
    train_ratio: float,
    norm_window_size: int,
    trading_state_history_len: int,
    *,
    n_val_folds: int = 3,
    val_action_days: int = 30,
    min_train_action_days: int = 150,
) -> Dict[str, Any]:
    """
    Extend plan_rl_train_test_split() with forward-rolling validation folds.

    - Outer train/test split is as in plan_rl_train_test_split (same dates/indices).
    - Inside the outer training action window, we build n_val_folds folds:
        For each fold:
          train actions: [outer_train_first_action .. train_last_action]
          val actions:   [val_start .. val_end]
      where val blocks roll forward near the end of the training window.

    Each train/val window is extended backward by warmup_len and forward by +1 day
    to obtain episode_start_idx/episode_end_idx, consistent with your existing logic.
    """
    # 1) Build the original outer split (unchanged logic)
    base = plan_rl_train_test_split(
        df_env=df_env,
        first_trade_date=first_trade_date,
        last_trade_date=last_trade_date,
        terminal_date=terminal_date,
        train_ratio=train_ratio,
        norm_window_size=norm_window_size,
        trading_state_history_len=trading_state_history_len,
    )

    warmup_len = int(base["meta"]["warmup_len"])
    terminal_idx = int(base["meta"]["terminal_idx"])

    outer_train_first_action = int(base["train"]["first_trade_idx"])
    outer_train_last_action = int(base["train"]["last_trade_idx"])

    outer_train_num_actions = outer_train_last_action - outer_train_first_action + 1
    if outer_train_num_actions < max(2, min_train_action_days):
        raise ValueError(
            "Training action window too small to construct validation folds. "
            f"outer_train_num_actions={outer_train_num_actions}, "
            f"min_train_action_days={min_train_action_days}. "
            "Increase train_ratio or relax val_min_train_action_days."
        )

    if val_action_days < 2:
        raise ValueError("val_action_days must be >= 2.")
    
    max_possible_folds = max(
        0,
        (outer_train_num_actions - min_train_action_days) // val_action_days,
    )
    if max_possible_folds <= 0:
        raise ValueError(
            "Cannot construct any validation folds with the given settings. "
            f"outer_train_num_actions={outer_train_num_actions}, "
            f"val_action_days={val_action_days}, "
            f"val_min_train_action_days={min_train_action_days}. "
            "Increase train_ratio or reduce val_action_days / val_min_train_action_days."
        )

    if n_val_folds < 1:
        base["val_folds"] = []
        base["meta"]["val_folds"] = {"enabled": False}
        return base

    n_val_folds_eff = int(min(n_val_folds, max_possible_folds))
    last_action = outer_train_last_action
    total_val_span = n_val_folds * val_action_days
    earliest_val_start = last_action - total_val_span + 1

    folds: List[Dict[str, Any]] = []
    for i in range(n_val_folds_eff):
        val_start = earliest_val_start + i * val_action_days
        val_end = val_start + val_action_days - 1

        if val_end > last_action:
            val_end = last_action
        if val_start > val_end:
            continue

        train_last_action = val_start - 1
        train_num_actions = train_last_action - outer_train_first_action + 1
        if train_num_actions < min_train_action_days:
            # too little training for this fold; skip
            continue

        # Train slice indices (with warmup and +1 terminal day)
        train_episode_start_idx = max(0, outer_train_first_action - warmup_len)
        train_episode_end_idx = min(train_last_action + 1, terminal_idx)

        # Val slice indices (with warmup and +1 terminal day)
        val_episode_start_idx = max(0, val_start - warmup_len)
        val_episode_end_idx = min(val_end + 1, terminal_idx)

        folds.append(
            {
                "fold_id": int(i),
                "train": {
                    "episode_start_idx": int(train_episode_start_idx),
                    "episode_end_idx": int(train_episode_end_idx),
                    "first_trade_idx": int(outer_train_first_action),
                    "last_trade_idx": int(train_last_action),
                },
                "val": {
                    "episode_start_idx": int(val_episode_start_idx),
                    "episode_end_idx": int(val_episode_end_idx),
                    "first_trade_idx": int(val_start),
                    "last_trade_idx": int(val_end),
                },
                "meta": {
                    "warmup_len": int(warmup_len),
                    "val_action_days": int(val_action_days),
                    "min_train_action_days": int(min_train_action_days),
                    "outer_train_first_action": int(outer_train_first_action),
                    "outer_train_last_action": int(outer_train_last_action),
                },
            }
        )

    if len(folds) == 0:
        raise ValueError(
            "Could not construct any validation folds. "
            f"outer_train_num_actions={outer_train_num_actions}, "
            f"n_val_folds={n_val_folds}, val_action_days={val_action_days}, "
            f"min_train_action_days={min_train_action_days}."
        )

    base["val_folds"] = folds
    base["meta"]["val_folds"] = {
        "enabled": True,
        "n_requested": int(n_val_folds),
        "n_possible": int(max_possible_folds),
        "n_built": int(len(folds)),
        "val_action_days": int(val_action_days),
        "min_train_action_days": int(min_train_action_days),
        "scope": "inside outer train action window",
    }
    return base
