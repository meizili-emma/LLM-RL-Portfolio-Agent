#!/usr/bin/env python
"""
build_weekly_indicators.py

Aggregate daily OHLCV + technical indicators into weekly features aligned with
the trading calendar in data/_CALENDAR.parquet.

For each calendar row (week_decision_date, curr_close_utc, prev_close_utc) and
each ticker, we:

- collect daily rows with prev_close_utc < close_ts_utc <= curr_close_utc,
- build a weekly OHLCV bar (open, high, low, close, volume),
- take the last day's technical indicators as the weekly technical snapshot
  (rename columns: drop 'ti_' prefix, no '_week' suffix),
- compute weekly log return relative to the last close before prev_close_utc.

Output: one row per (week_decision_date, ticker), with:

- calendar columns: week_decision_date, curr_close_utc, prev_close_utc
- ticker
- weekly OHLCV: open, high, low, close, volume, n_days_traded
- weekly log return: log_ret_1w
- weekly technical snapshot: rsi_14, macd, macd_signal, macd_hist, ... etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Mapping from daily technical columns to weekly feature names
TA_SNAPSHOT_MAP: Dict[str, str] = {
    "ti_ema_20": "ema_20",
    "ti_macd": "macd",
    "ti_macd_signal": "macd_signal",
    "ti_macd_hist": "macd_hist",
    "ti_rsi_14": "rsi_14",
    "ti_stoch_k": "stoch_k",
    "ti_stoch_d": "stoch_d",
    "ti_willr": "willr",
    "ti_atr_14": "atr_14",
    "ti_bb_low": "bb_low",
    "ti_bb_mid": "bb_mid",
    "ti_bb_high": "bb_high",
    "ti_bb_width": "bb_width",
    "ti_realized_vol_20": "realized_vol_20",
    "ti_atr_price_ratio": "atr_price_ratio",
    "ti_obv": "obv",
    "ti_volume_ma": "volume_ma",
    "ti_volume_zscore": "volume_zscore",
    "ti_volume_ma_ratio": "volume_ma_ratio",
    "ti_price_ma": "price_ma",
    "ti_price_zscore": "price_zscore",
    "ti_bb_pos": "bb_pos",
    "ti_dist_from_ema": "dist_from_ema",
    "ti_up_day_ratio_5": "up_day_ratio_5",
    "ti_kurtosis_20": "kurtosis_20",
}


def build_weekly_for_ticker(
    g: pd.DataFrame,
    calendar: pd.DataFrame,
    min_history_days: int,
) -> List[Dict[str, Any]]:
    """
    Build weekly features for a single ticker.

    Parameters
    ----------
    g : DataFrame
        Daily data + indicators for one ticker. Must have:
        ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
         'close_ts_utc', 'log_ret', ... ti_* columns ...].
    calendar : DataFrame
        Calendar with columns: week_decision_date, curr_close_utc, prev_close_utc.
        Already converted to datetime and sorted by week_decision_date.
    min_history_days : int
        Minimum number of daily observations with close_ts_utc <= curr_close_utc
        required before we emit any weekly feature.

    Returns
    -------
    List[dict]
        One dict per (week_decision_date, ticker).
    """
    g = g.sort_values("close_ts_utc").reset_index(drop=True).copy()

    results: List[Dict[str, Any]] = []

    for _, cal_row in calendar.iterrows():
        week_date = cal_row["week_decision_date"]
        curr_close = cal_row["curr_close_utc"]
        prev_close = cal_row["prev_close_utc"]

        # Check total history length up to curr_close_utc
        hist_until_curr = g[g["close_ts_utc"] <= curr_close]
        if len(hist_until_curr) < min_history_days:
            # Not enough history for stable indicators / weekly features
            continue

        # Select rows in the weekly window (prev_close_utc, curr_close_utc]
        week_rows = g[
            (g["close_ts_utc"] > prev_close) & (g["close_ts_utc"] <= curr_close)
        ]

        if week_rows.empty:
            # This ticker didn't trade in this weekly window (or has no data in it)
            continue

        week_rows = week_rows.sort_values("close_ts_utc")
        first = week_rows.iloc[0]
        last = week_rows.iloc[-1]

        # --- Weekly OHLCV ---
        open_week = first["open"]
        high_week = week_rows["high"].max()
        low_week = week_rows["low"].min()
        close_week = last["close"]
        volume_week = week_rows["volume"].sum()
        n_days_traded = len(week_rows)
        turbulence_week = last["turbulence"]

        # --- Weekly log return (close-to-close) ---
        prev_rows = g[g["close_ts_utc"] <= prev_close]
        if not prev_rows.empty:
            prev_close_price = prev_rows.iloc[-1]["close"]
            if prev_close_price > 0 and close_week > 0:
                log_ret_1w = np.log(close_week / prev_close_price)
            else:
                log_ret_1w = np.nan
        else:
            log_ret_1w = np.nan

        out: Dict[str, Any] = {
            "ticker": last["ticker"],
            "date": week_date,
            "week_decision_date": week_date,
            "curr_close_utc": curr_close,
            "prev_close_utc": prev_close,
            "open": open_week,
            "high": high_week,
            "low": low_week,
            "close": close_week,
            "volume": volume_week,
            "turbulence": turbulence_week,
            "n_days_traded": n_days_traded,
            "log_ret_1w": log_ret_1w,
        }

        # --- Weekly technical snapshot (last day in week) ---
        for src_col, dst_col in TA_SNAPSHOT_MAP.items():
            if src_col in last.index:
                out[dst_col] = last[src_col]
            else:
                out[dst_col] = np.nan

        results.append(out)

    return results


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    agg_cfg = cfg.get("aggregation", {})

    daily_path = Path(data_cfg["daily_indicators_path"])
    calendar_path = Path(data_cfg["weekly_calendar_path"])
    weekly_out_path = Path(data_cfg["weekly_indicators_path"])

    min_history_days = int(agg_cfg.get("min_history_days", 60))

    print(f"[build_weekly_indicators] Reading daily indicators from: {daily_path}")
    df_daily = pd.read_parquet(daily_path)

    required_daily_cols = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_ts_utc",
    ]

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily["close_ts_utc"] = pd.to_datetime(df_daily["close_ts_utc"])

    print(f"[build_weekly_indicators] Reading calendar from: {calendar_path}")
    calendar = pd.read_parquet(calendar_path)

    calendar["week_decision_date"] = pd.to_datetime(calendar["week_decision_date"])
    calendar["curr_close_utc"] = pd.to_datetime(calendar["curr_close_utc"])
    calendar["prev_close_utc"] = pd.to_datetime(calendar["prev_close_utc"])
    calendar = calendar.sort_values("week_decision_date").reset_index(drop=True)

    df_daily = df_daily.sort_values(["ticker", "close_ts_utc"])

    tickers = df_daily["ticker"].unique()
    print(f"[build_weekly_indicators] Processing {len(tickers)} tickers...")

    all_rows: List[Dict[str, Any]] = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"  - [{i}/{len(tickers)}] {ticker}")
        g = df_daily[df_daily["ticker"] == ticker]
        rows = build_weekly_for_ticker(
            g=g,
            calendar=calendar,
            min_history_days=min_history_days,
        )
        all_rows.extend(rows)

    if not all_rows:
        print(
            "[build_weekly_indicators] No weekly features produced "
            "(check data, calendar alignment, and min_history_days)."
        )
        weekly_df = pd.DataFrame()
    else:
        weekly_df = pd.DataFrame(all_rows)
        weekly_df = weekly_df.sort_values(
            ["week_decision_date", "ticker"]
        ).reset_index(drop=True)

    weekly_out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build_weekly_indicators] Writing weekly indicators to: {weekly_out_path}")
    weekly_df.to_parquet(weekly_out_path, index=False)
    print("[build_weekly_indicators] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate daily indicators into weekly features using calendar windows."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/raw/market/indicator_config.yaml"),
        help="Path to indicator config YAML.",
    )
    args = parser.parse_args()
    main(args.config)
