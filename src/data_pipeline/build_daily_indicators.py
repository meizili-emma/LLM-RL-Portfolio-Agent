#!/usr/bin/env python
"""
daily_processing.py

Pipeline:

1. Load raw daily OHLCV data.
2. Prepare clean env dataframe via prepare_env_dataframe:
   - filter to canonical tickers
   - deduplicate by (ticker, date, etc.)
   - handle NaNs in OHLCV appropriately
3. Compute daily technical indicators per ticker using pandas-ta-classic.
4. Write out:
   - cleaned env parquet
   - missing report parquet (optional)
   - env + technical indicators parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# pandas-ta-classic: indicator functions live under pandas_ta.classic
import pandas_ta_classic as ta  

from src.rl_agents.data import prepare_env_dataframe


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_indicators_for_ticker(
    g: pd.DataFrame,
    ind_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute technical indicators for a single ticker's daily data.

    Assumes:
        - g is already cleaned: one row per (ticker, date), no NaN OHLC prices,
          sorted by 'date'.
        - Columns: ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', ...]
    """
    g = g.sort_values("date").reset_index(drop=True).copy()

    # Ensure numeric dtype
    for col in ["open", "high", "low", "close", "volume"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")

    # --- Basic returns ---
    g["log_ret"] = np.log(g["close"]).diff()

    # === Trend indicators ===
    ema_len = ind_cfg["ema"]["length"]
    g[f"ti_ema_{ema_len}"] = ta.ema(g["close"], length=ema_len)

    macd_cfg = ind_cfg["macd"]
    macd_df = ta.macd(
        g["close"],
        fast=macd_cfg["fast"],
        slow=macd_cfg["slow"],
        signal=macd_cfg["signal"],
    )
    # MACD columns (typically): MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    g["ti_macd"] = macd_df.iloc[:, 0]
    g["ti_macd_signal"] = macd_df.iloc[:, 1]
    g["ti_macd_hist"] = macd_df.iloc[:, 2]

    # === Momentum indicators ===
    rsi_len = ind_cfg["rsi"]["length"]
    g[f"ti_rsi_{rsi_len}"] = ta.rsi(g["close"], length=rsi_len)

    stoch_cfg = ind_cfg["stoch"]
    stoch_df = ta.stoch(
        high=g["high"],
        low=g["low"],
        close=g["close"],
        k=stoch_cfg["k"],
        d=stoch_cfg["d"],
        smooth_k=stoch_cfg["smooth_k"],
    )
    g["ti_stoch_k"] = stoch_df.iloc[:, 0]
    g["ti_stoch_d"] = stoch_df.iloc[:, 1]

    willr_len = ind_cfg["willr"]["length"]
    g["ti_willr"] = ta.willr(
        high=g["high"],
        low=g["low"],
        close=g["close"],
        length=willr_len,
    )

    # === Volatility indicators ===
    atr_len = ind_cfg["atr"]["length"]
    g[f"ti_atr_{atr_len}"] = ta.atr(
        high=g["high"],
        low=g["low"],
        close=g["close"],
        length=atr_len,
    )

    bb_cfg = ind_cfg["bbands"]
    bb_df = ta.bbands(
        g["close"],
        length=bb_cfg["length"],
        std=bb_cfg["std"],
    )
    g["ti_bb_low"] = bb_df.iloc[:, 0]
    g["ti_bb_mid"] = bb_df.iloc[:, 1]
    g["ti_bb_high"] = bb_df.iloc[:, 2]
    # band width normalized by mid
    g["ti_bb_width"] = (g["ti_bb_high"] - g["ti_bb_low"]) / g["ti_bb_mid"]

    rv_len = ind_cfg["realized_vol"]["length"]
    g[f"ti_realized_vol_{rv_len}"] = (
        g["log_ret"].rolling(rv_len, min_periods=1).std()
    )

    # ATR / price ratio as normalized vol
    g["ti_atr_price_ratio"] = g[f"ti_atr_{atr_len}"] / g["close"]

    # === Volume / liquidity ===
    g["ti_obv"] = ta.obv(g["close"], g["volume"])

    vol_z_len = ind_cfg["volume_zscore"]["length"]
    vol_ma = g["volume"].rolling(vol_z_len, min_periods=1).mean()
    vol_std = g["volume"].rolling(vol_z_len, min_periods=1).std()
    g["ti_volume_ma"] = vol_ma
    g["ti_volume_zscore"] = (g["volume"] - vol_ma) / (vol_std.replace(0, np.nan))
    g["ti_volume_ma_ratio"] = g["volume"] / vol_ma.replace(0, np.nan)

    # === Mean reversion / stretch ===
    price_z_len = ind_cfg["price_zscore"]["length"]
    price_ma = g["close"].rolling(price_z_len, min_periods=1).mean()
    price_std = g["close"].rolling(price_z_len, min_periods=1).std()
    g["ti_price_ma"] = price_ma
    g["ti_price_zscore"] = (g["close"] - price_ma) / price_std.replace(0, np.nan)

    # Bollinger band position: 0 = at lower band, 1 = at upper band
    bb_range = (g["ti_bb_high"] - g["ti_bb_low"]).replace(0, np.nan)
    g["ti_bb_pos"] = (g["close"] - g["ti_bb_low"]) / bb_range

    # Distance from EMA (as % of price)
    g["ti_dist_from_ema"] = (g["close"] - g[f"ti_ema_{ema_len}"]) / g["close"]

    # === Market structure / regime ===
    up_len = ind_cfg["up_down_ratio"]["length"]
    up_flag = (g["close"] > g["close"].shift(1)).astype(float)
    g[f"ti_up_day_ratio_{up_len}"] = up_flag.rolling(up_len, min_periods=1).mean()

    kurt_len = ind_cfg["kurtosis"]["length"]
    g[f"ti_kurtosis_{kurt_len}"] = (
        g["log_ret"].rolling(kurt_len, min_periods=1).kurt()
    )

    return g


def add_turbulence_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily turbulence (Mahalanobis distance of returns) following
    Kritzman et al. (2010), using a 252-day rolling window.

    Input:
        df_daily: dataframe with columns ['date', 'ticker', 'close'] at DAILY frequency.
                  May contain duplicate (date, ticker) rows; we aggregate to one close
                  per (date, ticker) before pivoting.

    Output:
        turbulence_df: dataframe with columns ['date', 'turbulence'] (daily frequency)
    """
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ---- Ensure one close per (date, ticker) ----
    # If duplicates exist, we keep the last close for that (date, ticker).
    df = (
        df.groupby(["date", "ticker"], as_index=False)["close"]
          .last()
          .sort_values(["date", "ticker"])
    )

    # ---- Pivot to: index = daily date, columns = tickers ----
    price_mat = df.pivot(index="date", columns="ticker", values="close")
    price_mat = price_mat.sort_index()

    # ---- Daily returns ----
    ret_mat = price_mat.pct_change()

    dates = price_mat.index.to_list()
    n_dates = len(dates)

    # ---- We use 252 daily observations for rolling turbulence ----
    window = 252
    turbulence_vals = [0.0] * n_dates   # turbulence init: first 252 days = 0

    # Rolling turbulence calculation
    for i in range(window, n_dates):
        current_date = dates[i]

        # Current daily return vector (shape 1 × N)
        current_ret_vec = ret_mat.loc[current_date]

        # Historical returns (252 days)
        hist_window = ret_mat.iloc[i - window:i]

        # ---- Drop columns with too many NaNs ----
        valid_cols = hist_window.columns[hist_window.isna().sum() < window]
        hist_clean = hist_window[valid_cols].dropna(axis=0, how="any")

        if hist_clean.shape[0] < 20:
            turbulence_vals[i] = 0.0
            continue

        current_vec = current_ret_vec[valid_cols]
        hist_mean = hist_clean.mean(axis=0)
        centered = (current_vec - hist_mean).to_numpy()

        cov = hist_clean.cov().values
        try:
            inv_cov = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            turbulence_vals[i] = 0.0
            continue

        try:
            md2 = float(centered.T @ inv_cov @ centered)
        except Exception:
            md2 = 0.0

        if md2 > 0:
            turbulence_vals[i] = md2
        else:
            turbulence_vals[i] = 0.0

    turbulence_df = pd.DataFrame({
        "date": dates,
        "turbulence": turbulence_vals,
    })

    return turbulence_df


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    ind_cfg = cfg["indicators"]

    raw_daily_path = Path(data_cfg["raw_daily_path"])
    daily_env_path = Path(data_cfg["daily_env_path"])
    missing_report_path = Path(data_cfg["missing_report_path"])
    daily_indicators_path = Path(data_cfg["daily_indicators_path"])

    canonical_tickers = data_cfg["canonical_tickers"]

    print(f"[daily_processing] Reading raw daily OHLCV from: {raw_daily_path}")
    raw_df = pd.read_parquet(raw_daily_path)

    # Ensure date is datetime-like
    if "date" not in raw_df.columns:
        raise ValueError("Expected a 'date' column in raw daily data.")
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    # --- Step 1: env prep (dedup + NaN handling + ticker filtering) ---
    print("[daily_processing] Preparing env dataframe via prepare_env_dataframe(...)")
    df_env, missing_report = prepare_env_dataframe(
        raw_df=raw_df,
        canonical_tickers=canonical_tickers,
    )

    # Persist env dataframe and missing report (optional but useful)
    daily_env_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[daily_processing] Writing cleaned env dataframe to: {daily_env_path}")
    df_env.to_parquet(daily_env_path, index=False)

    if missing_report is not None:
        print(f"[daily_processing] Writing missing report to: {missing_report_path}")
        if isinstance(missing_report, pd.DataFrame):
            missing_report.to_parquet(missing_report_path, index=False)
        else:
            pd.DataFrame(missing_report).to_parquet(missing_report_path, index=False)

    # --- Step 2: compute daily technical indicators per ticker ---
    print("[daily_processing] Computing technical indicators per ticker...")

    required_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df_env.columns]
    if missing_cols:
        raise ValueError(f"Env dataframe is missing required columns: {missing_cols}")

    df_env = df_env.sort_values(["ticker", "date"])

    df_with_ta = (
        df_env.groupby("ticker", group_keys=False)
        .apply(add_indicators_for_ticker, ind_cfg=ind_cfg)
        .reset_index(drop=True)
    )

    # --- Step 3: compute daily turbulence from closes ---
    turb_daily = add_turbulence_daily(df_env[["date", "ticker", "close"]])
    # Merge turbulence into the indicator dataframe
    df_with_ta = df_with_ta.merge(turb_daily, on="date", how="left")
    df_with_ta["turbulence"] = df_with_ta["turbulence"].ffill()

    print(f"[daily_processing] Writing env + indicators to: {daily_indicators_path}")
    daily_indicators_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_ta.to_parquet(daily_indicators_path, index=False)

    print("[daily_processing] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare env dataframe and compute daily technical indicators."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/raw/market/indicator_config.yaml"),
        help="Path to config YAML.",
    )
    args = parser.parse_args()
    main(args.config)
