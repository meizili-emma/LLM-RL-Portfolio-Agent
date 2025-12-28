from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.llm_agents.utils import _iso, _structured_call
from src.llm_agents.technical_schema import TechnicalWeeklyAnalysis, tech_user_prompt, TECH_SYSTEM_PROMPT


DEFAULT_FIELDS_FOR_LLM: List[str] = [
    # Meta/time
    "date",
    "week_decision_date",
    "prev_close_utc",
    "curr_close_utc",
    # Price/return
    "close",
    "log_ret_1w",
    "dist_from_ema",
    "price_zscore",
    "bb_pos",
    # Trend/momentum
    "macd",
    "macd_hist",
    "rsi_14",
    "stoch_k",
    "willr",
    # Vol/risk/regime
    "realized_vol_20",
    "atr_price_ratio",
    "bb_width",
    "turbulence",
    "kurtosis_20",
    # Volume/participation
    "volume",
    "volume_zscore",
    "volume_ma_ratio",
]


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_required_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}. Found: {list(df.columns)}")


def _row_to_compact_dict(r: pd.Series, cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        v = r.get(c, None)
        if c in ("date", "week_decision_date", "curr_close_utc", "prev_close_utc"):
            try:
                out[c] = _iso(v)
            except Exception:
                out[c] = None
        else:
            if pd.isna(v):
                out[c] = None
            else:
                # convert numpy scalars to python
                if isinstance(v, (np.floating, np.integer)):
                    out[c] = float(v)
                else:
                    try:
                        out[c] = float(v) if isinstance(v, (int, float)) else v
                    except Exception:
                        out[c] = v
    return out


def _safe_float(x) -> float | None:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _compute_window_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rows are compact dicts oldest->newest. Compute deltas and summary stats to help the LLM.
    """
    if not rows:
        return {"valid_rows": 0}

    def _series(field: str) -> List[float]:
        vals: List[float] = []
        for rr in rows:
            v = _safe_float(rr.get(field))
            if v is not None:
                vals.append(v)
        return vals

    stats: Dict[str, Any] = {"valid_rows": int(len(rows))}

    # Deltas (end - start) using first/last non-null if possible
    def _delta(field: str) -> float | None:
        vals = [(_safe_float(rr.get(field))) for rr in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None
        return float(vals[-1] - vals[0])

    # Volatility clustering summaries
    for f in ["turbulence", "realized_vol_20", "atr_price_ratio", "bb_width", "kurtosis_20"]:
        vals = _series(f)
        stats[f"{f}_mean"] = float(np.mean(vals)) if vals else None
        stats[f"{f}_max"] = float(np.max(vals)) if vals else None
        stats[f"{f}_delta"] = _delta(f)

    # Directionality summaries
    for f in ["dist_from_ema", "macd_hist", "rsi_14", "price_zscore", "bb_pos", "log_ret_1w", "volume_zscore"]:
        stats[f"{f}_delta"] = _delta(f)

    # Up-week ratio from log_ret_1w
    rets = _series("log_ret_1w")
    if rets:
        stats["up_week_ratio"] = float(np.mean([1.0 if r > 0 else 0.0 for r in rets]))
    else:
        stats["up_week_ratio"] = None

    return stats


def _build_history_rows(
    df_ticker: pd.DataFrame,
    week_decision_date: pd.Timestamp,
    lookback_weeks: int,
    cols_for_llm: List[str],
) -> List[Dict[str, Any]]:
    d = pd.to_datetime(week_decision_date, utc=True, errors="coerce")
    if pd.isna(d):
        return []

    sub = df_ticker[df_ticker["week_decision_date"] <= d].copy()
    if sub.empty:
        return []

    sub = sub.sort_values("week_decision_date").tail(int(lookback_weeks))
    rows = [_row_to_compact_dict(r, cols_for_llm) for _, r in sub.iterrows()]
    return rows


def _analyze_single_ticker_week(
    *,
    model_cfg: Dict[str, Any],
    tech_cfg: Dict[str, Any],
    ticker: str,
    week_decision_date: pd.Timestamp,
    df_ticker: pd.DataFrame,
    cols_for_llm: List[str],
) -> TechnicalWeeklyAnalysis:
    lookback_weeks = int(tech_cfg.get("lookback_weeks", 6))
    retries = int(model_cfg.get("max_retries", 2))

    hist_rows = _build_history_rows(
        df_ticker=df_ticker,
        week_decision_date=week_decision_date,
        lookback_weeks=lookback_weeks,
        cols_for_llm=cols_for_llm,
    )

    current_week = hist_rows[-1] if hist_rows else {}

    window_stats = _compute_window_stats(hist_rows)

    user_prompt = tech_user_prompt(
        ticker=str(ticker),
        as_of=_iso(week_decision_date),
        lookback_weeks=lookback_weeks,
        current_week_json=json.dumps(current_week, ensure_ascii=False),
        recent_history_json=json.dumps(hist_rows, ensure_ascii=False),
        window_stats_json=json.dumps(window_stats, ensure_ascii=False),
    )

    out = _structured_call(
        model_cfg=model_cfg,
        schema=TechnicalWeeklyAnalysis,
        system_prompt=TECH_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        retries=retries,
    )
    return out


def build_weekly_technical_analysis(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Outputs weekly panel parquet with:
      ticker, week_decision_date, prev_close_utc, curr_close_utc,
      tech_json, tech_signal, tech_risk_score, tech_confidence, tech_rationale,
      discarded, discard_reason
    """
    raw_path = Path(cfg["technical"]["raw_path"])
    out_path = Path(cfg["technical"]["out_path"])

    raw = pd.read_parquet(raw_path)

    required_cols = ["ticker", "week_decision_date", "prev_close_utc", "curr_close_utc", "date"]
    _validate_required_cols(raw, required_cols, name=str(raw_path))

    raw = raw.copy()
    raw["ticker"] = raw["ticker"].astype(str)
    raw["week_decision_date"] = pd.to_datetime(raw["week_decision_date"], utc=True, errors="coerce")
    raw["prev_close_utc"] = pd.to_datetime(raw["prev_close_utc"], utc=True, errors="coerce")
    raw["curr_close_utc"] = pd.to_datetime(raw["curr_close_utc"], utc=True, errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")

    canonical = cfg.get("portfolio", {}).get("canonical_ticker_list", None)
    if canonical:
        canonical = [str(t).strip() for t in canonical if str(t).strip()]
        raw = raw[raw["ticker"].isin(canonical)].copy()

    raw.sort_values(["ticker", "week_decision_date"], inplace=True)

    # fields whitelist for LLM
    cols_for_llm = cfg.get("technical", {}).get("fields_for_llm", None)
    if not cols_for_llm:
        cols_for_llm = DEFAULT_FIELDS_FOR_LLM
    cols_for_llm = [str(c) for c in cols_for_llm]

    # ensure the chosen fields exist (warn instead of hard fail, but keep required)
    missing_llm_cols = [c for c in cols_for_llm if c not in raw.columns]
    if missing_llm_cols:
        raise KeyError(
            f"technical.fields_for_llm contains missing columns: {missing_llm_cols}. "
            f"Available columns include: {list(raw.columns)}"
        )

    # resume support
    if out_path.exists():
        out_df = pd.read_parquet(out_path)
    else:
        out_df = pd.DataFrame(
            columns=[
                "ticker",
                "week_decision_date",
                "prev_close_utc",
                "curr_close_utc",
                "tech_json",
                "tech_signal",
                "tech_risk_score",
                "tech_confidence",
                "tech_rationale",
                "discarded",
                "discard_reason",
            ]
        )

    if out_df.empty:
        processed_keys: set[tuple[str, str]] = set()
    else:
        processed_keys = set(
            zip(
                out_df["ticker"].astype(str),
                pd.to_datetime(out_df["week_decision_date"], utc=True, errors="coerce")
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        )

    groups = {tk: g.copy() for tk, g in raw.groupby("ticker", sort=False)}

    todo: List[Tuple[str, pd.Timestamp]] = []
    for tk, g in groups.items():
        for wd in g["week_decision_date"].dropna().unique():
            key = (str(tk), _iso(wd))
            if key not in processed_keys:
                todo.append((str(tk), pd.to_datetime(wd, utc=True)))

    if not todo:
        print("Technical analyst: nothing to do (all rows already processed).")
        return out_df

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Technical analyst: {len(todo)} ticker-week rows to process.")
    print("Technical out_path:", out_path.resolve())

    for tk, wd in tqdm(todo, desc="Technical analyst"):
        df_ticker = groups.get(tk)
        if df_ticker is None or df_ticker.empty:
            continue

        curr_row = df_ticker[df_ticker["week_decision_date"] == wd].tail(1)
        if curr_row.empty:
            continue
        prev_close_utc = curr_row["prev_close_utc"].iloc[0]
        curr_close_utc = curr_row["curr_close_utc"].iloc[0]

        try:
            analysis = _analyze_single_ticker_week(
                model_cfg=cfg["model"],
                tech_cfg=cfg["technical"],
                ticker=tk,
                week_decision_date=wd,
                df_ticker=df_ticker,
                cols_for_llm=cols_for_llm,
            )
        except Exception as e:
            print(f"[TECH] Skipping (ticker={tk}, week={_iso(wd)}) due to error: {e}")
            continue

        tech_json = analysis.model_dump_json()

        out_row = {
            "ticker": tk,
            "week_decision_date": wd,
            "prev_close_utc": prev_close_utc,
            "curr_close_utc": curr_close_utc,
            "tech_json": tech_json,
            "tech_signal": float(analysis.rl.signal),
            "tech_risk_score": float(analysis.rl.risk_score),
            "tech_confidence": float(analysis.rl.confidence),
            "tech_rationale": str(analysis.rl.rationale),
            "discarded": bool(analysis.discarded),
            "discard_reason": analysis.discard_reason,
        }

        out_df = pd.concat([out_df, pd.DataFrame([out_row])], ignore_index=True)
        out_df.to_parquet(out_path, index=False)

    print(f"Technical analyst: wrote {len(out_df)} total rows to {out_path}")
    return out_df


def run_technical_analyst(cfg: Dict[str, Any]) -> None:
    build_weekly_technical_analysis(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="data/raw/config/technical_analyst.yaml",
        help="Path to compression config YAML.",
        type=str,
    )
    args = parser.parse_args()
    cfg = _load_yaml(args.config)
    run_technical_analyst(cfg)


if __name__ == "__main__":
    main()

