from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _round4(x: Any) -> Any:
    """Round numeric scalars to 4dp; leave others unchanged."""
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            if np.isfinite(x):
                return float(np.round(float(x), 4))
            return float(x)
    except Exception:
        pass
    return x


def round_dict4(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively round floats inside a nested dict."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = round_dict4(v)
        elif isinstance(v, list):
            out[k] = [round_dict4(x) if isinstance(x, dict) else _round4(x) for x in v]
        else:
            out[k] = _round4(v)
    return out


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


# -----------------------------
# 1) Episode-level performance
# -----------------------------

def compute_episode_perf(df_eps: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """
    Add derived columns to the episode summary DataFrame.

    We only use fields if they exist; no assumptions about env.py schema
    beyond 'final_portfolio_value' if present.

    Adds:
      - nav_multiple = final_portfolio_value / initial_cash
      - episode_return_log = log(nav_multiple)
    """
    if df_eps.empty:
        return df_eps.copy()

    df = df_eps.copy()
    if "final_portfolio_value" in df.columns:
        df["nav_multiple"] = pd.to_numeric(df["final_portfolio_value"], errors="coerce") / float(initial_cash)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["episode_return_log"] = np.log(df["nav_multiple"])
    else:
        df["nav_multiple"] = np.nan
        df["episode_return_log"] = np.nan
    return df


def summarize_episode_perf(df_perf: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize a set of episodes (train or test).
    Output is JSON-friendly and float-rounded later.
    """
    if df_perf.empty:
        return {"empty": True}

    out: Dict[str, Any] = {"empty": False}
    for col in [
        "nav_multiple",
        "episode_return_log",
        "sharpe_ratio",
        "max_drawdown",
        "trade_count",
        "total_trade_cost",
        "avg_trade_cost_per_step",
    ]:
        s = _safe_series(df_perf, col)
        if not s.empty:
            out[col] = {
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
            }

    # Blowup heuristics (purely for sanity checks)
    nm = _safe_series(df_perf, "nav_multiple")
    if not nm.empty:
        out["flags"] = {
            "any_nav_leq_0_1": bool((nm <= 0.1).any()),
            "any_nav_geq_10": bool((nm >= 10.0).any()),
        }
    return out


# -----------------------------
# 2) Step-level: trading behavior & risk decomposition
# -----------------------------

def add_trade_shares_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive scalar trade-share summaries from vector 'trading.trade_shares' if present:
      - trade_shares_l1: sum_i |shares_i|
      - trade_shares_nnz: number of non-zero entries
    """
    if df.empty or "trading.trade_shares" not in df.columns:
        return df.copy()

    out = df.copy()
    l1 = []
    nnz = []
    for v in out["trading.trade_shares"]:
        if isinstance(v, (list, tuple, np.ndarray)):
            a = np.array(v, dtype=float)
            l1.append(float(np.abs(a).sum()))
            nnz.append(int(np.count_nonzero(a)))
        else:
            l1.append(np.nan)
            nnz.append(np.nan)
    out["trade_volume_abs"] = l1
    out["assets_traded_count"] = nnz
    return out


def add_turnover(df_steps: pd.DataFrame, weights_col: str) -> pd.DataFrame:
    """
    Turnover per step: 0.5 * sum_i |w_t,i - w_{t-1,i}|
    - For train: you likely want weights_col="trading.executed_weights"
    - For test:  weights_col="trading.executed_weights"

    Handles both single-stream and multi-env logs.
    """
    if df_steps.empty:
        return df_steps.copy()

    df = df_steps.copy()
    if weights_col not in df.columns:
        df["turnover"] = np.nan
        return df

    group_col = "episode_id" if "episode_id" in df.columns else (
        "meta.episode_id" if "meta.episode_id" in df.columns else (
            "env_id" if "env_id" in df.columns else ("meta.env_id" if "meta.env_id" in df.columns else None)
        )
    )

    turnover = np.full(len(df), np.nan, dtype=float)

    def _iter_rows(g: pd.DataFrame) -> None:
        w_prev = None
        for pos, (_, row) in enumerate(g.iterrows()):
            w = row.get(weights_col, None)
            if not isinstance(w, (list, tuple, np.ndarray)):
                w_prev = None
                continue
            w_arr = np.array(w, dtype=float)
            if w_prev is None:
                w_prev = w_arr
                continue
            t = 0.5 * float(np.abs(w_arr - w_prev).sum())
            turnover[g.index[pos]] = t
            w_prev = w_arr

    if group_col is None:
        _iter_rows(df)
    else:
        for _, g in df.groupby(group_col, sort=False):
            _iter_rows(g)

    df["turnover"] = turnover
    return df


def add_concentration_hhi(df_steps: pd.DataFrame, weights_col: str) -> pd.DataFrame:
    """
    Concentration proxy: HHI = sum_i w_i^2
    """
    if df_steps.empty:
        return df_steps.copy()

    df = df_steps.copy()
    if weights_col not in df.columns:
        df["concentration_hhi"] = np.nan
        return df

    vals: List[float] = []
    for w in df[weights_col]:
        if not isinstance(w, (list, tuple, np.ndarray)):
            vals.append(np.nan)
            continue
        w_arr = np.array(w, dtype=float)
        vals.append(float(np.square(w_arr).sum()))
    df["concentration_hhi"] = vals
    return df


def summarize_step_trading(df_steps: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize turnover / concentration from a step-level df.
    """
    if df_steps.empty:
        return {"empty": True}
    out: Dict[str, Any] = {"empty": False}

    t = _safe_series(df_steps, "turnover")
    if not t.empty:
        out["turnover"] = {
            "mean": float(t.mean()),
            "median": float(t.median()),
            "p90": float(t.quantile(0.9)),
            "max": float(t.max()),
        }

    c = _safe_series(df_steps, "concentration_hhi")
    if not c.empty:
        out["concentration_hhi"] = {
            "mean": float(c.mean()),
            "median": float(c.median()),
            "p90": float(c.quantile(0.9)),
            "max": float(c.max()),
        }

    # Optional: max single-name weight heuristic, if raw weights are present
    for col in ["trading.executed_weights", "trading.intent_weights"]:
        if col in df_steps.columns:
            mx = []
            for w in df_steps[col]:
                if isinstance(w, (list, tuple, np.ndarray)):
                    mx.append(float(np.max(np.array(w, dtype=float))))
            if mx:
                out[f"max_single_weight_from_{col}"] = float(np.max(mx))
    return out


def build_test_time_series(test_steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a date-indexed test time series for plotting & correlation checks.

    Assumptions (only if columns exist):
      - log_return: reward.log_return
      - nav: trading.nav_end_of_period
    """
    if test_steps_df.empty:
        return test_steps_df.copy()

    df = test_steps_df.copy()

    if "trading.nav_end_of_period" in df.columns:
        df["nav"] = pd.to_numeric(df["trading.nav_end_of_period"], errors="coerce")
    if "reward.log_return" in df.columns:
        df["log_return"] = pd.to_numeric(df["reward.log_return"], errors="coerce")

    if "log_return" in df.columns:
        lr = df["log_return"].fillna(0.0)
        df["cum_log_return"] = lr.cumsum()
        # time-series cumulative return from step log-returns
        df["cum_return"] = np.exp(df["cum_log_return"]) - 1.0
        # time-series NAV multiple implied by log-returns (endpoint should align with episode nav_multiple)
        df["nav_multiple_ts"] = np.exp(df["cum_log_return"])

    return df


def summarize_test_risk_relationships(ts_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Correlate step log_return with available penalty / risk terms.
    """
    if ts_df.empty:
        return {"empty": True}
    out: Dict[str, Any] = {"empty": False}

    if "log_return" not in ts_df.columns:
        out["note"] = "log_return not present (reward.log_return missing), skipping risk correlations."
        return out

    base = ts_df["log_return"]
    base = pd.to_numeric(base, errors="coerce")
    
    def _corr(col: str) -> Optional[float]:
        if col not in ts_df.columns:
            return None
        x = pd.to_numeric(ts_df[col], errors="coerce")
        m = base.notna() & x.notna()
        if m.sum() < 3:
            return None

        b = base[m]
        y = x[m]

        # Guard against zero-variance series (causes RuntimeWarning in numpy)
        if b.std() <= 1e-12 or y.std() <= 1e-12:
            return None

        return float(b.corr(y))

    # Keep this list conservative; add more later when you confirm env keys exist in your logs
    candidate_cols = [
        "reward.market_risk_penalty",
        "reward.llm_risk_penalty",
        "reward.endogenous_risk",
        "reward.normalized_turbulence",
        "reward.raw_turbulence",
        "reward.normalized_llm_risk",
        "reward.raw_llm_risk",
    ]
    corrs = {}
    for c in candidate_cols:
        v = _corr(c)
        if v is not None:
            corrs[c] = v
    if corrs:
        out["corr_log_return_vs_risks"] = corrs
    return out


# -----------------------------
# 3) Training dynamics (SB3 progress)
# -----------------------------

def summarize_training_progress(progress_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract and summarize training dynamics from SB3 progress CSV.

    We only use columns if present.
    """
    if progress_df.empty:
        return {"empty": True}
    out: Dict[str, Any] = {"empty": False}

    def _stats(col: str) -> Optional[Dict[str, float]]:
        if col not in progress_df.columns:
            return None
        s = pd.to_numeric(progress_df[col], errors="coerce").dropna()
        if s.empty:
            return None
        return {
            "first": float(s.iloc[0]),
            "last": float(s.iloc[-1]),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    for col in ["rollout/ep_rew_mean", "train/approx_kl", "train/explained_variance", "train/entropy_loss", "train/value_loss", "train/policy_gradient_loss"]:
        st = _stats(col)
        if st is not None:
            out[col] = st

    # Optional stability flags
    if "train/approx_kl" in progress_df.columns:
        kl = pd.to_numeric(progress_df["train/approx_kl"], errors="coerce").dropna()
        if not kl.empty:
            out["flags"] = {
                "kl_any_gt_0_2": bool((kl > 0.2).any()),
                "kl_any_gt_0_5": bool((kl > 0.5).any()),
            }

    return out

# -----------------------------
# 4) Generalization (train vs test)
# -----------------------------

def summarize_generalization(train_perf: pd.DataFrame, test_perf: pd.DataFrame) -> Dict[str, Any]:
    """
    Minimal, robust generalization summary:
      - delta in mean nav_multiple and mean sharpe_ratio (if available)
      - distribution overlap heuristics (quantiles)
    """
    if train_perf.empty or test_perf.empty:
        return {"empty": True}
    out: Dict[str, Any] = {"empty": False}

    def _mean(df: pd.DataFrame, col: str) -> Optional[float]:
        s = _safe_series(df, col)
        return float(s.mean()) if not s.empty else None

    for col in ["nav_multiple", "episode_return_log", "sharpe_ratio", "max_drawdown"]:
        mt = _mean(train_perf, col)
        ms = _mean(test_perf, col)
        if mt is not None and ms is not None:
            out[f"delta_mean_{col}"] = float(ms - mt)  # test - train

    # Quantile-based quick comparison on nav_multiple if present
    for col in ["nav_multiple", "episode_return_log"]:
        t = _safe_series(train_perf, col)
        s = _safe_series(test_perf, col)
        if not t.empty and not s.empty:
            out[f"{col}_quantiles"] = {
                "train": {q: float(t.quantile(q)) for q in [0.1, 0.5, 0.9]},
                "test": {q: float(s.quantile(q)) for q in [0.1, 0.5, 0.9]},
            }

    nav_gap = out.get("delta_mean_nav_multiple", None)
    sharpe_gap = out.get("delta_mean_sharpe_ratio", None)
    mdd_gap = out.get("delta_mean_max_drawdown", None)

    flags = {}
    if isinstance(nav_gap, (int, float)) and np.isfinite(nav_gap):
        flags["nav_degrades"] = bool(nav_gap < -0.05)   # you can tune thresholds
    if isinstance(sharpe_gap, (int, float)) and np.isfinite(sharpe_gap):
        flags["sharpe_degrades"] = bool(sharpe_gap < -0.2)
    if isinstance(mdd_gap, (int, float)) and np.isfinite(mdd_gap):
        # drawdown is negative; more negative means worse -> "degrades" if delta is negative
        flags["mdd_degrades"] = bool(mdd_gap < -0.05)
    if flags:
        out["flags"] = flags

    return out


# -----------------------------
# Canonical tables
# -----------------------------

def build_run_row(
    *,
    run_id: str,
    created_at: Optional[str],
    config: Dict[str, Any],
    train_eps_summary: Dict[str, Any],
    test_eps_summary: Dict[str, Any],
    training_summary: Dict[str, Any],
    test_step_summary: Dict[str, Any],
    generalization: Dict[str, Any],
) -> Dict[str, Any]:
    """
    One-row, JSON-friendly representation intended to become a DataFrame row.

    Keep this FLAT (no deep nesting) so concatenation is easy.
    """
    row: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "seed": config.get("experiment", {}).get("seed"),
        "algo": config.get("rl", {}).get("algo"),
        "total_timesteps": config.get("rl", {}).get("total_timesteps"),
        "n_envs": config.get("rl", {}).get("n_envs"),
    }

    # Pull some common headline metrics if present
    def _get(d: Dict[str, Any], key: str, sub: str) -> Any:
        if d.get("empty", True):
            return np.nan
        return d.get(key, {}).get(sub, np.nan)

    # Episode headline
    row["train_nav_multiple_mean"] = _get(train_eps_summary, "nav_multiple", "mean")
    row["train_sharpe_mean"] = _get(train_eps_summary, "sharpe_ratio", "mean")
    row["train_mdd_min"] = _get(train_eps_summary, "max_drawdown", "min")
    row["train_trade_count_mean"] = _get(train_eps_summary, "trade_count", "mean")
    row["train_avg_trade_cost_per_step_mean"] = _get(train_eps_summary, "avg_trade_cost_per_step", "mean")

    row["test_nav_multiple"] = _get(test_eps_summary, "nav_multiple", "mean")
    row["test_sharpe"] = _get(test_eps_summary, "sharpe_ratio", "mean")
    row["test_mdd"] = _get(test_eps_summary, "max_drawdown", "min")
    row["test_trade_count"] = _get(test_eps_summary, "trade_count", "mean")
    row["test_avg_trade_cost_per_step"] = _get(test_eps_summary, "avg_trade_cost_per_step", "mean")

    # Training stability
    if not training_summary.get("empty", True):
        row["approx_kl"] = training_summary.get("train/approx_kl", {}).get("last", np.nan)
        row["explained_variance"] = training_summary.get("train/explained_variance", {}).get("last", np.nan)
        row["ep_rew_mean"] = training_summary.get("rollout/ep_rew_mean", {}).get("last", np.nan)
        row["loss"] = training_summary.get("train/loss", {}).get("last", np.nan)
        row["value_loss"] = training_summary.get("train/value_loss", {}).get("last", np.nan)
        row["clip_fraction"] = training_summary.get("train/clip_fraction", {}).get("last", np.nan)
        row["entropy_loss"] = training_summary.get("train/entropy_loss", {}).get("last", np.nan)
        row["policy_gradient_loss"] = training_summary.get("train/policy_gradient_loss", {}).get("last", np.nan)

    # Test trading behavior
    if not test_step_summary.get("empty", True):
        row["test_turnover_median"] = test_step_summary.get("turnover", {}).get("median", np.nan)
        row["test_turnover_p90"] = test_step_summary.get("turnover", {}).get("p90", np.nan)
        row["test_concentration_hhi_median"] = test_step_summary.get("concentration_hhi", {}).get("median", np.nan)
        row["test_concentration_hhi_p90"] = test_step_summary.get("concentration_hhi", {}).get("p90", np.nan)

    # Generalization headline
    if not generalization.get("empty", True):
        for k, v in generalization.items():
            if k.startswith("delta_mean_"):
                row[k] = v

    return row
