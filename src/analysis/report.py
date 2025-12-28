from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.paths import get_run_id, get_run_paths
from src.analysis.loaders import (
    load_meta,
    load_train_steps,
    load_train_episodes,
    load_test_steps,
    load_test_episodes,
    load_train_progress,
)
from src.analysis.metrics import (
    add_trade_shares_scalars,
    add_concentration_hhi,
    add_turnover,
    build_run_row,
    build_test_time_series,
    compute_episode_perf,
    round_dict4,
    summarize_episode_perf,
    summarize_generalization,
    summarize_step_trading,
    summarize_test_risk_relationships,
    summarize_training_progress,
)


# -----------------------------
# Minimal plotting helpers
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_series(ts_df: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str, ylabel: str) -> None:
    if ts_df.empty or xcol not in ts_df.columns or ycol not in ts_df.columns:
        return
    plt.figure()
    plt.plot(ts_df[xcol], pd.to_numeric(ts_df[ycol], errors="coerce"))
    plt.xlabel(xcol)
    plt.ylabel(ylabel)
    plt.title(title)
    _savefig(out_path)


def plot_nav(ts_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if ts_df.empty or "date" not in ts_df.columns or "nav" not in ts_df.columns:
        return
    plt.figure()
    plt.plot(ts_df["date"], ts_df["nav"])
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title(title)
    _savefig(out_path)


def plot_cum_return(ts_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if ts_df.empty or "date" not in ts_df.columns or "cum_return" not in ts_df.columns:
        return
    plt.figure()
    plt.plot(ts_df["date"], ts_df["cum_return"])
    plt.xlabel("Date")
    plt.ylabel("Cumulative return (exp(cumsum(log_return)) - 1)")
    plt.title(title)
    _savefig(out_path)


def plot_risk_terms(ts_df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot risk-related terms (if present) on a shared x-axis.
    To avoid scale confusion, we plot each series in its own subplot-like figure
    (separate PNG per term), but keep this function minimal.
    """
    if ts_df.empty or "date" not in ts_df.columns:
        return

    # Only include columns that exist
    cols = [
        ("reward.log_return", "Log return"),
        ("reward.market_risk_penalty", "Market risk penalty"),
        ("reward.endogenous_risk", "Endogenous risk"),
        ("reward.normalized_turbulence", "Normalized turbulence"),
        ("reward.normalized_llm_risk", "Normalized LLM risk"),
    ]
    existing = [(c, label) for c, label in cols if c in ts_df.columns]
    if not existing:
        return

    # One figure, multiple lines, but only if they appear to be in similar scales.
    # Otherwise, you can split later.
    plt.figure()
    for c, label in existing:
        plt.plot(ts_df["date"], pd.to_numeric(ts_df[c], errors="coerce"), label=label)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Risk terms vs. Log return (test)")
    plt.legend(loc="upper left")
    _savefig(out_path)


def plot_concentration_heatmap(
    test_steps_aug: pd.DataFrame,
    out_path: Path,
    weight_labels: list[str] | None = None,
) -> None:
    """
    Minimal heatmap: weights over time, if 'trading.executed_weights' exists.
    We do not assume ticker names are available in logs; x-axis is step index, y-axis is asset index.
    """
    if test_steps_aug.empty or "trading.executed_weights" not in test_steps_aug.columns:
        return

    W = []
    dates = None
    if "date" in test_steps_aug.columns:
        dates = test_steps_aug["date"]
    for w in test_steps_aug["trading.executed_weights"]:
        if isinstance(w, (list, tuple, np.ndarray)):
            W.append(np.array(w, dtype=float))
    if not W:
        return
    mat = np.vstack(W)  # [T, A]

    plt.figure()

    finite = np.isfinite(mat)
    if not finite.any():
        return
    vmin = 0.0
    # robust upper bound (e.g., typical max weight), avoids a single spike flattening everything
    vmax = float(np.nanpercentile(mat[finite], 99))
    # safety: if vmax is tiny, fall back to a small ceiling
    vmax = max(vmax, 1e-6)

    plt.figure()
    im = plt.imshow(
        mat.T,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label(f"Portfolio weight (clipped to [0, {vmax:.3f}])")
    
    if "date" in test_steps_aug.columns:
        dates = pd.to_datetime(test_steps_aug["date"], errors="coerce")
        T = len(dates)

        # choose ~6 ticks max
        n_ticks = min(24, T)
        tick_idx = np.linspace(0, T - 1, n_ticks, dtype=int)

        tick_labels = []
        for i in tick_idx:
            d = dates.iloc[i]
            tick_labels.append(d.strftime("%Y-%m-%d") if pd.notna(d) else "")

        plt.xticks(tick_idx, tick_labels, rotation=45)
        plt.xlabel("Date")
    else:
        plt.xlabel("Time step")

    labels = ["CASH"] + list(weight_labels)
    if labels is not None:
        plt.yticks(np.arange(len(labels)), labels)
        plt.ylabel("Asset")
    else:
        plt.ylabel("Asset index")
        plt.title("Executed weights heatmap (test)")
    _savefig(out_path)


def _get_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_training_progress_curves(progress_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot PPO/SB3 training curves into out_dir.
    Robust to both SB3-prefixed columns (train/..., rollout/...) and un-prefixed ones.
    """
    if progress_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # X-axis preference: total timesteps (SB3) else fallback to row index.
    x_col = _get_first_existing_col(progress_df, ["time/total_timesteps", "total_timesteps"])
    if x_col is not None:
        x = pd.to_numeric(progress_df[x_col], errors="coerce")
        x_label = "Total timesteps"
    else:
        x = pd.Series(np.arange(len(progress_df)))
        x_label = "Update index"

    # Metric mapping: friendly_name -> candidate column names in CSV
    metric_specs: list[tuple[str, list[str], str]] = [
        ("approx_kl", ["train/approx_kl", "approx_kl"], "Approx KL"),
        ("explained_variance", ["train/explained_variance", "explained_variance"], "Explained variance"),
        ("ep_rew_mean", ["rollout/ep_rew_mean", "ep_rew_mean"], "Episode reward mean"),
        ("clip_fraction", ["train/clip_fraction", "clip_fraction"], "Clip fraction"),
        ("entropy_loss", ["train/entropy_loss", "entropy_loss"], "Entropy loss"),
        ("value_loss", ["train/value_loss", "value_loss"], "Value loss"),
        ("policy_gradient_loss", ["train/policy_gradient_loss", "policy_gradient_loss"], "Policy gradient loss"),
    ]

    for key, candidates, title in metric_specs:
        col = _get_first_existing_col(progress_df, candidates)
        if col is None:
            continue

        y = pd.to_numeric(progress_df[col], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 2:
            continue

        plt.figure()
        plt.plot(x[m], y[m])
        plt.xlabel(x_label)
        plt.ylabel(key)
        plt.title(f"Training: {title}")
        plt.tight_layout()
        plt.savefig(out_dir / f"train_{key}.png")
        plt.close()
# -----------------------------
# Per-run analysis entry point
# -----------------------------

def analyze_run(run_dir: str | Path) -> Dict[str, Any]:
    """
    Analyze a single run directory.

    Outputs under run_root/analysis:
      - summary.json                (nested, human-readable)
      - tables/run_row.json         (flat row, multi-run concat)
      - tables/episodes_train.csv   (episode table)
      - tables/episodes_test.csv
      - figures/*.png
    """
    paths = get_run_paths(run_dir)
    run_root: Path = paths["run_dir"]  # type: ignore[assignment]
    run_id = get_run_id(run_root)

    analysis_dir: Path = paths["analysis_dir"]  # type: ignore[assignment]
    figures_dir: Path = paths["figures_dir"]  # type: ignore[assignment]
    tables_dir: Path = paths["tables_dir"]  # type: ignore[assignment]
    _ensure_dir(analysis_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(tables_dir)

    # Load meta/config (for initial_cash, seeds, etc.)
    meta = load_meta(run_root)
    config = meta.get("config", {}) or {}
    created_at = meta.get("created_at", None)

    initial_cash = float(config.get("env", {}).get("initial_cash", np.nan))

    # Load core logs
    train_steps = load_train_steps(run_root)
    test_steps = load_test_steps(run_root)
    train_eps = load_train_episodes(run_root)
    test_eps = load_test_episodes(run_root)
    progress = load_train_progress(run_root)

    def _canonicalize_steps(df: pd.DataFrame, *, episode_col: str, date_col: str, step_col: str | None) -> pd.DataFrame:
        if df.empty:
            return df

        out = df.copy()

        # Rename to canonical names if needed
        if episode_col != "episode_id" and episode_col in out.columns and "episode_id" not in out.columns:
            out = out.rename(columns={episode_col: "episode_id"})
        if date_col != "date" and date_col in out.columns and "date" not in out.columns:
            out = out.rename(columns={date_col: "date"})

        # Type coercions (nullable-safe)
        if "episode_id" in out.columns:
            out["episode_id"] = pd.to_numeric(out["episode_id"], errors="coerce").astype("Int64")
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")

        # Sorting for chronological correctness (critical for turnover and plotting)
        sort_cols: list[str] = []
        if "episode_id" in out.columns:
            sort_cols.append("episode_id")
        if "date" in out.columns:
            sort_cols.append("date")
        if step_col and step_col in out.columns:
            sort_cols.append(step_col)

        if sort_cols:
            out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        return out

    # train_steps originally uses meta.* keys
    train_steps = _canonicalize_steps(
        train_steps,
        episode_col="meta.episode_id",
        date_col="meta.date",
        step_col="meta.step_idx" if "meta.step_idx" in train_steps.columns else None,
    )

    # test_steps already uses top-level episode_id/date; keep canonicalization consistent
    test_steps = _canonicalize_steps(
        test_steps,
        episode_col="episode_id",
        date_col="date",
        step_col="step_idx" if "step_idx" in test_steps.columns else None,
    )

    # Episode perf
    train_perf = compute_episode_perf(train_eps, initial_cash)
    test_perf = compute_episode_perf(test_eps, initial_cash)
    train_eps_summary = summarize_episode_perf(train_perf)
    test_eps_summary = summarize_episode_perf(test_perf)

    # Step trading metrics (train/test separately)
    train_steps_aug = add_concentration_hhi(add_turnover(train_steps, "trading.executed_weights"), "trading.executed_weights")
    test_steps_aug = add_concentration_hhi(add_turnover(test_steps, "trading.executed_weights"), "trading.executed_weights")
    train_step_summary = summarize_step_trading(train_steps_aug)
    test_step_summary = summarize_step_trading(test_steps_aug)

    # Test time series + risk relationships
    ts_test = build_test_time_series(test_steps_aug)
    ts_test = add_trade_shares_scalars(ts_test)

    risk_rel = summarize_test_risk_relationships(ts_test)
    # --- Consistency check: endpoint of step log-returns should match episode nav_multiple ---
    # (Only meaningful for 1-episode test runs; keep it defensive.)
    if (not ts_test.empty) and ("nav_multiple_ts" in ts_test.columns) and (not test_perf.empty) and ("nav_multiple" in test_perf.columns):
        nav_ts_end = float(pd.to_numeric(ts_test["nav_multiple_ts"].iloc[-1], errors="coerce"))
        nav_ep = float(pd.to_numeric(test_perf["nav_multiple"].iloc[0], errors="coerce"))
        if np.isfinite(nav_ts_end) and np.isfinite(nav_ep):
            summary_nav_gap = nav_ts_end - nav_ep  # should be near 0
        else:
            summary_nav_gap = np.nan
    else:
        summary_nav_gap = np.nan

    # Training progress summary
    training_summary = summarize_training_progress(progress)

    # Generalization summary
    generalization = summarize_generalization(train_perf, test_perf)

    # Plots (date-based)
    plot_nav(ts_test, figures_dir / "test_nav.png", "Test NAV over time")
    plot_cum_return(ts_test, figures_dir / "test_cum_return.png", "Test time-series cumulative return (from step log_return)")
    plot_risk_terms(ts_test, figures_dir / "test_risk_terms.png")

    # Trading behaviour plots
    plot_series(test_steps_aug, "date", "turnover",
                figures_dir / "test_turnover.png", "Test turnover over time", "Turnover")
    plot_series(test_steps_aug, "date", "concentration_hhi",
                figures_dir / "test_concentration_hhi.png", "Test concentration(HHI) over time", "Turnover")
    plot_series(ts_test, "date", "trading.trade_cost",
                figures_dir / "test_trade_cost.png", "Test trade cost over time", "Trade cost")
    plot_series(ts_test, "date", "trade_volume_abs",
                figures_dir / "test_trade_volume_abs.png",  "Test trade volume (sum |shares|) over time", "Sum |shares|")
    plot_series(ts_test, "date", "assets_traded_count",
                figures_dir / "test_assets_traded_count.png", "Test assets traded count over time", "# assets traded")

    weight_labels = (config.get("data", {}) or {}).get("canonical_tickers", None)
    plot_concentration_heatmap(
        test_steps_aug,
        figures_dir / "test_weights_heatmap.png",
        weight_labels=weight_labels,
    )

    plot_training_progress_curves(progress, figures_dir / "training_progress")
    # Save episode /test tables 
    if not train_perf.empty:
        train_perf.to_csv(tables_dir / "episodes_train.csv", index=False)
    if not test_perf.empty:
        test_perf.to_csv(tables_dir / "episodes_test.csv", index=False)
    if not len(test_steps_aug) == 0:
        ts_test.to_csv(tables_dir / "test_steps_behaviour.csv", index=False)

    # Flat run-row for multi-run comparison
    run_row = build_run_row(
        run_id=run_id,
        created_at=created_at,
        config=config,
        train_eps_summary=train_eps_summary,
        test_eps_summary=test_eps_summary,
        training_summary=training_summary,
        test_step_summary=test_step_summary,
        generalization=generalization,
    )

    # Nested summary.json (human readable)
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "files_present": {
            "train_steps": bool(paths["train_steps"].exists()),  
            "test_steps": bool(paths["test_steps"].exists()),   
            "train_episodes": bool(paths["train_episodes"].exists()),  
            "test_episodes": bool(paths["test_episodes"].exists()),   
            "train_progress": bool(paths["train_progress"].exists()),  
            "train_monitor_files_num": int(len(paths["train_monitor_files"])),  
            "test_monitor_files_num": int(len(paths["test_monitor_files"])),   
        },
        "episode_performance": {
            "train": train_eps_summary,
            "test": test_eps_summary,
        },
        "step_trading": {
            "train": train_step_summary,
            "test": test_step_summary,
        },
        "training_metrics": training_summary,
        "test_risk_relationships": risk_rel,
        "generalization": generalization,
        "consistency_checks": {
            "test_nav_multiple_ts_minus_episode": summary_nav_gap,
        },
        "run_row": run_row,
        # Keep config so summary.json is self-contained (you can trim later)
        "config": config,
    }

    summary = round_dict4(summary)
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    run_row = summary["run_row"]
    df = pd.DataFrame([run_row])
    out = tables_dir / "run_row.csv"
    df.to_csv(out, index=False)

    return summary
