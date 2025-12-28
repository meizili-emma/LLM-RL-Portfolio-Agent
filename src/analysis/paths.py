from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union, TypedDict


PathOrPaths = Union[Path, List[Path]]


def get_run_id(run_dir: str | Path) -> str:
    """Run id is the leaf directory name (run_root basename)."""
    return Path(run_dir).resolve().name


def get_run_paths(run_dir: str | Path) -> Dict[str, PathOrPaths]:
    """
    Given a single RL run directory (run_root), return the standard file layout
    as a dict of Paths and lists of Paths.

    This is the single place where we encode assumptions about filenames.
    """
    run_dir = Path(run_dir)

    train_dir = run_dir / "train"
    test_dir = run_dir / "test"
    model_dir = run_dir / "models"

    # Core files
    meta = run_dir / "meta.json"
    final_model = model_dir / "final_model.zip"

    # Train logs
    train_steps = train_dir / "train_steps.jsonl"
    train_episodes = train_dir / "train_episodes.jsonl"

    # SB3 training progress: prefer sb3_train_progress.csv if present
    sb3_progress = train_dir / "sb3_train_progress.csv"
    legacy_progress = train_dir / "progress.csv"
    train_progress = sb3_progress if sb3_progress.exists() else legacy_progress

    # Monitor CSVs: one per env, globbed
    train_monitor_files = sorted(train_dir.glob("StockPortfolioEnv_env*.monitor.csv"))
    test_monitor_files = sorted(test_dir.glob("StockPortfolioEnv_env*.monitor.csv"))

    # Test logs
    test_steps = test_dir / "test_steps.jsonl"
    test_episodes = test_dir / "test_episodes.jsonl"

    # Analysis outputs
    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    tables_dir = analysis_dir / "tables"

    return {
        "run_dir": run_dir,
        "meta": meta,
        "final_model": final_model,
        "train_steps": train_steps,
        "train_episodes": train_episodes,
        "train_progress": train_progress,
        "train_monitor_files": train_monitor_files,
        "test_steps": test_steps,
        "test_episodes": test_episodes,
        "test_monitor_files": test_monitor_files,
        "analysis_dir": analysis_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
    }
