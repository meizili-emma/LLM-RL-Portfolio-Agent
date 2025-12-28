from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def load_run_row(run_dir: str | Path) -> Dict[str, Any]:
    """
    Load analysis/tables/run_row.json from a run directory.
    """
    run_dir = Path(run_dir)
    p = run_dir / "analysis" / "tables" / "run_row.json"
    if not p.exists():
        raise FileNotFoundError(f"run_row.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def aggregate_runs(run_root: str | Path, *, pattern: str = "*", require_analysis: bool = True) -> pd.DataFrame:
    """
    Aggregate multiple runs under log_root into a single DataFrame.

    Args:
      run_root: e.g., data/rl_runs
      pattern: glob pattern for run directories
      require_analysis: if True, only include runs that already have run_row.json
    """
    run_root = Path(run_root)
    rows: List[Dict[str, Any]] = []
    for rd in sorted(run_root.glob(pattern)):
        if not rd.is_dir():
            continue
        p = rd / "analysis" / "tables" / "run_row.json"
        if require_analysis and (not p.exists()):
            continue
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("run_id", drop=False)
