# src/analysis/run_batch.py
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


# -----------------------------
# YAML IO
# -----------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# -----------------------------
# Dotted-path set/get
# -----------------------------
def set_by_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_by_path(d: Dict[str, Any], dotted: str) -> Any:
    keys = dotted.split(".")
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# -----------------------------
# Naming
# -----------------------------
def _fmt_val(v: Any, float_fmt: str) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # keep stable names (avoid trailing .0 etc.)
        return format(v, float_fmt)
    return str(v)


def make_run_name(
    prefix: str,
    cfg: Dict[str, Any],
    include_keys: List[str],
    float_fmt: str = "g",
    max_len: Optional[int] = None,
) -> str:
    """
    Build a human-readable run_id suffix from a prefix + selected config keys.

    Example:
      prefix="pure_price_no_risk_v1"
      include_keys = ["rl.hyperparams.learning_rate", "rl.hyperparams.n_steps"]

    gives something like:
      "pure_price_no_risk_v1_learning_rate_0.0001_n_steps_1024"
    """
    parts: List[str] = []
    if prefix:
        parts.append(prefix)

    for k in include_keys:
        v = get_by_path(cfg, k)
        if v is None:
            continue
        # Only use the leaf name after the last dot, e.g.
        # "rl.hyperparams.learning_rate" -> "learning_rate"
        leaf = k.split(".")[-1]
        safe_k = leaf.replace(".", "_")
        safe_v = _fmt_val(v, float_fmt).replace("/", "_")
        parts.append(f"{safe_k}_{safe_v}")

    name = "_".join(parts)
    if max_len is not None and len(name) > max_len:
        name = name[:max_len]
    return name


# -----------------------------
# Reading per-run outputs (simplified per your rule)
# -----------------------------
def read_run_row_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    p = run_dir / "analysis" / "tables" / "run_row.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# -----------------------------
# Scoring (weights from delta.yaml)
# -----------------------------
def score_row(row: pd.Series, weights: Dict[str, float]) -> float:
    def _num(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            if pd.isna(v):
                return default
            return v
        except Exception:
            return default

    s = 0.0
    for col, w in (weights or {}).items():
        s += float(w) * _num(row.get(col), default=0.0)
    return float(s)


# -----------------------------
# Grid expansion
# -----------------------------
def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    grid:
      {"a.b": [1,2], "x.y": [3,4]} -> list of override dicts
    """
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        d = {k: v for k, v in zip(keys, combo)}
        out.append(d)
    return out


def latest_run_dir(stage_root: Path, run_name: str) -> Optional[Path]:
    """
    run.py creates: stage_root / f"{exp_name}_{timestamp}"
    Here exp_name == run_name (cfg["experiment"]["name"])
    """
    candidates = sorted(
        stage_root.glob(f"{run_name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
# -----------------------------
# Main batch
# -----------------------------
def main(experiments_root: str) -> None:
    base_cfg_path = Path(experiments_root) / "config.yaml"
    delta_path = Path(experiments_root) / "delta.yaml"

    base_cfg = load_yaml(base_cfg_path)
    delta = load_yaml(delta_path)
   
    stage_root = Path(base_cfg["experiment"]["log_root"])
    stage_root.mkdir(parents=True, exist_ok=True)
    
    batch_root = stage_root / "_batch"
    cfg_snap_dir = batch_root / "configs"
    cfg_snap_dir.mkdir(parents=True, exist_ok=True)

    # save batch meta for reproducibility
    meta = {
        "base_cfg_path": str(Path(base_cfg_path).resolve()),
        "delta_path": str(Path(delta_path).resolve()),
        "stage_root": str(stage_root),
        "created_at": datetime.now().isoformat(),
    }
    (batch_root).mkdir(parents=True, exist_ok=True)
    with open(batch_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    batch_cfg = delta.get("batch", {}) or {}
    prefix = str(batch_cfg.get("prefix", "batch"))
    include_keys = list(batch_cfg.get("include_keys", []))
    float_fmt = str(batch_cfg.get("float_fmt", "g"))
    max_name_len = batch_cfg.get("max_name_len", None)
    max_name_len = int(max_name_len) if max_name_len is not None else None

    runner = delta.get("runner", {}) or {}
    cmd = list(runner.get("cmd", ["python", "-m", "src.rl_agents.run"]))
    continue_on_error = bool(runner.get("continue_on_error", True))

    export = delta.get("export", {}) or {}
    summary_csv = str(export.get("summary_csv", "summary.csv"))
    ranked_csv = str(export.get("ranked_csv", "ranked.csv"))
    failures_csv = str(export.get("failures_csv", "failures.csv"))

    score_cfg = (delta.get("score", {}) or {})
    weights = dict(score_cfg.get("weights", {}) or {})
    if not weights:
        weights = {
            "test_nav_multiple_mean": 10.0,
            "test_sharpe_mean": 1.0,
            "test_mdd_min": 1.0,
        }

    grid = delta.get("grid", {}) or {}
    overrides_list = expand_grid(grid)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for overrides in tqdm(overrides_list, desc="Batch runs"):
        cfg = deepcopy(base_cfg)

        # apply overrides
        for k, v in overrides.items():
            set_by_path(cfg, k, v)

        # set experiment.name (run_name)
        run_name = make_run_name(
            prefix=prefix,
            cfg=cfg,
            include_keys=include_keys,
            float_fmt=float_fmt,
            max_len=max_name_len,
        )
        cfg["experiment"]["name"] = run_name

        # snapshot config into batch configs folder
        tmp_cfg_path = cfg_snap_dir / f"{run_name}.yaml"
        dump_yaml(cfg, tmp_cfg_path)

        # run
        run_cmd = cmd + ["--config", str(tmp_cfg_path)]
        try:
            subprocess.run(run_cmd, check=True)
        except subprocess.CalledProcessError as e:
            failures.append(
                {
                    "run_name": run_name,
                    "error": f"subprocess_failed(returncode={e.returncode})",
                    "overrides": json.dumps(overrides),
                }
            )
            if not continue_on_error:
                break
            continue

         # locate actual run directory created by run.py
        run_dir = latest_run_dir(stage_root, run_name)
        if run_dir is None:
            failures.append(
                {
                    "run_name": run_name,
                    "error": "cannot_find_run_dir",
                    "overrides": json.dumps(overrides),
                }
            )
            if not continue_on_error:
                break
            continue

        df_row = read_run_row_csv(run_dir)
        if df_row is None or df_row.empty:
            failures.append(
                {
                    "run_name": run_name,
                    "error": "missing_or_empty_run_row_csv",
                    "run_id_dir": str(run_dir),
                }
            )
            if not continue_on_error:
                break
            continue

        r = df_row.iloc[0].to_dict()
        r["batch_prefix"] = prefix
        r["run_name"] = run_name
        r["root_dir"] = str(stage_root)     # per your instruction
        r["run_id_dir"] = str(run_dir)      # keep for drill-down; remove if you insist
        for k, v in overrides.items():
            r[f"ovr::{k}"] = v
        rows.append(r)

    # write summary.csv
    df_sum = pd.DataFrame(rows)
    summary_path = batch_root / summary_csv
    df_sum.to_csv(summary_path, index=False)

    # write failures.csv
    df_fail = pd.DataFrame(failures)
    fail_path = batch_root / failures_csv
    df_fail.to_csv(fail_path, index=False)

    # ranked.csv
    ranked_path = batch_root / ranked_csv
    if not df_sum.empty:
        df_rank = df_sum.copy()
        df_rank["score"] = df_rank.apply(lambda x: score_row(x, weights=weights), axis=1)
        df_rank = df_rank.sort_values("score", ascending=False)
        df_rank.to_csv(ranked_path, index=False)
    else:
        pd.DataFrame([]).to_csv(ranked_path, index=False)

    print(f"[batch] stage_root: {stage_root}")
    print(f"[batch] batch_root: {batch_root}")
    print(f"[batch] wrote: {summary_path}")
    print(f"[batch] wrote: {ranked_path}")
    print(f"[batch] wrote: {fail_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Directory to base config.yaml and delta.yaml")
    args = ap.parse_args()
    main(args.config)
