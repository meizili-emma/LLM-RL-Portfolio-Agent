from stable_baselines3.common.callbacks import BaseCallback
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

def _to_serializable(obj):
    """
    Recursively convert numpy / pandas / datetime objects into JSON-serializable
    Python types (lists, floats, ints, strings, dicts).
    """
    # numpy scalars
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas Timestamp or python datetime
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # dict: recurse on values
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # list / tuple: recurse on elements
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    # everything else: leave as-is (json will handle str, int, float, bool, None)
    return obj


class PortfolioLoggingCallback(BaseCallback):
    """
    [NEW] SB3 callback to log per-step and per-episode records from StockPortfolioEnv.

    It expects the environment's `info` dict to contain:
      - "meta": {...}
      - "reward_components": {...}
      - "trading": {...}
      - optionally "episode_summary" at terminal steps

    It also uses "env_id" and "seed_used" keys injected by the _MetaInfo wrapper
    in `StockPortfolioEnv.build_vec_env`.
    """

    def __init__(self,
                 log_dir: str,
                 step_filename: str = "steps.jsonl",
                 episode_filename: str = "episodes.jsonl",
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.step_path = self.log_dir / step_filename
        self.episode_path = self.log_dir / episode_filename
        self.step_f = None
        self.episode_f = None

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # open files in append mode to support multiple runs if needed
        self.step_f = self.step_path.open("w", encoding="utf-8")
        self.episode_f = self.episode_path.open("w", encoding="utf-8")
        if self.verbose:
            print(f"[PortfolioLoggingCallback] Logging steps to {self.step_path}")
            print(f"[PortfolioLoggingCallback] Logging episodes to {self.episode_path}")

    def _on_step(self) -> bool:
        # For VecEnv, infos is a list of dicts, one per sub-env
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue

            # Per-step logging if "meta" present
            if "meta" in info and "reward_components" in info and "trading" in info:
                record = {
                    "meta": dict(info["meta"]),
                    "reward_components": dict(info["reward_components"]),
                    "trading": {},
                }

                # propagate env_id / seed_used if added by _MetaInfo
                for k in ("env_id", "seed_used"):
                    if k in info:
                        record["meta"][k] = info[k]

                # trading section: convert arrays to lists for JSON
                trading = info["trading"]
                record["trading"] = {
                    "intent_weights": trading["intent_weights"].tolist()
                        if trading.get("intent_weights") is not None else None,
                    "executed_weights": trading["executed_weights"].tolist()
                        if trading.get("executed_weights") is not None else None,
                    "trade_shares": trading["trade_shares"].tolist()
                        if trading.get("trade_shares") is not None else None,
                    "trade_cost": float(trading.get("trade_cost", 0.0)),
                    "pre_trade_value": float(trading.get("pre_trade_value", 0.0)),
                    "post_trade_value_t": float(trading.get("post_trade_value_t", 0.0)),
                    "nav_end_of_period": float(trading.get("nav_end_of_period", 0.0)),
                }

                safe_record = _to_serializable(record)
                self.step_f.write(json.dumps(safe_record) + "\n")

            # Per-episode logging if episode_summary present
            if "episode_summary" in info:
                summary = dict(info["episode_summary"])
                # attach env_id/seed_used if present
                for k in ("env_id", "seed_used"):
                    if k in info:
                        summary[k] = info[k]

                safe_summary = _to_serializable(summary)
                self.episode_f.write(json.dumps(safe_summary) + "\n")

        # Make sure logs are flushed reasonably often
        self.step_f.flush()
        self.episode_f.flush()
        return True

    def _on_training_end(self) -> None:
        if self.step_f is not None:
            self.step_f.close()
            self.step_f = None
        if self.episode_f is not None:
            self.episode_f.close()
            self.episode_f = None
