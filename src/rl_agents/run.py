# src/rl_agents/run.py

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import shutil

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

# Adjust these imports to your actual package structure
from src.rl_agents.env import StockPortfolioEnv
from src.rl_agents.data import (prepare_env_dataframe, 
                                plan_rl_train_test_split, 
                                plan_rl_train_test_split_with_forward_rolling_val,) 
from src.rl_agents.callback import PortfolioLoggingCallback
from src.rl_agents.feature_extractor import PortfolioFeatureExtractor
from src.analysis.report import analyze_run
from src.analysis.paths import get_run_paths

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")

    def write_record(self, record: Dict[str, Any]) -> None:
        """
        JSONL with robust handling for numpy / pandas / datetime types.
        """
        def _default(o):
            import numpy as _np
            import pandas as _pd
            from datetime import datetime as _dt

            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, (_np.floating, _np.integer)):
                return o.item()
            if isinstance(o, (_pd.Timestamp, _dt)):
                return o.isoformat()
            return str(o)

        self._f.write(json.dumps(record, default=_default) + "\n")

    def close(self) -> None:
        if self._f and not self._f.closed:
            self._f.close()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(cfg: dict) -> Path:
    log_root = Path(cfg["experiment"]["log_root"])
    exp_name = cfg["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{exp_name}_{timestamp}"
    run_dir = log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_meta(run_dir: Path, cfg: dict, extra_meta: dict | None = None) -> None:
    def _json_default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, pd.Index):
            return o.tolist()
        return str(o)

    meta = {
        "config": cfg,
        "created_at": datetime.now().isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = run_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

# --------------------------------------------------------------------
# Environment construction helpers
# --------------------------------------------------------------------

def apply_llm_feature_mask(df_env: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply per-feature scaling/masking to LLM feature columns, based on env.llm_feature_list
    and env.llm_feature_mask in the config.

    - If a column is in llm_feature_list but not in llm_feature_mask -> factor = 1.0 (no change).
    - If factor == 0.0 -> we hard-set the column to 0.0 (curriculum 'off' stage).
    - Otherwise -> we multiply the column by the factor.
    """
    env_cfg = cfg.get("env", {})
    llm_cols = env_cfg.get("llm_feature_list", []) or []
    mask_cfg = env_cfg.get("llm_feature_mask", {}) or {}

    if not llm_cols or not mask_cfg:
        # nothing to do
        return df_env

    df = df_env.copy()

    for col in llm_cols:
        if col not in df.columns:
            print(f"[mask] Warning: LLM feature column '{col}' not found in df_env; skipping.")
            continue

        factor = float(mask_cfg.get(col, 1.0))
        if factor == 1.0:
            continue

        # numeric, NaNs -> 0, then scale
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if factor == 0.0:
            df[col] = 0.0
        else:
            df[col] = s * factor

        print(f"[mask] Applied factor {factor} to LLM feature '{col}'.")

    return df


def build_env_kwargs(df_slice: pd.DataFrame, cfg: dict) -> dict:
    
    env_cfg = cfg["env"]
    data_cfg = cfg["data"]

    return dict(
        df=df_slice,
        initial_cash=float(env_cfg["initial_cash"]),
        tech_indicator_list=list(env_cfg.get("tech_indicator_list", [])),
        fundamental_indicator_list=list(env_cfg.get("fundamental_indicator_list", [])),
        llm_feature_list=list(env_cfg.get("llm_feature_list", [])),
        global_feature_list=list(env_cfg.get("global_feature_list", [])),
        buy_cost_pct=float(env_cfg.get("buy_cost_pct", 0.0)),
        sell_cost_pct=float(env_cfg.get("sell_cost_pct", 0.0)),
        min_cash_weight=float(env_cfg.get("min_cash_weight", 0.0)),
        allow_fractional=bool(env_cfg.get("allow_fractional", True)),
        trading_state_history_len=int(data_cfg["trading_state_history_len"]),
        norm_window_size=int(data_cfg["norm_window_size"]),
        rolling_risk_window=int(env_cfg.get("rolling_risk_window", 26)),
        llm_risk_exposure_col=env_cfg.get("llm_risk_exposure_col", None),
        market_risk_exposure_col=env_cfg.get("market_risk_exposure_col", None),
        max_llm_risk_score=env_cfg.get("max_llm_risk_score", None),
        max_turbulence=env_cfg.get("max_turbulence", None),
        beta=float(env_cfg.get("beta", 1.0)),
        gamma=float(env_cfg.get("gamma", 1.0)),
        reward_scaling=float(env_cfg.get("reward_scaling", 1.0)),
        render_mode=None,
        render_topk=5,
        print_render=False,
    )


def make_vec_env(
    df_slice: pd.DataFrame,
    cfg: dict,
    phase: str,
    run_dir: Path,
    base_seed: int,
):
    """
    Build a vectorized StockPortfolioEnv using the classmethod build_vec_env.
    """
    rl_cfg = cfg["rl"]
    vec_cfg = rl_cfg["vec_env"]
    log_cfg = cfg["logging"][phase]

    monitor_dir = run_dir / log_cfg["monitor_dir"]
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = build_env_kwargs(df_slice=df_slice, cfg=cfg)

    vec_env = StockPortfolioEnv.build_vec_env(
        env_kwargs=env_kwargs,
        n_envs=int(rl_cfg["n_envs"] if phase == "train" else 1),
        base_seed=int(base_seed),
        use_subproc=bool(vec_cfg.get("use_subproc", True)) if phase == "train" else False,
        monitor_dir=str(monitor_dir),
        norm_reward=bool(vec_cfg.get("norm_reward", False)),
        clip_reward=float(vec_cfg.get("clip_reward", 10.0)),
        monitor_info_keys=None,
    )
    return vec_env


# --------------------------------------------------------------------
# Train
# --------------------------------------------------------------------


def train(cfg: dict, run_dir: Path, df_env: pd.DataFrame, split_plan: dict) -> str:
    """
    Train PPO on the train split. Returns path to final model.
    """
    seed = int(cfg["experiment"]["seed"])
    rl_cfg = cfg["rl"]

    # Slice train window by index (plan_rl_train_test_split defines start/end indices)
    train_start = int(split_plan["train"]["episode_start_idx"])
    train_end = int(split_plan["train"]["episode_end_idx"])
    df_train = df_env.loc[train_start:train_end]

    vec_env = make_vec_env(df_train, cfg, phase="train", run_dir=run_dir, base_seed=seed)

    # Prepare logging directories
    train_log_dir = run_dir / cfg["logging"]["train"]["monitor_dir"]
    train_log_dir.mkdir(parents=True, exist_ok=True)

    # Portfolio logging callback (train phase)
    logging_cfg = cfg["logging"]
    portfolio_cb = PortfolioLoggingCallback(
        log_dir=str(train_log_dir),
        step_filename=logging_cfg.get("step_filename", "train_steps.jsonl"),
        episode_filename=logging_cfg.get("episode_filename", "train_episodes.jsonl"),
        verbose=int(logging_cfg.get("verbose", 1)),
    )

    callbacks = CallbackList([portfolio_cb])

    # Instantiate PPO
    hp = rl_cfg["hyperparams"]

    fe_cfg = cfg.get("feature_extractor", {})
    use_fe = bool(fe_cfg.get("enabled", True))

    policy_kwargs = {}
    if use_fe:
        # don't pass the control flag into the module
        fe_kwargs = {k: v for k, v in fe_cfg.items() if k != "enabled"}
        policy_kwargs = dict(
            features_extractor_class=PortfolioFeatureExtractor,
            features_extractor_kwargs=fe_kwargs,
        )

    logger = configure(str(train_log_dir), ["stdout", "csv"])

    pretrained_path = cfg["experiment"].get("pretrained_model_path")
    if pretrained_path:
        pretrained_path = str(pretrained_path)
        if Path(pretrained_path).exists():
            print(f"[train] Loading pretrained PPO model from: {pretrained_path}")
            model = PPO.load(pretrained_path, env=vec_env)
        else:
            print(f"[train] WARNING: pretrained_model_path not found: {pretrained_path}, training from scratch.")
            model = PPO(
                rl_cfg["policy"],
                vec_env,
                learning_rate=float(hp["learning_rate"]),
                n_steps=int(hp["n_steps"]),
                batch_size=int(hp["batch_size"]),
                n_epochs=int(hp["n_epochs"]),
                gamma=float(hp["gamma"]),
                gae_lambda=float(hp["gae_lambda"]),
                clip_range=float(hp["clip_range"]),
                ent_coef=float(hp["ent_coef"]),
                vf_coef=float(hp["vf_coef"]),
                max_grad_norm=float(hp["max_grad_norm"]),
                verbose=1,
                seed=seed,
                policy_kwargs=policy_kwargs,
                )
    else:
        model = PPO(
            rl_cfg["policy"],
            vec_env,
            learning_rate=float(hp["learning_rate"]),
            n_steps=int(hp["n_steps"]),
            batch_size=int(hp["batch_size"]),
            n_epochs=int(hp["n_epochs"]),
            gamma=float(hp["gamma"]),
            gae_lambda=float(hp["gae_lambda"]),
            clip_range=float(hp["clip_range"]),
            ent_coef=float(hp["ent_coef"]),
            vf_coef=float(hp["vf_coef"]),
            max_grad_norm=float(hp["max_grad_norm"]),
            verbose=1,
            seed=seed,
            policy_kwargs=policy_kwargs,
            )
    
    model.set_logger(logger)

    model_dir = run_dir / cfg["logging"]["train"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(rl_cfg["total_timesteps"])
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_model_path = model_dir / "final_model.zip"
    model.save(str(final_model_path))
    print(f"[train] Saved final model to: {final_model_path}")
    progress_src = train_log_dir / "progress.csv"
    progress_dst = train_log_dir / "sb3_train_progress.csv"
    if progress_src.exists():
        shutil.copy2(progress_src, progress_dst)
        print(f"[train] SB3 progress copied to: {progress_dst}")
    return str(final_model_path)


# --------------------------------------------------------------------
# Test: deterministic rollout with JSONL logging
# --------------------------------------------------------------------


def run_test_episodes(
    model: PPO,
    df_test: pd.DataFrame,
    cfg: dict,
    log_root: str | Path,
) -> None:
    """
    Run test episodes with a trained PPO model on df_test and
    log per-step and per-episode statistics to JSONL files.

    Matches the design you sketched:
      - deterministic rollout
      - rich per-step info (meta, reward_components, trading)
      - per-episode summary via episode_summary in info
    """
    from src.rl_agents.env import StockPortfolioEnv  # ensure same env class

    test_cfg = cfg.get("test", {})
    logging_cfg = cfg.get("logging", {})

    if not test_cfg.get("enabled", True):
        print("[test] Test stage disabled in config.")
        return

    n_episodes: int = int(test_cfg.get("n_episodes", 1))
    deterministic: bool = bool(test_cfg.get("deterministic", True))
    max_steps: Optional[int] = test_cfg.get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)

    log_root = Path(log_root)
    test_log_dir = log_root / str(test_cfg.get("log_subdir", "test"))
    test_log_dir.mkdir(parents=True, exist_ok=True)

    # JSONL paths
    step_path = test_log_dir / logging_cfg.get("test_step_filename", "test_steps.jsonl")
    episode_path = test_log_dir / logging_cfg.get("test_episode_filename", "test_episodes.jsonl")

    steps_writer = JsonlWriter(step_path)
    episodes_writer = JsonlWriter(episode_path)

    # Build env_kwargs for test env
    env_kwargs = build_env_kwargs(df_slice=df_test, cfg=cfg)

    # Optional: monitor for test
    monitor_root = test_log_dir
    monitor_root.mkdir(parents=True, exist_ok=True)

    test_vec_env = StockPortfolioEnv.build_vec_env(
        env_kwargs=env_kwargs,
        n_envs=1,
        base_seed=int(test_cfg.get("seed", cfg["experiment"]["seed"])),
        use_subproc=False,
        monitor_dir=str(monitor_root),
        norm_reward=False,   # usually no VecNormalize at test
        clip_reward=float(cfg["rl"]["vec_env"].get("clip_reward", 10.0)),
        monitor_info_keys=None,
    )

    # For a DummyVecEnv of size 1, vec_env.reset() -> obs only
    for ep in range(n_episodes):
        obs = test_vec_env.reset()
        done = False
        step_counter = 0

        print(f"[test] Starting episode {ep + 1}/{n_episodes} (deterministic={deterministic})")

        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = test_vec_env.step(action)

            step_counter += 1
            done = bool(dones[0])
            info = infos[0]

            # Extract info fields we agreed on in the env
            meta = info.get("meta", {}) or {}
            reward_components = info.get("reward_components", {}) or {}
            trading = info.get("trading", {}) or {}

            reward_scalar = float(rewards[0])

            env_id = info.get("env_id", meta.get("env_id"))
            date_idx = meta.get("date_idx", meta.get("day_idx"))

            step_record = {
                "episode_id": meta.get("episode_id"),
                "env_id": env_id,
                "step_idx": meta.get("step_idx"),
                "day_idx": date_idx,
                "date": str(meta.get("date")) if meta.get("date") is not None else None,
                "close_ts_utc": meta.get("close_ts_utc"),
                "reward": reward_scalar,
                "reward_components": reward_components,
                "trading": trading,
            }
            steps_writer.write_record(step_record)

            if done or (max_steps is not None and step_counter >= max_steps):
                # On terminal, env should have attached episode_summary
                episode_summary = info.get("episode_summary", None)
                if episode_summary is not None:
                    episodes_writer.write_record(episode_summary)
                break

        print(f"[test] Episode {ep + 1} finished after {step_counter} steps.")

    steps_writer.close()
    episodes_writer.close()
    test_vec_env.close()

    print(f"[test] Step logs written to: {step_path}")
    print(f"[test] Episode logs written to: {episode_path}")


def test(cfg: dict, run_dir: Path, df_env: pd.DataFrame, split_plan: dict, model_path: str) -> None:
    """
    Test stage: deterministic rollout on test split with JSONL logging.
    """
    # Slice test window
    test_start = int(split_plan["test"]["episode_start_idx"])
    test_end = int(split_plan["test"]["episode_end_idx"])
    df_test = df_env.loc[test_start:test_end]

    # Load model without binding env (we will do manual rollout)
    model = PPO.load(model_path)
    run_test_episodes(
        model=model,
        df_test=df_test,
        cfg=cfg,
        log_root=run_dir,
    )


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    stage = str(cfg.get("experiment", {}).get("stage", "final")).lower()
    
    run_dir = make_run_dir(cfg)
    shutil.copy2(config_path, run_dir / "config.yaml")
    # 1) Load and prepare dataframe for the environment
    data_cfg = cfg["data"]
    raw_df = pd.read_parquet(data_cfg["file_path"])
    df_env, missing_report = prepare_env_dataframe(
        raw_df=raw_df,
        canonical_tickers=data_cfg["canonical_tickers"],
    )

    print(f"[data] Env df shape: {df_env.shape}")
    if not missing_report.empty:
        print("[data] Missing ticker report (first few rows):")
        print(missing_report.head())

    #  --- Curriculum / LLM feature masking ---
    df_env = apply_llm_feature_mask(df_env, cfg)

    # 2) Plan train/test split with warmup logic
    # Be robust to train_frac vs train_ratio
    train_ratio = float(data_cfg.get("train_ratio", data_cfg.get("train_frac", 0.7)))

    if stage == "select":
        print("[split] Stage='select': using forward-rolling validation folds.")
        n_val_folds = int(data_cfg.get("val_n_folds", 3))
        val_action_days = int(data_cfg.get("val_action_days", 30))
        min_train_action_days = int(data_cfg.get("val_min_train_action_days", 150))
        selected_fold_id = int(data_cfg.get("val_fold_id", 0))

        # Build outer train/test + inner val folds on the masked df_env
        full_plan = plan_rl_train_test_split_with_forward_rolling_val(
            df_env=df_env,
            first_trade_date=data_cfg["first_trade_date"],
            last_trade_date=data_cfg["last_trade_date"],
            terminal_date=data_cfg["terminal_date"],
            train_ratio=train_ratio,
            norm_window_size=int(data_cfg["norm_window_size"]),
            trading_state_history_len=int(data_cfg["trading_state_history_len"]),
            n_val_folds=n_val_folds,
            val_action_days=val_action_days,
            min_train_action_days=min_train_action_days,
        )

        folds = full_plan.get("val_folds", [])
        if not folds:
            raise ValueError("Forward-rolling validation requested, but no validation folds were built.")

        if not (0 <= selected_fold_id < len(folds)):
            raise ValueError(
                f"Requested validation fold index={selected_fold_id}, "
                f"but only {len(folds)} folds were built. "
                f"Use 0..{len(folds)-1} for data.val_fold_id."
                )

        fold = folds[selected_fold_id]
        split_plan = {
            "meta": {
                **full_plan["meta"],
                "stage": stage,
                "selected_val_fold_id": selected_fold_id,
            },
            "train": fold["train"],
            "test": fold["val"],
            "all_dates": full_plan.get("all_dates", {}),
        }
        split_plan_full = full_plan

    else: 
        print("[split] Stage='final': using standard train/test split.")
        split_plan = plan_rl_train_test_split(
            df_env=df_env,
            first_trade_date=data_cfg["first_trade_date"],
            last_trade_date=data_cfg["last_trade_date"],
            terminal_date=data_cfg["terminal_date"],
            train_ratio=train_ratio,
            norm_window_size=int(data_cfg["norm_window_size"]),
            trading_state_history_len=int(data_cfg["trading_state_history_len"]),)
        split_plan["meta"]["stage"] = stage
        split_plan_full = split_plan

    env_tickers_order = sorted(pd.Series(df_env["ticker"].unique()).dropna().astype(str).tolist())
    weight_labels = ["CASH"] + env_tickers_order

    save_meta(
        run_dir,
        cfg,
        extra_meta={
            "split_plan": split_plan,
            "split_plan_full": split_plan_full,
            "env_tickers_order": env_tickers_order,
            "weight_labels": weight_labels,
        },
    )

    # 3) Train
    final_model_path = train(cfg, run_dir, df_env, split_plan)

    # 4) Test
    test(cfg, run_dir, df_env, split_plan, final_model_path)

    try:
        paths = get_run_paths(run_dir)
        analysis_dir: Path = paths["analysis_dir"]  # type: ignore[assignment]
        figures_dir: Path = paths["figures_dir"]  # type: ignore[assignment]
        tables_dir: Path = paths["tables_dir"] 
        _ = analyze_run(run_dir)
        print(
            f"[analysis] Wrote analysis outputs under:{analysis_dir} \n"
            f"[figures] Wrote plots outputs under:{figures_dir} \n "
            f"[table] Wrote summary outputs under:{tables_dir}."
            )
    except Exception as e:
        print(f"[analysis] Failed (non-fatal): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
