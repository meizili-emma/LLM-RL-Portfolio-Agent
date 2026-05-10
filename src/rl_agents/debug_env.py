import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from env import StockPortfolioEnv

def make_synthetic_df(T=60, tickers=None,
                      tech_cols=("mom", "rsi", "vol"),
                      fund_cols=("pe", "pb"),
                      llm_cols=("llm_overall", "llm_conf"),
                      global_cols=("cpi", "turbulence"),
                      llm_risk_col="llm_risk_exposure",
                      mkt_risk_col="turbulence"):
    """
    Create a DataFrame that matches the env's expectations:
      - index: 0..T-1 (int), *repeated* for each ticker
      - columns: 'date','ticker','close', feature cols, llm_risk_col, mkt_risk_col
      - exactly N rows per day (sorted by ['date','ticker'])
    """
    if tickers is None:
        tickers = [f"T{i}" for i in range(5)]
    N = len(tickers)
    dates = [datetime(2020,1,3) + timedelta(weeks=i) for i in range(T)]

    rows = []
    price_levels = {tic: 100.0 + 5.0*np.random.randn() for tic in tickers}
    for t in range(T):
        # globals are same for all tickers on day t
        gvals = {
            "cpi": 2.0 + 0.1*np.sin(t/10),
            "turbulence": max(0.0, np.random.randn()*0.5 + 1.5)
        }
        for tic in tickers:
            # simple random walk
            price_levels[tic] *= np.exp(0.001 + 0.02*np.random.randn())
            row = {
                "date": dates[t],
                "ticker": tic,
                "close": float(price_levels[tic]),
                # tech
                "mom": np.random.randn(),
                "rsi": np.clip(50 + 10*np.random.randn(), 0, 100),
                "vol": abs(np.random.randn()),
                # fund
                "pe": np.clip(15 + 5*np.random.randn(), 0, 60),
                "pb": np.clip(2 + 0.4*np.random.randn(), 0, 10),
                # llm features (bounded)
                "llm_overall": np.clip(np.random.uniform(-1, 1), -1, 1),
                "llm_conf": np.clip(np.random.uniform(0, 1), 0, 1),
                # risk exposures
                llm_risk_col: np.clip(np.random.uniform(0, 1), 0, 1),
                # globals (copied; env will take first row)
                "cpi": gvals["cpi"],
                mkt_risk_col: gvals["turbulence"],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # critical: sort & reindex as 0..T-1, repeated per ticker
    df = df.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    # build day index 0..T-1 repeated N times
    day_index = np.repeat(np.arange(T), len(tickers))
    df.index = day_index
    return df

def check_single_env():
    print("== single env test ==")
    T = 80
    tickers = [f"T{i}" for i in range(6)]
    df = make_synthetic_df(T=T, tickers=tickers)
    env = StockPortfolioEnv(
        df=df,
        initial_cash=1_000_000.0,
        tech_indicator_list=["mom", "rsi", "vol"],
        fundamental_indicator_list=["pe", "pb"],
        llm_feature_list=["llm_overall", "llm_conf"],
        global_feature_list=["cpi", "turbulence"],
        max_llm_risk_score=1.0,
        max_turbulence=5.0,
        action_temperature=1.0,
        trading_state_history_len=8,
        norm_window_size=26,
        rolling_risk_window=26,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        min_cash_weight=0.05,
        allow_fractional=True,
        llm_risk_exposure_col="llm_risk_exposure",
        market_risk_exposure_col="turbulence",
        reward_scaling=1.0,
        render_mode="ansi",
        print_render=False,
    )
    obs, info = env.reset()
    assert isinstance(obs, dict)
    # space checks
    assert env.observation_space.contains(obs), "obs not in space"

    # run random policy until done
    a_dim = env.action_space.shape[0]
    steps = 0
    while True:
        action = np.random.randn(a_dim).astype(np.float32)
        obs, r, term, trunc, info = env.step(action)
        # invariants
        w = env._current_weights_vector()
        assert np.all(np.isfinite(w))
        s = float(w.sum())
        assert abs(s - 1.0) < 1e-5, f"weights sum {s}"
        assert w[0] >= 0.0
        if env.min_cash_weight > 0:
            assert w[0] + env.EPS_WEIGHT >= env.min_cash_weight - 1e-6
        steps += 1

        if steps % 10 == 0:
            snap = env.render("ansi")
            print(snap)

        if term or trunc:
            print("episode end ->", info.get("episode_summary", {}))
            break

def check_vec_env(use_subproc=False):
    print(f"== vec env test (Subproc={use_subproc}) ==")
    from stable_baselines3.common.vec_env import VecEnv
    T = 70
    tickers = [f"T{i}" for i in range(4)]
    df = make_synthetic_df(T=T, tickers=tickers)

    env_kwargs = dict(
        df=df,
        initial_cash=1_000_000.0,
        tech_indicator_list=["mom", "rsi", "vol"],
        fundamental_indicator_list=["pe", "pb"],
        llm_feature_list=["llm_overall", "llm_conf"],
        global_feature_list=["cpi", "turbulence"],
        max_llm_risk_score=1.0,
        max_turbulence=5.0,
        trading_state_history_len=8,
        norm_window_size=26,
        rolling_risk_window=26,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        min_cash_weight=0.1,
        allow_fractional=False,  # test integer branch too
        llm_risk_exposure_col="llm_risk_exposure",
        market_risk_exposure_col="turbulence",
        reward_scaling=1.0,
        render_mode="none",
    )
    venv = StockPortfolioEnv.build_vec_env(
        env_kwargs=env_kwargs,
        n_envs=3,
        base_seed=123,
        use_subproc=use_subproc,
        monitor_dir="./_mon_debug",
        norm_reward=False,
    )
    assert isinstance(venv, VecEnv)
    obs = venv.reset()
    assert isinstance(obs, (dict, np.ndarray)), "VecEnv obs type unexpected"

    a_dim = env_kwargs["df"].loc[0:0].shape[0]  # wrong; easier: instantiate a temp env to read action dim
    tmp_env = StockPortfolioEnv(**env_kwargs)
    a_dim = tmp_env.action_space.shape[0]
    del tmp_env

    for _ in range(5):
        actions = np.random.randn(venv.num_envs, a_dim).astype(np.float32)
        obs, rewards, dones, infos = venv.step(actions)
        # print a compact status line
        print(f"vec step -> rewards: {rewards}")

    venv.close()

def sb3_check_env():
    print("== SB3 check_env ==")
    try:
        from stable_baselines3.common.env_checker import check_env
    except Exception as e:
        print("SB3 not available or old version:", e)
        return
    df = make_synthetic_df(T=60)
    env = StockPortfolioEnv(
        df=df,
        initial_cash=1_000_000.0,
        tech_indicator_list=["mom","rsi","vol"],
        fundamental_indicator_list=["pe","pb"],
        llm_feature_list=["llm_overall","llm_conf"],
        global_feature_list=["cpi","turbulence"],
        max_llm_risk_score=1.0,
        max_turbulence=5.0,
        allow_fractional=True,
        llm_risk_exposure_col="llm_risk_exposure",
        market_risk_exposure_col="turbulence",
    )
    check_env(env, warn=True, skip_render_check=True)
    print("check_env passed")

if __name__ == "__main__":
    np.random.seed(0)
    check_single_env()
    # DummyVec
    check_vec_env(use_subproc=False)
    # SubprocVec (safe under __main__)
    check_vec_env(use_subproc=True)
    sb3_check_env()

