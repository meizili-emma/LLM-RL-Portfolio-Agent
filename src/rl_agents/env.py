from collections import deque, OrderedDict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.stats import norm


class StockPortfolioEnv(gym.Env):
    """
    A stock portfolio management environment for reinforcement learning.

    This environment simulates the weekly trading of a portfolio of stocks.
    It is designed to be flexible for research, allowing for the inclusion of
    both ticker-specific and global features.

    **Action Space:**
    The action space is a continuous vector of size `num_tickers + 1`, representing
    the raw logits for the target portfolio weights (including cash). A softmax
    function is applied to these logits to get the final target weights.

    **Observation Space (State):**
    The observation is a `gymnasium.spaces.Dict` with four components:
    
    - "portfolio_state": shape (num_tickers + 1,), containing cash weight and per-ticker weights.
    - "valuation_state": shape (num_tickers, V), containing normalized technical, LLM, and
      fundamental features per ticker.
    - "trading_state": shape (num_tickers, history_len), containing a rolling history of
      normalized prices per ticker.
    - "global_state": shape (G,), containing pre-processed global features (e.g., macro, turbulence).

    **Reward Function:**
    At each step, the reward is the **log return** of total portfolio value over the period,
    penalized by:
      - an exogenous market risk term (e.g., turbulence),
      - an LLM-based risk exposure term,
      - and an endogenous risk term derived from recent downside statistics.
    """
    metadata = {"render_modes": ["ansi", "none"]}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_cash: float,
                 tech_indicator_list: list[str],
                 fundamental_indicator_list: list[str],
                 llm_feature_list: list[str],
                 global_feature_list: list[str],
                 max_llm_risk_score: float | None = None,
                 max_turbulence: float | None = None,
                 action_temperature: float = 1.0,
                 trading_state_history_len: int = 8,
                 norm_window_size: int = 26, 
                 rolling_risk_window: int = 26,
                 buy_cost_pct: float = 0.001,
                 sell_cost_pct: float = 0.001,
                 min_cash_weight: float = 0.0,
                 allow_fractional: bool = False,
                 llm_risk_exposure_col: str | None = 'llm_risk_exposure',
                 market_risk_exposure_col: str | None = 'turbulence', 
                 w_vol: float = 0.33, 
                 w_mag: float = 0.33, 
                 w_freq: float = 0.34,
                 beta: float = 1.0, 
                 gamma: float = 1.0,  
                 scale_downside: float = 0.03,
                 scale_cvar: float = 0.05,
                 scale_loss_freq: float = 1.0,
                 reward_scaling: float = 1.0,
                 render_mode: str | None = None,
                 render_topk: int = 5,
                 print_render: bool = False):
        """
        Initializes the portfolio environment.

        Args:
            df (pd.DataFrame): A pandas DataFrame with a single-index of day indicating the order of days in the df. 
                               It must contain ’date','ticker', ‘close' prices and all specified features.
                               It assumes df is sorted by ['date','ticker'] and there are exactly N rows per day. 
            initial_cash (float): The starting cash balance.
            buy_cost_pct (float): Transaction cost for buying as a percentage (e.g., 0.001 for 0.1%).
            sell_cost_pct (float): Transaction cost for selling as a percentage.
            min_cash_weight (float): Minimum cash weight to maintain in the portfolio.
            tech_indicator_list (list[str]): List of column names for technical indicators.
            global_feature_list (list[str]): List of column names for global features.
            llm_feature_list (list[str]): List of column names for LLM-derived features.
            fundamental_indicator_list (list[str]): List of column names for fundamental indicators.
            action_temperature (float): Temperature parameter for softmax action scaling.
            max_llm_risk_score (float | None): Normalization constant for the LLM risk exposure. 
                                If None and llm_risk_exposure_col is not None, it will be inferred from df; 
                                if llm_risk_exposure_col is None, this is ignored.
            max_turbulence (float | None): The maximum turbulence value for normalization, derived from training data.
            trading_state_history_len (int): Number of past days to include in the trading state.
            norm_window_size (int): Window size for rolling normalization of dynamic features.
            llm_risk_exposure_col (str | None): Column name for LLM risk exposure. If None, LLM risk penalties are disabled.
            market_risk_exposure_col (str | None): Column name for market risk exposure (e.g., turbulence). 
                                If None, market risk penalties are disabled.
            rolling_risk_window (int): Window size for calculating endogenous risk metrics.
            beta (float): Scaling factor for market risk penalty in the reward function.
            gamma (float): Scaling factor for LLM risk penalty in the reward function.
            scale_downside (float): Baseline scale for downside deviation.
            scale_cvar (float): Baseline scale for CVaR penalty.
            scale_loss_freq (float): Baseline scale for loss frequency (usually 1.0).
            w_vol (float): Weight for volatility component in endogenous risk.
            w_mag (float): Weight for magnitude component in endogenous risk.    
            w_freq (float): Weight for frequency component in endogenous risk.
            reward_scaling (float): Scaling factor for the final reward.       
        """
        super().__init__()

        # --- DATA AND FEATURE SETUP ---
        self.df = df
        self.tech_indicator_list = tech_indicator_list
        self.fundamental_indicator_list = fundamental_indicator_list
        self.llm_feature_list = llm_feature_list
        self.global_feature_list = global_feature_list
        self.tickers = self.df['ticker'].unique().tolist()
        self.num_tickers = len(self.tickers)
        
        # --- TRADING SETUP ---
        self.initial_cash = initial_cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.min_cash_weight = min_cash_weight
        self.allow_fractional = allow_fractional

        # --- RISK CALCULATION SETUP ---
        self.llm_risk_exposure_col = llm_risk_exposure_col
        self.market_risk_exposure_col = market_risk_exposure_col
        self.use_llm_risk = self.llm_risk_exposure_col is not None
        self.use_market_risk = self.market_risk_exposure_col is not None
        if self.use_llm_risk:
            if self.llm_risk_exposure_col not in self.df.columns:
                raise KeyError(f"llm_risk_exposure_col='{self.llm_risk_exposure_col}' not found in df.columns")
            if max_llm_risk_score is None:
                arr = self.df[self.llm_risk_exposure_col].to_numpy()
                if arr.size == 0 or np.all(np.isnan(arr)):
                    max_llm_risk_score = 1.0
                else:
                    max_llm_risk_score = float(np.nanmax(np.abs(arr)))
            self.max_llm_risk_score = float(max_llm_risk_score)
        else:
            self.max_llm_risk_score = 1.0  # dummy, never used

        if self.use_market_risk:
            if self.market_risk_exposure_col not in self.df.columns:
                raise KeyError(f"market_risk_exposure_col='{self.market_risk_exposure_col}' not found in df.columns")
            if max_turbulence is None:
                arr = self.df[self.market_risk_exposure_col].to_numpy()
                if arr.size == 0 or np.all(np.isnan(arr)):
                    max_turbulence = 1.0
                else:
                    max_turbulence = float(np.nanmax(np.abs(arr)))
            self.max_turbulence = float(max_turbulence)
        else:
            self.max_turbulence = 1.0  # dummy, never used

        self.rolling_risk_window = rolling_risk_window # Rolling windows for endogenous risk metrics
        self.beta = beta # Market risk penalty scaling factor
        self.gamma = gamma # LLM risk penalty scaling factor
        self.w_vol = w_vol
        self.w_mag = w_mag
        self.w_freq = w_freq
        self.reward_scaling = reward_scaling
        self.EPS_VALUE = 1e-9      # for financial values (prices, portfolio totals)
        self.EPS_WEIGHT = 1e-12
        self.scale_downside = float(scale_downside)
        self.scale_cvar = float(scale_cvar)
        self.scale_loss_freq = float(scale_loss_freq)

        # --- OBSERVATION & ACTION SPACE DEFINITION ---
        self._precompute_panel()
        self.history_len = trading_state_history_len # Length of trading state history
        portfolio_shape = (self.num_tickers + 1,) # cash_w, holdings_w
        self._valuation_dim = (
            len(tech_indicator_list)
            + len(fundamental_indicator_list)
            + len(llm_feature_list)
        )
        valuation_shape = (self.num_tickers, max(self._valuation_dim, 1))
        trading_shape = (self.num_tickers, self.history_len)
        self._global_dim = len(global_feature_list)
        global_shape = (max(self._global_dim, 1),)
        obs_space_range = 10.0 # All scaled features will be clipped to this range
        self.observation_space = spaces.Dict({
            "portfolio_state": spaces.Box(low=0, high=1, shape=portfolio_shape, dtype=np.float32),
            "valuation_state": spaces.Box(low=-obs_space_range, high=obs_space_range, shape=valuation_shape, dtype=np.float32),
            "trading_state": spaces.Box(low=-obs_space_range, high=obs_space_range, shape=trading_shape, dtype=np.float32),
            "global_state": spaces.Box(low=-obs_space_range, high=obs_space_range, shape=global_shape, dtype=np.float32),
        })
        self._action_low = -10.0
        self._action_high = 10.0
        self.action_space = spaces.Box(low=self._action_low, high=self._action_high, shape=(self.num_tickers + 1,), dtype=np.float32)
        self.action_temperature = action_temperature

        # --- STATEFUL NORMALIZATION & HISTORY SETUP ---
        self.norm_window_size = norm_window_size # Stateful rolling normalization window 
        self.dynamic_features = ['close'] + self.tech_indicator_list
        self.feature_normalizers = {
            col: PerTickerRollingNormalizer(num_tickers=self.num_tickers, window_size=self.norm_window_size)
            for col in self.dynamic_features
            }
        self.rolling_return_deque = deque(maxlen=self.rolling_risk_window)
        self.normalized_price_history_buffer = deque(maxlen=self.history_len) 

        # --- INTERNAL STATE VARIABLES ---
        self.episode = 0 
        self.day = 0
        self.cash_balance = self.initial_cash 
        self.total_value = self.initial_cash
        self.share_holdings = np.zeros(self.num_tickers, dtype=np.float32)
        self.asset_log = []
        self.reward_log = []
        self.state_log = [] 
        self.action_intent_log = []
        self.action_execution_log = []
        self.trade_shares_log = []
        self.return_log = [] 
        self.trade_costs_log: list[float] = []
        self.last_trade_cost: float = 0.0
        self.step_in_episode: int = 0
        self.reset()

        # --- RENDERING SETUP ---
        self.render_mode  = (render_mode or "none")
        self.render_topk  = int(render_topk)
        self.print_render = bool(print_render)
        try: # fast date lookup if you have df['date']
            self._dates = self.df['date'].drop_duplicates().tolist()
        except Exception:
            self._dates = None
    
    def _precompute_panel(self):
        """ Build fast NumPy tensors:
        - _price_ts: (T, N)
        - _val_raw_ts: (T, N, V) for valuation features (tech + llm + fundamental)
        - _global_ts: (T, G)
        - _llm_risk_exposure_ts: (T, N)
        - _turbulence_ts: (T,)
        Assumes df is sorted by ['date','ticker'] and index is 0..T-1 with exactly N rows per step."""
        tech = list(self.tech_indicator_list)
        llm  = list(self.llm_feature_list)
        fund = list(self.fundamental_indicator_list)
        all_val_feats = tech + llm + fund
        G = len(self.global_feature_list)
        V = len(tech) + len(llm) + len(fund)

        # stable ticker order (whatever is in this slice)
        tickers = list(self.df["ticker"].unique())
        tickers.sort()
        self.tickers = tickers
        self.num_tickers = len(tickers)
        N = self.num_tickers

         # ----- 1. clean up duplicates on (date, ticker) -----
        df = self.df.copy()
        df = df.sort_values(["date", "ticker"])
        df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

        # ----- 2. keep only dates with full ticker coverage -----
        counts = df.groupby("date")["ticker"].nunique()
        full_dates = counts[counts == N].index
        if len(full_dates) == 0:
            raise ValueError(
                f"No dates with full coverage for {N} tickers. "
                f"Got group sizes:\n{counts.describe()}"
            )

        df = df[df["date"].isin(full_dates)].copy()
        df = df.sort_values(["date", "ticker"])

         # canonical date order
        dates = pd.Index(sorted(full_dates))
        T = len(dates)
        self.max_day = T

        # we also keep these dates around for logging in info["meta"]["date"]
        self._dates = list(dates)

        # ----- 3. indexing for valuation features -----
        self._val_index = {f: j for j, f in enumerate(all_val_feats)}

        # ----- 4. allocate tensors -----
        self._price_ts = np.zeros((T, N), dtype=np.float32)
        self._val_raw_ts = np.zeros((T, N, max(1, V)), dtype=np.float32)
        self._global_ts = np.zeros((T, max(1, G)), dtype=np.float32)
        self._llm_risk_exposure_ts = np.zeros((T, N), dtype=np.float32)
        self._turbulence_ts = np.zeros((T,), dtype=np.float32)

        # ----- 5. fill tensors, one date at a time -----
        for t, d in enumerate(dates):
            block = df[df["date"] == d]

            # enforce ticker order and full coverage
            block = (
                block
                .set_index("ticker")
                .reindex(tickers)
            )

            if block["close"].isna().any():
                raise ValueError(
                    f"NaNs in 'close' after reindexing for date={d}. "
                    "Check data prep for missing prices."
                )

            # prices
            self._price_ts[t] = block["close"].to_numpy(np.float32)

            # valuation features
            for f in all_val_feats:
                if f not in block.columns:
                    raise KeyError(f"Valuation feature '{f}' not found in df columns.")
                self._val_raw_ts[t, :, self._val_index[f]] = block[f].to_numpy(np.float32)

            # global features: same for all tickers on a given date
            if G > 0:
                first_row = block.iloc[0]
                self._global_ts[t] = first_row[self.global_feature_list].to_numpy(np.float32)

            # risk exposures
            if self.use_llm_risk:
                if self.llm_risk_exposure_col not in block.columns:
                    raise KeyError(
                        f"llm_risk_exposure_col='{self.llm_risk_exposure_col}' not found in df columns."
                    )
                self._llm_risk_exposure_ts[t] = block[self.llm_risk_exposure_col].to_numpy(np.float32)

            if self.use_market_risk:
                if self.market_risk_exposure_col not in block.columns:
                    raise KeyError(
                        f"market_risk_exposure_col='{self.market_risk_exposure_col}' not found in df columns."
                    )
                first_row = block.iloc[0]
                self._turbulence_ts[t] = np.float32(first_row[self.market_risk_exposure_col])

        # finally, store the cleaned panel back (optional but nice for debugging)
        self.df = df.reset_index(drop=True)

    def _vec(self, feature_name: str, day: int) -> np.ndarray:
        """Return the (N,) vector for a valuation feature at a specific day using precomputed tensors.
        Only valid for features in tech/llm/fund lists. """
        j = self._val_index[feature_name]
        return self._val_raw_ts[day, :, j]
    
    def _assemble_state(self) -> OrderedDict:
        """Orchestrates the fetching, scaling, and assembly of the observation dict."""
    
        portfolio_state = self._get_scaled_portfolio_state()
        valuation_state = self._get_scaled_valuation_state()
        trading_state = self._get_scaled_trading_state()
        global_state = self._get_scaled_global_state()
        
        observation = OrderedDict([
            ("portfolio_state", portfolio_state.astype(np.float32)),
            ("valuation_state", valuation_state.astype(np.float32)),
            ("trading_state", trading_state.astype(np.float32)),
            ("global_state", global_state.astype(np.float32)),
        ])
        return observation

    # --- Helper methods for scaling and assembly ---
    def _get_scaled_portfolio_state(self):
        """ Scales cash and holdings to weights."""
        prices = self._price_ts[self.day]
        if self.total_value > 1e-9:
            cash_weight = self.cash_balance / self.total_value
            holdings_weights = (self.share_holdings * prices) / self.total_value
        else:
            cash_weight = 1.0
            holdings_weights = np.zeros(self.num_tickers, dtype=np.float32)
        return np.append(np.float32(cash_weight), holdings_weights.astype(np.float32))
    
    def _get_scaled_valuation_state(self):
        # tech: rolling z-score per ticker
        if self._valuation_dim == 0:
            return np.zeros((self.num_tickers, 1), dtype=np.float32)
        
        # tech: rolling z-score per ticker
        tech_scaled = []
        for col in self.tech_indicator_list:
            v = self.feature_normalizers[col].update_and_normalize(self._vec(col, self.day))
            tech_scaled.append(np.clip(v, -10, 10).astype(np.float32))
        tech_scaled = np.stack(tech_scaled, axis=0) if tech_scaled else np.empty((0, self.num_tickers), dtype=np.float32)

        # llm: clip only
        llm_scaled = []
        for col in self.llm_feature_list:
            v = self._vec(col, self.day)
            llm_scaled.append(np.clip(v, -10, 10).astype(np.float32))
        llm_scaled = np.stack(llm_scaled, axis=0) if llm_scaled else np.empty((0, self.num_tickers), dtype=np.float32)

        # fundamental: cross-sectional rank (per feature)
        fund_scaled = []
        for col in self.fundamental_indicator_list:
            v = self._vec(col, self.day)
            fund_scaled.append(cross_sectional_rank_scale(v).astype(np.float32))
        fund_scaled = np.stack(fund_scaled, axis=0) if fund_scaled else np.empty((0, self.num_tickers), dtype=np.float32)

        # concat along feature axis and transpose → [tickers, features]
        all_feats = (
            np.concatenate([tech_scaled, llm_scaled, fund_scaled], axis=0)
            if any([tech_scaled.size, llm_scaled.size, fund_scaled.size])
            else np.empty((0, self.num_tickers), dtype=np.float32)
        )
        return all_feats.T  # (N, V)

    def _get_scaled_trading_state(self) -> np.ndarray:
        """Assembles the trading state directly from the normalized history buffer."""
        current_prices_raw = self._price_ts[self.day] 
        current_prices_normalized = self.feature_normalizers['close'].update_and_normalize(current_prices_raw)
        self.normalized_price_history_buffer.append(current_prices_normalized)
        if len(self.normalized_price_history_buffer) < self.history_len:
            # Handle warm-up period by padding with zeros
            return np.zeros((self.num_tickers, self.history_len), dtype=np.float32)
        # np.array() converts deque of arrays -> (history_len, num_tickers)
        # .T transposes to -> [num_tickers, history_len]
        scaled_history = np.array(self.normalized_price_history_buffer).T
        return np.clip(scaled_history, -10.0, 10.0).astype(np.float32)

    def _get_scaled_global_state(self):
        # Assumes global features in df are pre-processed to be pct_change
        # We only need one row since they are the same for all tickers
        if self._global_dim == 0:
            return np.zeros((1,), dtype=np.float32)
        g = self._global_ts[self.day].astype(np.float32)
        g = np.clip(g, self._action_low, self._action_high)
        return g.astype(np.float32)
    
    def reset(self, *, seed=None, options=None) -> tuple[OrderedDict, dict]:
        super().reset(seed=seed)
        
        # Determine the first day the agent can actually start trading
        warmup_len = self.norm_window_size + self.history_len - 2
        if warmup_len >= self.max_day:
            raise ValueError("Dataset is too short for the required warm-up period.")
        
        # reset stateful buffers
        self.rolling_return_deque.clear()
        self.normalized_price_history_buffer.clear()
        for normalizer in self.feature_normalizers.values():
            normalizer.reset()
        
        # warmup normalizers & history
        for i in range(warmup_len):
            _ = self.feature_normalizers['close'].update_and_normalize(self._price_ts[i])
            for col in self.tech_indicator_list:
                _ = self.feature_normalizers[col].update_and_normalize(self._vec(col, i))
            # After the normalizer window is full, start populating the trading state history
            if i >= self.norm_window_size - 1:
                cur_norm_prices = self.feature_normalizers['close'].normalize(self._price_ts[i])
                self.normalized_price_history_buffer.append(cur_norm_prices.astype(np.float32))
        
         # init episode state
        self.episode += 1
        self.day = warmup_len 
        self.cash_balance = np.float32(self.initial_cash)                   
        self.total_value  = np.float32(self.initial_cash)                  
        self.share_holdings = np.zeros(self.num_tickers, dtype=np.float32) 

        self.asset_log = [np.float32(self.total_value)] 
        self.return_log.clear() 
        self.reward_log.clear()
        self.action_execution_log.clear()
        self.action_intent_log.clear()
        self.trade_shares_log.clear()
        self.trade_costs_log.clear()
        self.last_trade_cost = 0.0
        self.step_in_episode = 0

        observation = self._assemble_state()
        self.state_log = [observation]
        return observation, {}

    def _execute_rebalancing_trades(self, target_weights: np.ndarray):
        """
        Rebalance toward target weights (cash + N assets) with:
        - min cash weight enforcement (self.min_cash_weight)
        - sells first, then scaled buys
        - integer or fractional shares (self.allow_fractional)
        - transaction costs on both sides
        - numerically safe post-trade weights (>=0, sum=1)
        """
        def _safe_div(a, b, eps=self.EPS_WEIGHT):
            return a / max(b, eps)
        
        current_prices = self._price_ts[self.day]
        N = self.num_tickers
        # ---- pre-trade value ----
        total_trade_cost = 0.0
        pre_trade_value = float(self.cash_balance + np.dot(self.share_holdings, current_prices))
        if pre_trade_value <= self.EPS_VALUE: 
            self.last_trade_cost = 0.0
            return
        # --------------- Current weights before trade ----------------
        pre_assets_value = self.share_holdings * current_prices
        pre_weights_assets = pre_assets_value / pre_trade_value
        pre_sell_weights = np.concatenate(([self.cash_balance / pre_trade_value], pre_weights_assets)).astype(np.float32)
        
        weight_diff = target_weights - pre_sell_weights
        trade_shares_count = np.zeros(self.num_tickers, dtype=np.float32) # For logging purposes
        # --------------- SELL PASS ----------------
        for i in range(self.num_tickers):
            if weight_diff[i + 1] < 0:
                sell_dollar_value = -weight_diff[i + 1] * pre_trade_value
                if self.allow_fractional:
                    shares_to_sell = min(sell_dollar_value/ max(current_prices[i], self.EPS_VALUE), self.share_holdings[i])
                else:
                    shares_to_sell = min(np.floor(sell_dollar_value/ max(current_prices[i], self.EPS_VALUE)), self.share_holdings[i])
                if shares_to_sell > 0:
                    gross_proceeds = float(shares_to_sell * current_prices[i])
                    proceeds = gross_proceeds * (1 - self.sell_cost_pct)
                    total_trade_cost += float(gross_proceeds - proceeds)

                    self.cash_balance +=  np.float32(proceeds)
                    self.share_holdings[i] -= np.float32(shares_to_sell)
                    trade_shares_count[i] -= np.float32(shares_to_sell)

        # --------------- State after sells ----------------
        post_sell_value = float(self.cash_balance + float(np.dot(self.share_holdings, current_prices)))
        cur_after_sell_assets_w = (self.share_holdings * current_prices) / max(post_sell_value, 1e-9)
        cur_after_sell_weights = np.concatenate(([self.cash_balance / max(post_sell_value, 1e-9)], cur_after_sell_assets_w))
        # --------------- BUY PLAN (gap after sells) ----------------
        gap = target_weights - cur_after_sell_weights
        desired_buys = {} # Store {ticker_idx: pre_cost_dollar_amount}
        total_cash_required_post_cost = 0.0
        for i in range(N):
            if gap[i + 1] > 0:
                pre_cost_dollar = gap[i + 1] * post_sell_value
                desired_buys[i] = pre_cost_dollar
                total_cash_required_post_cost += pre_cost_dollar * (1 + self.buy_cost_pct)

        # Reserve target cash (based on post-sell value)
        target_cash_reserve = float(target_weights[0] * post_sell_value)
        cash_available_for_buys = max(0.0, float(self.cash_balance) - target_cash_reserve)

        allocation_ratio = 0.0
        if total_cash_required_post_cost > 0.0:
            allocation_ratio = min(1.0, cash_available_for_buys / total_cash_required_post_cost)

        # --------------- BUY EXECUTION ----------------
        for idx, pre_cost_dollar in desired_buys.items():
            spendable = pre_cost_dollar * (1 + self.buy_cost_pct) * allocation_ratio
            price_with_cost = current_prices[idx] * (1 + self.buy_cost_pct)
            if self.allow_fractional:
                shares_to_buy = spendable / max(price_with_cost, self.EPS_VALUE)
            else:
                shares_to_buy = np.floor(spendable / max(price_with_cost, self.EPS_VALUE))
            if shares_to_buy > 0:
                cost_of_buy = shares_to_buy * price_with_cost
                if self.cash_balance >= cost_of_buy: # Final check to prevent rounding errors from overspending
                    gross_notional = float(shares_to_buy * current_prices[idx])
                    total_trade_cost += float(cost_of_buy - gross_notional)
                    self.cash_balance -= np.float32(cost_of_buy)
                    self.share_holdings[idx] += np.float32(shares_to_buy)
                    trade_shares_count[idx] = np.float32(shares_to_buy)
        
        # --------------- Post-trade weights (safe & renormalized) ----------------
        post_trade_value = float(self.cash_balance + np.sum(self.share_holdings * current_prices))
        asset_val = self.share_holdings * current_prices
        asset_w = _safe_div(asset_val, post_trade_value).astype(np.float32)
        cash_w = np.float32(_safe_div(self.cash_balance, post_trade_value))
        w = np.concatenate(([cash_w], asset_w))
        w = np.clip(w, 0.0, None, dtype=np.float32)
        s = float(w.sum())
        w = (w / s).astype(np.float32) if s > 0.0 else np.concatenate(([np.float32(1.0)], np.zeros_like(asset_w)))
        if getattr(self, "min_cash_weight", 0.0) > 0.0 and (w[0] + self.EPS_WEIGHT) < self.min_cash_weight:
            # replace with your logger as needed:
            # print(f"[warn] cash below floor: got {w[0]:.4f}, floor {self.min_cash_weight:.4f} @ day {self.day}")
            pass
        self.action_execution_log.append(w)
        self.trade_shares_log.append(trade_shares_count.astype(np.float32))
        self.last_trade_cost = float(total_trade_cost)

    def step(self, action: np.ndarray):
        """Executes one weekly step in the environment, including trading, reward calculation,
        and state assembly, following a chronologically correct causal chain."""
        # --- 0. Sanitize and clip action for numerical stability ---
        # SB3 should already respect action_space, but we make it explicit and robust
        # to lists / torch tensors in case of non-SB3 callers.
        step_idx = self.step_in_episode
        pre_trade_value = float(self.total_value)
        current_prices = self._price_ts[self.day]

        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)
        else:
            action = action.astype(np.float32, copy=False)
        # Clip logits into a safe range before softmax.
        action = np.clip(action, self._action_low, self._action_high)
        raw_logits = np.asarray(action, dtype=np.float32)
        intent_weights = self._softmax(raw_logits / self.action_temperature)
        target_weights = self._project_with_min_cash(intent_weights)
        self.action_intent_log.append(intent_weights.astype(np.float32))
        self._execute_rebalancing_trades(target_weights)
        # --- 2. Calculate Exogenous (External) Risk Context at Week t ---
        # These calculations use information available only at the end of week t.
        post_trade_value_t = float(self.cash_balance + np.sum(self.share_holdings * current_prices))
        post_trade_weights_t = (self.share_holdings * current_prices) / max(post_trade_value_t, self.EPS_VALUE)

        normalized_llm_risk = 0.0
        normalized_turbulence = 0.0
        raw_llm_risk_t = 0.0
        raw_turbulence_t = 0.0
        if self.use_llm_risk:
            raw_llm_risk_t = float(np.dot(post_trade_weights_t, self._llm_risk_exposure_ts[self.day]))
            normalized_llm_risk = float(np.clip(raw_llm_risk_t / max(self.max_llm_risk_score, self.EPS_VALUE), 0, 1))
        if self.use_market_risk:
            raw_turbulence_t = float(self._turbulence_ts[self.day])
            normalized_turbulence = float(np.clip(raw_turbulence_t / max(self.max_turbulence, self.EPS_VALUE), 0, 1))
            
        # --- 3. Advance Time and Measure Outcome at Week t+1 ---
        self.day += 1
        terminated = self.day >= self.max_day - 1
        new_prices = self._price_ts[self.day]
        self.total_value = np.float32(self.cash_balance + np.sum(self.share_holdings * new_prices))
        self.asset_log.append(np.float32(self.total_value))
        log_return = np.float32(np.log(max(self.total_value, self.EPS_VALUE) / max(pre_trade_value, self.EPS_VALUE)))
        self.return_log.append(log_return)
        self.rolling_return_deque.append(log_return)
        # --- 4. Calculate Endogenous (Behavioral) Risk from Recent Performance ---
        downside_dev = self._calculate_downside_deviation()
        cvar = self._calculate_cvar()
        loss_freq = self._calculate_loss_frequency()
        if self.scale_downside > 0:
            downside_norm = downside_dev / self.scale_downside
        else:
            downside_norm = 0.0

        if self.scale_cvar > 0:
            cvar_norm = cvar / self.scale_cvar
        else:
            cvar_norm = 0.0

        if self.scale_loss_freq > 0:
            loss_freq_norm = loss_freq / self.scale_loss_freq
        else:
            loss_freq_norm = 0.0
        endogenous_risk = float(
            self.w_vol * downside_norm
            + self.w_mag * cvar_norm
            + self.w_freq * loss_freq_norm
        )
        # --- 5. Calculate Final Reward and Prepare Logs ---
        market_risk_penalty = self.beta * normalized_turbulence * endogenous_risk
        llm_risk_penalty = self.gamma * normalized_llm_risk
        reward = float((log_return - market_risk_penalty - llm_risk_penalty) * self.reward_scaling)
        self.reward_log.append(np.float32(reward))
        self.trade_costs_log.append(float(self.last_trade_cost))
        info = {
            "meta": {
                "episode_id": int(self.episode),
                "step_idx": int(step_idx),
                "date_idx": int(self.day),
                "date": self._dates[self.day] if getattr(self, "_dates", None) and 0 <= self.day < len(self._dates) else None,
            },
            "reward_components": {
                "log_return": log_return,
                "downside_deviation": downside_dev,
                "cvar": cvar,
                "loss_frequency": loss_freq,
                "downside_norm": downside_norm,
                "cvar_norm": cvar_norm,
                "loss_freq_norm": loss_freq_norm,
                "endogenous_risk": endogenous_risk,
                "raw_turbulence": raw_turbulence_t,
                "normalized_turbulence": normalized_turbulence,
                "raw_llm_risk": raw_llm_risk_t,
                "normalized_llm_risk": normalized_llm_risk,
                "market_risk_penalty": market_risk_penalty,
                "llm_risk_penalty": llm_risk_penalty,
                "total_reward": reward,
            },
            "trading": {
                "intent_weights": intent_weights.astype(np.float32),
                "executed_weights": self.action_execution_log[-1] if self.action_execution_log else None,
                "trade_shares": self.trade_shares_log[-1] if self.trade_shares_log else None,
                "trade_cost": float(self.last_trade_cost),
                "cash_balance": float(self.cash_balance),
                "pre_trade_value": pre_trade_value,
                "post_trade_value_t": post_trade_value_t,
                "nav_end_of_period": float(self.total_value),
            },
        }
        # --- 6. Assemble Next State and Return ---
        observation = self._assemble_state()
        self.state_log.append(observation)
        # Optional: numeric sanity checks during development
        # Comment this out in production if it becomes a bottleneck.
        self._check_numerics(observation, float(reward))
        if terminated:
            info["episode_summary"] = self.get_episode_summary()
        self.step_in_episode += 1
        return observation, reward, terminated, False, info # obs, reward, terminated, truncated, info
    
    def _calculate_downside_deviation(self) -> float:
        """Calculates the volatility of losses."""
        if len(self.rolling_return_deque) < 2:
            return 0.0
        returns = np.array(self.rolling_return_deque)
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        return np.std(negative_returns)
    
    def _calculate_cvar(self, percentile: int = 20) -> float:
        """
        Robust Conditional Value at Risk (Expected Shortfall).

        Returns:
        A non-negative scalar equal to the average magnitude of tail losses
        (in log-return units) over the specified lower percentile.
        If there are no negative returns in the window, returns 0.0.
        """
        if len(self.rolling_return_deque) < self.rolling_risk_window:
            return 0.0

        returns = np.array(self.rolling_return_deque)

        # 1. Keep only negative returns (true losses)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0  # No downside in window → no CVaR penalty

        # 2. Compute VAR on negative returns only
        var_threshold = np.percentile(negative_returns, percentile)

        # 3. Tail losses: returns worse (more negative) than VaR
        tail_losses = negative_returns[negative_returns <= var_threshold]
        if len(tail_losses) == 0:
            return 0.0

        # 4. CVaR must be positive: magnitude of average downside
        return float(-np.mean(tail_losses))

    def _calculate_loss_frequency(self) -> float:
        """Calculates the proportion of losing periods in the window."""
        if not self.rolling_return_deque:
            return 0.0
        returns = np.array(self.rolling_return_deque)
        return np.sum(returns < 0) / len(returns)
    
    def close(self):
        pass 

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for a vector x."""
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum(axis=0)
        return out.astype(np.float32)    
    
    def _project_with_min_cash(self, w):
        """
        Project raw intent weights onto the simplex, enforcing a soft minimum cash weight.
        This guarantees the *intended* cash allocation is ≥ min_cash_weight.
        Executed post-trade cash may differ slightly due to discrete shares and costs.
        """
        # enforce simplex + min cash
        w = np.maximum(w, self.EPS_WEIGHT)
        w = w / w.sum()
        if self.min_cash_weight > 0.0 and w[0] < self.min_cash_weight:
            delta = self.min_cash_weight - w[0]
            w[0] = self.min_cash_weight
            w[1:] = np.maximum(w[1:] - delta * (w[1:] / (w[1:].sum() + self.EPS_WEIGHT)), self.EPS_WEIGHT)
            w = w / w.sum()
        return w.astype(np.float32)
    
    def get_episode_summary(self):
        """ Returns a dictionary containing the performance summary of the completed episode. """
        sharpe = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        episode_steps = len(self.return_log)
        # total number of non-zero trades across all assets & steps
        trade_count = 0
        for ts in self.trade_shares_log:
            if ts is not None:
                trade_count += int(np.count_nonzero(ts))
        total_trade_cost = float(sum(self.trade_costs_log)) if self.trade_costs_log else 0.0
        avg_trade_cost_per_step = float(total_trade_cost / episode_steps) if episode_steps > 0 else 0.0
        summary = {
            "episode_id": int(self.episode),
            "final_portfolio_value": self.asset_log[-1],
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "episode_steps": int(episode_steps),
            "trade_count": int(trade_count),
            "total_trade_cost": total_trade_cost,
            "avg_trade_cost_per_step": avg_trade_cost_per_step,
            "value_history": np.array(self.asset_log, dtype=np.float32).tolist(), 
            "return_history":  np.array(self.return_log, dtype=np.float32).tolist(),
            "reward_history": np.array(self.reward_log, dtype=np.float32).tolist(),
            "intent_history": [np.array(arr, dtype=np.float32).tolist() for arr in self.action_intent_log],
            "execution_result_history": [np.array(arr, dtype=np.float32).tolist() for arr in self.action_execution_log],
            "trade_shares_history": [np.array(arr, dtype=np.float32).tolist() for arr in self.trade_shares_log]
        }
        return summary
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculates the annualized Sharpe ratio from the episode's raw returns. """
        returns = np.array(self.return_log)
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        sharpe = (returns.mean() - risk_free_rate) / returns.std()
        annualized_sharpe = sharpe * np.sqrt(52)
        return annualized_sharpe
    
    def _calculate_max_drawdown(self):
        """Calculates the maximum drawdown from the portfolio value history, 
         Maximum drawdown is the largest percentage drop from a peak to a subsequent trough."""
        asset_series = pd.Series(self.asset_log)
        running_max = asset_series.cummax()
        drawdown = (asset_series - running_max) / running_max
        return drawdown.min()

    def _current_weights_vector(self) -> np.ndarray:
        """Return (cash + per-ticker) weights at current prices."""
        prices = self._price_ts[self.day]
        total  = float(self.cash_balance + np.sum(self.share_holdings * prices))
        if total <= 1e-12:
            w = np.zeros(self.num_tickers + 1, dtype=np.float32)
            w[0] = 1.0
            return w
        asset_w = (self.share_holdings * prices) / total
        cash_w  = self.cash_balance / total
        return np.concatenate(([cash_w], asset_w.astype(np.float32))).astype(np.float32)

    def _format_topk_weights(self, w: np.ndarray, k: int) -> str:
        labels = ["CASH"] + self.tickers
        k = min(k, len(w))
        idx = np.argsort(-np.abs(w))[:k]
        return "  ".join(f"{labels[i]}:{w[i]:+.3f}" for i in idx)
    
    def _check_numerics(self, obs: OrderedDict, reward: float):
        """
        Debug helper: raise AssertionError if obs or reward contain NaN/inf.
        Call this only in development; you can disable it later for speed.
        """
        if not np.isfinite(reward):
            raise AssertionError(f"Non-finite reward encountered: {reward}")
        for k, v in obs.items():
            arr = np.asarray(v)
            if not np.all(np.isfinite(arr)):
                raise AssertionError(f"Non-finite values in obs['{k}']: min={np.nanmin(arr)}, max={np.nanmax(arr)}")

    def render(self, mode: str | None = None):
        """ Minimal text render for sanity checks (no plotting).
            Shows: step, NAV, last reward, cash weight, top-K weights.
            Returns the string when mode='ansi'; otherwise no-op.
        """
        mode = (mode or self.render_mode or "none")
        if mode != "ansi":
            return  # no-op

        # Step index (logged steps; safe if empty)
        step_idx = max(0, len(self.asset_log) - 1)

        # Day & optional date
        day_i = int(self.day)
        date_str = ""
        if getattr(self, "_dates", None) and 0 <= day_i < len(self._dates):
            date_str = f" ({self._dates[day_i]})"

        # NAV and last reward (fallbacks if logs are empty)
        if self.asset_log:
            nav = float(self.asset_log[-1])
        else:
            nav = float(self.cash_balance + np.sum(self.share_holdings * self._price_ts[self.day]))
        last_r = float(self.reward_log[-1]) if self.reward_log else 0.0

        # Weights & top-K
        w = self._current_weights_vector()
        cash_w = float(w[0])
        topk_str = self._format_topk_weights(w, getattr(self, "render_topk", 5))

        # Assemble snapshot
        text = (
            f"[Step {step_idx:>4}] Day {day_i}{date_str}\n"
            f"  NAV: {nav:,.2f}   LastReward: {last_r:+.6f}   CashW: {cash_w:.3f}\n"
            f"  Top-{getattr(self, 'render_topk', 5)} weights: {topk_str}"
        )
        if getattr(self, "print_render", False):
            print(text)
        return text

    @classmethod
    def build_vec_env(
        cls,
        env_kwargs: dict,
        n_envs: int = 1,
        base_seed: int = 0,
        use_subproc: bool | None = None,
        monitor_dir: str | None = None,
        norm_reward: bool = False,
        clip_reward: float = 10.0,
        monitor_info_keys: tuple[str, ...] | None = None,
        ):
        """
        Create a vectorized environment for Stable-Baselines3 with:
        - robust, per-env seeding (python/numpy/torch + action/obs spaces + gymnasium reset(seed))
        - per-env Monitor CSVs (episode returns/length/time + selected episode_summary fields)
        - EpisodeSummaryKeys wrapper that flattens episode_summary[...] into top-level info at 'done'
        - optional reward normalization via VecNormalize (obs already normalized in this env)

        Args:
        env_kwargs: kwargs passed to StockPortfolioEnv(...)
        n_envs: number of parallel envs
        base_seed: master seed; env k uses (base_seed + k)
        use_subproc: True -> SubprocVecEnv, False -> DummyVecEnv, None -> auto (Subproc if n_envs>1)
        monitor_dir: if set, writes one Monitor csv per env under this directory
        norm_reward: wrap VecNormalize(norm_obs=False, norm_reward=True)
        clip_reward: reward clipping for VecNormalize
        monitor_info_keys: extra keys (besides defaults) to pull from info at episode end

        Returns:
        vec_env: DummyVecEnv or SubprocVecEnv (possibly wrapped by VecNormalize)
        """
        import random
        from pathlib import Path
        import numpy as np
        import gymnasium as gym
        try:
            import torch
            _has_torch = True
        except Exception:
            _has_torch = False

        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

        # -------- small wrappers (defined here to keep all logic local) --------
        class _MetaInfo(gym.Wrapper):
            """Injects {'env_id', 'seed_used'} into info each step (Monitor reads them at episode end)."""
            def __init__(self, env: gym.Env, env_id: int, seed_used: int):
                super().__init__(env)
                self._meta = {"env_id": int(env_id), "seed_used": int(seed_used)}

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                info.update(self._meta)
                return obs, info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                info.update(self._meta)
                return obs, reward, terminated, truncated, info

        class _EpisodeSummaryKeys(gym.Wrapper):
            """
            On the terminal step, copies selected fields from info['episode_summary'] to top-level info
            so Monitor(info_keywords=...) can write them.
            """
            def __init__(self, env: gym.Env, keys: tuple[str, ...]):
                super().__init__(env)
                self._keys = tuple(keys)

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                if (terminated or truncated) and isinstance(info.get("episode_summary"), dict):
                    es = info["episode_summary"]
                    for k in self._keys:
                        if k in es:
                            info[k] = es[k]
                return obs, reward, terminated, truncated, info

        # -------- config & defaults --------
        if use_subproc is None:
            use_subproc = (n_envs > 1)

        # episode_summary fields we almost always want in Monitor CSVs
        default_episode_keys = (
            "final_portfolio_value",
            "sharpe_ratio",
            "max_drawdown",
            "episode_id",
            "episode_steps",
            "trade_count",
            "total_trade_cost",
            "avg_trade_cost_per_step",
        )
        # also log which env/seed produced each episode row
        default_meta_keys = ("env_id", "seed_used")

        info_keys = tuple(dict.fromkeys(  # de-dup while keeping order
            (monitor_info_keys or ()) + default_episode_keys + default_meta_keys).keys())

        mon_dir = None
        if monitor_dir is not None:
            mon_dir = Path(monitor_dir)
            mon_dir.mkdir(parents=True, exist_ok=True)

        def _seed_everything(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            if _has_torch:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        def make_env_fn(rank: int):
            """Factory → thunk; SB3 expects a no-arg callable that returns an Env."""
            env_seed = int(base_seed + rank)

            def _thunk():
                # 1) create env
                env = cls(**env_kwargs)

                # 2) global PRNGs + spaces
                _seed_everything(env_seed)
                try:
                    env.action_space.seed(env_seed); 
                    env.observation_space.seed(env_seed)
                except Exception:
                    pass

                # 3) reset with seed for gymnasium reproducibility
                try:
                    env.reset(seed=env_seed)
                except TypeError:
                    env.reset()

                # 4) add meta tags every step (so Monitor can record them)
                env = _MetaInfo(env, env_id=rank, seed_used=env_seed)

                # 5) lift episode_summary[...] to top-level info at episode end
                episode_keys_for_lift = tuple(k for k in info_keys if k not in default_meta_keys)
                env = _EpisodeSummaryKeys(env, keys=episode_keys_for_lift)

                # 6) per-env Monitor CSV (episode return/len/time + our info_keys)
                if mon_dir is not None:
                    fname = mon_dir / f"{cls.__name__}_env{rank}.monitor.csv"
                    env = Monitor(env, filename=str(fname), info_keywords=info_keys)

                return env
            return _thunk

        thunks = [make_env_fn(rank) for rank in range(n_envs)]
        vec_env = SubprocVecEnv(thunks) if (use_subproc and n_envs > 1) else DummyVecEnv(thunks)

        # Optional reward normalization (obs already normalized/clipped inside the env)
        if norm_reward:
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=clip_reward)

        # Attach a tiny provenance note (handy when saving models & logs together)
        vec_env._spenv_meta = {
            "env_class": cls.__name__,
            "n_envs": int(n_envs),
            "base_seed": int(base_seed),
            "backend": "SubprocVecEnv" if (use_subproc and n_envs > 1) else "DummyVecEnv",
            "monitor_dir": str(mon_dir) if mon_dir else None,
            "norm_reward": bool(norm_reward),
            "clip_reward": float(clip_reward),
            "monitor_info_keys": info_keys,
            }
        return vec_env


class PerTickerRollingNormalizer:
    """
    Normalizes features on a per-ticker basis using a fixed-size rolling window.
    """
    def __init__(self, num_tickers: int, window_size: int):
        self.num_tickers = num_tickers
        self.window_size = window_size 
        self.buffers = [deque(maxlen=self.window_size) for _ in range(num_tickers)]
        self.EPS_WEIGHT = 1e-12

    def _update_buffers(self, x: np.ndarray):
        """Internal method to update the deques with new data."""
        for i in range(self.num_tickers):
            self.buffers[i].append(x[i])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes a vector of observations using the current buffer statistics.
        This method DOES NOT update the buffers. It is "read-only."
        """
        normalized_x = np.zeros(self.num_tickers, dtype=np.float32)
        for i in range(self.num_tickers):
            if not self.buffers[i]: # Handle empty buffer at the start
                normalized_x[i] = np.float32(x[i]); continue
            m = np.mean(self.buffers[i])
            s = np.std(self.buffers[i])
            normalized_x[i] = np.float32((x[i] - m) / (s + self.EPS_WEIGHT))
        return normalized_x

    def update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        First updates the buffers with new data, then returns the normalized values.
        This is the standard method for processing the current day's data.
        """
        self._update_buffers(x)
        return self.normalize(x)
    
    def reset(self):
        """Clears the internal buffers to prepare for a new episode."""
        self.buffers = [deque(maxlen=self.window_size) for _ in range(self.num_tickers)]


def cross_sectional_rank_scale(data: np.ndarray) -> np.ndarray:
    """
    Performs cross-sectional ranking and scales to a Gaussian distribution.
    :param data: A 1D numpy array of feature values for all tickers at one timestep.
    :return: A 1D numpy array of Gauss-ranked values.
    """
    # Compute ranks (from 0 to N-1)
    ranks = data.argsort().argsort()
    # Scale ranks to [0, 1]
    scaled_ranks = (ranks + 1) / (len(ranks) + 1)
    # Apply inverse CDF of a normal distribution (Gauss Rank)
    return norm.ppf(scaled_ranks) # Scale to avoid inf at the edges


