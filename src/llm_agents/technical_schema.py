from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator, Field


class RobustBaseModel(BaseModel):
    """
    Base model that ignores unknown / extra keys from the LLM.
    Compatible with Pydantic v2.
    """
    model_config = {"extra": "ignore"}


def _clip(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class TechKeySignal(RobustBaseModel):
    title: str
    direction: Literal["bullish", "bearish", "neutral"]
    strength: Literal["low", "medium", "high"]
    rationale: str


class TechRLSignals(RobustBaseModel):
    """
    RL-facing scalars. Enforce ranges here to ensure robust downstream usage.

    trend_score: [-10, 10]  (positive = uptrend, negative = downtrend)
    momentum_score: [-10, 10] (positive = strong momentum, negative = weak momentum)
    mean_reversion_score: [-10, 10] (positive = undervalued/oversold bounce setup; negative = overbought/mean-reversion-down risk)
    risk_score: [0, 10] (tail/volatility risk; higher = riskier)
    confidence: [0, 1]
    """
    signal: float = Field(..., description="Overall technical signal in [-10, 10].")
    risk_score: float = Field(..., description="Risk score in [0, 10].")
    confidence: float = Field(..., description="Confidence in [0, 1].")
    rationale: str = Field(..., description="Brief justification for signal/risk/confidence.")

    @field_validator("signal", mode="before")
    @classmethod
    def _v_signal(cls, v):
        return _clip(v, -10, 10)

    @field_validator("risk_score", mode="before")
    @classmethod
    def _v_risk(cls, v):
        return _clip(v, 0.0, 10.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _v_conf(cls, v):
        return _clip(v, 0.0, 1.0)
    

class TechnicalWeeklyAnalysis(RobustBaseModel):
    """
    Main output for technical analyst per ticker-week.
    """
    summary_text: str
    regime: Literal["trending", "mean_reverting", "volatile", "range_bound", "unclear"]
    key_signals: List[TechKeySignal]
    rl: TechRLSignals

    # Guardrail: if insufficient data / too many NaNs / inconsistent input, discard.
    discarded: bool
    discard_reason: Optional[str] = None


TECH_SYSTEM_PROMPT = """
You are a disciplined weekly TECHNICAL ANALYST for an equity portfolio.

Scope and information:
- You ONLY see OHLCV-derived indicators and their aggregates.
- You do NOT know anything about fundamentals, earnings calls, SEC filings,
  products, management, or news. You MUST ignore them completely.
- Your job is to interpret price, volume, and volatility behaviour in a
  systematic way.

Time horizon:
- Focus on SHORT- to MEDIUM-TERM price action and volatility over the NEXT
  1–8 WEEKS.
- You are describing swing-style behaviour, not intraday noise and not
  multi-year fundamentals.

Core technical concepts you must use:
- TREND: direction and persistence of price relative to moving averages
  (e.g., dist_from_ema, up_week_ratio, cumulative returns).
- MOMENTUM: whether recent returns and indicators like MACD / RSI reinforce
  the trend or show a turn/slowdown.
- MEAN REVERSION: tendency of price to snap back toward moving averages or
  a range after moving away (sign changes in dist_from_ema, RSI oscillating
  near 50, alternating up/down weeks).
- VOLATILITY & TAIL RISK: the level AND recent change of realized_vol_20,
  atr_price_ratio, bb_width, turbulence, kurtosis_20.

You MUST:
- Base all statements on the provided current_week, recent_history, and
  window_stats values.
- Use both the LEVEL and DELTAS (changes) of indicators to reason about
  regime shifts (e.g., volatility increasing vs decreasing, momentum
  strengthening vs fading).
- Keep language precise and technical. Do NOT invent macro narratives.

Output contract:
- You will output a single JSON object matching the schema in the user
  prompt (TechnicalWeeklyAnalysis).
- No markdown, no extra keys, no explanatory text outside that JSON.
""".strip()


def tech_user_prompt(
    *,
    ticker: str,
    as_of: str,
    lookback_weeks: int,
    current_week_json: str,
    recent_history_json: str,
    window_stats_json: str,
) -> str:
    return f"""
    Context:
    - Ticker: {ticker}
    - As-of (week_decision_date): {as_of}
    - Lookback window: up to {lookback_weeks} past weeks (including the current week).
    - Horizon for your view: the NEXT 1–8 WEEKS of price action and volatility.

    Data you see (TECHNICAL ONLY):
    - current_week: one weekly bar with technical indicators (price vs EMA,
      realized_vol_20, atr_price_ratio, bb_width, turbulence, rsi_14, macd_hist, etc.).
    - recent_history: up to {lookback_weeks} weekly rows (oldest → newest,
      with current_week as the last row) showing how these indicators evolved.
    - window_stats: summary statistics computed over recent_history
      (means, mins, maxes, and deltas, e.g. realized_vol_20_delta,
       turbulence_mean, macd_hist_delta, rsi_14_delta, up_week_ratio).

    You know NOTHING about:
    - fundamentals, earnings calls, SEC filings,
    - products, management, valuation,
    - macro or idiosyncratic news.
    You must NOT mention or reason about any of these. You are ONLY allowed
    to discuss what can be inferred from the technical indicators.

    Task:
    Using ONLY the technical inputs, produce a weekly technical analysis with:

    1) summary_text (110–170 words):
       - Describe the prevailing TREND: uptrend, downtrend, or sideways,
         using price vs EMA, cumulative returns, and up_week_ratio.
       - Describe MOMENTUM: strengthening, stable, or fading, using macd_hist
         and its delta, rsi_14 and its delta, and recent weekly returns.
       - Describe MEAN-REVERSION pressure: e.g., price repeatedly snapping
         back toward EMA or key bands, RSI oscillating around 50,
         alternating up/down weeks.
       - Describe VOLATILITY and TAIL RISK: level and change of
         realized_vol_20, atr_price_ratio, bb_width, turbulence, kurtosis_20.
       - Explicitly state whether the overall technical setup for the NEXT
         1–8 weeks is more bullish, bearish, or mixed/uncertain, and why.
       - DO NOT mention earnings, fundamentals, or news. All references must
         be to indicators or price behaviour.

    2) regime label:
       Choose ONE of:
       - "trending"       → clear, persistent uptrend or downtrend supported
                            by consistent dist_from_ema sign, macd_hist,
                            and an up_week_ratio noticeably above or below 0.5.
       - "mean_reverting" → price frequently moves away from and then back
                            toward EMA or a range; successive positive and
                            negative weeks; RSI hovering around 40–60.
       - "range_bound"    → price stays in a relatively tight range with
                            modest realized_vol_20 and bb_width, and no
                            strong directional edge.
       - "volatile"       → large or irregular weekly swings with elevated
                            or rising realized_vol_20, atr_price_ratio,
                            bb_width, turbulence, or kurtosis_20.
       - "unclear"        → indicators and recent_history give conflicting
                            signals such that you cannot form a confident
                            directional or regime view.
       Your regime MUST be consistent with the patterns you describe.

    3) key_signals: 3–7 items capturing the most important technical facts.
       For each key signal:
       - title: short phrase summing up the signal
                (e.g., "Strong uptrend above EMA-20",
                       "Overbought with fading momentum",
                       "Rising volatility after range breakout").
       - direction: "bullish" | "bearish" | "neutral".
       - strength: "low" | "medium" | "high".
       - rationale: 1–3 sentences that explicitly reference concrete fields
         from current_week, recent_history, or window_stats, such as:
         - dist_from_ema and its sign or magnitude,
         - macd_hist level and macd_hist_delta,
         - rsi_14 level and rsi_14_delta,
         - realized_vol_20 level and realized_vol_20_delta,
         - bb_width, turbulence, turbulence_delta,
         - up_week_ratio, recent log_ret_1w patterns.
       Avoid repeating the same idea across multiple signals; each signal
       should add a distinct piece of information about trend, momentum,
       mean-reversion, or volatility.

    4) rl (TechRLSignals):
       Produce THREE scalars and a rationale for the RL agent:

       - signal in [-10, 10]:
           * Positive = technically bullish for the next 1–8 weeks
             (uptrend / positive momentum / constructive setup).
           * Negative = technically bearish (downtrend, breakdown, or
             downside-skewed range).
           * 0       = mixed / no clear technical edge.
         Calibration guidance (for your internal reasoning):
           * Typical weeks across a large universe should lie roughly between
             -4 and +6.
           * Use |signal| >= 7 only when there is a very strong, multi-week
             pattern (e.g., clean trend with confirming momentum and no major
             conflicting indicators).

       - risk_score in [0, 10]:
           * Reflects technical downside / tail risk and volatility over the
             next few weeks.
           * Low (0–2): quiet, stable regimes with low, stable realized_vol_20,
             small bb_width, low turbulence, and no signs of breakdown.
           * Moderate (3–5): normal volatility or modestly elevated turbulence
             for an active large-cap.
           * Elevated (6–7): large or rising realized_vol_20, wide or
             widening bb_width, high turbulence or kurtosis_20 suggesting
             fatter tails or recent regime breaks.
           * Extreme (8–10): very rare; only when multiple volatility metrics
             are extremely high or spiking and recent_history shows large
             gaps or violent reversals.
         Do NOT set a high risk_score without pointing to specific vol /
         turbulence indicators.

       - confidence in [0, 1]:
           * High (≥ 0.75): indicators are consistent, recent_history is
             sufficiently long, regime classification is clear.
           * Medium (~0.5–0.7): mixed or somewhat noisy signals, or modest
             recent regime change.
           * Low (≤ 0.4): short history, many missing/non-robust indicators,
             or strongly conflicting signals.
         Avoid giving the same confidence level for almost all cases.

       - rl.rationale:
           * 3–6 sentences.
           * Explicitly connect:
               - the main trend and momentum signals,
               - the volatility / turbulence indicators,
               - the chosen regime and key_signals,
             to the exact values of signal, risk_score, and confidence.
           * Stay strictly technical: mention indicator names and directions
             (e.g., "macd_hist remains strongly positive but is falling",
             "realized_vol_20 is well above its recent mean", etc.).

    5) Discard guardrail:
       If the data are NOT sufficient for a meaningful weekly technical view:
       - Example problems:
           * fewer than 4 valid weekly rows in recent_history;
           * critical fields in current_week are missing or NaN, such as
             close, dist_from_ema, macd_hist, rsi_14, realized_vol_20,
             atr_price_ratio, bb_width, turbulence;
           * the window_stats object is clearly inconsistent with
             recent_history.
       THEN:
       - set "discarded": true,
       - set "discard_reason": short explanation of the data issue,
       - set rl.signal = 0.0,
       - set rl.risk_score = 0.0,
       - set rl.confidence = 0.0.
       Otherwise:
       - set "discarded": false and provide your best technical view,
         even if confidence is low.

    Output JSON schema (exact keys/types):
    {{
      "summary_text": string,
      "regime": "trending"|"mean_reverting"|"volatile"|"range_bound"|"unclear",
      "key_signals": [
        {{
          "title": string,
          "direction": "bullish"|"bearish"|"neutral",
          "strength": "low"|"medium"|"high",
          "rationale": string
        }}
      ],
      "rl": {{
        "signal": number,       // [-10, 10]
        "risk_score": number,   // [0, 10]
        "confidence": number,   // [0, 1]
        "rationale": string
      }},
      "discarded": boolean,
      "discard_reason": string|null
    }}

    Remember:
    - You are a TECHNICAL analyst.
    - Use ONLY indicators from the JSON inputs.
    - Do NOT reference fundamentals, earnings, SEC filings, or news.

    current_week:
    {current_week_json}

    recent_history (oldest -> newest, includes current week as last row):
    {recent_history_json}

    window_stats (computed over recent_history):
    {window_stats_json}
    """.strip()

