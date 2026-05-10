from __future__ import annotations

from typing import List, Literal
from pydantic import Field, confloat, field_validator

from src.llm_agents.utils import RobustBaseModel


class NewsEvent(RobustBaseModel):
    """
    A single materially relevant news event for the ticker during the week.
    """
    title: str = Field(
        ...,
        description="Short label for the event (e.g. 'DOJ launches investigation', 'Q3 earnings beat').",
    )
    category: Literal[
        "earnings",
        "guidance",
        "product",
        "mna",
        "litigation",
        "regulatory",
        "macro",
        "industry",
        "management",
        "financing",
        "other",
    ] = Field(
        ...,
        description="High-level category of the event."
    )
    tone: Literal["positive", "negative", "mixed", "unclear"] = Field(
        ...,
        description="Overall tone of this event for the ticker's fundamentals."
    )
    materiality: Literal["low", "medium", "high"] = Field(
        ...,
        description="How material this event is for the ticker in the near/medium term."
    )
    rationale: str = Field(
        ...,
        description="One or two sentences explaining why this event matters and its tone."
    )

    @field_validator("category", mode="before")
    def _normalize_category(cls, v):
        """
        Make the category field robust against slight variations:
        - case-insensitive
        - map common synonyms to the allowed set
        - default to 'other' when unsure
        """
        if v is None:
            return "other"
        s = str(v).strip().lower()

        mapping = {
            "earnings": "earnings",
            "results": "earnings",
            "q3 results": "earnings",
            "q4 results": "earnings",

            "guidance": "guidance",
            "outlook": "guidance",
            "forecast": "guidance",

            "product": "product",
            "launch": "product",
            "products": "product",

            "mna": "mna",
            "m&a": "mna",
            "merger": "mna",
            "acquisition": "mna",
            "deal": "mna",

            "litigation": "litigation",
            "lawsuit": "litigation",
            "legal": "litigation",

            "regulatory": "regulatory",
            "regulation": "regulatory",
            "compliance": "regulatory",

            "macro": "macro",
            "market": "macro",
            "economy": "macro",

            "industry": "industry",
            "sector": "industry",

            "management": "management",
            "leadership": "management",
            "executive": "management",

            "financing": "financing",
            "capital": "financing",
        }

        if s in mapping:
            return mapping[s]
        return "other"


# =========================
#   RL Signal Schema
# =========================

class NewsRLSignals(RobustBaseModel):
    """
    Compact RL-oriented signals derived from the week's news flow for a ticker.

    Aligns with EC/SEC semantics:
      - signal in [-10, 10] : direction & magnitude of fundamental impact (near-term)
      - risk_score in [0, 10]: probability-weighted downside tail risk from news alone
      - confidence in [0, 1] : confidence in (signal, risk_score) given evidence quality/volume
    """
    signal: confloat(ge=-10, le=10) = Field(
        ...,
        description=(
            "Signed directional impact of this week's news on fundamentals over the next few quarters. "
            "-10 = extremely bearish, 0 = mixed/neutral/unclear, +10 = extremely bullish. "
            "Do NOT encode tail risk into signal."
        ),
    )
    risk_score: confloat(ge=0, le=10) = Field(
        ...,
        description=(
            "Probability-weighted downside tail risk implied by the week's news alone. "
            "0 = no clear incremental tail risk; 10 = extreme tail risk (credible severe downside path)."
        ),
    )
    confidence: confloat(ge=0, le=1) = Field(
        ...,
        description=(
            "Confidence in (signal, risk_score) given news volume/clarity/relevance. "
            "0 = unusable/noisy/mismatched; 1 = very clear, consistent, specific evidence."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Brief justification (3–6 sentences) referencing the dominant themes/events "
            "and why signal and risk_score are set as such."
        ),
    )

    @field_validator("signal", mode="before")
    def _coerce_legacy_signal(cls, v, info):
        # If someone passes the old key names, keep schema stable.
        if v is not None:
            return v
        data = info.data or {}
        # Legacy: "sentiment_score" was used as signal proxy
        if "sentiment_score" in data:
            return data["sentiment_score"]
        return 0.0

    @field_validator("risk_score", mode="before")
    def _coerce_legacy_risk(cls, v, info):
        if v is not None:
            return v
        data = info.data or {}
        # Legacy: "event_risk_score" was used as risk proxy
        if "event_risk_score" in data:
            return data["event_risk_score"]
        return 0.0


class NewsWeeklyReduce(RobustBaseModel):
    """
    Final weekly news analysis for one (ticker, decision week).

    This object will be written to a parquet and later joined into the RL
    weekly panel by (ticker, week_decision_date).
    """
    summary_text: str = Field(
        ...,
        description=(
            "120–180 word summary of the week's news for this ticker. "
            "If there is effectively no meaningful or relevant news, this MUST be ''."
        ),
    )
    key_events: List[NewsEvent] = Field(
        default_factory=list,
        description="List of the most important news events this week (may be empty).",
    )
    rl: NewsRLSignals = Field(
        ...,
        description="Compact RL features and risk score derived from the weekly news.",
    )


NEWS_SYSTEM_PROMPT = """
You are a fundamental equity news analyst working for a systematic portfolio manager.

Your outputs will be used as:
- machine-readable RL signals (for a weekly trading agent), and
- inputs to a senior analyst who also sees EC and SEC summaries.

ROLE RELATIVE TO EC / SEC:
- Treat the SEC and EC analyses as the SLOWER baseline:
  * SEC: 1–3 year structural view (business model, balance sheet, structural risks).
  * EC: 1–4 quarter fundamental view at the time of the last earnings call.
- Your NEWS job is to decide whether THIS WEEK’S NEWS materially UPDATES that
  1–4 quarter outlook, not to re-estimate the business from scratch.

HORIZON AND SCOPE:
- Horizon: think in terms of the next 1–4 quarters (≈3–12 months).
- Focus on news that affects the company’s fundamentals, expectations, or risk profile
  over that horizon (earnings power, guidance, balance sheet, structural risks),
  not on short-lived intraday price noise.
- If a headline does not clearly change the 1–4 quarter view, treat it as noise.

STRICT OUTPUT RULES:
- You MUST return a single JSON object matching the NewsWeeklyReduce schema.
- Do NOT output markdown, comments, or explanations outside JSON.
- Do NOT add extra top-level keys or change key names.
- All numeric fields MUST respect their ranges:
  * rl.signal ∈ [-10, 10]
  * rl.risk_score ∈ [0, 10]
  * rl.confidence ∈ [0, 1]

EMPTY / GUARDRAIL CASES (MUST FOLLOW EXACTLY):
- If there is effectively NO meaningful, ticker-relevant news in the window
  (only generic macro headlines, duplicates, or trivia), you MUST output:
    summary_text = ""
    key_events = []
    rl.signal = 0
    rl.risk_score = 0
    rl.confidence = 0
    rl.rationale = "NO_MEANINGFUL_NEWS"
- If the snippets are clearly about a DIFFERENT company or entity and cannot be
  reconciled with the given ticker, you MUST output:
    summary_text = ""
    key_events = []
    rl.signal = 0
    rl.risk_score = 0
    rl.confidence = 0
    rl.rationale = "INCONSISTENT_METADATA"
- Do NOT try to “fix” the ticker or guess the correct company. If in doubt, follow
  the guardrail behaviour above.

KEY_EVENTS CONSTRUCTION:
- key_events should capture the 3–8 most important, ticker-specific news items this week.
- Prefer concrete, fundamental events:
  * earnings results and guidance
  * product launches / failures
  * M&A, large contracts, capital raises/buybacks
  * regulatory actions, litigation, investigations
  * management changes with real strategic impact
  * material macro / industry news that clearly links to this company
- For each event:
  * category: use the closest label (earnings, guidance, product, mna,
              litigation, regulatory, macro, industry, management, financing, other).
    If nothing fits well, use "other".
  * tone: "positive", "negative", "mixed", or "unclear" for the company’s fundamentals.
  * materiality: "high" only for events that could plausibly move the
    1–4 quarter outlook in a meaningful way by themselves; otherwise "medium" or "low".
  * rationale: 1–2 sentences explaining WHY this event matters and WHY the tone is what it is.

RL SIGNAL SEMANTICS (aligned with EC / SEC):
- signal ([-10, 10]): signed directional impact of THIS WEEK’S NEWS on fundamentals
  and expectations over the next 1–4 quarters, relative to the prior EC/SEC baseline.
  - Positive = net bullish update.
  - Negative = net bearish update.
  - 0 ≈ neutral/mixed/unclear update.
  - Do NOT encode tail risk into signal; that belongs in risk_score.

- risk_score ([0, 10]): incremental downside / tail risk implied by THIS WEEK’S NEWS
  relative to a typical large-cap with no special news that week.
  - 0–2: almost no incremental downside risk from the news.
  - 3–5: normal large-cap risk; some medium-intensity risks or mild controversies.
  - 6–8: elevated risk; multiple high-severity or credible downside paths
         (e.g., serious investigations, accidents, liquidity concerns, large execution failures).
  - 9–10: extreme; very serious, unresolved situations where severe downside is plausible.

- confidence ([0, 1]): reliability of (signal, risk_score) given the week’s news:
  - 0.8–1.0: several clear, consistent, specific items directly about this ticker.
  - 0.5–0.8: typical week; some ambiguity or gaps but overall usable.
  - 0.0–0.5: very sparse, noisy, or conflicting coverage, or guardrail cases.

CROSS-SECTIONAL CALIBRATION (for your internal reasoning, not for output labels):
- Imagine applying this rubric across many large, liquid stocks and many weeks:
  - Only a minority of (ticker, week) pairs should get |signal| ≥ 7 or risk_score ≥ 7.
  - Most routine weeks with modest or mixed news should have:
    * signal in roughly [-3, +3], and
    * risk_score in roughly [2, 6].
- Major positive weeks (clear beat-and-raise, big positive strategic news) justify
  signal in [+4, +8] but only rarely [+9, +10].
- Major negative weeks (large scandals, accidents, acute liquidity fears)
  justify signal in [-4, -8] and risk_score in [7, 10].

RL RATIONALE:
- rl.rationale MUST be 3–6 sentences.
- It MUST:
  - reference the most important key_events (by content, not by JSON key),
  - explain why the net impact is positive/negative/mixed over 1–4 quarters,
  - explain why the risk_score is low/normal/elevated based on downside mechanisms,
  - briefly justify the confidence level (evidence amount and clarity).

Do NOT restate every minor headline. Focus on the dominant themes and mechanisms.
""".strip()


def news_reduce_user_prompt(
    ticker: str,
    week_decision_date: str,
    window_start_utc: str,
    window_end_utc: str,
    news_context: str,
) -> str:
    """
    Build the user prompt for the weekly news analyst for one (ticker, week).
    `news_context` is the concatenated [ARTICLE] blocks from the selector.
    """
    return f"""
    Context:
    - Ticker: {ticker}
    - Decision week (trading calendar): {week_decision_date}
    - News window (UTC): {window_start_utc} to {window_end_utc}

    You are given a set of short, noisy news snippets (titles plus brief text)
    for this ticker during the specified time window. Some items may be low-quality,
    generic, duplicated, or only weakly relevant.

    Treat the latest SEC and EC analyses as the slower baseline view of this company.
    Your job is to decide whether THIS WEEK'S NEWS materially updates the
    1–4 quarter outlook implied by that baseline, and if so, in which direction
    and with what incremental downside risk.

    Task:
    1) Identify the main ticker-relevant themes and events in this window.
    2) Discard obviously irrelevant, duplicated, or purely generic macro headlines
       that have no clear implication for this specific ticker's 1–4 quarter outlook.
    3) Produce:
       - a compact weekly summary, and
       - RL-friendly signals (signal, risk_score, confidence, rationale)
         consistent with the NEWS_SYSTEM_PROMPT instructions and a 1–4 quarter horizon.
    4) Follow the EMPTY-output guardrails and JSON schema below EXACTLY.

    EMPTY-output rules (must follow exactly):
    - If there is effectively NO meaningful or relevant news for this ticker
      in this window (i.e., none of the snippets clearly changes the 1–4 quarter view),
      you MUST output:
        summary_text = ""
        key_events = []
        rl.signal = 0
        rl.risk_score = 0
        rl.confidence = 0
        rl.rationale = "NO_MEANINGFUL_NEWS"
    - If the snippets are clearly about a different company / entity and cannot
      be reconciled with the given ticker, you MUST output:
        summary_text = ""
        key_events = []
        rl.signal = 0
        rl.risk_score = 0
        rl.confidence = 0
        rl.rationale = "INCONSISTENT_METADATA"

    Summary and key events:
    - summary_text:
      * 120–180 words describing the dominant themes of the week for this ticker,
        focused on how they update the 1–4 quarter fundamental and risk outlook.
      * If you trigger NO_MEANINGFUL_NEWS or INCONSISTENT_METADATA, summary_text MUST be "".
    - key_events:
      * Prefer 3–8 high-quality events when there is enough material.
      * Each event should be specific, ticker-relevant, and non-duplicative.
      * Avoid creating many nearly-identical events for the same underlying story.

    Output JSON schema (exact keys/types):
    {{
      "summary_text": string,   // 120–180 words, or "" if NO_MEANINGFUL_NEWS / INCONSISTENT_METADATA
      "key_events": [
        {{
          "title": string,
          "category": "earnings"|"guidance"|"product"|"mna"|"litigation"|
                       "regulatory"|"macro"|"industry"|"management"|
                       "financing"|"other",
          "tone": "positive"|"negative"|"mixed"|"unclear",
          "materiality": "low"|"medium"|"high",
          "rationale": string
        }}
      ],
      "rl": {{
        "signal": number,        // [-10, 10], net directional impact over the next 1–4 quarters
        "risk_score": number,    // [0, 10], incremental downside risk from this week's news
        "confidence": number,    // [0, 1], reliability of (signal, risk_score)
        "rationale": string      // 3–6 sentences referencing the main key_events and mechanisms
      }}
    }}

    News items (short, noisy snippets):
    {news_context}
    """.strip()
