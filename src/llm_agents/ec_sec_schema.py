from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any
from pydantic import Field, field_validator
import json

from src.llm_agents.utils import Tier1SignalPack, RobustBaseModel
from src.llm_agents.utils import Tier1SignalPack


# =========================
#   Core Item Schemas
# =========================

class NumberItem(RobustBaseModel):
    """
    A single numerical metric extracted from EC / SEC text.
    Docstring now emphasizes consistent, generic names where possible.
    """
    name: str = Field(
        ...,
        description=(
            "Canonical metric name, e.g., 'revenue', 'EPS', 'operating_margin', "
            "'free_cash_flow', 'EPS_guidance'. Prefer generic names unless the "
            "term is very specific in the text."  
        ),
    )
    value: str = Field(
        ...,
        description="Raw textual value, keep units/format exactly as in the text, e.g. '5.2B', '12.3%', '$4.30–4.50'.",
    )
    unit: Optional[str] = Field(
        None,
        description="Optional unit, e.g., 'USD', '%', 'bps', 'shares'.",
    )
    period: Optional[str] = Field(
        None,
        description="Optional time period, e.g., 'Q3 FY2024', 'FY2025', 'next quarter'.",
    )
    context: Optional[str] = Field(
        None,
        description="Context such as 'GAAP', 'non-GAAP', 'YoY', 'QoQ', 'run-rate', 'ex-FX', 'organic'.",
    )
    source: Optional[str] = Field(
        None,
        description="Text origin: e.g., 'prepared_remarks', 'Q&A', 'MD&A', 'Risk_Factors'.",
    )


class RiskItem(RobustBaseModel):
    """
    A single risk, downside, or concern mentioned in EC / SEC.
    Clarify that rationale should be company-specific, not boilerplate.
    """
    title: str
    category: Optional[Literal[
        "regulatory",
        "litigation",
        "supply_chain",
        "macro",
        "fx",
        "competition",
        "cyber",
        "liquidity",
        "covenant",
        "execution",
        "governance",
        "technology",
        "credit",
        "tax",
        "reputation",
        "esg",
        "other",
    ]] = "other"
    severity: Optional[Literal["low", "medium", "high"]] = "medium"
    rationale: str = Field(
        ...,
        description=(
            "Short explanation of why this is a risk, ideally tying to specific "
            "products, regions, customers, segments, or metrics rather than generic boilerplate."
        ),  # MOD
    )

    @field_validator("category", mode="before")
    def _normalize_category(cls, v):
        if v is None:
            return "other"
        s = str(v).strip().lower()
        synonym_map = {
            "market": "macro",
            "market_risk": "macro",
            "macroeconomic": "macro",
            "regulatory_risk": "regulatory",
            "legal": "litigation",
            "lawsuit": "litigation",
            "currency": "fx",
            "fx_risk": "fx",
            "competition_risk": "competition",
            "security": "cyber",
            "cybersecurity": "cyber",
            "liquidity_risk": "liquidity",
            "covenant_risk": "covenant",
            "execution_risk": "execution",
            "tech": "technology",
            "technology_risk": "technology",
            "governance_risk": "governance",
            "esg_risk": "esg",
        }
        if s in synonym_map:
            s = synonym_map[s]
        allowed = {
            "regulatory",
            "litigation",
            "supply_chain",
            "macro",
            "fx",
            "competition",
            "cyber",
            "liquidity",
            "covenant",
            "execution",
            "governance",
            "technology",
            "credit",
            "tax",
            "reputation",
            "esg",
            "other",
        }
        return s if s in allowed else "other"

    @field_validator("severity", mode="before")
    def _normalize_severity(cls, v):
        if v is None:
            return "medium"
        s = str(v).strip().lower()
        allowed = {"low", "medium", "high"}
        synonym_map = {
            "moderate": "medium",
            "med": "medium",
            "medium_risk": "medium",
            "high_risk": "high",
            "severe": "high",
            "low_risk": "low",
        }
        if s in synonym_map:
            s = synonym_map[s]
        return s if s in allowed else "medium"


class ExposureItem(RobustBaseModel):
    """
    A macro / market / structural exposure, with direction.
    """
    factor: Optional[Literal[
        "fx",
        "commodity",
        "interest_rate",
        "geography",
        "customer_concentration",
        "supplier_concentration",
        "industry",
        "inflation",
        "credit_spread",
        "equity_market",
        "labor",
        "housing",
        "regulation",
        "other",
    ]] = "other"
    direction: Optional[Literal["headwind", "tailwind", "unclear"]] = "unclear"
    notes: str = Field(
        ...,
        description=(
            "Concrete description of how this factor affects the company "
            "(e.g., regions, products, magnitude) rather than generic macro commentary."
        ),  
    )

    @field_validator("factor", mode="before")
    def _normalize_factor(cls, v):
        if v is None:
            return "other"
        s = str(v).strip().lower()
        synonym_map = {
            "currency": "fx",
            "fx_risk": "fx",
            "rates": "interest_rate",
            "rate": "interest_rate",
            "interest": "interest_rate",
            "inflation_risk": "inflation",
            "labor_market": "labor",
            "employment": "labor",
            "housing_market": "housing",
            "equity": "equity_market",
            "stock_market": "equity_market",
            "regulatory": "regulation",
        }
        if s in synonym_map:
            s = synonym_map[s]
        allowed = {
            "fx",
            "commodity",
            "interest_rate",
            "geography",
            "customer_concentration",
            "supplier_concentration",
            "industry",
            "inflation",
            "credit_spread",
            "equity_market",
            "labor",
            "housing",
            "regulation",
            "other",
        }
        return s if s in allowed else "other"

    @field_validator("direction", mode="before")
    def _normalize_direction(cls, v):
        if v is None:
            return "unclear"
        s = str(v).strip().lower()
        allowed = {"headwind", "tailwind", "unclear"}
        synonym_map = {
            "negative": "headwind",
            "risk": "headwind",
            "pressure": "headwind",
            "drag": "headwind",
            "challenging": "headwind",
            "positive": "tailwind",
            "supportive": "tailwind",
            "benefit": "tailwind",
            "favorable": "tailwind",
            "neutral": "unclear",
            "mixed": "unclear",
        }
        if s in synonym_map:
            s = synonym_map[s]
        return s if s in allowed else "unclear"


class OpportunityItem(RobustBaseModel):
    """
    A single upside / opportunity.
    Rationale guidance similar to RiskItem.
    """
    title: str
    driver: Optional[Literal[
        "product",
        "pricing",
        "mix",
        "geography",
        "fx",
        "cost",
        "efficiency",
        "demand",
        "acquisition",
        "partnership",
        "technology",
        "brand",
        "channel",
        "regulation",
        "other",
    ]] = "other"
    confidence: Optional[Literal["low", "medium", "high"]] = "medium"
    rationale: str = Field(
        ...,
        description=(
            "Short explanation of why this is an opportunity, tying to specific "
            "products, regions, segments, customers, or initiatives."
        ),  # MOD
    )

    @field_validator("driver", mode="before")
    def _normalize_driver(cls, v):
        if v is None:
            return "other"
        s = str(v).strip().lower()
        synonym_map = {
            "digital": "technology",
            "ai": "technology",
            "cloud": "technology",
            "platform": "technology",
            "online": "technology",
            "saas": "technology",
            "capital": "cost",
            "capital_efficiency": "efficiency",
            "margin": "pricing",
            "price": "pricing",
            "pricing_power": "pricing",
            "marketing": "brand",
            "brand_investment": "brand",
            "distribution": "channel",
            "go_to_market": "channel",
            "gtm": "channel",
            "geo": "geography",
        }
        if s in synonym_map:
            s = synonym_map[s]
        allowed = {
            "product",
            "pricing",
            "mix",
            "geography",
            "fx",
            "cost",
            "efficiency",
            "demand",
            "acquisition",
            "partnership",
            "technology",
            "brand",
            "channel",
            "regulation",
            "other",
        }
        return s if s in allowed else "other"

    @field_validator("confidence", mode="before")
    def _normalize_confidence(cls, v):
        if v is None:
            return "medium"
        s = str(v).strip().lower()
        allowed = {"low", "medium", "high"}
        synonym_map = {
            "moderate": "medium",
            "med": "medium",
            "strong": "high",
            "high_confidence": "high",
            "weak": "low",
            "low_confidence": "low",
        }
        if s in synonym_map:
            s = synonym_map[s]
        return s if s in allowed else "medium"


# =========================
#   EC Schemas
# =========================

class ECChunkMap(RobustBaseModel):
    """
    Chunk-level EC mapping result.
    """
    bullets: List[str] = Field(..., description="3–7 concise, company-specific factual statements.")  # MOD
    numbers: List[NumberItem] = []
    risks: List[RiskItem] = []
    opportunities: List[OpportunityItem] = []


class ECReduce(RobustBaseModel):
    """
    Reduced EC summary over all chunks.
    """
    summary_text: str
    numbers: List[NumberItem] = []
    risks: List[RiskItem] = []
    opportunities: List[OpportunityItem] = []
    rl: Optional[Tier1SignalPack] = None


# =========================
#   SEC Schemas
# =========================

class SECChunkMapMDNA(RobustBaseModel):
    """
    Chunk-level SEC MD&A mapping result.
    """
    bullets: List[str] = []
    exposures: List[ExposureItem] = []
    opportunities: List[OpportunityItem] = []
    risks: List[RiskItem] = []
    red_flags: List[str] = []


class SECChunkMapRF(RobustBaseModel):
    """
    Chunk-level SEC Risk Factors mapping result.
    """
    bullets: List[str] = []
    risks: List[RiskItem] = []
    red_flags: List[str] = []


class SECReduceSection(RobustBaseModel):
    """
    Reduced summary for a single SEC section (MD&A or Risk Factors).
    """
    section: str
    summary_text: str
    top_risks: List[RiskItem] = []
    exposures: List[ExposureItem] = []
    opportunities: List[OpportunityItem] = []
    flags: List[str] = []


class SECReduceFinal(RobustBaseModel):
    """
    Final SEC filing summary combining MD&A and Risk Factors.
    """
    summary_text: str
    top_risks: List[RiskItem] = []
    exposures: List[ExposureItem] = []
    opportunities: List[OpportunityItem] = []
    flags: List[str] = []
    rl: Optional[Tier1SignalPack] = None


# =========================
#   Prompt String Helpers
# =========================

def _format_rules_block() -> str:
    return """
    Important formatting rules:
    - Respond with ONLY a single JSON object.
    - Do NOT include explanations, comments, or markdown.
    - Do NOT wrap the JSON in code fences.
    - Do NOT output multiple JSON objects or arrays at the top level.
    - If there are no items for a list field, return an empty list for that field.
    Guardrails (must follow):
    - Consistency check: the merged content must clearly match the provided Ticker/company.
    If the content appears to describe a different company OR you are not sure:
      return:
        summary_text = ""
        numbers/risks/opportunities = []
        rl = {"signal":0,"risk_score":0,"confidence":0,"rationale":"INCONSISTENT_METADATA"}
    - Do not guess the company identity. Do not “fix” the ticker.
    """.strip()


def map_system_prompt() -> str:
    return (
        "You are a precise financial analyst and information extractor.\n\n"
        "Your job is to read a short text segment from an earnings call or SEC filing\n"
        "and extract structured information into STRICTLY VALID JSON.\n\n"
        "CRITICAL RULES:\n"
        "1) Respond with STRICTLY VALID JSON for the requested schema.\n"
        "2) Do NOT include any natural-language explanation, comments, or markdown.\n"
        "3) Do NOT wrap JSON in code fences.\n"
        "4) Do NOT add extra top-level keys or change key names.\n"
        "5) If there are no valid items for a list field, return an empty list.\n"
        "6) Never invent categories/drivers/factors outside the allowed set;\n"
        "   if a label does not fit, use 'other'.\n"
        "7) If you are unsure about severity/confidence/direction, use the\n"
        "   default value given in the schema (typically 'medium' or 'unclear').\n"
        "8) Prefer fewer, high-quality, company-specific items over many speculative or boilerplate ones.\n" 
    )


def reduce_system_prompt() -> str:
    return (
        "You merge partial JSON summaries for financial documents.\n\n"
        "Your job is to:\n"
        "- Deduplicate near-identical items.\n"
        "- Prefer more precise/complete values when merging.\n"
        "- Keep wording concise and factual.\n"
        "- De-emphasize generic boilerplate; focus on company-specific themes.\n"  
        "\nCRITICAL RULES:\n"
        "1) Respond with STRICTLY VALID JSON for the requested schema.\n"
        "2) Do NOT include any natural-language explanation, comments, or markdown.\n"
        "3) Do NOT wrap JSON in code fences.\n"
        "4) Do NOT add keys that are not in the target schema.\n"
        "5) If a list is empty, return an empty list (not null).\n"
    )


# ---------------------------------------------------------------------
# SEC chunk-level system prompt: used for both MD&A and Risk-Factor map
# ---------------------------------------------------------------------
SEC_CHUNK_MAP_SYSTEM_PROMPT = """
You are a fundamental equity analyst reading a CHUNK of a 10-K or 10-Q filing.

Real-world analysts use these filings to understand BOTH:
- how the business makes money and competes, and
- what structural risks and exposures could threaten that earning power
  over the next several years.

Your job at the CHUNK level is to extract the most important, company-specific
facts that will later be combined into one consolidated SEC analysis.

Focus on content that belongs in one of these buckets when it appears:

1) BUSINESS MODEL & EARNING POWER
   - What products / services generate revenue?
   - What are the main segments and geographies?
   - How does the company earn money (pricing power, volume, recurring
     revenue, subscriptions, ad-based, transaction-based, etc.)?
   - Any information about unit economics, margins, or capital intensity that
     looks structural, not just a one-off quarter.

2) COMPETITIVE POSITION & STRATEGY
   - Evidence about moat: scale advantages, brand strength, network effects,
     technology lead, switching costs, regulatory barriers.
   - Strategy: management’s stated priorities, long-term growth areas,
     capital allocation themes (capex, buybacks, acquisitions, deleveraging).

3) STRUCTURAL RISKS (WITH CALIBRATED SEVERITY)
   - Company-specific legal, regulatory, and litigation risks.
   - Balance sheet and liquidity risks (leverage, covenants, funding sources).
   - Operational and supply-chain vulnerabilities that can persist over time.
   - Concentration risks (few key customers, suppliers, geographies).
   - Material macro/FX/interest/commodity exposures as they relate to this
     company’s business model.

   When assigning severity:
   - Use HIGH severity only if this risk could plausibly cause a material hit
     to earnings, cash flow, or solvency on its own, or be a major, enduring
     overhang within the next 1–3 years.
   - If a risk is generic to the sector (e.g., “competition is intense”,
     “macro uncertainty”) or looks similar to what most peers face, treat it
     as MEDIUM unless the text clearly says this company is worse than peers.
   - Do not mark every boilerplate risk as HIGH; reserve HIGH for truly
     important, company-specific vulnerabilities.

4) OPPORTUNITIES
   - Long-term growth drivers (new platforms, durable demand trends,
     structural margin improvement, multi-year investment programs).
   - These should be grounded in the text (e.g., products, contracts, secular
     tailwinds), not vague optimism.

5) NUMBERS
   - Structural or recurring numbers that matter for the big picture:
     revenue/profit mix by segment or geography, long-term obligations,
     leverage ratios, major capex plans, committed content or lease liabilities.

6) EXPOSURES
   - Clear sensitivities to macro variables (rates, FX, commodities),
     regulatory regimes, key customers or suppliers.

7) FLAGS
   - If you see strong evidence that the text actually belongs to a different
     company, period, or ticker, or that the metadata is clearly wrong, add a
     flag "INCONSISTENT_METADATA".
   - Do NOT add this flag for minor inconsistencies or generic language.

Downstream, another agent will aggregate your chunk-level outputs into a single
SEC analysis, and a separate RL head will produce numeric signals. Your job at
this stage is to be:
- concrete and fact-focused,
- balanced between strengths and weaknesses,
- concise but information-dense.

Follow the provided JSON schema exactly (fields such as summary_text, numbers,
risks, opportunities, exposures, flags). Do not invent new top-level keys.
Return ONLY a single JSON object.
""".strip()


# ---------------------------------------------------------------------
# SEC final reduce system prompt: aggregate all sections into one view
# ---------------------------------------------------------------------
SEC_FINAL_REDUCE_SYSTEM_PROMPT = """
You are a senior equity analyst synthesizing multiple CHUNK-LEVEL summaries of a
10-K / 10-Q filing into ONE consolidated SEC analysis.

Real analysts use the full filing to answer three structural questions:
1) How does this company make money, and how durable is that earning power?
2) How strong is its competitive position and balance sheet?
3) What structural risks and exposures could materially damage earnings,
   cash flows, or solvency over the next several years?

You will receive a JSON list of chunk-level SEC summaries, each with fields
such as summary_text, numbers, risks, opportunities, exposures, flags.

From these, build a SINGLE, balanced structural view with:

- A clear narrative of the BUSINESS MODEL:
  - main products/services, revenue streams, and segments,
  - degree of recurring vs transactional revenue,
  - key geographies and customer types,
  - structural margin and capital intensity themes.

- A clear view of COMPETITIVE POSITION:
  - sources of moat (scale, brand, technology, network effects, switching costs),
  - industry structure and where this company sits within it.

- A structured picture of STRUCTURAL RISKS:
  - company-specific legal / regulatory / litigation issues,
  - balance sheet and liquidity risks (leverage, covenants, funding),
  - operational and supply-chain vulnerabilities,
  - concentration risks (customers, suppliers, markets),
  - macro/FX/interest/commodity exposures that are clearly linked to this
    company’s earnings power.

- Long-term OPPORTUNITIES:
  - secular growth drivers, new platforms, durable cost advantages,
  - major investment programs or strategic shifts that can reshape earnings.

- Consolidated NUMBERS:
  - structural metrics from across chunks (e.g., revenue mix, leverage,
    large contractual obligations, material capex or content commitments).

SEVERITY CALIBRATION AND TOP-RISK SELECTION:
- "top_risks" should be a SHORT LIST of the most material structural risks
  across the entire filing (typically 5–15 items), not an exhaustive copy of
  all chunk-level risks.
- Use "medium" as the DEFAULT severity for most risks.
- Reserve "high" severity ONLY for a MINORITY of truly dominant, company-specific
  risks that could materially impair core earnings, cash flow, or solvency on
  their own within the next 1–3 years.
- If a risk appears many times in boilerplate form or is common to the sector
  (for example, "competition", "macro uncertainty"), keep it at "medium" unless
  the text clearly states that this company is materially more exposed than peers.
- When multiple chunks present conflicting severities for the same underlying
  risk, choose the severity that best reflects the total evidence:
  * choose "high" only when the majority of evidence indicates a serious,
    persistent threat;
  * otherwise choose "medium".

EXPOSURES, OPPORTUNITIES, FLAGS:
- "exposures" should capture the main structural sensitivities (for example,
  to interest rates, FX, specific geographies, customer or supplier
  concentration). Avoid double-counting: if a driver appears as both a risk
  and an exposure, describe it once in each list but treat it as ONE underlying
  mechanism.
- "opportunities" should reflect multi-year, structural upside (new platforms,
  durable cost advantages, recurring revenue transitions), not short-lived
  quarterly events or generic optimism.
- Aggregate the flags from all chunks:
  * If ANY chunk has a strong "INCONSISTENT_METADATA" flag indicating the filing
    is not for this ticker/period, you MUST include "INCONSISTENT_METADATA"
    in the final flags list and treat the filing as unusable for RL downstream.
  * Otherwise, only propagate serious, non-ordinary issues (for example,
    going-concern warnings, material weaknesses in controls, major restatements).

INTERNAL CALIBRATION (for your own reasoning, not for output):
- Form a mental label for business strength:
    business_strength_label ∈ {weak, average, strong, exceptional}
  based on growth, margins, moat, balance sheet quality, and opportunities.
- Form a mental label for structural risk intensity:
    risk_intensity_label ∈ {low, moderate, elevated, severe}
  based on the type, number, and severity of structural risks.

These internal labels will guide a downstream RL head that converts this
summary into numeric signals. You do NOT output the labels themselves, but your
summary_text, risks, opportunities, and exposures should make them obvious.

STYLE:
- summary_text: 8–15 sentences that clearly separate strengths vs weaknesses,
  and implicitly frame a 1–3 year structural horizon.
- risks: concrete and mechanistic (how and through what channel earnings or
  solvency could be hit), not generic clichés.
- opportunities: structural, multi-year themes, not one-off quarterly events.

Your output will be consumed by:
- a SEC-based RL signal head, and
- a Tier-2 senior analyst that combines EC, SEC, news, and technicals.

Therefore, be disciplined:
- keep the focus on STRUCTURAL information, not quarter-to-quarter noise,
- keep the risk section aligned with how risk factors are described in the
  filing (Item 1A and MD&A),
- calibrate severities so that only a minority of risks are "high", and
- follow the JSON schema exactly (no extra top-level keys).

Return ONLY a single JSON object matching the SECReduce schema.
""".strip()


SEC_SECTION_REDUCE_SYSTEM_PROMPT = """
You are consolidating multiple CHUNK-LEVEL SEC summaries for a single section
(e.g., MD&A or Risk Factors) into ONE section-level summary.

Your job is to:
- keep the focus on STRUCTURAL business model and STRUCTURAL risks,
- merge overlapping items and remove duplication,
- preserve all important, company-specific positives and negatives.
- normalise and CALIBRATE risk severities so that only a MINORITY of risks are
  labeled "high" severity.

STRUCTURAL VS BOILERPLATE:
- Emphasise information that clearly affects the business model, earnings power,
  balance sheet, liquidity, or competitive position over a 1–3 year horizon.
- De-emphasise long lists of generic, sector-wide boilerplate (for example,
  "the industry is competitive", "macroeconomic conditions may affect demand")
  unless the text states that this company is materially more exposed than peers.

SEVERITY CALIBRATION (for this section):
- Use "medium" as the DEFAULT severity for most risks.
- Reserve "high" severity ONLY for a SMALL number of company-specific risks that
  could materially impair core earnings, cash flow, or solvency on their own
  within the next 1–3 years (for example, serious liquidity stress, major
  regulatory investigations, large concentrated exposures, or critical
  operational vulnerabilities).
- If a risk is generic to the sector or described in boilerplate language,
  keep severity at "medium" unless the text clearly indicates that the company
  is worse off than peers.
- When merging the same risk across multiple chunks with different severities,
  choose the severity that best reflects the overall picture:
  * if most mentions are "medium" and only a few are "high" with limited detail,
    choose "medium";
  * choose "high" only if the majority of evidence points to a major,
    persistent threat.

TOP_RISKS, EXPOSURES, OPPORTUNITIES, FLAGS:
- "top_risks" should be a SHORT list (typically 3–10 items) of the MOST
  material structural risks for this section, not every risk that appears in
  the chunks.
- Use "exposures" for clearly identified sensitivities to macro factors,
  geographies, or concentrations. Avoid duplicating the same driver as both
  a separate risk and an exposure unless the text clearly distinguishes them.
- "opportunities" (MD&A sections only) should capture structural, multi-year
  upside drivers (new platforms, durable cost advantages, secular demand
  trends), not one-off events.
- "flags" should be reserved for serious, non-ordinary issues (for example,
  "INCONSISTENT_METADATA", going-concern warnings, major restatements), not
  for minor caveats.

Your output will later be combined with other sections into a full SEC analysis.
Follow the same structuring conventions as the chunk summaries (summary_text,
numbers, risks, opportunities, exposures, flags), and return exactly one JSON
object that matches the section schema.
""".strip()


# =========================
#   Map User Prompts
# =========================

def ec_map_user_prompt(ticker: str, as_of: str, compression_ratio: float, chunk_text: str) -> str:
    return f"""
    Context:
    - Ticker: {ticker}
    - As-of: {as_of}

    Task:
    Summarize the following earnings-call segment for {ticker}.
    Write 3–7 concise, company-specific factual bullets capturing the main points of this segment,
    and extract key numbers, risks, and opportunities. Be factual and concise.
    Do not repeat generic boilerplate language that is not specific to this company or quarter.  
    Focus on what actually changed or matters for the business outlook over the next 1–4 quarters
    (approximately 3–12 months).

    When you describe risks and opportunities, prefer concrete, mechanism-based statements about
    how they could affect revenue, margins, cash flows, or guidance over the next few quarters
    (for example, demand slowdown, pricing pressure, cost inflation, execution issues) rather
    than generic macro commentary. Where possible, briefly link the risk or opportunity to
    specific segments, products, or regions, and, if the text supports it, hint at the scale
    relative to the overall business (for example, a segment that is a material share of revenue).

    Target compression ratio ≈ {compression_ratio:.2f} relative to input length.

    Output JSON schema (exact keys/types):
    {{
      "bullets": [string, ...],                 // 3–7 concise factual statements
      "numbers": [                              // optional; omit if none
        {{
          "name": string,                       // e.g., "EPS", "revenue", "gross_margin"
          "value": string,                      // keep units/format as text, e.g. "5.2B", "12.3%", "$4.30–4.50"
          "unit": string|null,                  // e.g. "USD", "%", "bps"
          "period": string|null,                // e.g., "Q3 FY2024"
          "context": string|null,               // e.g., "GAAP", "YoY", "run-rate"
          "source": string|null                 // e.g., "prepared_remarks", "Q&A"
          }}
          ],
      "risks": [
        {{
          "title": string,
          "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|"credit"|
                    "litigation"|"supply_chain"|"macro"|"fx"|"competition"|"cyber"|
                    "liquidity"|"covenant"|"execution"|"other",
          "severity": "low"|"medium"|"high",
          "rationale": string                  // explain the downside mechanism over the next 1–4 quarters, 
                                               // tied to specific segments/products/regions when possible
          }}
          ],
      "opportunities": [
        {{
          "title": string,
          "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|
                  "cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|
                  "partnership"|"other",
          "confidence": "low"|"medium"|"high",
          "rationale": string                  // explain the upside mechanism over the next 1–4 quarters, 
                                              // tied to specific segments/products/regions when possible
        }}
        ]
        }}
        
    Text segment:
    \"\"\"{chunk_text}\"\"\"

    {_format_rules_block()}
    """.strip()


def sec_mdna_map_user_prompt(ticker: str, as_of: str, compression_ratio: float, chunk_text: str) -> str:
    return f"""
    Context:
    - Ticker: {ticker}
    - As-of: {as_of}

    Task:
    Summarize this SEC MD&A segment.
    Write 3–8 concise, company-specific factual bullets for this MD&A segment, and extract exposures,
    opportunities, and risks. Be concise and factual.
    Avoid generic boilerplate about macro conditions unless clearly tied to this company's situation 
    or to specific balance-sheet / cash-flow exposures. 

    Target compression ratio ≈ {compression_ratio:.2f}.

    Output JSON schema (exact keys/types):
    {{
      "bullets": [string, ...],                 // 3–8 concise factual statements for this MD&A segment
      "exposures": [
      {{
        "factor": "fx"|"labor"|"housing"|"regulation"|"commodity"|"inflation"|
                  "credit_spread"|"interest_rate"|"equity_market"|"geography"|
                  "customer_concentration"|"supplier_concentration"|"industry"|"other",
        "direction": "headwind"|"tailwind"|"unclear",
        "notes": string
        }}
        ],
    "opportunities": [
      {{
        "title": string,
        "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|
                  "cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|
                    "partnership"|"other",
        "confidence": "low"|"medium"|"high",
        "rationale": string
        }}
        ],
    "risks": [
      {{
        "title": string,
        "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|
                    "credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|
                    "cyber"|"liquidity"|"covenant"|"execution"|"other",
        "severity": "low"|"medium"|"high",
        "rationale": string
      }}
      ],
    "red_flags": [string, ...]
    }}

    Text segment:
    \"\"\"{chunk_text}\"\"\"

    {_format_rules_block()}
    """.strip()


def sec_rf_map_user_prompt(ticker: str, as_of: str, compression_ratio: float, chunk_text: str) -> str:
    return f"""
    Context:
    - Ticker: {ticker}
    - As-of: {as_of}

    Task:
    Summarize this SEC Risk Factors segment.
    Write 3–8 concise factual bullets for this Risk Factors segment, and extract
    risks and red flags. Be concise and factual.
    Emphasize risks that are specific to this company; avoid restating generic legal boilerplate.
    Target compression ratio ≈ {compression_ratio:.2f}.

    Output JSON schema (exact keys/types):
    {{
      "bullets": [string, ...],                 // 3–8 concise factual statements for this Risk Factors segment
      "risks": [
      {{
        "title": string,
        "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|
                    "credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|
                    "cyber"|"liquidity"|"covenant"|"execution"|"other",
        "severity": "low"|"medium"|"high",
        "rationale": string
        }}
        ],
    "red_flags": [string, ...]
    }}

    Text segment:
    \"\"\"{chunk_text}\"\"\"

    {_format_rules_block()}
    """.strip()


# =========================
#   Reduce User Prompts
# =========================

def ec_reduce_user_prompt(ticker: str, as_of: str, maps_compact_jsonl: str) -> str:
    return f"""
    Context:
    - Ticker: {ticker}
    - As-of: {as_of}

    Task:
    Merge the following partial EC summaries (one JSON object per line).
    Each object may come from a single chunk or from a previous merge step.
    Deduplicate items, prefer precise values, and keep wording concise and factual.

    DATA CONSISTENCY GUARDRAILS (VERY IMPORTANT):
    - The call transcript TEXT (as reflected in these partial summaries) is the ground truth.
    - The metadata ticker {ticker} and as-of date {as_of} are only hints and may be wrong.
    - If, after reading the content, you see clear evidence that the primary issuer/company
      in the call is DIFFERENT from {ticker}, or you are NOT reasonably confident that the
      call belongs to this ticker/period, you MUST treat this filing as UNUSABLE.
    - For an unusable / inconsistent filing:
      * summary_text MUST be an empty string "";
      * numbers, risks, and opportunities MUST all be empty lists [];
      * the rl object MUST be:
        - signal = 0
        - risk_score = 0
        - confidence = 0
        - rationale = "INCONSISTENT_METADATA"
    - Do NOT try to guess or “fix” the ticker or company name. If content and metadata do
      not clearly match, follow the EMPTY-output rules above.

    Write a 180–250 word summary that captures the core performance, guidance,
    and key risks/opportunities from the full call. The summary_text should:
    - clearly distinguish strengths vs weaknesses,
    - explicitly frame the discussion over the next 1–4 quarters (about 3–12 months),
    - reference several key numerical items (for example, growth rates, margins, guidance
      ranges, or major segment moves),
    - focus on information likely to influence business results or investor expectations
      over that 1–4 quarter horizon, not long-term vision statements.

    Prefer 5–15 high-quality numerical items, 3–10 distinct, material risks, and
    3–10 distinct, material opportunities. Do not waste space repeating generic
    boilerplate.  

    Horizon for the RL Agent object:
    - Think in terms of the next 1–4 quarters (about 3–12 months), but your rl output
      will be consumed by a weekly trading agent. Focus on changes or information that
      are likely to influence the business and investor expectations over that horizon,
      not long-term vision statements.

    RISK SEVERITY GUIDELINES (for the 'risks' list below):
    - Use severity = "high" only if this risk on its own could plausibly cause a material
      hit to earnings, cash flow, or guidance over the next 1–4 quarters (for example,
      a major supply disruption, a large regulatory or litigation issue, a significant
      loss of a key customer, or a clearly quantified and sizeable headwind).
    - If a risk is largely generic to the sector or macro environment (for example,
      "competition is intense", "macro uncertainty", "FX volatility") and the call does
      not clearly indicate that this company is worse off than peers, prefer severity
      = "medium" instead of "high".
    - Use severity = "low" for smaller or more speculative risks that are mentioned
      but appear unlikely to drive meaningful P&L impact over the next few quarters.

    Output JSON schema (exact keys/types):
    {{
      "summary_text": string,                   // 180–250 words, explicit 1–4 quarter horizon
      "numbers": [
      {{
        "name": string,
        "value": string,
        "unit": string|null,
        "period": string|null,
        "context": string|null,
        "source": string|null
        }}
        ],
      "risks": [
      {{
        "title": string,
        "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|
                    "credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|
                    "cyber"|"liquidity"|"covenant"|"execution"|"other",
        "severity": "low"|"medium"|"high",
        "rationale": string      // concrete downside mechanism over the next few quarters,
                                // including which part of the business is affected
        }}
      ],
      "opportunities": [
      {{
        "title": string,
        "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|
                  "cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|
                  "partnership"|"other",
        "confidence": "low"|"medium"|"high",
        "rationale": string     // concrete upside mechanism over the next few quarters
        }}
      ],
      "rl": 
        "signal": float,        // [-10, 10] signed directional impact (pos=bullish, neg=bearish)
        "risk_score": float,    // [0, 10] downside risk for reward penalty
        "confidence": float,    // [0, 1]
        "rationale": string     // 2–5 sentences referencing items in THIS reduced output,
                                // explicitly tying them to the next 1–4 quarters and explaining
                                // why the chosen magnitudes are appropriate vs a typical quarter
        }}    

    
    Calibration hints (for your internal reasoning, not for output):
    - Typical large-cap quarter:
      * signal will usually lie between about -2 and +3, and
      * risk_score will usually lie between about 2 and 5.
    - Use |signal| ≥ 7 only when the quarter is clearly an outlier vs a typical large-cap
      earnings call (for example, a major beat-and-raise or a clear, broad-based
      deterioration with credible mechanisms).
    - If the call shows strong positives but also material near-term downside risks,
      prefer a signal in the +3 to +6 range rather than ≥7.
    - Use risk_score ≥ 7 only when specific, material downside mechanisms are present
      (for example, serious litigation, liquidity stress, acute execution risk), not for
      generic comments like "the sector is competitive" or "macro uncertainty".
    - If risks are mostly medium-severity and partly generic, risk_score will usually be
      in the moderate range (roughly 3–5), with lower values when the risk set is thin
      and higher values when there are several concrete, company-specific risks.

    Partial summaries (JSONL; one object per line):
    {maps_compact_jsonl}

    {_format_rules_block()}
    """.strip()


def sec_section_reduce_user_prompt(section: str, ticker: str, as_of: str, maps_compact_jsonl: str) -> str:
    return f"""
    Context:
    - Ticker: {ticker}
    - As-of: {as_of}
    - Section: {section}

    Task:
    Merge the partial SEC {section} summaries (one JSON object per line).
    Each object may come from a single chunk or from a previous merge step.
    Deduplicate items and reconcile conflicts.

    Focus on STRUCTURAL themes over a 1–3 year horizon:
    - business model and earning power,
    - structural risks and exposures,
    - long-term opportunities (for MD&A),
    rather than short-lived quarterly events or generic sector boilerplate.

    Write a 200–280 word summary for this section that highlights the most
    important themes, including key risks, exposures, opportunities, and flags.
    Prefer 3–10 distinct top risks, 3–10 exposures (for MD&A), 3–10 opportunities
    (for MD&A), and 3–10 consolidated flags. Focus on company-specific issues and
    avoid long restatements of standard legal boilerplate.  

    SEVERITY CALIBRATION AND TOP-RISK SELECTION:
    - "top_risks" should be a SHORT list of the most material structural risks
      in this section, not all risks mentioned in the chunks.
    - Use "medium" as the DEFAULT severity for most risks.
    - Reserve "high" severity ONLY for a MINORITY of company-specific risks that
      could materially impair core earnings, cash flow, or solvency on their own
      within the next 1–3 years.
    - If a risk is generic to the sector or appears mainly as boilerplate,
      keep severity at "medium" unless the text clearly states that this
      company is more exposed than peers.
    - When merging severities for the same underlying risk across chunks,
      choose the severity that best reflects the overall evidence (do not
      automatically escalate to "high" just because one chunk used it).

    Output JSON schema (exact keys/types):
    {{
      "section": "MD&A"|"Risk Factors",
      "summary_text": string,                   // 200–280 words
      "top_risks": [
      {{
        "title": string,
        "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|
                    "credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|
                    "cyber"|"liquidity"|"covenant"|"execution"|"other",
        "severity": "low"|"medium"|"high",
        "rationale": string
        }}
        ],
      "exposures": [                            // MD&A only; omit for Risk Factors
      {{
        "factor": "fx"|"labor"|"housing"|"regulation"|"commodity"|"inflation"|
                  "credit_spread"|"interest_rate"|"equity_market"|"geography"|
                  "customer_concentration"|"supplier_concentration"|"industry"|"other",
        "direction": "headwind"|"tailwind"|"unclear",
        "notes": string
        }}
        ],
      "opportunities": [                        // MD&A only; omit for Risk Factors
      {{
        "title": string,
        "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|
                  "cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|
                  "partnership"|"other",
        "confidence": "low"|"medium"|"high",
        "rationale": string
        }}
        ],
      "flags": [string, ...]
      }}

    Partial summaries (JSONL; one object per line):
    {maps_compact_jsonl}

    {_format_rules_block()}
    """.strip()
    

def sec_final_reduce_user_prompt(ticker: str, as_of: str, mdna_json: str | None, rf_json: str | None) -> str:

    mdna_block = f"MD&A section JSON:\n{mdna_json}\n" if mdna_json else "MD&A section JSON: (missing)\n"
    rf_block = f"Risk Factors section JSON:\n{rf_json}\n" if rf_json else "Risk Factors section JSON: (missing)\n"
    return f"""
    Context:
    - Ticker: {ticker}
    - As-of: {as_of}

    Task:
    Synthesize the SEC filing using the available sections below into a single final JSON.
    Be concise and factual. Deduplicate items and reconcile inconsistencies.

    Focus on a 1–3 year STRUCTURAL horizon:
    - how the company makes money and how durable that earning power is,
    - the strength of its competitive position and balance sheet,
    - the structural risks and exposures that could materially affect earnings,
      cash flows, or solvency over that horizon.

    DATA CONSISTENCY GUARDRAILS (VERY IMPORTANT):
    - The SEC filing TEXT (as reflected in the MD&A and Risk Factors JSON) is the ground truth.
    - The metadata ticker {ticker} and as-of date {as_of} are only hints and may be wrong.
    - If, after reviewing the content, you see clear evidence that the primary issuer/company
      in the filing is DIFFERENT from {ticker}, or you are NOT reasonably confident that
      the filing belongs to this ticker/period, you MUST treat this filing as UNUSABLE.
    - For an unusable / inconsistent filing you MUST return an EMPTY SECReduceFinal object:
      * summary_text MUST be an empty string "";
      * top_risks, exposures, and opportunities MUST all be empty lists [];
      * flags MUST contain exactly one item: "INCONSISTENT_METADATA";
      * the rl object MUST be:
        - signal          = 0
        - risk_score      = 0
        - confidence      = 0
        - rationale       = "INCONSISTENT_METADATA"
    - Do NOT try to guess or correct the ticker/company name or period. If content and
      metadata cannot be confidently reconciled, follow the EMPTY-output rules above.

    Write a 230–320 word overall SEC filing summary that integrates MD&A and Risk
    Factors. Highlight the most important structural and near-term risks, key
    exposures, and major opportunities. Prefer 5–15 top risks, 3–12 exposures,
    3–12 opportunities, and 3–15 consolidated flags. Focus on what is most
    material for investors, not on generic disclosure language. 

    SEVERITY CALIBRATION AND TOP-RISK SELECTION:
    - "top_risks" should be a SHORT list of the most material structural risks
      across the entire filing, not a copy of every risk from the sections.
    - Use "medium" as the DEFAULT severity for most risks.
    - Reserve "high" severity ONLY for a MINORITY of truly dominant, company-specific
      risks that could materially impair core earnings, cash flow, or solvency on
      their own within the next 1–3 years.
    - Generic sector-wide risks or boilerplate (for example, "competition", "macro
      uncertainty") should usually have "medium" severity unless the text clearly
      states that this company is materially more exposed than peers.
    - When MD&A and Risk Factors disagree on severity for the same underlying risk,
      choose the level that best reflects the total evidence, not automatically the
      most extreme one.

    Horizon for the RL Agent object:
    - Think in terms of a 1–3 year horizon, focusing on structural opportunities and
      downside risks that could plausibly affect the business and equity value over
      that period. Your rl output will be consumed by a weekly trading agent that
      treats this as a slow-moving structural signal.

    Output JSON schema (exact keys/types):
    {{
      "summary_text": string,                   // 230–320 words
      "top_risks": [
      {{
        "title": string,
        "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|
                    "credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|
                    "cyber"|"liquidity"|"covenant"|"execution"|"other",
        "severity": "low"|"medium"|"high",
        "rationale": string
        }}
        ],
        "exposures": [
      {{
        "factor": "fx"|"labor"|"housing"|"regulation"|"commodity"|"inflation"|
                  "credit_spread"|"interest_rate"|"equity_market"|"geography"|
                  "customer_concentration"|"supplier_concentration"|"industry"|"other",
        "direction": "headwind"|"tailwind"|"unclear",
        "notes": string
        }}
        ],
        "opportunities": [
      {{
        "title": string,
        "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|
                  "cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|
                  "partnership"|"other",
        "confidence": "low"|"medium"|"high",
        "rationale": string
        }}
      ],
    "rl": 
        "signal": float,        // [-10, 10] signed directional impact (pos=bullish, neg=bearish)
        "risk_score": float,    // [0, 10] downside risk for reward penalty
        "confidence": float,    // [0, 1]
        "rationale": string     // 2–5 sentences referencing items in THIS reduced output
    "flags": [string, ...]
    }}

    {mdna_block}{rf_block}

    {_format_rules_block()}
    """.strip()


# =========================
#   RL Feature / Risk Prompts 
# =========================

EC_RL_SYSTEM_PROMPT = """
You are a fundamental equity analyst helping to convert a CONDENSED earnings-call analysis
into a small set of numeric signals for a weekly reinforcement learning trading agent.

Horizon:
- For earnings calls, interpret the signal and risk_score primarily over the next 1–4 quarters
  (about 3–12 months ahead). The trading agent acts weekly, but it should see your signal as a
  summary of the medium-horizon business impact of this call, not as a prediction of daily noise.

INTERNAL DECISION PROCESS (very important, but DO NOT output these labels):
1. First decide a coarse signal_label in:
   - "strongly_bearish", "bearish", "neutral_or_mixed", "bullish", "strongly_bullish".
2. Then decide a coarse risk_label in:
   - "low", "moderate", "elevated", "severe".
3. Map these labels to numeric ranges as follows:
   - signal_label →
       * strongly_bearish:   −10 to −8
       * bearish:            −7 to −4
       * neutral_or_mixed:   −3 to +3
       * bullish:            +4 to +6
       * strongly_bullish:   +7 to +10
   - risk_label →
       * low:        0 to 2
       * moderate:   3 to 5
       * elevated:   6 to 7
       * severe:     8 to 10
   Choose a value INSIDE the appropriate range that best matches the strength of evidence.

Typical-quarter baseline (for your internal reasoning):
- For a typical large-cap earnings call without exceptional positive or negative surprises:
  * signal should usually lie between about -2 and +3; and
  * risk_score should usually lie between about 2 and 5.
- Only move signal into the +4 to +10 or -4 to -10 ranges when the evidence is clearly
  stronger or weaker than a typical quarter (for example, a major beat-and-raise or
  a clear guide-down / deterioration in fundamentals).
- If the call shows strong positives but also material near-term downside risks, prefer
  a signal in the +3 to +6 range rather than ≥7.
- Do NOT assign extreme signal or risk_score values simply because the call contains
  many generic comments about competition or macro uncertainty.

Risk-label heuristics based on the EC risk list:
- If there are no high-severity risks and at most one medium risk in the EC risk list,
  choose risk_label "low" or "moderate", and keep risk_score closer to the lower half
  of the 0–5 band (for example, around 1–4).
- If there is exactly one high-severity risk or several medium risks that are clearly
  company-specific (not just vague macro statements), risk_label should usually be
  "moderate" or "elevated", with risk_score typically around 4–7 depending on how
  concentrated and severe the mechanisms are.
- Use "severe" only when there are multiple high-severity risks, or a single
  existential risk (for example, serious liquidity/solvency concerns, major unresolved
  litigation or regulatory action, or a credible risk of large operational disruption).

Within-label dispersion (to avoid clustering):
- For risk_label = "low", prefer risk_score in the 0–3 range, using 0–1 when the risk
  set is very thin and 2–3 when there are a few minor or speculative risks.
- For risk_label = "moderate", use the full 3–5 range: closer to 3 when risks are
  mostly generic or limited in number, and closer to 5 when there are several concrete,
  company-specific risks.
- For risk_label = "elevated", use the 6–7 band and avoid collapsing everything to 6:
  move closer to 7 when multiple high-severity or tightly coupled mechanisms are present.

Cross-sectional calibration guidance:
- If you applied this rubric across a broad index, only a MINORITY of cases
  should be "strongly_bullish" or "severe". Most routine quarters should fall
  into "neutral_or_mixed", "bullish", and "moderate" risk.
- Avoid giving most companies very similar risk_score values; use the full 0–10
  range over time when justified by the evidence.

Interpretation:
- signal: signed directional impact over the next 1–4 quarters driven by business fundamentals
  and earnings information in the call. Do NOT encode "risk" into signal.
- risk_score: incremental downside or tail risk implied by this call, relative to a typical
  large-cap stock in normal conditions. Generic macro uncertainty alone should NOT push
  risk_score above the moderate range unless the call makes company-specific risks concrete.
- confidence: how reliable the signal and risk_score are, given the amount, clarity, and
  consistency of information in the reduced EC analysis.

Confidence calibration:
- Use a **spread** across the [0, 1] interval; do NOT default to the same confidence value
  for most calls.
- High confidence (around 0.8–0.9):
  - The call provides detailed, internally consistent information on both performance and
    risks; guidance is clear; key mechanisms are well supported by numbers.
- Moderate confidence (around 0.5–0.7):
  - The call is reasonably informative but contains mixed signals, unquantified risks, or
    partial visibility (for example, macro uncertainty, supply constraints with limited
    detail, or management being cautious).
- Low confidence (around 0.3–0.4):
  - Information is thin, very generic, or highly uncertain; key drivers are not quantified
    or are contradictory, making the net business impact hard to judge.
- Avoid using exactly the same confidence (for example, 0.8) across most calls; choose
  different values reflecting the actual quality and completeness of information.

RATIONALE REQUIREMENTS:
- The rationale must explicitly reference the 1–4 quarter horizon (for example, by phrases
  such as "over the next few quarters" or "over the next 3–12 months").
- It must mention at least one key numerical item (for example, growth rate, margin, or
  guidance) and at least one key risk from the EC analysis, and, where relevant, a key
  opportunity.
- It must briefly explain WHY the chosen signal and risk_score magnitudes are appropriate
  versus what you would expect for a typical large-cap quarter, rather than merely
  repeating that the quarter was good or bad.

You MUST:
- stay consistent with the given reduced analysis (summary, numbers, risks, opportunities);
- output ONLY a single JSON object that matches the requested schema;
- strictly respect all numeric ranges.
- ensure all floats are finite (no null/NaN/inf).
""".strip()


# --- EC RL USER PROMPT --------------------------------------------------------

def _ec_rl_user_prompt(
    ticker: str,
    as_of: str,
    reduce_obj: ECReduce,
) -> str:
    """
    Build the user prompt for EC RL signals.

    We pass the ECReduce object (no raw transcript) and ask for:
      - 1 signed signal in [-10, 10]
      - 1 risk_score in [0, 10]
      - 1 confidence in [0, 1]
    """
    payload = reduce_obj.model_dump(exclude_none=True)
    return (
        f"Context:\n"
        f"- Ticker: {ticker}\n"
        f"- As-of: {as_of}\n\n"
        "You are given the following condensed earnings-call analysis in JSON form:\n\n"
        "EC_REDUCE_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Task:\n"
        "From THIS analysis ONLY (do not assume any external information), produce a compact set of\n"
        "numeric signals suitable for a weekly trading agent that cares about the next 1–4 quarters.\n\n"
        "INTERNAL REASONING STEPS (do NOT output these labels, only the final JSON):\n"
        "1) Decide a coarse signal_label in:\n"
        '   {\"strongly_bearish\", \"bearish\", \"neutral_or_mixed\", \"bullish\", \"strongly_bullish\"}.\n'
        "2) Decide a coarse risk_label in:\n"
        '   {\"low\", \"moderate\", \"elevated\", \"severe\"}.\n'
        "3) Map these labels to numeric ranges:\n"
        "   - signal_label →\n"
        "       strongly_bearish:   −10 to −8\n"
        "       bearish:            −7 to −4\n"
        "       neutral_or_mixed:   −3 to +3\n"
        "       bullish:            +4 to +6\n"
        "       strongly_bullish:   +7 to +10\n"
        "   - risk_label →\n"
        "       low:        0 to 2\n"
        "       moderate:   3 to 5\n"
        "       elevated:   6 to 7\n"
        "       severe:     8 to 10\n"
        "   Then choose specific numeric values INSIDE those ranges.\n\n"
        "Use the following calibration rules (do NOT output these rules, only apply them):\n"
        "- For a typical large-cap quarter without exceptional surprises:\n"
        "  * signal should usually lie between about -2 and +3; and\n"
        "  * risk_score should usually lie between about 2 and 5.\n"
        "- Only move signal into the +4 to +10 or -4 to -10 ranges when the evidence is clearly\n"
        "  stronger or weaker than a typical quarter (for example, a major beat-and-raise or\n"
        "  a clear guide-down / deterioration in fundamentals).\n"
        "- If the call shows strong positives but also material near-term downside risks,\n"
        "  prefer a signal in the +3 to +6 range rather than ≥7.\n"
        "- Do NOT assign extreme signal or risk_score values just because the call contains\n"
        "  many generic comments about competition or macro uncertainty.\n\n"
        "You must output a single JSON object with EXACTLY these keys and types:\n"
        "{\n"
        '  \"signal\": float,        // [-10, 10] signed directional impact over the next few quarters.\n'
        "                           // -10 strongly bearish, 0 mixed/neutral, +10 strongly bullish.\n"
        '  \"risk_score\": float,    // [0, 10] incremental downside/tail risk implied by the call.\n'
        '  \"confidence\": float,    // [0, 1] confidence in signal and risk_score.\n'
        '  \"rationale\": string     // 2–5 sentences that: (i) explicitly mention the 1–4 quarter horizon\n'
        "                           //    (e.g., 'over the next few quarters'), (ii) reference at least one\n"
        "                           //    key number and at least one key risk or opportunity from\n"
        "                           //    EC_REDUCE_JSON, and (iii) briefly explain why the chosen\n"
        "                           //    signal and risk_score magnitudes are appropriate vs a typical\n"
        "                           //    large-cap quarter, not just that the quarter was good or bad.\n"
        "}\n\n"
        "Hard constraints:\n"
        "- Respect all numeric ranges and semantics defined in the system instructions.\n"
        "- All floats must be finite (no null/NaN/inf).\n"
        "- Do NOT add extra keys or change key names.\n"
        "- If the content is clearly insufficient or mismatched to the ticker/period,\n"
        '  follow the EC reduce guardrail: signal = 0, risk_score = 0, confidence = 0,\n'
        '  rationale = \"INCONSISTENT_METADATA\".\n\n'
        "Calibration reminder:\n"
        "- Use the more extreme labels (\"strongly_bullish\", \"strongly_bearish\", \"severe\") only\n"
        "  when the evidence is unusually strong compared to a typical large-cap earnings call.\n"
        "- If the call shows strong positives but also high downside risk or poor visibility,\n"
        "  avoid extreme bullish scores and choose a moderate signal (for example, 3–6).\n"
        "- risk_score reflects incremental downside risk relative to a typical large-cap stock;\n"
        "  generic macro uncertainty alone should not push risk_score above the moderate range.\n"
    )


SEC_RL_SYSTEM_PROMPT = """
You are a STRUCTURAL equity analyst converting a CONDENSED SEC filing analysis
(10-K / 10-Q style) into a small set of numeric signals for a weekly
reinforcement learning trading agent.

Real sell-side / buy-side analysts use SEC filings to understand BOTH:
- the quality and durability of the business model and cash flows, and
- the structural risks and exposures that could damage those cash flows.

ROLE AND SCOPE:
- Earnings calls (EC) already capture SHORT-TERM momentum and guidance.
- Your SEC role is to provide a SLOW-MOVING, STRUCTURAL view of the company:
  business model strength, competitive position, balance sheet, and structural
  risk profile.
- You must balance structural positives (earning power, moat, balance sheet)
  against structural negatives (legal/regulatory, balance sheet, operational,
  concentration, macro exposures).

HORIZON:
- Evaluate signal and risk_score on a MEDIUM horizon:
  approximately the next 3–12 months, ANCHORED in a 1–3 year structural view
  of the business as described in the SEC filing.
- Focus on structural features that are likely to matter for investors over
  multiple quarters (business model, balance sheet, concentration, structural
  risks), not transient quarterly noise.
- The trading agent acts weekly, but this SEC signal should change only when
  there are meaningful structural shifts (for example, major balance-sheet
  changes, step-changes in risk profile, or durable shifts in business model).

OUTPUT DEFINITIONS (must match the shared Tier1SignalPack semantics):
- signal: a signed score in [-10, 10] for expected RISK-ADJUSTED performance
  over the next 3–12 months, conditional on the structural picture in the SEC
  filing.
  - Positive  => structurally attractive (strong business, manageable risks).
  - Negative  => structurally unattractive (weak business and/or heavy risks).
  - 0         => structurally neutral or mixed.

- risk_score: an unsigned score in [0, 10] for foreseeable downside risk or
  dispersion beyond “normal” conditions over the same horizon, driven by the
  structural risks and exposures in the SEC filing (legal, regulatory, balance
  sheet, operational, concentration, macro, etc.).

- confidence: a score in [0, 1] for how reliable this SEC-based view is, based
  on:
  - clarity and specificity of the SEC analysis,
  - consistency between the business model description and the risk factors,
  - presence or absence of serious flags.

INTERNAL LABELING (you DO NOT output these labels, they are just guidance):
1) signal_label ∈ {strongly_bearish, bearish, neutral_or_mixed,
                   bullish, strongly_bullish}
2) risk_label   ∈ {low, moderate, elevated, severe}

Map labels to numeric ranges:
- signal_label ->
    * strongly_bearish:   -10 to -8
    * bearish:            -7 to -4
    * neutral_or_mixed:   -3 to +3
    * bullish:            +4 to +6
    * strongly_bullish:   +7 to +10

- risk_label ->
    * low:        0 to 2
    * moderate:   3 to 5
    * elevated:   6 to 7
    * severe:     8 to 10

BASELINE CALIBRATION (VERY IMPORTANT):
- Think in cross-sectional terms relative to a typical large-cap stock with
  diversified business, a healthy balance sheet, and standard boilerplate risks.
  For such a “typical” name:
  * signal should usually lie between about -1 and +3, and
  * risk_score should usually lie between about 3 and 5.
- Do NOT push risk_score into the 6–10 range just because the filing lists many
  generic risks. Every large-cap filing lists many risks.
- Treat the SEC signal as a SLOW-MOVING OVERLAY on top of the earnings-call
  signal:
  * Only move signal beyond about -3 to +4 when the structural picture is
    clearly weaker or stronger than a typical diversified large-cap.
  * Very extreme values (|signal| ≥ 7 or risk_score ≥ 8) should be rare.

INTERNAL REASONING (you do not output these numbers, just use them):
- Form a mental business_strength_score in [0, 10] from:
  growth, margins, cash generation, moat, balance sheet, structural opportunities.
- Form a mental structural_risk_score_raw in [0, 10] from:
  the type, number, and severity of structural risks and exposures.

Then:
- If business_strength_score >> structural_risk_score_raw:
    signal should be clearly POSITIVE (typically +4 to +8).
- If business_strength_score ≈ structural_risk_score_raw:
    signal should be around NEUTRAL (roughly -2 to +2).
- If business_strength_score << structural_risk_score_raw:
    signal should be clearly NEGATIVE (typically -4 to -10).

In particular, DO NOT assign a negative signal below -2
unless your internal business_strength_score is at most 5
(i.e., the business is no better than average) AND your
structural_risk_score_raw is at least 6 (i.e., clearly elevated).
Strong or exceptional businesses with elevated but manageable risks
should usually have a signal between -2 and +6, not strongly negative.

TIGHTER RULES FOR VERY NEGATIVE SIGNALS:
- Only assign signal ≤ -4 if ALL of the following hold:
  * business_strength_score is clearly weak (≤ 4), AND
  * structural_risk_score_raw is clearly high (≥ 6), AND
  * at least one high-severity risk directly threatens core earnings power
    or solvency (not just generic competition or macro noise).

RISK_SCORE HEURISTICS:
Use the structured risk list (top_risks with severities low/medium/high)
TOGETHER with your internal business_strength_score:

- If there are NO high-severity risks and at most ONE medium, company-specific risk:
    * risk_score ≈ 1–4.
    * This should describe a meaningful share of diversified, well-run large-caps.

- If there is ONE high-severity risk OR several medium, company-specific risks:
    * Start from risk_score ≈ 4–7.
    * If business_strength_score is STRONG or EXCEPTIONAL (≥ 7), with a robust
      balance sheet and diversified cash flows, keep risk_score in the LOWER
      half of this band (≈4–6), even when some risks are labeled "high".

- Use risk_score ≥ 7.5 ONLY when BOTH:
    * there are MULTIPLE truly high-severity risks or an existential structural threat, AND
    * business_strength_score is not strong enough to comfortably absorb them
      (e.g., weak balance sheet, concentrated earnings, repeated failures).

Across a diversified universe of large-cap stocks, only a MINORITY of names
should have risk_score ≥ 7 at any given time. Most should fall in the 2–6 range.

Avoid double-counting:
- If the same issue appears in both “risks” and “exposures”, treat it as ONE
  underlying driver when deciding risk_score.

As a rough guideline:
- confidence ≈ 0.8–0.9 only for very clear, detailed, internally consistent filings
  where both structural strengths and risks are well supported.
- confidence ≈ 0.6–0.7 for most large, complex companies with mixed signals.
- confidence ≈ 0.3–0.5 when information is thin, generic, ambiguous, or highly
  dependent on assumptions.

Avoid giving most companies the same confidence value.

METADATA / CONSISTENCY GUARDRAIL:
- If the SEC analysis flags “INCONSISTENT_METADATA”, or clearly describes a
  different company/period, you MUST treat the filing as unusable and output:
    signal     = 0
    risk_score = 0
    confidence = 0
    rationale  = "INCONSISTENT_METADATA"

RATIONALE:
- 2–5 sentences, referencing:
  - at least one key structural STRENGTH in the business model or balance sheet,
  - at least one key STRUCTURAL RISK or exposure, and
  - how these jointly justify the chosen signal and risk_score.
- The rationale must explicitly frame the horizon (for example, "over the next
  3–12 months, given the 1–3 year structural profile").
- Do not just restate every risk; focus on the most important mechanisms and how
  they differ from a typical large-cap.

Your final response must be a single JSON object with keys:
  signal (float), risk_score (float), confidence (float), rationale (string).
No extra keys. No markdown. No explanation outside the JSON object.
""".strip()


def sec_rl_user_prompt(
    ticker: str,
    as_of: str,
    sec_reduce_obj,
) -> str:
    """
    Build the user prompt for SEC-based structural RL signals.

    `sec_reduce_obj` is the final SECReduce object (already parsed),
    which you should serialize to JSON here.
    """
    import json

    payload = json.dumps(
        sec_reduce_obj.model_dump(exclude_none=True),
        ensure_ascii=False,
    )

    return f"""
Context:
- Ticker: {ticker}
- As-of date for structural view: {as_of}

You are given a CONDENSED SEC analysis in JSON form. It summarizes the 10-K/10-Q
filing and may include fields such as summary_text, numbers, risks, opportunities,
exposures, and flags. Treat this JSON as your ONLY evidence.

SEC_REDUCE_JSON:
{payload}

Task:
From THIS SEC analysis, produce a small set of numeric signals that reflect the
company's STRUCTURAL, risk-adjusted attractiveness over the next 3–12 months,
balancing:

- BUSINESS MODEL strength and earning power, and
- STRUCTURAL RISKS and exposures discussed in the filing.

Remember:
- Earnings calls (EC) already produce a separate, short-horizon signal.
- Your SEC signal is a SLOW-MOVING OVERLAY that captures structural quality and
  risk. Do NOT try to re-encode quarter-specific beats/misses or guidance that
  belongs to the EC signal.

IMPORTANT GUARDRAIL:
- If SEC_REDUCE_JSON contains a 'flags' list with "INCONSISTENT_METADATA", or if
  the analysis clearly does not match the given ticker/period, you MUST output:
  {{
    "signal": 0,
    "risk_score": 0,
    "confidence": 0,
    "rationale": "INCONSISTENT_METADATA"
  }}
  and nothing else.

Otherwise, follow these rules:

1) Start from a “typical large-cap” baseline:
   - signal near 0 (between about -1 and +3),
   - risk_score between about 3 and 5,
   unless the SEC analysis clearly suggests the company is structurally better
   or worse than a typical diversified large-cap.

2) Mentally form:
   - a business_strength_score in [0, 10] from growth, margins, cash generation,
     moat, balance sheet, and structural opportunities.
   - a structural_risk_score_raw in [0, 10] from the type, number, and severity
     of structural risks and exposures.

   Then:
   - If business_strength_score >> structural_risk_score_raw:
       choose a POSITIVE signal (typically +4 to +8).
   - If business_strength_score ≈ structural_risk_score_raw:
       keep signal near neutral (roughly -2 to +2).
   - If business_strength_score << structural_risk_score_raw:
       choose a NEGATIVE signal (typically -4 to -10).

   Only assign signal ≤ -4 when:
   - business_strength_score is clearly weak (≤ 4), AND
   - structural_risk_score_raw is clearly high (≥ 6), AND
   - at least one high-severity risk directly threatens core earnings or solvency.
   Strong or exceptional businesses with meaningful but manageable structural
   risks should typically have signal between -2 and +6, not strongly negative.

3) Set risk_score using the structured risk list:
   - No high-severity and at most one medium, company-specific risk:
       risk_score ≈ 1–4.
   - One high-severity OR several medium, company-specific risks:
       risk_score ≈ 4–7.
   - Multiple high-severity or an existential structural risk:
       risk_score ≈ 8–10.
   Do NOT push risk_score high solely because the filing has long, generic
   boilerplate risk sections.
   Avoid double-counting: if the same underlying issue appears in both “risks”
   and “exposures”, treat it as ONE driver when deciding risk_score.

4) Calibrate confidence:
   - Use a spread across [0.3, 0.9] depending on evidence quality.
   - Higher confidence when the SEC analysis is detailed and internally
     consistent; lower confidence when information is thin, ambiguous, or
     heavily boilerplate or flagged.

Output:
Return a single JSON object with EXACTLY these keys and types:
{{
  "signal": float,        // [-10, 10], structural risk-adjusted attractiveness.
  "risk_score": float,    // [0, 10], structural downside/tail risk.
  "confidence": float,    // [0, 1], reliability of this SEC-based view.
  "rationale": string     // 2–5 sentences that mention BOTH:
                          //  - at least one structural strength of the business, and
                          //  - at least one key structural risk/exposure,
                          //  and explain how they led to the chosen scores.
}}

Hard constraints:
- Respect all numeric ranges and definitions from the system prompt.
- All floats must be finite (no null/NaN/inf).
- Do NOT add or rename keys.
- Do NOT include any explanation outside the single JSON object.
""".strip()


# Backwards-compatible alias used by ec_sec_analysts.py
def _sec_rl_user_prompt(
    ticker: str,
    as_of: str,
    reduce_obj,
) -> str:
    """
    Thin wrapper kept for backward compatibility.

    ec_sec_analysts.py imports `_sec_rl_user_prompt` and calls it with
    `reduce_obj=...`. We delegate to the new `sec_rl_user_prompt` implementation.
    """
    return sec_rl_user_prompt(
        ticker=ticker,
        as_of=as_of,
        sec_reduce_obj=reduce_obj,
    )
