from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# =========================
#   Robust BaseModel
# =========================

class RobustBaseModel(BaseModel):
    """
    Base model that:
    - ignores unknown / extra keys from the LLM,
    - works in both Pydantic v1 and v2.
    """
    model_config = {"extra": "ignore"}

# =========================
#   Pydantic Schemas
# =========================

class NumberItem(RobustBaseModel):
    name: str = Field(..., description="e.g., EPS, revenue, gross_margin, guidance_eps")
    value: str = Field(..., description="raw textual value, keep units/format")
    unit: Optional[str] = None
    period: Optional[str] = Field(None, description="e.g., Q3 FY2024, FY2025, next quarter")
    context: Optional[str] = Field(None, description="e.g., GAAP, non-GAAP, YoY, QoQ, etc.")
    source: Optional[str] = Field(None, description="prepared_remarks, Q&A, MD&A, Risk Factors")


class RiskItem(RobustBaseModel):
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
        "other"
    ]] = "other"
    severity: Optional[Literal["low", "medium", "high"]] = "medium"
    rationale: str

    @field_validator("category", mode="before")
    def _normalize_category(cls, v):
        if v is None:
            return "other"
        s = str(v).strip().lower()
        # simple synonym mapping for common LLM outputs
        synonym_map = {
            "market": "macro",
            "market_risk": "macro",
            "regulatory_risk": "regulatory",
            "credit_risk": "credit",
            "liquidity_risk": "liquidity",
            "fx_risk": "fx",
            "currency": "fx",
            "technology_risk": "technology",
            "it": "technology",
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
        # allow things like "moderate" → "medium"
        synonym_map = {
            "moderate": "medium",
            "med": "medium",
            "high_risk": "high",
            "low_risk": "low",
        }
        if s in synonym_map:
            s = synonym_map[s]
        return s if s in allowed else "medium"


class ExposureItem(RobustBaseModel):
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
        "other"
    ]] = "other"
    direction: Optional[Literal["headwind", "tailwind", "unclear"]] = "unclear"
    notes: str

    @field_validator("factor", mode="before")
    def _normalize_factor(cls, v):
        if v is None:
            return "other"
        s = str(v).strip().lower()
        synonym_map = {
            "currency": "fx",
            "fx_risk": "fx",
            "rates": "interest_rate",
            "interest": "interest_rate",
            "rate": "interest_rate",
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
            "positive": "tailwind",
            "supportive": "tailwind",
            "benefit": "tailwind",
            "neutral": "unclear",
            "mixed": "unclear",
        }
        if s in synonym_map:
            s = synonym_map[s]
        return s if s in allowed else "unclear"
    
class OpportunityItem(RobustBaseModel):
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
        "other"
    ]] = "other"
    confidence: Optional[Literal["low", "medium", "high"]] = "medium"
    rationale: str

    @field_validator("driver", mode="before")
    def _normalize_driver(cls, v):
        if v is None:
            return "other"
        s = str(v).strip().lower()
        # map common LLM outputs to canonical buckets
        synonym_map = {
            "digital": "technology",
            "ai": "technology",
            "cloud": "technology",
            "platform": "technology",
            "online": "technology",
            "capital": "cost",
            "capital_efficiency": "efficiency",
            "margin": "pricing",
            "price": "pricing",
            "marketing": "brand",
            "distribution": "channel",
            "go_to_market": "channel",
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
            "weak": "low",
        }
        if s in synonym_map:
            s = synonym_map[s]
        return s if s in allowed else "medium"

# ---- EC ----

class ECChunkMap(RobustBaseModel):
    bullets: List[str] = Field(..., description="3–6 concise factual statements")
    numbers: List[NumberItem] = []
    risks: List[RiskItem] = []
    opportunities: List[OpportunityItem] = []

class ECReduce(RobustBaseModel):
    summary_text: str
    numbers: List[NumberItem] = []
    risks: List[RiskItem] = []
    opportunities: List[OpportunityItem] = []


# ---- SEC ----

class SECChunkMapMDNA(RobustBaseModel):
    bullets: List[str] = []
    exposures: List[ExposureItem] = []
    opportunities: List[OpportunityItem] = []
    risks: List[RiskItem] = []
    red_flags: List[str] = []

class SECChunkMapRF(RobustBaseModel):
    bullets: List[str] = []
    risks: List[RiskItem] = []
    red_flags: List[str] = []

class SECReduceSection(RobustBaseModel):
    section: str
    summary_text: str
    top_risks: List[RiskItem] = []
    exposures: List[ExposureItem] = []
    opportunities: List[OpportunityItem] = []
    flags: List[str] = []

class SECReduceFinal(RobustBaseModel):
    summary_text: str
    top_risks: List[RiskItem] = []
    exposures: List[ExposureItem] = []
    opportunities: List[OpportunityItem] = []
    flags: List[str] = []


# =========================
#   Prompt Strings
# =========================

def _format_rules_block() -> str:
    return """
Important formatting rules:
- Respond with ONLY a single JSON object.
- Do NOT include explanations, comments, or markdown.
- Do NOT wrap the JSON in code fences.
- Do NOT output multiple JSON objects or arrays at the top level.
""".strip()


def map_system_prompt() -> str:
    return (
        "You are a precise financial analyst. "
        "Return STRICTLY valid JSON matching the requested schema. "
        "Do not include any text outside JSON. "
        "Do not invent keys not listed in the schema. Omit null/unknown fields."
    )


def reduce_system_prompt() -> str:
    return (
        "Merge JSON snippets deterministically. Deduplicate and reconcile conflicts. "
        "Return STRICTLY valid JSON matching the requested schema. "
        "No text outside JSON. Do not add keys not in the schema; omit nulls."
    )


def ec_map_user_prompt(ticker: str, as_of: str, compression_ratio: float, chunk_text: str) -> str:
    return f"""
Context:
- Ticker: {ticker}
- As-of: {as_of}

Task:
Summarize the following earnings-call segment for {ticker}. Be factual.
Target compression ratio ≈ {compression_ratio:.2f} relative to input length.

Output JSON schema (exact keys/types):
{{
  "bullets": [string, ...],                 // 3–6 concise factual statements
  "numbers": [                              // optional; omit if none
    {{
      "name": string,                       // e.g., "EPS", "revenue", "gross_margin"
      "value": string,                      // keep units/format as text
      "unit": string|null,
      "period": string|null,                // e.g., "Q3 FY2024"
      "context": string|null,               // e.g., "GAAP", "YoY"
      "source": string|null                 // e.g., "prepared_remarks", "Q&A"
    }}
  ],
  "risks": [
    {{
      "title": string,
      "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|"credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|"cyber"|"liquidity"|"covenant"|"execution"|"other",
      "severity": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "opportunities": [
    {{
      "title": string,
      "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|"cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|"partnership"|"other",
      "confidence": "low"|"medium"|"high",
      "rationale": string
    }}
  ]
}}

Text:
\"\"\"{chunk_text}\"\"\"

{_format_rules_block()}
""".strip()


def sec_mdna_map_user_prompt(ticker: str, as_of: str, compression_ratio: float, chunk_text: str) -> str:
    return f"""
Context:
- Ticker: {ticker}
- As-of: {as_of}

Task:
Summarize this SEC MD&A segment. Extract exposures, opportunities, and risks. Be concise and factual.
Target compression ratio ≈ {compression_ratio:.2f}.

Output JSON schema (exact keys/types):
{{
  "bullets": [string, ...],
  "exposures": [
    {{
      "factor": "fx"|"labor"|"housing"|"regulation"|"commodity"|"inflation"|"credit_spread"|"interest_rate"|"equity_market"|"geography"|"customer_concentration"|"supplier_concentration"|"industry"|"other",
      "direction": "headwind"|"tailwind"|"unclear",
      "notes": string
    }}
  ],
  "opportunities": [
    {{
      "title": string,
      "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|"cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|"partnership"|"other",
      "confidence": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "risks": [
    {{
      "title": string,
      "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|"credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|"cyber"|"liquidity"|"covenant"|"execution"|"other",
      "severity": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "red_flags": [string, ...]
}}

Text:
\"\"\"{chunk_text}\"\"\"

{_format_rules_block()}
""".strip()


def sec_rf_map_user_prompt(ticker: str, as_of: str, compression_ratio: float, chunk_text: str) -> str:
    return f"""
Context:
- Ticker: {ticker}
- As-of: {as_of}

Task:
Summarize this SEC Risk Factors segment. Extract risks and red flags. Be concise and factual.
Target compression ratio ≈ {compression_ratio:.2f}.

Output JSON schema (exact keys/types):
{{
  "bullets": [string, ...],
  "risks": [
    {{
      "title": string,
      "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|"credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|"cyber"|"liquidity"|"covenant"|"execution"|"other",
      "severity": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "red_flags": [string, ...]
}}

Text:
\"\"\"{chunk_text}\"\"\"

{_format_rules_block()}
""".strip()


def ec_reduce_user_prompt(ticker: str, as_of: str, maps_compact_jsonl: str) -> str:
    return f"""
Context:
- Ticker: {ticker}
- As-of: {as_of}

Task:
Merge the following partial EC summaries (one JSON object per line). Each object may come
from a single chunk or from a previous merge step. Deduplicate items, prefer precise values,
and keep wording concise and factual.

Output JSON schema (exact keys/types):
{{
  "summary_text": string,                   // 120–180 words
  "numbers": [                              // optional; omit if none
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
      "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|"credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|"cyber"|"liquidity"|"covenant"|"execution"|"other",
      "severity": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "opportunities": [
    {{
      "title": string,
      "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|"cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|"partnership"|"other",
      "confidence": "low"|"medium"|"high",
      "rationale": string
    }}
  ]
}}

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
Merge the partial SEC {section} summaries (one JSON object per line). Each object may come
from a single chunk or from a previous merge step. Deduplicate and reconcile.

Output JSON schema (exact keys/types):
{{
  "section": "MD&A"|"Risk Factors",
  "summary_text": string,                   // 120–180 words
  "top_risks": [                            // omit if none
    {{
      "title": string,
      "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|"credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|"cyber"|"liquidity"|"covenant"|"execution"|"other",
      "severity": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "exposures": [                            // MD&A only; omit for Risk Factors
    {{
      "factor": "fx"|"labor"|"housing"|"regulation"|"commodity"|"inflation"|"credit_spread"|"interest_rate"|"equity_market"|"geography"|"customer_concentration"|"supplier_concentration"|"industry"|"other",
      "direction": "headwind"|"tailwind"|"unclear",
      "notes": string
    }}
  ],
  "opportunities": [                        // MD&A only; omit for Risk Factors
    {{
      "title": string,
      "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|"cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|"partnership"|"other",
      "confidence": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "flags": [string, ...]                    // consolidated red flags, if any
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

Output JSON schema (exact keys/types):
{{
  "summary_text": string,                   // 150–220 words
  "top_risks": [
    {{
      "title": string,
      "category": "regulatory"|"tax"|"reputation"|"esg"|"governance"|"technology"|"credit"|"litigation"|"supply_chain"|"macro"|"fx"|"competition"|"cyber"|"liquidity"|"covenant"|"execution"|"other",
      "severity": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "exposures": [
    {{
      "factor": "fx"|"labor"|"housing"|"regulation"|"commodity"|"inflation"|"credit_spread"|"interest_rate"|"equity_market"|"geography"|"customer_concentration"|"supplier_concentration"|"industry"|"other",
      "direction": "headwind"|"tailwind"|"unclear",
      "notes": string
    }}
  ],
  "opportunities": [
    {{
      "title": string,
      "driver": "product"|"pricing"|"mix"|"geography"|"fx"|"regulation"|"channel"|"cost"|"brand"|"efficiency"|"demand"|"technology"|"acquisition"|"partnership"|"other",
      "confidence": "low"|"medium"|"high",
      "rationale": string
    }}
  ],
  "flags": [string, ...]
}}

{mdna_block}{rf_block}

{_format_rules_block()}
""".strip()
