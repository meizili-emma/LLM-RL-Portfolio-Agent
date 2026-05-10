from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

import json
from pathlib import Path
import pandas as pd

from src.llm_agents.utils import _structured_call, _iso
from src.llm_agents.tier2_schema import (
    BusinessAnalystOutput,
    RiskAnalystOutput,
    SkepticOutput,
    SeniorAnalystVerdict,
    BUSINESS_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
    SKEPTIC_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT 

)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class Tier2Config:
    tier1_panel_path: str
    output_path: str
    model: Dict[str, Any]
    llm_retries: int = 1
    max_rows: Optional[int] = None


def load_config(path: str) -> Tier2Config:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    return Tier2Config(
        tier1_panel_path=raw["tier1_panel_path"],
        output_path=raw["output_path"],
        model=raw["model"],
        llm_retries=int(raw.get("llm_retries", 1)),
        max_rows=raw.get("max_rows"),
    )


# -----------------------------------------------------------------------------
# Context construction helpers
# -----------------------------------------------------------------------------

def build_tier2_context_from_row(row: pd.Series) -> Dict[str, Any]:
    """
    Build the JSON-serializable context dict passed into Tier-2 LLM agents
    for a single ticker-week.
    """

    def _to_iso_date(x):
        if pd.isna(x):
            return None
        ts = pd.to_datetime(x)
        return ts.date().isoformat()

    def _to_iso_datetime(x):
        if x is None or pd.isna(x):
            return None
        return _iso(x)

    # Tier-1 per-source views
    tier1_ec = None
    if "ec_signal" in row and not pd.isna(row["ec_signal"]):
        tier1_ec = {
            "signal": float(row["ec_signal"]),
            "risk_score": float(row.get("ec_risk_score", 0.0) or 0.0),
            "confidence": float(row.get("ec_confidence", 0.0) or 0.0),
            "rationale": str(row.get("ec_rationale", "") or ""),
            "summary": str(row.get("ec_summary", "") or ""),
            "event_ts_utc": _to_iso_datetime(row.get("ec_event_ts_utc")),
        }

    tier1_sec = None
    if "sec_signal" in row and not pd.isna(row["sec_signal"]):
        tier1_sec = {
            "signal": float(row["sec_signal"]),
            "risk_score": float(row.get("sec_risk_score", 0.0) or 0.0),
            "confidence": float(row.get("sec_confidence", 0.0) or 0.0),
            "rationale": str(row.get("sec_rationale", "") or ""),
            "summary": str(row.get("sec_summary", "") or ""),
            "filed_at_utc": _to_iso_datetime(row.get("sec_filed_at_utc")),
            "form_type": str(row.get("sec_form_type", "") or ""),
        }

    tier1_news = None
    if "news_signal" in row and not pd.isna(row["news_signal"]):
        tier1_news = {
            "signal": float(row["news_signal"]),
            "risk_score": float(row.get("news_risk_score", 0.0) or 0.0),
            "confidence": float(row.get("news_confidence", 0.0) or 0.0),
            "rationale": str(row.get("news_rationale", "") or ""),
            "summary": str(row.get("news_summary", "") or ""),
            # The actual news window boundaries are not stored here;
            # we document their semantics in the prompts.
        }

    tier1_tech = None
    if "tech_signal" in row and not pd.isna(row["tech_signal"]):
        tier1_tech = {
            "signal": float(row["tech_signal"]),
            "risk_score": float(row.get("tech_risk_score", 0.0) or 0.0),
            "confidence": float(row.get("tech_confidence", 0.0) or 0.0),
            "rationale": str(row.get("tech_rationale", "") or ""),
            "summary": str(row.get("tech_summary", "") or ""),
            # Technical indicators are computed from daily OHLCV and summarized weekly.
        }

    # Timeliness features
    days_since_ec = row.get("days_since_ec") if "days_since_ec" in row else None
    days_since_sec = row.get("days_since_sec") if "days_since_sec" in row else None

    # Coverage / conflict features
    n_sources_available = int(row.get("n_sources_available", 0) or 0)
    signal_sign_disagreement = bool(row.get("signal_sign_disagreement", False))

    ctx: Dict[str, Any] = {
        "ticker": str(row["ticker"]),
        "week_decision_date": _to_iso_date(row["week_decision_date"]),
        "curr_close_utc": _to_iso_datetime(row.get("curr_close_utc")),
        "tier1": {
            "ec": tier1_ec,
            "sec": tier1_sec,
            "news": tier1_news,
            "tech": tier1_tech,
        },
        "timing": {
            "days_since_ec": days_since_ec,
            "days_since_sec": days_since_sec,
            # Descriptive notes about how windows are constructed
            "notes": {
                "news_window": (
                    "news_summary and news_signal are computed over a rolling window "
                    "covering approximately the previous 7 calendar days ending at the "
                    "current decision date."
                ),
                "tech_window": (
                    "tech_summary and tech_signal are derived from daily OHLCV data "
                    "but summarized at weekly frequency at the decision date."
                ),
            },
        },
        "coverage": {
            "n_sources_available": n_sources_available,
            "signal_sign_disagreement": signal_sign_disagreement,
        },
    }

    return ctx


# -----------------------------------------------------------------------------
# LLM call wrappers
# -----------------------------------------------------------------------------

def _call_business_analyst(
    model_cfg: Dict[str, Any],
    context: Dict[str, Any],
    retries: int,
) -> BusinessAnalystOutput:
    """
    Call the Business Analyst role.

    The system prompt already defines the JSON schema:
      {
        "signal_proposal": float,
        "thesis": string,
        "key_drivers": [string, ...]
      }
    """
    ticker = context.get('ticker', '')
    user_prompt = (
        "You are evaluating ONE stock for a weekly decision.\n"
        f"{ticker}\n"
        "Use the context below to produce a BUSINESS-FOCUSED signal for the NEXT 1–4 WEEKS.\n\n"
        "IMPORTANT:\n"
        "- Follow the JSON schema described in the system prompt.\n"
        "- Return ONLY a single JSON object, no extra text.\n\n"
        "CONTEXT:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    return _structured_call(
        model_cfg=model_cfg,
        schema=BusinessAnalystOutput,
        system_prompt=BUSINESS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        retries=retries,
    )


def _call_risk_analyst(
    model_cfg: Dict[str, Any],
    context: Dict[str, Any],
    retries: int,
) -> RiskAnalystOutput:
    """
    Call the Risk Analyst role.

    JSON schema (defined in system prompt):
      {
        "risk_score_proposal": float,
        "risk_factors": [string, ...],
        "tail_events": [string, ...]
      }
    """

    ticker = context.get("ticker", "")
    user_prompt = (
        "You are evaluating FORESEEABLE DOWNSIDE and UNCERTAINTY for ONE stock "
        "over the NEXT 1–4 WEEKS.\n\n"
        f"{ticker}\n"
        "IMPORTANT:\n"
        "- Follow the JSON schema described in the system prompt.\n"
        "- Return ONLY a single JSON object, no extra text.\n\n"
        "CONTEXT:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    return _structured_call(
        model_cfg=model_cfg,
        schema=RiskAnalystOutput,
        system_prompt=RISK_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        retries=retries,
    )


def _call_skeptic(
    model_cfg: Dict[str, Any],
    context: Dict[str, Any],
    business_out: BusinessAnalystOutput,
    risk_out: RiskAnalystOutput,
    retries: int,
) -> SkepticOutput:
    """
    Call the Skeptic role.

    JSON schema (defined in system prompt):
      {
        "disagreement_points": [string, ...],
        "disagreement_score": float
      }
    """
    ticker = context.get("ticker", "")
    skeptic_context = {
        **context,
        "business": business_out.model_dump(),
        "risk": risk_out.model_dump(),
    }
    user_prompt = (
        "You are reviewing the Business Analyst and Risk Analyst proposals and looking for weaknesses.\n"
        f"{ticker}\n"
        "Stress-test their reasoning using the shared context.\n\n"
        "IMPORTANT:\n"
        "- Follow the JSON schema described in the system prompt.\n"
        "- Return ONLY a single JSON object, no extra text.\n\n"
        "CONTEXT + PROPOSALS:\n"
        f"{json.dumps(skeptic_context, ensure_ascii=False, indent=2)}"
    )
    return _structured_call(
        model_cfg=model_cfg,
        schema=SkepticOutput,
        system_prompt=SKEPTIC_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        retries=retries,
    )


def _call_judge(
    model_cfg: Dict[str, Any],
    context: Dict[str, Any],
    business_out: BusinessAnalystOutput,
    risk_out: RiskAnalystOutput,
    skeptic_out: SkepticOutput,
    retries: int,
) -> SeniorAnalystVerdict:
    """
    Call the Judge role.

    JSON schema (defined in system prompt):
      {
        "ticker": string,
        "week_decision_date": string,
        "curr_close_utc": string | null,
        "senior_signal": float,
        "senior_risk_score": float,
        "senior_confidence": float,
        "senior_rationale": string,
      }
    """
    ticker = context.get("ticker", "")

    judge_context = {
        **context,
        "business": business_out.model_dump(),
        "risk": risk_out.model_dump(),
        "skeptic": skeptic_out.model_dump(),
        "signal_definition": (
            "Signal is expected directional performance over the next 1–4 weeks, centered at 0, "
            "driven primarily by business trajectory and catalysts."
        ),
        "risk_definition": (
            "Risk score in [0,10] measures foreseeable asymmetric downside or uncertainty beyond "
            "normal conditions over the same horizon. Generic industry risks alone should not "
            "push this score high."
        ),
        "confidence_definition": (
            "Confidence in [0,1] reflects evidence strength and coherence across EC/SEC/news/tech, "
            "not optimism. High disagreement or speculative reasoning lowers confidence."
        ),
    }
    user_prompt = (
        "You are the final senior decision-maker. Produce a SINGLE integrated verdict for the NEXT 1–4 WEEKS.\n\n"
        f"{ticker}\n"
        "IMPORTANT:\n"
        "- Follow the JSON schema described in the system prompt exactly.\n"
        "- Return ONLY a single JSON object, no extra text.\n\n"
        "FULL CONTEXT + PROPOSALS:\n"
        f"{json.dumps(judge_context, ensure_ascii=False, indent=2)}"
    )
    return _structured_call(
        model_cfg=model_cfg,
        schema=SeniorAnalystVerdict,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        retries=retries,
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def run_tier2_for_row(
    row: pd.Series,
    model_cfg: Dict[str, Any],
    retries: int,
) -> Dict[str, Any]:
    """
    Run the full Tier-2 mechanism (Business, Risk, Skeptic, Judge) for a single ticker-week row.

    Returns a flat dict that includes:
      - all SeniorAnalystVerdict fields (senior_signal, senior_risk_score, senior_confidence, senior_rationale, etc.),
      - plus JSON-serialized business_out, risk_out, skeptic_out for traceability.
    """
    context = build_tier2_context_from_row(row)

    # 1) Business & Risk analysts
    business_out = _call_business_analyst(model_cfg, context, retries)
    risk_out = _call_risk_analyst(model_cfg, context, retries)

    # 2) Skeptic
    skeptic_out = _call_skeptic(model_cfg, context, business_out, risk_out, retries)

    # 3) Judge
    verdict = _call_judge(model_cfg, context, business_out, risk_out, skeptic_out, retries)

    # Ensure identifiers are consistent with the actual row keys
    verdict.ticker = str(row["ticker"])
    verdict.week_decision_date = row["week_decision_date"]

    if verdict.curr_close_utc is None and "curr_close_utc" in row:
        if row["curr_close_utc"] is not None and not pd.isna(row["curr_close_utc"]):
            verdict.curr_close_utc = _iso(row["curr_close_utc"])

    # Build flat dict for DataFrame / parquet
    result: Dict[str, Any] = {
        "senior_signal": verdict.senior_signal,
        "senior_risk_score": verdict.senior_risk_score,
        "senior_confidence": verdict.senior_confidence,
        "senior_rationale": verdict.senior_rationale,
    }

    # Add JSON-serialized Tier-2 intermediate outputs for clarity/debugging
    result["business_out_json"] = json.dumps(
        business_out.model_dump(), ensure_ascii=False
    )
    result["risk_out_json"] = json.dumps(
        risk_out.model_dump(), ensure_ascii=False
    )
    result["skeptic_out_json"] = json.dumps(
        skeptic_out.model_dump(), ensure_ascii=False
    )

    if "tier2_row_id" in row:
        result["tier2_row_id"] = int(row["tier2_row_id"])

    return result


def build_tier2_panel(cfg: Tier2Config) -> None:
    df = pd.read_parquet(cfg.tier1_panel_path)

    if cfg.max_rows is not None:
        df = df.head(cfg.max_rows).copy()

    # Create a stable row id (required for resume)
    df = df.reset_index(drop=True)
    df["tier2_row_id"] = df.index

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # Tier-2 columns we expect to produce (no identifiers here)
    tier2_cols = [
        "senior_signal",
        "senior_risk_score",
        "senior_confidence",
        "senior_rationale",
        "business_out_json",
        "risk_out_json",
        "skeptic_out_json",
    ]

    # If there is an existing output parquet, load it and index by tier2_row_id,
    # but only use it as a cache for Tier-2 columns.
    existing_tier2_index = None
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        if "tier2_row_id" in existing_df.columns:
            existing_tier2_index = existing_df.set_index("tier2_row_id")

    # Base (Tier-1) columns: these define the canonical identifiers
    base_cols = list(df.columns)

    # Combined column order: Tier-1 first, then Tier-2
    combined_cols: List[str] = list(base_cols)
    for col in tier2_cols:
        if col not in combined_cols:
            combined_cols.append(col)

    writer: Optional[pq.ParquetWriter] = None
    last_num_cols = len(combined_cols)

    try:
        # tqdm progress bar over ticker-weeks
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Running Tier-2 senior analyst",
        ):
            row_id = int(row["tier2_row_id"])

            base_dict = row.to_dict()

            # If we have a cached Tier-2 row, reuse those Tier-2 values
            tier2_vals: Dict[str, Any] = {}
            if existing_tier2_index is not None and row_id in existing_tier2_index.index:
                exist_row = existing_tier2_index.loc[row_id]
                for col in tier2_cols:
                    tier2_vals[col] = exist_row.get(col, None)
            else:
                # New row: run Tier-2 now
                tier2_result = run_tier2_for_row(row, cfg.model, cfg.llm_retries)
                for col in tier2_cols:
                    tier2_vals[col] = tier2_result.get(col, None)

            # Merge Tier-1 + Tier-2 into a single row with canonical column names
            merged_dict = {**base_dict, **tier2_vals}
            combined = {col: merged_dict.get(col, None) for col in combined_cols}

            combined_df = pd.DataFrame([combined], columns=combined_cols)

            # First row: infer schema and create writer
            if writer is None:
                table = pa.Table.from_pandas(combined_df, preserve_index=False)
                writer = pq.ParquetWriter(str(temp_path), table.schema)
            else:
                # Subsequent rows: coerce to the same schema to avoid type drift
                table = pa.Table.from_pandas(
                    combined_df,
                    schema=writer.schema,
                    preserve_index=False,
                    safe=False,
                )

            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    # Atomically replace old output (if any) with the new completed file
    temp_path.replace(out_path)

    print(
        f"[tier2_senior_analyst] Wrote: {cfg.output_path} rows={len(df)}, cols={last_num_cols}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Tier-2 senior analyst over Tier-1 panel.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config containing tier1_panel_path, output_path, and model settings.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_tier2_panel(cfg)


if __name__ == "__main__":
    main()
