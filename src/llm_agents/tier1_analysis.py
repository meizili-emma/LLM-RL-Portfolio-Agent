from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


# -------------------------
# Config
# -------------------------

@dataclass
class Tier1SyncConfig:
    calendar_path: str
    output_path: str

    tickers: Optional[List[str]] = None

    # EC
    ec_analysis_source_path: str = ""
    ec_history_window_days: int = 180

    # SEC
    sec_analysis_source_path: str = ""
    sec_history_window_days: int = 365

    # NEWS 
    news_analysis_source_path: str = ""

    tech_analysis_source_path: str = ""


def load_config(path: str) -> Tier1SyncConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return Tier1SyncConfig(
        calendar_path=cfg["calendar_path"],
        output_path=cfg["output_path"],
        tickers=cfg.get("tickers"),
        ec_analysis_source_path=cfg.get("ec_analysis_source_path", ""),
        ec_history_window_days=int(cfg.get("ec_history_window_days", 240)),
        sec_analysis_source_path=cfg.get("sec_analysis_source_path", ""),
        sec_history_window_days=int(cfg.get("sec_history_window_days", 240)),
        news_analysis_source_path=cfg.get("news_analysis_source_path", ""),
        tech_analysis_source_path=cfg.get("tech_analysis_source_path", ""),
    )


# -------------------------
# Utilities
# -------------------------

def _to_utc_ts(x) -> pd.Timestamp:
    """
    Normalize timestamps to tz-aware UTC pd.Timestamp.
    Accepts strings, datetime-like, pd.Timestamp.
    """
    if pd.isna(x):
        return pd.NaT
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    return ts


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        s = str(x)
        return s
    except Exception:
        return default


def _parse_json_maybe(obj: Any) -> Dict[str, Any]:
    """
    Accept dict or JSON string; return dict; fallback {}.
    """
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}


def _get_nested(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        if k not in cur:
            return None
        cur = cur[k]
    return cur


def _extract_rl_from_ec_summary(ec_summary: Any) -> Tuple[float, float, float, str]:
    """
    Extract (signal, risk_score, confidence, rationale) from ec_summary_json.

    This is intentionally tolerant: it tries multiple key paths commonly used in your tier-1 schemas.
    If nothing found, returns (0,0,0,"").
    """
    d = _parse_json_maybe(ec_summary)

    # Candidate locations for RL dict
    rl_candidates = [
        ("rl",),
        ("rl_signals",),
        ("signals", "rl"),
        ("analysis", "rl"),
        ("result", "rl"),
    ]

    rl: Dict[str, Any] = {}
    for p in rl_candidates:
        val = _get_nested(d, p)
        if isinstance(val, dict):
            rl = val
            break

    # If RL dict still empty, allow top-level legacy flat keys
    # (keeps pipeline running even under partial schema drift)
    if not rl:
        rl = d

    # Canonical
    signal = rl.get("signal", None)
    risk = rl.get("risk_score", None)
    conf = rl.get("confidence", None)
    rat = rl.get("rationale", None)

    # Legacy fallbacks (if your older EC used different naming)
    if signal is None:
        signal = rl.get("sentiment_score", rl.get("ec_signal", 0.0))
    if risk is None:
        risk = rl.get("event_risk_score", rl.get("risk", rl.get("ec_risk_score", 0.0)))
    if conf is None:
        conf = rl.get("conf", rl.get("ec_confidence", 0.0))
    if rat is None:
        rat = rl.get("reasoning", rl.get("justification", rl.get("ec_rationale", "")))

    return (
        _safe_float(signal, 0.0),
        _safe_float(risk, 0.0),
        _safe_float(conf, 0.0),
        _safe_str(rat, ""),
    )


# -------------------------
# Calendar + Base Panel
# -------------------------

def load_calendar(calendar_path: str) -> pd.DataFrame:
    cal = pd.read_parquet(calendar_path)

    # Required: week_decision_date, curr_close_utc, prev_close_utc
    missing = [c for c in ["week_decision_date", "curr_close_utc", "prev_close_utc"] if c not in cal.columns]
    if missing:
        raise ValueError(f"CALENDAR missing columns: {missing}. Found: {list(cal.columns)}")

    cal = cal.copy()
    cal["week_decision_date"] = pd.to_datetime(cal["week_decision_date"], errors="coerce").dt.date
    cal["curr_close_utc"] = cal["curr_close_utc"].apply(_to_utc_ts)
    cal["prev_close_utc"] = cal["prev_close_utc"].apply(_to_utc_ts)

    # anchor columns for RL agent
    cal["date"] = cal["week_decision_date"]  
    cal = cal.sort_values("curr_close_utc").reset_index(drop=True)
    return cal


def build_base_panel(calendar_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Create full (ticker x week) anchor table so downstream merges never drop weeks.
    """
    cal = calendar_df[["week_decision_date", "date", "curr_close_utc", "prev_close_utc"]].copy()
    tk = pd.DataFrame({"ticker": tickers})

    # cross join
    cal["_k"] = 1
    tk["_k"] = 1
    panel = cal.merge(tk, on="_k", how="inner").drop(columns=["_k"])

    panel = panel.sort_values(["ticker", "curr_close_utc"]).reset_index(drop=True)
    return panel


# -------------------------
# EC Attachment
# -------------------------

def load_ec_analysis(ec_path: str) -> pd.DataFrame:
    df = pd.read_parquet(ec_path)

    required = ["ticker", "call_date", "event_ts_utc", "doc_id", "ec_summary_json"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"EC analysis missing columns: {missing}. Found: {list(df.columns)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    out["event_ts_utc"] = out["event_ts_utc"].apply(_to_utc_ts)

    # call_date may be date-like; keep but not required for joining
    out["call_date"] = pd.to_datetime(out["call_date"], errors="coerce").dt.date
    return out


def load_sec_analysis(sec_path: str) -> pd.DataFrame:
    """
    SEC analysis parquet columns:
      ['ticker', 'form_type', 'filing_date', 'filed_at_utc', 'doc_id', 'sec_summary_json']
    """
    p = Path(sec_path)
    if not p.exists():
        print(f"[tier1_sync][WARN] SEC analysis file not found: {sec_path}. Proceeding with empty SEC.")
        return pd.DataFrame(
            columns=["ticker", "form_type", "filing_date", "filed_at_utc", "doc_id", "sec_summary_json"]
        )

    df = pd.read_parquet(p)

    required = ["ticker", "form_type", "filing_date", "filed_at_utc", "doc_id", "sec_summary_json"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SEC analysis missing columns: {missing}. Found: {list(df.columns)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    out["form_type"] = out["form_type"].astype(str)
    out["filed_at_utc"] = out["filed_at_utc"].apply(_to_utc_ts)
    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce").dt.date
    out["doc_id"] = out["doc_id"].astype(str)
    return out

def load_news_analysis(news_path: str) -> pd.DataFrame:
    """
    News weekly analysis parquet columns:
      ['ticker', 'week_decision_date', 'window_start', 'window_end', 'summary_text',
       'news_rl_signal', 'news_rl_risk_score', 'news_rl_confidence', 'news_rl_rationale', 'key_events_json']
    """
    p = Path(news_path)
    if not p.exists():
        print(f"[tier1_sync][WARN] News analysis file not found: {news_path}. Proceeding with empty news.")
        return pd.DataFrame(
            columns=[
                "ticker",
                "week_decision_date",
                "news_rl_signal",
                "news_rl_risk_score",
                "news_rl_confidence",
                "news_rl_rationale",
            ]
        )

    df = pd.read_parquet(p)

    required = [
        "ticker",
        "week_decision_date",
        "news_rl_signal",
        "news_rl_risk_score",
        "news_rl_confidence",
        "news_rl_rationale",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"News analysis missing columns: {missing}. Found: {list(df.columns)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    if "summary_text" in out.columns:
        out["summary_text"] = out["summary_text"].fillna("").astype(str)

    # Normalize to date for reliable join with base panel
    out["week_decision_date"] = pd.to_datetime(out["week_decision_date"], errors="coerce").dt.date

    # Ensure numeric robustness
    out["news_rl_signal"] = pd.to_numeric(out["news_rl_signal"], errors="coerce").fillna(0.0)
    out["news_rl_risk_score"] = pd.to_numeric(out["news_rl_risk_score"], errors="coerce").fillna(0.0)
    out["news_rl_confidence"] = pd.to_numeric(out["news_rl_confidence"], errors="coerce").fillna(0.0)
    out["news_rl_rationale"] = out["news_rl_rationale"].fillna("").astype(str)
    out["summary_text"] = out["summary_text"].fillna("").astype(str)

    # Deduplicate defensively: if multiple rows exist per ticker-week, keep the last one
    out = out.sort_values(["ticker", "week_decision_date"]).drop_duplicates(
        subset=["ticker", "week_decision_date"],
        keep="last",
    )

    return out


def load_tech_analysis(tech_path: str) -> pd.DataFrame:
    """
    Technical weekly analysis parquet columns:
      ['ticker', 'week_decision_date', 'prev_close_utc', 'curr_close_utc',
       'tech_json', 'tech_signal', 'tech_risk_score', 'tech_confidence',
       'tech_rationale', 'discarded', 'discard_reason']
    """
    p = Path(tech_path)
    if not p.exists():
        print(f"[tier1_sync][WARN] Tech analysis file not found: {tech_path}. Proceeding with empty tech.")
        return pd.DataFrame(
            columns=[
                "ticker",
                "week_decision_date",
                "tech_json",
                "tech_signal",
                "tech_risk_score",
                "tech_confidence",
                "tech_rationale",
                "discarded",
                "discard_reason",
            ]
        )

    df = pd.read_parquet(p)

    required = [
        "ticker",
        "week_decision_date",
        "tech_signal",
        "tech_risk_score",
        "tech_confidence",
        "tech_rationale",
        "discarded",
        "discard_reason",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tech analysis missing columns: {missing}. Found: {list(df.columns)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    out["week_decision_date"] = pd.to_datetime(out["week_decision_date"], errors="coerce").dt.date

    out["tech_signal"] = pd.to_numeric(out["tech_signal"], errors="coerce").fillna(0.0)
    out["tech_risk_score"] = pd.to_numeric(out["tech_risk_score"], errors="coerce").fillna(0.0)
    out["tech_confidence"] = pd.to_numeric(out["tech_confidence"], errors="coerce").fillna(0.0)
    out["tech_rationale"] = out["tech_rationale"].fillna("").astype(str)
    out["tech_json"] = out["tech_json"].fillna("").astype(str)

    # discarded is boolean-like; normalize
    out["discarded"] = out["discarded"].fillna(False).astype(bool)
    out["discard_reason"] = out["discard_reason"].fillna("").astype(str)

    # Deduplicate defensively: keep last per ticker-week
    out = out.sort_values(["ticker", "week_decision_date"]).drop_duplicates(
        subset=["ticker", "week_decision_date"],
        keep="last",
    )

    return out


def attach_ec_features(
    panel: pd.DataFrame,
    ec_df: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """
    For each (ticker, curr_close_utc anchor), pick the latest EC event where:
      event_ts_utc <= curr_close_utc
      and (curr_close_utc - event_ts_utc) <= window_days

    Never drops rows. Out-of-window or missing EC => zeros/empty rationale.
    """
    panel = panel.copy()

    # Defaults for ALL rows (your requirement)
    panel["ec_signal"] = 0.0
    panel["ec_risk_score"] = 0.0
    panel["ec_confidence"] = 0.0
    panel["ec_rationale"] = ""
    panel["ec_doc_id"] = ""
    panel["ec_summary"] = ""
    panel["ec_event_ts_utc"] = pd.NaT

    if ec_df is None or ec_df.empty:
        return panel

    # Ensure join keys are tz-aware UTC timestamps (critical)
    panel["curr_close_utc"] = panel["curr_close_utc"].apply(_to_utc_ts)
    ec_df = ec_df.copy()
    ec_df["event_ts_utc"] = ec_df["event_ts_utc"].apply(_to_utc_ts)

    # Extract RL fields from ec_summary_json
    extracted = ec_df["ec_summary_json"].apply(_extract_rl_from_ec_summary)
    ec_df["__signal"] = extracted.apply(lambda t: t[0])
    ec_df["__risk"] = extracted.apply(lambda t: t[1])
    ec_df["__conf"] = extracted.apply(lambda t: t[2])
    ec_df["__rat"] = extracted.apply(lambda t: t[3])

    # Drop rows without timestamps on the EC side (cannot asof-match them)
    ec_df = ec_df.dropna(subset=["event_ts_utc"]).copy()

    window = pd.to_timedelta(int(window_days), unit="D")

    pieces = []
    for tk, p_tk in panel.groupby("ticker", sort=False):
        p_tk = p_tk.sort_values("curr_close_utc").copy()

        e_tk = ec_df[ec_df["ticker"].astype(str) == str(tk)]
        if e_tk.empty:
            pieces.append(p_tk)  # keep defaults
            continue

        e_tk = e_tk.sort_values("event_ts_utc").copy()

        m = pd.merge_asof(
            p_tk,
            e_tk[["event_ts_utc", "doc_id", "ec_summary_json", "__signal", "__risk", "__conf", "__rat"]],
            left_on="curr_close_utc",
            right_on="event_ts_utc",
            direction="backward",
            allow_exact_matches=True,
        )

        ok = (m["event_ts_utc"].notna()) & ((m["curr_close_utc"] - m["event_ts_utc"]) <= window)

        # Clear stale/out-of-window matches (prevents leakage)
        m.loc[~ok, ["doc_id"]] = ""
        m.loc[~ok, ["event_ts_utc"]] = pd.NaT
        m.loc[~ok, ["__signal", "__risk", "__conf"]] = 0.0
        m.loc[~ok, "__rat"] = ""
        m.loc[~ok, "ec_summary_json"] = ""

        # Assign final columns (every row has value; out-of-window already zeroed)
        m["ec_signal"] = m["__signal"].astype(float)
        m["ec_risk_score"] = m["__risk"].astype(float)
        m["ec_confidence"] = m["__conf"].astype(float)
        m["ec_rationale"] = m["__rat"].astype(str)
        m["ec_doc_id"] = m["doc_id"].astype(str)
        m["ec_event_ts_utc"] = m["event_ts_utc"]
        m["ec_summary"] = m["ec_summary_json"].fillna("").astype(str)


        m = m.drop(columns=["doc_id", "event_ts_utc", "ec_summary_json", "__signal", "__risk", "__conf", "__rat"], errors="ignore")
        pieces.append(m)

    out = pd.concat(pieces, axis=0, ignore_index=False)
    out = out.sort_values(["ticker", "curr_close_utc"]).reset_index(drop=True)
    return out


def attach_sec_features(
    panel: pd.DataFrame,
    sec_df: pd.DataFrame,
    window_days: int, ) -> pd.DataFrame:
    """
    For each (ticker, curr_close_utc anchor), pick the latest SEC filing where:
      filed_at_utc <= curr_close_utc
      and (curr_close_utc - filed_at_utc) <= window_days

    Latest-whichever-form-type semantics. Never drops rows.
    Out-of-window or missing SEC => zeros/empty rationale.
    """
    panel = panel.copy()

    # Defaults for ALL rows (no missing rows)
    panel["sec_signal"] = 0.0
    panel["sec_risk_score"] = 0.0
    panel["sec_confidence"] = 0.0
    panel["sec_rationale"] = ""
    panel["sec_doc_id"] = ""
    panel["sec_filed_at_utc"] = pd.NaT
    panel["sec_form_type"] = ""
    panel["sec_summary"] = ""

    if sec_df is None or sec_df.empty:
        return panel

    # Normalize time keys (critical for asof)
    panel["curr_close_utc"] = panel["curr_close_utc"].apply(_to_utc_ts)
    sec_df = sec_df.copy()
    sec_df["filed_at_utc"] = sec_df["filed_at_utc"].apply(_to_utc_ts)

    # Extract RL fields from sec_summary_json (same rl logic as EC)
    extracted = sec_df["sec_summary_json"].apply(_extract_rl_from_ec_summary)
    sec_df["__signal"] = extracted.apply(lambda t: t[0])
    sec_df["__risk"] = extracted.apply(lambda t: t[1])
    sec_df["__conf"] = extracted.apply(lambda t: t[2])
    sec_df["__rat"] = extracted.apply(lambda t: t[3])

    # Cannot match rows without filed_at_utc
    sec_df = sec_df.dropna(subset=["filed_at_utc"]).copy()

    window = pd.to_timedelta(int(window_days), unit="D")

    pieces = []
    for tk, p_tk in panel.groupby("ticker", sort=False):
        p_tk = p_tk.sort_values("curr_close_utc").copy()

        s_tk = sec_df[sec_df["ticker"].astype(str) == str(tk)]
        if s_tk.empty:
            pieces.append(p_tk)  # keep defaults
            continue

        s_tk = s_tk.sort_values("filed_at_utc").copy()

        m = pd.merge_asof(
            p_tk,
            s_tk[["filed_at_utc", "doc_id", "form_type", "sec_summary_json", "__signal", "__risk", "__conf", "__rat"]],
            left_on="curr_close_utc",
            right_on="filed_at_utc",
            direction="backward",
            allow_exact_matches=True,
        )

        ok = (m["filed_at_utc"].notna()) & ((m["curr_close_utc"] - m["filed_at_utc"]) <= window)

        # Clear stale/out-of-window matches to prevent leakage
        m.loc[~ok, ["doc_id", "form_type"]] = ["", ""]
        m.loc[~ok, ["filed_at_utc"]] = pd.NaT
        m.loc[~ok, ["__signal", "__risk", "__conf"]] = 0.0
        m.loc[~ok, "__rat"] = ""
        m.loc[~ok, "sec_summary_json"] = ""

        # Assign SEC columns (every row has value; out-of-window already zeroed)
        m["sec_signal"] = m["__signal"].astype(float)
        m["sec_risk_score"] = m["__risk"].astype(float)
        m["sec_confidence"] = m["__conf"].astype(float)
        m["sec_rationale"] = m["__rat"].astype(str)
        m["sec_doc_id"] = m["doc_id"].astype(str)
        m["sec_filed_at_utc"] = m["filed_at_utc"]
        m["sec_form_type"] = m["form_type"].astype(str)
        m["sec_summary"] = m["sec_summary_json"].fillna("").astype(str)

        m = m.drop(
            columns=["doc_id", "form_type", "sec_summary_json", "filed_at_utc", "__signal", "__risk", "__conf", "__rat"],
            errors="ignore",
        )
        pieces.append(m)

    out = pd.concat(pieces, axis=0, ignore_index=False)
    out = out.sort_values(["ticker", "curr_close_utc"]).reset_index(drop=True)
    return out

  
def attach_news_features(panel: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join weekly news RL features by (ticker, week_decision_date).
    Never drops rows. Missing news => zeros/empty rationale.
    """
    panel = panel.copy()

    # Defaults for ALL rows
    panel["news_signal"] = 0.0
    panel["news_risk_score"] = 0.0
    panel["news_confidence"] = 0.0
    panel["news_rationale"] = ""
    panel["news_summary"] = ""

    if news_df is None or news_df.empty:
        return panel

    # Ensure join keys match types
    panel["week_decision_date"] = pd.to_datetime(panel["week_decision_date"], errors="coerce").dt.date
    news_df = news_df.copy()
    news_df["ticker"] = news_df["ticker"].astype(str)
    news_df["week_decision_date"] = pd.to_datetime(news_df["week_decision_date"], errors="coerce").dt.date

    news_df = news_df.sort_values(["ticker", "week_decision_date"]).drop_duplicates(
        subset=["ticker", "week_decision_date"], keep="last"
    )

    base_cols = [
        "ticker",
        "week_decision_date",
        "news_rl_signal",
        "news_rl_risk_score",
        "news_rl_confidence",
        "news_rl_rationale",
    ]
    payload_cols = base_cols + (["summary_text"] if "summary_text" in news_df.columns else [])

    merged = panel.merge(news_df[payload_cols], on=["ticker", "week_decision_date"], how="left")

    # Overwrite defaults where present; keep defaults where missing
    merged["news_signal"] = merged["news_rl_signal"].fillna(0.0).astype(float)
    merged["news_risk_score"] = merged["news_rl_risk_score"].fillna(0.0).astype(float)
    merged["news_confidence"] = merged["news_rl_confidence"].fillna(0.0).astype(float)
    merged["news_rationale"] = merged["news_rl_rationale"].fillna("").astype(str)
    
    if "summary_text" in merged.columns:
        merged["news_summary"] = merged["summary_text"].fillna("").astype(str)
    else:
        merged["news_summary"] = ""

    merged = merged.drop(
        columns=["news_rl_signal", "news_rl_risk_score", "news_rl_confidence", "news_rl_rationale", "summary_text"],
        errors="ignore",
    )

    return merged


def attach_tech_features(panel: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join weekly technical RL features by (ticker, week_decision_date).
    Never drops rows. Missing or discarded tech => zeros/empty rationale.
    """
    panel = panel.copy()

    # Defaults for ALL rows (RL-facing)
    panel["tech_signal"] = 0.0
    panel["tech_risk_score"] = 0.0
    panel["tech_confidence"] = 0.0
    panel["tech_rationale"] = ""
    panel["tech_discarded"] = False
    panel["tech_discard_reason"] = ""
    panel["tech_summary"] = ""

    if tech_df is None or tech_df.empty:
        return panel

    # Normalize join keys
    panel["week_decision_date"] = pd.to_datetime(panel["week_decision_date"], errors="coerce").dt.date
    tech_df = tech_df.copy()
    tech_df["week_decision_date"] = pd.to_datetime(tech_df["week_decision_date"], errors="coerce").dt.date
    tech_df["ticker"] = tech_df["ticker"].astype(str)

    # Deduplicate defensively (last per ticker-week)
    tech_df = tech_df.sort_values(["ticker", "week_decision_date"]).drop_duplicates(
        subset=["ticker", "week_decision_date"], keep="last"
    )

    # Rename source columns to avoid any collision with panel columns
    tech_src = tech_df[
        [
            "ticker",
            "week_decision_date",
            "tech_json",
            "tech_signal",
            "tech_risk_score",
            "tech_confidence",
            "tech_rationale",
            "discarded",
            "discard_reason",
        ]
    ].rename(
        columns={
            "tech_signal": "tech_src_signal",
            "tech_risk_score": "tech_src_risk_score",
            "tech_confidence": "tech_src_confidence",
            "tech_rationale": "tech_src_rationale",
            "discarded": "tech_src_discarded",
            "discard_reason": "tech_src_discard_reason",
            "tech_json": "tech_src_summary",
        }
    )

    merged = panel.merge(tech_src, on=["ticker", "week_decision_date"], how="left")

    # Robust types + avoid FutureWarning
    discarded = merged["tech_src_discarded"].astype("boolean").fillna(False)

    merged["tech_discarded"] = discarded.astype(bool)
    merged["tech_discard_reason"] = merged["tech_src_discard_reason"].fillna("").astype(str)

    # Fill scalars from source when present and not discarded; else keep defaults
    merged["tech_signal"] = pd.to_numeric(merged["tech_src_signal"], errors="coerce").fillna(0.0).astype(float)
    merged["tech_risk_score"] = pd.to_numeric(merged["tech_src_risk_score"], errors="coerce").fillna(0.0).astype(float)
    merged["tech_confidence"] = pd.to_numeric(merged["tech_src_confidence"], errors="coerce").fillna(0.0).astype(float)
    merged["tech_rationale"] = merged["tech_src_rationale"].fillna("").astype(str)
    merged["tech_summary"] = merged["tech_src_summary"].fillna("").astype(str)
    merged.loc[merged["tech_discarded"], "tech_summary"] = ""

    # Drop source helper columns so final panel is clean
    merged = merged.drop(
        columns=[
            "tech_src_signal",
            "tech_src_risk_score",
            "tech_src_confidence",
            "tech_src_rationale",
            "tech_src_discarded",
            "tech_src_discard_reason",
            'tech_discarded', 
            'tech_src_summary',
            'tech_discard_reason',
        ],
        errors="ignore",
    )

    return merged


def add_global_tier1_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-source features used by Tier-2 and RL:
      - days_since_ec, days_since_sec (in days; NaN if no event in window)
      - n_sources_available: number of Tier-1 sources with a non-null signal
      - signal_sign_disagreement: True if at least one positive and one negative signal
        among the available sources.
    """
    # --- Timeliness: days since last EC and SEC events ---

    # Ensure both sides are proper datetimes before subtracting
    if "curr_close_utc" in panel.columns and "ec_event_ts_utc" in panel.columns:
        curr_ec = pd.to_datetime(panel["curr_close_utc"], utc=True, errors="coerce")
        ec_ts = pd.to_datetime(panel["ec_event_ts_utc"], utc=True, errors="coerce")
        diff_ec = curr_ec - ec_ts  # Timedelta series
        panel["days_since_ec"] = diff_ec.dt.days

    if "curr_close_utc" in panel.columns and "sec_filed_at_utc" in panel.columns:
        curr_sec = pd.to_datetime(panel["curr_close_utc"], utc=True, errors="coerce")
        sec_ts = pd.to_datetime(panel["sec_filed_at_utc"], utc=True, errors="coerce")
        diff_sec = curr_sec - sec_ts  # Timedelta series
        panel["days_since_sec"] = diff_sec.dt.days

    # --- Coverage / conflict across Tier-1 sources ---

    source_signal_cols = [
        col for col in ["ec_signal", "sec_signal", "news_signal", "tech_signal"]
        if col in panel.columns
    ]

    if source_signal_cols:
        # How many Tier-1 sources actually produced a signal
        panel["n_sources_available"] = (
            panel[source_signal_cols].notna().sum(axis=1).astype("int64")
        )

        # True if at least one positive and one negative signal across sources
        def _sign_disagreement(row) -> bool:
            vals = [row[c] for c in source_signal_cols if pd.notna(row[c])]
            if not vals:
                return False
            has_pos = any(v > 0 for v in vals)
            has_neg = any(v < 0 for v in vals)
            return has_pos and has_neg

        panel["signal_sign_disagreement"] = panel.apply(_sign_disagreement, axis=1)

    return panel

# -------------------------
# Main
# -------------------------

def run_sync(cfg: Tier1SyncConfig) -> None:
    cal = load_calendar(cfg.calendar_path)

    # Load EC first; infer tickers if not provided
    ec_df = load_ec_analysis(cfg.ec_analysis_source_path) if cfg.ec_analysis_source_path else pd.DataFrame()
    sec_df = load_sec_analysis(cfg.sec_analysis_source_path) if cfg.sec_analysis_source_path else pd.DataFrame()
    news_df = load_news_analysis(cfg.news_analysis_source_path) if cfg.news_analysis_source_path else pd.DataFrame()
    tech_df = load_tech_analysis(cfg.tech_analysis_source_path) if cfg.tech_analysis_source_path else pd.DataFrame()

    if cfg.tickers and len(cfg.tickers) > 0:
        tickers = [str(t) for t in cfg.tickers]
    else:
        if ec_df.empty:
            raise ValueError("tickers not provided in config and cannot infer because EC file is empty.")
        tickers = sorted(ec_df["ticker"].dropna().astype(str).unique().tolist())

    panel = build_base_panel(cal, tickers)

    # Attach EC features (first modality)
    panel = attach_ec_features(panel, ec_df, window_days=cfg.ec_history_window_days)
    panel = attach_sec_features(panel, sec_df, window_days=cfg.sec_history_window_days)
    panel = attach_news_features(panel, news_df)
    panel = attach_tech_features(panel, tech_df)

    panel = add_global_tier1_features(panel)
    
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"[tier1_sync] Wrote: {out_path} rows={len(panel):,} cols={len(panel.columns):,}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to tier1_sync.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_sync(cfg)
