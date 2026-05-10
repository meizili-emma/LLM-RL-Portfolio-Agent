from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------- small utils ----------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _iso(s) -> str:
    return pd.to_datetime(s, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")

def _read_calendar(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"week_decision_date","prev_close_utc","curr_close_utc"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Calendar missing columns: {miss}")
    df["week_decision_date"] = df["week_decision_date"].astype(str)
    return df.sort_values("curr_close_utc").reset_index(drop=True)

def _recency(ts_iso: str, ref_iso: str, half_life_days: float = 7.0) -> float:
    a = pd.to_datetime(ts_iso, utc=True); b = pd.to_datetime(ref_iso, utc=True)
    d = max((b - a).total_seconds(), 0.0) / 86400.0
    if half_life_days <= 0: return 1.0
    return float(math.exp(-math.log(2) * (d / float(half_life_days))))

def _minmax(x: pd.Series) -> pd.Series:
    if x.empty: return x
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo + 1e-12:
        return pd.Series(np.ones(len(x), dtype=np.float32), index=x.index)
    return (x - lo) / (hi - lo)

def _window_from_end(end_iso: str, back_days: int) -> Tuple[str, str]:
    end = pd.to_datetime(end_iso, utc=True)
    start = end - pd.Timedelta(days=int(back_days))
    return _iso(start), _iso(end)


# ---------- loaders ----------

def _load_raws(cfg: dict) -> Dict[str, pd.DataFrame]:
    news = pd.read_parquet(cfg["raw"]["news_path"]).copy()
    ec   = pd.read_parquet(cfg["raw"]["ec_path"]).copy()
    sec  = pd.read_parquet(cfg["raw"]["sec_path"]).copy()
    # normalize key cols
    if "published_at_utc" in news.columns:
        news["published_at_utc" ] = pd.to_datetime(news["published_at_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "event_ts_utc" in ec.columns:
        ec["event_ts_utc"] = pd.to_datetime(ec["event_ts_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ec["ticker"] = ec["ticker"].astype(str).str.strip().str.upper()
        ec["event_dt"] = pd.to_datetime(ec["event_ts_utc"], utc=True, errors="coerce").dt.floor("s")
    if "filed_at_utc" in sec.columns:
        sec["filed_at_utc"] = pd.to_datetime(sec["filed_at_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"news": news, "ec": ec, "sec": sec}

def _infer_tickers(cfg_tickers: Optional[List[str]], raws: Dict[str, pd.DataFrame]) -> List[str]:
    if cfg_tickers: return list(cfg_tickers)
    s = set()
    for k in ("news","ec","sec"):
        if "ticker" in raws[k].columns:
            s |= set(raws[k]["ticker"].dropna().astype(str).unique().tolist())
    return sorted(s)


# ---------- NEWS selection ----------

def _score_news(df: pd.DataFrame, end_iso: str, w: Dict[str, float]) -> pd.DataFrame:
    # df has: ticker, published_at_utc, doc_id, text, tok_len, source, source_tier, source_score
    df = df.copy()
    df["rec"] = df["published_at_utc"].map(lambda ts: _recency(ts, end_iso, 7.0)).astype(np.float32)
    if "tok_len" in df.columns:
        df["len_norm"] = _minmax(pd.to_numeric(df["tok_len"], errors="coerce").fillna(0.0))
    else:
        df["len_norm"] = 1.0
    # source_score: 3/2/1 → normalize (1.0, 0.667, 0.333) by dividing by 3
    src = pd.to_numeric(df.get("source_score", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    df["src_norm"] = (src / 3.0).clip(lower=0.0, upper=1.0).astype(np.float32)

    df["score"] = (
        float(w["recency"]) * df["rec"] +
        float(w["length"])  * df["len_norm"] +
        float(w["source"])  * df["src_norm"]
    ).astype(np.float32)

    # stable tie-breakers later when sorting
    return df

def _cap_and_dynamic_top(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    caps = cfg["news"]["caps"]
    dyn  = cfg["news"]["dynamic_topN"]
    # per-source cap; per-day cap
    out_rows = []
    per_src = {}
    per_day = {}
    for _, r in df.sort_values(["score","rec","published_at_utc","tok_len"], ascending=[False,False,False,False]).iterrows():
        s = r.get("source","")
        d = r["published_at_utc"][:10]
        per_src[s] = per_src.get(s, 0) + 0
        per_day[d] = per_day.get(d, 0) + 0
        if per_src.get(s, 0) >= int(caps["per_source"]): continue
        if per_day.get(d, 0) >= int(caps["per_day"]):     continue
        out_rows.append(r)
        per_src[s] = per_src.get(s, 0) + 1
        per_day[d] = per_day.get(d, 0) + 1

    cand = pd.DataFrame(out_rows)
    C = len(cand)
    if C == 0:
        return cand

    N_min, N_max, tau, min_score = int(dyn["N_min"]), int(dyn["N_max"]), float(dyn["tau"]), float(dyn["min_score"])
    N = max(N_min, min(N_max, int(round(N_min + (N_max - N_min) * (1 - math.exp(-C / max(tau,1e-6)))))))
    cand = cand[cand["score"] >= min_score].copy()
    if len(cand) < N_min:
        cand = pd.DataFrame(out_rows).sort_values("score", ascending=False).head(N_min)
    else:
        cand = cand.sort_values("score", ascending=False).head(N)

    cand = cand.reset_index(drop=True)
    cand["rank"] = cand.index + 1
    return cand


# ---------- EC / SEC selection ----------

def _select_ec_latest(ec_df: pd.DataFrame, ticker: str, end_iso: str, backoff_days: int) -> Optional[dict]:
    tk = str(ticker).strip().upper()
    end_dt = pd.to_datetime(end_iso, utc=True)
    start_dt = end_dt - pd.Timedelta(days=int(backoff_days))
    sub = ec_df.loc[
        (ec_df["ticker"] == tk) &
        (ec_df["event_dt"].notna()) &
        (ec_df["event_dt"].between(start_dt, end_dt, inclusive="both")),
        ["call_date", "event_dt", "text", "text_len", "doc_id"]
    ].sort_values("event_dt", ascending=False)

    if sub.empty:
        # Gentle fallback: widen window once (e.g., 540 days) to detect data alignment issues,
        # but do not write if still empty (we still return None).
        wider = ec_df.loc[
            (ec_df["ticker"] == tk) &
            (ec_df["event_dt"].notna()) &
            (ec_df["event_dt"].between(end_dt - pd.Timedelta(days=540), end_dt, inclusive="both")),
            ["call_date", "event_dt", "text", "text_len", "doc_id"]
        ].sort_values("event_dt", ascending=False)
        if wider.empty:
            return None
        sub = wider

    r = sub.iloc[0]
    return {
        "ticker": tk,
        "call_date": str(r["call_date"]),
        "event_ts_utc": r["event_dt"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "doc_id": str(r["doc_id"]),
        "token_len": int(pd.to_numeric(r.get("text_len", 0), errors="coerce") or 0),
        "text": str(r["text"]),
        }

def _select_sec_latest(sec_df: pd.DataFrame, ticker: str, end_iso: str, q_backoff: int, k_backoff: int) -> dict:
    def pick(form: str, back: int) -> Optional[dict]:
        end = pd.to_datetime(end_iso, utc=True); start = end - pd.Timedelta(days=int(back))
        m = (sec_df["ticker"] == ticker) & (sec_df["form_type"] == form) & \
            (sec_df["filed_at_utc"] <= end_iso) & (sec_df["filed_at_utc"] >= _iso(start))
        cols = ["ticker","filing_date","filed_at_utc","form_type","management_discussion","risk_factors"]
        if not set(cols).issubset(sec_df.columns): return None
        meta = sec_df.loc[m, cols].sort_values("filed_at_utc", ascending=False)
        if meta.empty: return None
        r = meta.iloc[0]
        # token_len rough estimates if you want (deterministic)
        tok_mda  = int(len(str(r["management_discussion"])) // 4)
        tok_risk = int(len(str(r["risk_factors"])) // 4)
        return {
            "form_type": form,
            "filing_date": str(r["filing_date"]),
            "filed_at_utc": _iso(r["filed_at_utc"]),
            "management_discussion": str(r["management_discussion"]),
            "risk_factors": str(r["risk_factors"]),
            "tok_len_mda": tok_mda,
            "tok_len_risk_factors": tok_risk,
        }
    out = {"filings": []}
    q = pick("10-Q", q_backoff);  k = pick("10-K", k_backoff)
    if q: out["filings"].append(q)
    if k: out["filings"].append(k)
    return out


# ---------- per (week, ticker) ----------

def build_week_ticker(cfg: dict, raws: Dict[str, pd.DataFrame], week_row: pd.Series, ticker: str, out_root: Path) -> Dict[str, int]:
    week = str(week_row["week_decision_date"])
    end_iso = _iso(week_row["curr_close_utc"])
    start_news_iso, _ = _window_from_end(end_iso, int(cfg["news"]["lookback_days"]))

    tdir = out_root / week / ticker
    _ensure_dir(tdir)

    # NEWS
    news_df = raws["news"]
    m = (news_df["ticker"] == ticker) & (news_df["published_at_utc"] <= end_iso) & (news_df["published_at_utc"] > start_news_iso)
    cand = news_df.loc[m, ["ticker","published_at_utc","doc_id","text","tok_len","source","source_tier","source_score"]].copy()
    scored = _score_news(cand, end_iso=end_iso, w={
        "recency": cfg["news"]["weights"]["recency"],
        "length":  cfg["news"]["weights"]["length"],
        "source":  cfg["news"]["weights"]["source"],
    })
    picked = _cap_and_dynamic_top(scored, cfg)

    news_path = tdir / "news.jsonl"
    with news_path.open("w", encoding="utf-8") as f:
        for _, r in picked.iterrows():
            row = {
                "doc_id": str(r["doc_id"]),
                "ticker": ticker,
                "source": str(r.get("source","")),
                "ts_utc": str(r["published_at_utc"]),
                "rank": int(r["rank"]),
                "score": float(r["score"]),
                "rec": float(r["rec"]),
                "len_norm": float(r["len_norm"]),
                "source_score": float(r.get("source_score", 1.0)),
                "tok_len": int(pd.to_numeric(r.get("tok_len", 0), errors="coerce") or 0),
                "text": str(r.get("text","")),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # EC
    ec_obj = _select_ec_latest(raws["ec"], ticker, end_iso, backoff_days=int(cfg["ec"]["backoff_days"]))
    (tdir / "ec.json").write_text(json.dumps(ec_obj or {}, ensure_ascii=False, indent=2), encoding="utf-8")

    # SEC
    sec_obj = _select_sec_latest(raws["sec"], ticker, end_iso,
                                 q_backoff=int(cfg["sec"]["q_backoff_days"]),
                                 k_backoff=int(cfg["sec"]["k_backoff_days"]))
    (tdir / "sec.json").write_text(json.dumps(sec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # COUNTS
    counts = {
        "ticker": ticker,
        "week_decision_date": week,
        "news_candidates": int(len(cand)),
        "news_selected": int(len(picked)),
        "news_tok_sum": int(pd.to_numeric(picked.get("tok_len", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not picked.empty else 0,
        "ec_present": bool(ec_obj),
        "ec_token_len": int(ec_obj.get("token_len", 0)) if ec_obj else 0,
        "sec_10Q_present": any(f.get("form_type")=="10-Q" for f in sec_obj.get("filings",[])),
        "sec_10K_present": any(f.get("form_type")=="10-K" for f in sec_obj.get("filings",[])),
        "sec_tok_sum": int(sum(f.get("tok_len_mda",0)+f.get("tok_len_risk_factors",0) for f in sec_obj.get("filings",[]))),
    }
    (tdir / "_COUNT.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


# ---------- driver ----------

def build_all(cfg: dict):
    raws = _load_raws(cfg)
    cal = _read_calendar(Path(cfg["calendar_path"]))
    if cfg.get("weeks"):
        cal = cal[cal["week_decision_date"].astype(str).isin(list(cfg["weeks"]))].copy()
    tickers = _infer_tickers(cfg.get("tickers"), raws)

    out_root = Path(cfg["out_root"])
    _ensure_dir(out_root)

    all_counts = []
    for _, row in cal.iterrows():
        for tk in tickers:
            cnt = build_week_ticker(cfg, raws, row, tk, out_root)
            all_counts.append(cnt)

    if all_counts:
        pd.DataFrame(all_counts).to_csv(out_root / "_BUILD_INFO.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="data/weekly/v1/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    build_all(cfg)

if __name__ == "__main__":
    main()
