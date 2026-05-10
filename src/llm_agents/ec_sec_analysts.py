from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel
from tqdm.auto import tqdm

from src.llm_agents.ec_sec_schema import (
    ECChunkMap,
    ECReduce,
    SECChunkMapMDNA,
    SECChunkMapRF,
    SECReduceSection,
    SECReduceFinal,
    SEC_CHUNK_MAP_SYSTEM_PROMPT,
    SEC_FINAL_REDUCE_SYSTEM_PROMPT,
    SEC_SECTION_REDUCE_SYSTEM_PROMPT,
    SEC_RL_SYSTEM_PROMPT,
    EC_RL_SYSTEM_PROMPT,
    map_system_prompt,
    reduce_system_prompt,
    ec_map_user_prompt,
    ec_reduce_user_prompt,
    sec_mdna_map_user_prompt,
    sec_rf_map_user_prompt,
    sec_section_reduce_user_prompt,
    sec_final_reduce_user_prompt,
    _ec_rl_user_prompt,
    _sec_rl_user_prompt,
)

from src.llm_agents.utils import (
    _iso,
    _structured_call,
    _simple_char_chunks,
    Tier1SignalPack,
)

def _extract_ec_rl_signals(
    model_cfg: Dict[str, Any],
    reduce_obj: ECReduce,
    ticker: str,
    as_of: str,
) -> Tier1SignalPack:
    """
    Use the same model_cfg and _structured_call machinery to obtain Tier1SignalPack
    from a ECReduce object.
    """
    if not (reduce_obj.summary_text or "").strip():
        return Tier1SignalPack(signal=0, risk_score=0, confidence=0, rationale="INCONSISTENT_METADATA")
    retries = int(model_cfg.get("max_retries", 2))
    user_prompt = _ec_rl_user_prompt(ticker=ticker, as_of=as_of, reduce_obj=reduce_obj)
    
    return _structured_call(
            model_cfg=model_cfg,
            schema=Tier1SignalPack,
            system_prompt=EC_RL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            retries=retries,
            )


def _extract_sec_rl_signals(
    model_cfg: Dict[str, Any],
    reduce_obj: SECReduceFinal,
    ticker: str,
    as_of: str,
    ) -> Tier1SignalPack:
    """
    Use the same model_cfg and _structured_call machinery to obtain Tier1SignalPack
    from a SECReduceFinal object.
    """
    if not (reduce_obj.summary_text or "").strip():
        return Tier1SignalPack(signal=0, risk_score=0, confidence=0, rationale="INCONSISTENT_METADATA")
    retries = int(model_cfg.get("max_retries", 2))
    user_prompt = _sec_rl_user_prompt(ticker=ticker, as_of=as_of, reduce_obj=reduce_obj)
    return _structured_call(
            model_cfg=model_cfg,
            schema=Tier1SignalPack,
            system_prompt=SEC_RL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            retries=retries,
            )


# =========================
#   Hierarchical Reduce Helpers
# =========================

def _hierarchical_reduce_ec_maps(
    maps: List[ECChunkMap],
    model_cfg: Dict[str, Any],
    ec_cfg: Dict[str, Any],
    ticker: str,
    as_of: str,
) -> ECReduce:
    """
    Two-tier hierarchical reduce for EC:

    - If len(maps) <= group_size: single reduce call.
    - Else: group maps into batches of group_size, reduce each to ECReduce,
      then reduce the group-level summaries again.
    """
    retries = int(model_cfg.get("max_retries", 3))
    group_size = int(ec_cfg.get("reduce_group_size", 15))

    def _reduce_jsonl(lines: List[str]) -> ECReduce:
        jsonl = "\n".join(lines)
        up = ec_reduce_user_prompt(ticker=ticker, as_of=as_of, maps_compact_jsonl=jsonl)
        return _structured_call(
            model_cfg=model_cfg,
            schema=ECReduce,
            system_prompt=reduce_system_prompt(),
            user_prompt=up,
            retries=retries,
        )

    if not maps:
        return ECReduce(summary_text="", numbers=[], risks=[], opportunities=[])

    lines = [m.model_dump_json() for m in maps]
    if len(maps) <= group_size:
        return _reduce_jsonl(lines)

    # Stage 1: per-group reduce
    stage1: List[ECReduce] = []
    for i in range(0, len(lines), group_size):
        group_lines = lines[i : i + group_size]
        red = _reduce_jsonl(group_lines)
        stage1.append(red)

    # Stage 2: final reduce over group-level summaries
    stage1_lines = [r.model_dump_json() for r in stage1]
    return _reduce_jsonl(stage1_lines)


def _hierarchical_reduce_sec_section(
    maps_json_models: List[BaseModel],
    model_cfg: Dict[str, Any],
    sec_cfg: Dict[str, Any],
    ticker: str,
    as_of: str,
    section: str,
) -> SECReduceSection:
    """
    Two-tier hierarchical reduce for a SEC section (MD&A or Risk Factors).

    maps_json_models: list of SECChunkMapMDNA or SECChunkMapRF objects.
    """
    retries = int(model_cfg.get("max_retries", 3))
    group_size = int(sec_cfg.get("reduce_group_size", 15))

    def _reduce_jsonl(lines: List[str]) -> SECReduceSection:
        jsonl = "\n".join(lines)
        up = sec_section_reduce_user_prompt(
            section=section,
            ticker=ticker,
            as_of=as_of,
            maps_compact_jsonl=jsonl,
        )
        return _structured_call(
            model_cfg=model_cfg,
            schema=SECReduceSection,
            system_prompt=SEC_SECTION_REDUCE_SYSTEM_PROMPT,
            user_prompt=up,
            retries=retries,
        )

    if not maps_json_models:
        return SECReduceSection(
            section=section if section in ("MD&A", "Risk Factors") else "MD&A",
            summary_text="",
            top_risks=[],
            exposures=[],
            opportunities=[],
            flags=[],
        )

    lines = [m.model_dump_json() for m in maps_json_models]
    if len(maps_json_models) <= group_size:
        return _reduce_jsonl(lines)

    # Stage 1: per-group reduce
    stage1: List[SECReduceSection] = []
    for i in range(0, len(lines), group_size):
        group_lines = lines[i : i + group_size]
        red = _reduce_jsonl(group_lines)
        stage1.append(red)

    # Stage 2: final reduce over group-level summaries
    stage1_lines = [r.model_dump_json() for r in stage1]
    return _reduce_jsonl(stage1_lines)


# =========================
#   EC Compression
# =========================

def _compress_single_ec_row(
    row: pd.Series,
    model_cfg: Dict[str, Any],
    ec_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compress a single EC transcript row into a structured JSON summary.
    Returns a dict to be written as one row in the condensed parquet.
    """
    ticker = str(row["ticker"])
    call_date = str(row.get("call_date", ""))
    event_ts_utc = _iso(row["event_ts_utc"])
    doc_id = str(row.get("doc_id", ""))

    text = str(row.get("text", "") or "")
    if not text.strip():
        return {
            "ticker": ticker,
            "call_date": call_date,
            "event_ts_utc": event_ts_utc,
            "doc_id": doc_id,
            "ec_summary_json": json.dumps(
                {"summary_text": "", "numbers": [], "risks": [], "opportunities": []},
                ensure_ascii=False,
            ),
        }

    max_len = int(ec_cfg.get("chunk_chars", 60000))
    overlap = int(ec_cfg.get("chunk_overlap_chars", 5000))
    max_chunks = int(ec_cfg.get("max_chunks", 5))
    compression_ratio = float(ec_cfg.get("compression_ratio", 0.3))

    chunks = _simple_char_chunks(text, max_len=max_len, overlap=overlap, max_chunks=max_chunks)
    retries = int(model_cfg.get("max_retries", 2))
    as_of = event_ts_utc

    maps: List[ECChunkMap] = []
    for ch in chunks:
        up = ec_map_user_prompt(
            ticker=ticker,
            as_of=as_of,
            compression_ratio=compression_ratio,
            chunk_text=ch,
        )
        m = _structured_call(
            model_cfg=model_cfg,
            schema=ECChunkMap,
            system_prompt=map_system_prompt(),
            user_prompt=up,
            retries=retries,
        )
        maps.append(m)

    red = _hierarchical_reduce_ec_maps(
        maps=maps,
        model_cfg=model_cfg,
        ec_cfg=ec_cfg,
        ticker=ticker,
        as_of=as_of,
    )

    try:
        rl_signals = _extract_ec_rl_signals(
            model_cfg=model_cfg,
            reduce_obj=red,
            ticker=ticker,
            as_of=as_of,
        )
        red.rl = rl_signals
    except Exception as e:
        # Do not break the whole pipeline if RL extraction fails.
        print(
            f"[EC-RL] Failed RL extraction for ticker={ticker}, "
            f"call_date={call_date}, doc_id={doc_id}: {e}"
        )

    return {
        "ticker": ticker,
        "call_date": call_date,
        "event_ts_utc": event_ts_utc,
        "doc_id": doc_id,
        "ec_summary_json": json.dumps(red.model_dump(), ensure_ascii=False),
    }


def run_ec_compression(cfg: Dict[str, Any]) -> None:
    raw_path = Path(cfg["ec"]["raw_path"])
    out_path = Path(cfg["ec"]["out_path"])

    raw = pd.read_parquet(raw_path)

    if out_path.exists():
        out_df = pd.read_parquet(out_path)
    else:
        out_df = pd.DataFrame(
            columns=[
                "ticker",
                "call_date",
                "event_ts_utc",
                "doc_id",
                "ec_summary_json",
            ]
        )

    if out_df.empty:
        processed_keys: set[tuple[str, str]] = set()
    else:
        processed_keys = set(
            zip(
                out_df["ticker"].astype(str),
                out_df["event_ts_utc"].astype(str),
            )
        )

    raw = raw.copy()
    raw["event_ts_utc"] = pd.to_datetime(raw["event_ts_utc"], utc=True, errors="coerce")

    todo_rows: List[pd.Series] = []
    for _, r in raw.iterrows():
        tk = str(r["ticker"])
        ts = _iso(r["event_ts_utc"])
        if (tk, ts) not in processed_keys:
            todo_rows.append(r)

    if not todo_rows:
        print("EC compression: nothing to do (all rows already processed).")
        return

    print(f"EC compression: {len(todo_rows)} rows to process.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("EC out_path:", out_path.resolve())

    for r in tqdm(todo_rows, desc="EC compression"):
        try:
            out_row = _compress_single_ec_row(
                row=r,
                model_cfg=cfg["model"],
                ec_cfg=cfg["ec"],
            )
        except Exception as e:
            print(
                f"[EC] Skipping row (ticker={r.get('ticker','')}, "
                f"call_date={r.get('call_date','')}) due to error: {e}"
            )
            continue

        out_df = pd.concat(
            [out_df, pd.DataFrame([out_row])],
            ignore_index=True,
        )
        out_df.to_parquet(out_path, index=False)

    print(f"EC compression: wrote {len(out_df)} total rows to {out_path}")


# =========================
#   SEC Compression
# =========================

def _compress_single_sec_row(
    row: pd.Series,
    model_cfg: Dict[str, Any],
    sec_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compress a single SEC filing row into a structured JSON summary.
    We handle MD&A and Risk Factors separately with hierarchical map-reduce,
    then synthesize both sections.
    """
    ticker = str(row["ticker"])
    form_type = str(row["form_type"])
    filing_date = str(row["filing_date"])
    filed_at_utc = _iso(row["filed_at_utc"])
    doc_id = str(row.get("doc_id", ""))

    mdna_text = str(row.get("management_discussion", "") or "")
    rf_text = str(row.get("risk_factors", "") or "")

    max_len = int(sec_cfg.get("chunk_chars", 80000))
    overlap = int(sec_cfg.get("chunk_overlap_chars", 7000))
    max_chunks = int(sec_cfg.get("max_chunks", 5))
    compression_ratio = float(sec_cfg.get("compression_ratio", 0.3))

    as_of = filed_at_utc
    retries = int(model_cfg.get("max_retries", 2))

    mdna_reduce: SECReduceSection | None = None
    rf_reduce: SECReduceSection | None = None

    # MD&A map-reduce if text exists
    if mdna_text.strip():
        mdna_chunks = _simple_char_chunks(
            mdna_text,
            max_len=max_len,
            overlap=overlap,
            max_chunks=max_chunks,
        )
        mdna_maps: List[SECChunkMapMDNA] = []
        for ch in mdna_chunks:
            up = sec_mdna_map_user_prompt(
                ticker=ticker,
                as_of=as_of,
                compression_ratio=compression_ratio,
                chunk_text=ch,
            )
            m = _structured_call(
                model_cfg=model_cfg,
                schema=SECChunkMapMDNA,
                system_prompt=SEC_CHUNK_MAP_SYSTEM_PROMPT,
                user_prompt=up,
                retries=retries,
            )
            mdna_maps.append(m)
        mdna_reduce = _hierarchical_reduce_sec_section(
            maps_json_models=mdna_maps,
            model_cfg=model_cfg,
            sec_cfg=sec_cfg,
            ticker=ticker,
            as_of=as_of,
            section="MD&A",
        )

    # Risk Factors map-reduce if text exists
    if rf_text.strip():
        rf_chunks = _simple_char_chunks(
            rf_text,
            max_len=max_len,
            overlap=overlap,
            max_chunks=max_chunks,
        )
        rf_maps: List[SECChunkMapRF] = []
        for ch in rf_chunks:
            up = sec_rf_map_user_prompt(
                ticker=ticker,
                as_of=as_of,
                compression_ratio=compression_ratio,
                chunk_text=ch,
            )
            m = _structured_call(
                model_cfg=model_cfg,
                schema=SECChunkMapRF,
                system_prompt=SEC_CHUNK_MAP_SYSTEM_PROMPT,
                user_prompt=up,
                retries=retries,
            )
            rf_maps.append(m)
        rf_reduce = _hierarchical_reduce_sec_section(
            maps_json_models=rf_maps,
            model_cfg=model_cfg,
            sec_cfg=sec_cfg,
            ticker=ticker,
            as_of=as_of,
            section="Risk Factors",
        )

    mdna_json = json.dumps(mdna_reduce.model_dump(), ensure_ascii=False) if mdna_reduce else None
    rf_json = json.dumps(rf_reduce.model_dump(), ensure_ascii=False) if rf_reduce else None

    final = _structured_call(
        model_cfg=model_cfg,
        schema=SECReduceFinal,
        system_prompt=SEC_FINAL_REDUCE_SYSTEM_PROMPT,
        user_prompt=sec_final_reduce_user_prompt(
            ticker=ticker,
            as_of=as_of,
            mdna_json=mdna_json,
            rf_json=rf_json,
        ),
        retries=retries,
    )

    try:
        rl_signals = _extract_sec_rl_signals(
            model_cfg=model_cfg,
            reduce_obj=final,
            ticker=ticker,
            as_of=as_of,
        )
        final.rl = rl_signals
    except Exception as e:
        print(
            f"[SEC-RL] Failed RL extraction for ticker={ticker}, "
            f"form_type={form_type}, filing_date={filing_date}, doc_id={doc_id}: {e}"
        )

    return {
        "ticker": ticker,
        "form_type": form_type,
        "filing_date": filing_date,
        "filed_at_utc": filed_at_utc,
        "doc_id": doc_id,
        "sec_summary_json": json.dumps(final.model_dump(), ensure_ascii=False),
    }


def run_sec_compression(cfg: Dict[str, Any]) -> None:
    raw_path = Path(cfg["sec"]["raw_path"])
    out_path = Path(cfg["sec"]["out_path"])

    raw = pd.read_parquet(raw_path)

    if out_path.exists():
        out_df = pd.read_parquet(out_path)
    else:
        out_df = pd.DataFrame(
            columns=[
                "ticker",
                "form_type",
                "filing_date",
                "filed_at_utc",
                "doc_id",
                "sec_summary_json",
            ]
        )

    if out_df.empty:
        processed_keys: set[tuple[str, str, str]] = set()
    else:
        processed_keys = set(
            zip(
                out_df["ticker"].astype(str),
                out_df["form_type"].astype(str),
                out_df["filing_date"].astype(str),
            )
        )

    raw = raw.copy()
    raw["filed_at_utc"] = pd.to_datetime(raw["filed_at_utc"], utc=True, errors="coerce")

    todo_rows: List[pd.Series] = []
    for _, r in raw.iterrows():
        tk = str(r["ticker"])
        ft = str(r["form_type"])
        fd = str(r["filing_date"])
        if (tk, ft, fd) not in processed_keys:
            todo_rows.append(r)

    if not todo_rows:
        print("SEC compression: nothing to do (all rows already processed).")
        return

    print(f"SEC compression: {len(todo_rows)} rows to process.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("SEC out_path:", out_path.resolve())

    for r in tqdm(todo_rows, desc="SEC compression"):
        try:
            out_row = _compress_single_sec_row(
                row=r,
                model_cfg=cfg["model"],
                sec_cfg=cfg["sec"],
            )
        except Exception as e:
            print(
                f"[SEC] Skipping row (ticker={r.get('ticker','')}, "
                f"form_type={r.get('form_type','')}, "
                f"filing_date={r.get('filing_date','')}) due to error: {e}"
            )
            continue

        out_df = pd.concat(
            [out_df, pd.DataFrame([out_row])],
            ignore_index=True,
        )
        out_df.to_parquet(out_path, index=False)

    print(f"SEC compression: wrote {len(out_df)} total rows to {out_path}")


# =========================
#   Driver
# =========================

def _load_cfg(path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="data/raw/compression_config/config_azure.yaml",
        help="Path to compression config YAML.",
    )
    ap.add_argument(
        "--mode",
        choices=["ec", "sec", "both"],
        default="both",
        help="Which domain(s) to compress.",
    )
    args = ap.parse_args()
    cfg = _load_cfg(Path(args.config))

    if args.mode in ("sec", "both"):
        run_sec_compression(cfg)
    if args.mode in ("ec", "both"):
        run_ec_compression(cfg)


if __name__ == "__main__":
    main()


