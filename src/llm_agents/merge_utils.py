from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

KEY_MERGE = ["date", "ticker"]
KEY_DEDUP = ["ticker", "date", "week_decision_date", "curr_close_utc", "prev_close_utc"]


def merge_tier1_into_market_keep_original_names(
    tier1_df: pd.DataFrame,
    market_df: pd.DataFrame,
    *,
    fill_value: float = 0.0,
    enforce_datetime: bool = True,
    # If True, raise if overlapping non-key columns disagree on overlapping keys.
    # If False, keep market_df values and drop tier1 overlapping columns even if they disagree.
    strict_overlap_check: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Merge tier1_df into market_df on (ticker, date) without suffixes.

    Policy:
      - Keep ALL non-overlapping columns from both dataframes exactly as-is.
      - For overlapping non-key columns:
          * Perform a surgical match on overlapping (ticker,date) rows.
          * If strict_overlap_check=True: raise if any disagreements.
          * Otherwise: keep market_df version and drop tier1 version of overlapping columns.

      - market_df is the backbone (left join).
      - Sort by ticker A->Z and date old->new.
      - Drop duplicates on KEY_DEDUP (keep last).
      - Fill missing tier1-only columns with `fill_value` (0.0) instead of NaN.

    Returns:
      merged_df, report (diagnostics)
    """
    t1 = tier1_df.copy()
    mk = market_df.copy()

    # --- Schema checks ---
    for col in KEY_MERGE:
        if col not in t1.columns:
            raise ValueError(f"tier1_df missing required merge key column: {col}")
        if col not in mk.columns:
            raise ValueError(f"market_df missing required merge key column: {col}")

    # --- Normalize types ---
    if enforce_datetime:
        t1["date"] = pd.to_datetime(t1["date"], errors="coerce")
        mk["date"] = pd.to_datetime(mk["date"], errors="coerce")
        if "week_decision_date" in t1.columns:
            t1["week_decision_date"] = pd.to_datetime(t1["week_decision_date"], errors="coerce")
        if "week_decision_date" in mk.columns:
            mk["week_decision_date"] = pd.to_datetime(mk["week_decision_date"], errors="coerce")

    t1["ticker"] = t1["ticker"].astype(str)
    mk["ticker"] = mk["ticker"].astype(str)

    # --- Pre-merge dedup on merge keys to avoid many-to-many explosion ---
    t1_dup_before = int(t1.duplicated(subset=KEY_MERGE).sum())
    mk_dup_before = int(mk.duplicated(subset=KEY_MERGE).sum())

    t1 = t1.sort_values(["date", "ticker"]).drop_duplicates(subset=KEY_MERGE, keep="last")
    mk = mk.sort_values(["date", "ticker"]).drop_duplicates(subset=KEY_MERGE, keep="last")

    # --- Identify overlapping non-key columns ---
    overlap_non_key = sorted((set(t1.columns) & set(mk.columns)) - set(KEY_MERGE))

    # --- Surgical match for overlapping columns on overlapping keys ---
    disagreements: Dict[str, int] = {}
    if overlap_non_key:
        # compare only on overlapping keys (ticker,date) that exist in both
        overlap_keys = mk[KEY_MERGE].merge(t1[KEY_MERGE], on=KEY_MERGE, how="inner")

        if len(overlap_keys) > 0:
            # bring the overlapping columns from each side for comparison
            mk_cmp = mk.merge(overlap_keys, on=KEY_MERGE, how="inner")[KEY_MERGE + overlap_non_key]
            t1_cmp = t1.merge(overlap_keys, on=KEY_MERGE, how="inner")[KEY_MERGE + overlap_non_key]

            # align rows by keys
            mk_cmp = mk_cmp.sort_values(KEY_MERGE).reset_index(drop=True)
            t1_cmp = t1_cmp.sort_values(KEY_MERGE).reset_index(drop=True)

            # Compare column by column, treating NaN == NaN as equal
            for c in overlap_non_key:
                a = mk_cmp[c]
                b = t1_cmp[c]

                # equality with NaN-safe logic
                eq = (a == b) | (a.isna() & b.isna())

                # For floats, you may want tolerant comparison; handle numeric columns:
                if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                    # tolerate tiny fp differences where both are finite
                    both_finite = np.isfinite(a.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(
                        b.to_numpy(dtype=float, na_value=np.nan)
                    )
                    tol_eq = np.zeros(len(eq), dtype=bool)
                    if both_finite.any():
                        aa = a.to_numpy(dtype=float, na_value=np.nan)
                        bb = b.to_numpy(dtype=float, na_value=np.nan)
                        tol_eq[both_finite] = np.isclose(aa[both_finite], bb[both_finite], rtol=1e-9, atol=1e-12)
                    eq = eq | pd.Series(tol_eq, index=eq.index)

                n_bad = int((~eq).sum())
                if n_bad > 0:
                    disagreements[c] = n_bad

        if strict_overlap_check and disagreements:
            # Provide a concise but actionable error
            bad_cols = ", ".join([f"{k}({v})" for k, v in disagreements.items()])
            raise ValueError(
                "Overlapping non-key columns disagree between market_df and tier1_df "
                f"on overlapping (ticker,date) rows: {bad_cols}. "
                "Either fix upstream, or set strict_overlap_check=False to keep market_df values."
            )

        # No suffixes: keep market_df version and drop tier1 overlapping columns (safe if strict passed)
        t1 = t1.drop(columns=overlap_non_key)

    # --- Merge without suffix (now guaranteed no overlapping non-key columns remain) ---
    merged = mk.merge(t1, on=KEY_MERGE, how="left")

    # --- Fill missing tier1-only columns with 0 ---
    tier1_only_cols = [c for c in t1.columns if c not in KEY_MERGE]
    merged[tier1_only_cols] = merged[tier1_only_cols].fillna(fill_value)

    # --- Dedup on requested composite key ---
    dedup_cols = [c for c in KEY_DEDUP if c in merged.columns]
    before = len(merged)
    merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    after = len(merged)

    # --- Sort: ticker A->Z, date old->new ---
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    report: Dict[str, object] = {
        "tier1_rows_in": int(len(tier1_df)),
        "market_rows_in": int(len(market_df)),
        "tier1_(ticker,date)_dups_before": t1_dup_before,
        "market_(ticker,date)_dups_before": mk_dup_before,
        "overlap_non_key_cols": overlap_non_key,
        "overlap_disagreements_counts": disagreements,
        "tier1_only_cols_added": tier1_only_cols,
        "filled_value": fill_value,
        "final_dedup_cols_used": dedup_cols,
        "rows_removed_by_final_dedup": int(before - after),
    }

    if verbose:
        if overlap_non_key:
            print("[merge] overlap non-key cols:", overlap_non_key)
        if disagreements:
            print("[merge] overlap disagreements:", disagreements)
        print("[merge] tier1 (ticker,date) dups before:", t1_dup_before)
        print("[merge] market (ticker,date) dups before:", mk_dup_before)
        print("[merge] rows removed by final dedup:", int(before - after))

    col = "ec_event_ts_utc"
    merged[col] = pd.to_datetime(merged[col], utc=True, errors="coerce")
    return merged, report
