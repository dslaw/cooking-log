from typing import Optional
import pandas as pd


def deduplicate_and_count(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate a DataFrame by `canonical_dish_id` and add a `frequency` column.

    The returned DataFrame contains one row per `canonical_dish_id` with representative
    values for the other columns and a `frequency` column indicating how many rows
    in the original `df` corresponded to that canonical id.

    Columns expected in `df`: dish_id, raw_text, cleaned_text, canonical_dish_id,
    entry_date, meal_type
    """
    if df is None or df.empty:
        # Return empty frame with expected columns
        cols = ["dish_id", "raw_text", "cleaned_text", "canonical_dish_id", "entry_date", "meal_type", "frequency"]
        return pd.DataFrame(columns=cols)

    # Ensure required columns exist
    required = {"dish_id", "raw_text", "cleaned_text", "canonical_dish_id", "entry_date", "meal_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    group = df.groupby("canonical_dish_id")

    # We'll construct a representative row per canonical_dish_id.
    rows = []
    for canon_id, sub in group:
        # Frequency
        freq = len(sub)

        # Preferred representative: row where dish_id == canonical_dish_id, if any
        preferred = sub[sub["dish_id"] == canon_id]
        if not preferred.empty:
            rep = preferred.iloc[0]
        else:
            # Fall back to: earliest entry_date, then first
            sub_sorted = sub.sort_values("entry_date")
            rep = sub_sorted.iloc[0]

        rows.append({
            "dish_id": rep["dish_id"],
            "raw_text": rep.get("raw_text"),
            "cleaned_text": rep.get("cleaned_text"),
            "canonical_dish_id": canon_id,
            "entry_date": rep.get("entry_date"),
            "meal_type": rep.get("meal_type"),
            "frequency": freq,
        })

    merged = pd.DataFrame(rows)

    # Reorder columns for readability
    cols = ["dish_id", "raw_text", "cleaned_text", "canonical_dish_id", "entry_date", "meal_type", "frequency"]
    # Some DataFrames may not have canonical_dish_id as the middle column; ensure it's present
    merged = merged[[c for c in cols if c in merged.columns]]

    # Sort by frequency descending
    merged = merged.sort_values("frequency", ascending=False).reset_index(drop=True)

    return merged
