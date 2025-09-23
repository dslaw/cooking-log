import pandas as pd
from datetime import date

from app.utils import deduplicate_and_count


def test_deduplicate_counts():
    data = [
        {"dish_id": 1, "raw_text": "A", "cleaned_text": "a", "canonical_dish_id": 1, "entry_date": date(2025, 1, 1), "meal_type": "Lunch"},
        {"dish_id": 2, "raw_text": "A alt", "cleaned_text": "a", "canonical_dish_id": 1, "entry_date": date(2025, 1, 2), "meal_type": "Dinner"},
        {"dish_id": 3, "raw_text": "B", "cleaned_text": "b", "canonical_dish_id": 2, "entry_date": date(2025, 1, 3), "meal_type": "Lunch"},
    ]
    df = pd.DataFrame(data)
    out = deduplicate_and_count(df)
    # Two canonical ids
    assert len(out) == 2
    # canonical 1 frequency == 2
    row = out[out["canonical_dish_id"] == 1].iloc[0]
    assert int(row["frequency"]) == 2


def test_deduplicate_empty():
    df = pd.DataFrame(columns=["dish_id", "raw_text", "cleaned_text", "canonical_dish_id", "entry_date", "meal_type"]) 
    out = deduplicate_and_count(df)
    assert out.empty


def test_tiebreak_prefers_canonical_dish_id():
    data = [
        {"dish_id": 1, "raw_text": "Alias A", "cleaned_text": "a", "canonical_dish_id": 10, "entry_date": date(2025, 1, 1), "meal_type": "Lunch"},
        {"dish_id": 10, "raw_text": "Canonical A", "cleaned_text": "a-can", "canonical_dish_id": 10, "entry_date": date(2025, 1, 2), "meal_type": "Dinner"},
    ]
    df = pd.DataFrame(data)
    out = deduplicate_and_count(df)
    assert len(out) == 1
    row = out.iloc[0]
    # Representative dish_id should be the canonical id (10)
    assert int(row["dish_id"]) == 10
    # Frequency should be 2
    assert int(row["frequency"]) == 2
