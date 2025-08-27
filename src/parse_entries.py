from pathlib import Path

import pandas as pd

from src.clean import FILTER_LINES, clean, tokenize
from src.parse import parse_cooking_log, read_cooking_log

print("--- Reading cooking log ---")
data_dir = Path("data")
input_file = data_dir / "cooking-log.md"

print("--- Parsing cooking log ---")
cooking_log = read_cooking_log(input_file)
entries = parse_cooking_log(cooking_log)
print(f"--- Parsed {len(entries):,} ---")


print("--- Creating entries table ---")
df_entries = pd.DataFrame.from_records(
    [
        {
            "date": entry.date,
            "meal": str(entry.meal),
            "notes": entry.notes,
        }
        for entry in entries
    ]
)
df_entries["date"] = pd.to_datetime(df_entries.date)
df_entries["id"] = range(1, len(df_entries) + 1)

# Create a table with a row for each log entry dish.
print("--- Creating dishes table ---")
dish_id = 1
dish_records = []
for entry_id, entry in df_entries.iterrows():
    dishes = entries[entry_id].dishes
    for dish in dishes:
        cleaned = clean(dish)

        if cleaned in FILTER_LINES:
            continue

        tokens = tokenize(cleaned)

        dish_records.append(
            {
                "dish_id": dish_id,
                "entry_id": entry_id,
                "raw_text": dish,
                "cleaned_text": cleaned,
                "tokens": tokens,
            }
        )
        dish_id += 1

df_dishes = pd.DataFrame.from_records(dish_records)

print("--- Writing to disk ---")
df_entries.to_parquet(data_dir / "entries.parquet", index=True)
df_dishes.to_parquet(data_dir / "dishes.parquet", index=True)
