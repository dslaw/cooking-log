from pathlib import Path
from time import time

import pandas as pd

from src.clean import FILTER_LINES, clean, tokenize
from src.parse import parse_cooking_log, read_cooking_log

PROGRESS_INTERVAL = 1_000_000
CHECKPOINT_INTERVAL = 4_000_000


def write_distances(data_dir: Path, checkpoint_num: int, distances: list[dict]) -> None:
    output_file = data_dir / f"distances_{checkpoint_num}.parquet"
    df = pd.DataFrame.from_records(distances)
    df.to_parquet(output_file, index=False)
    return


def similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard similarity between token sets."""
    unique_tokens_a = set(tokens_a)
    unique_tokens_b = set(tokens_b)
    n_overlap = len(unique_tokens_a.intersection(unique_tokens_b))
    n_total = len(unique_tokens_a.union(unique_tokens_b))
    return n_overlap / n_total if n_total else 0.0


print("--- Reading cooking log ---")
data_dir = Path("data")
input_file = data_dir / "cooking-log.md"
cooking_log = read_cooking_log(input_file)
entries = parse_cooking_log(cooking_log)

print(f"--- Read {len(entries):,} entries ---")

# Create a table with a row for each log entry.
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


# Create a table with pairwise distances between dishes.
print("--- Creating distances table ---")
distances: list[dict] = []
n_processed = 0
checkpoint_num = 1
cache: dict[tuple[int, int], float] = {}
start_time = time()
for _, dish_a in df_dishes.iterrows():
    for _, dish_b in df_dishes.iterrows():
        dish_a_id = dish_a["dish_id"]
        dish_b_id = dish_b["dish_id"]

        if dish_a_id == dish_b_id:
            continue

        if n_processed > 0 and ((n_processed % PROGRESS_INTERVAL) == 0):
            print(f"Calculated {n_processed:,} distances")

        # Flush in-memory data to avoid OOM.
        if distances and ((len(distances) % CHECKPOINT_INTERVAL) == 0):
            write_distances(data_dir, checkpoint_num, distances)
            print(f"Checkpointed {len(distances):,} distances")

            checkpoint_num += 1
            distances.clear()
            cache.clear()

        cache_key = tuple(sorted([dish_a_id, dish_b_id]))
        if cache_key in cache:
            distance = cache[cache_key]
        elif dish_a.tokens == dish_b.tokens:
            distance = 1.0
        else:
            distance = similarity(dish_a.tokens, dish_b.tokens)

        distances.append(
            {
                "dish_a_id": dish_a_id,
                "dish_b_id": dish_b_id,
                "distance": distance,
            }
        )
        cache[cache_key] = distance
        n_processed += 1

if distances:
    write_distances(data_dir, checkpoint_num, distances)

elapsed_time = time() - start_time
print(
    f"--- Finished calculating {n_processed:,} distances in {int(elapsed_time):,} seconds"
)

print("--- Writing to disk ---")
df_entries.to_parquet("data/entries.parquet", index=True)
df_dishes.to_parquet("data/dishes.parquet", index=True)
