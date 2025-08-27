from collections import Counter
from pathlib import Path

import pandas as pd


def hash_tokens(tokens: list[str]) -> str:
    token_counts = Counter(tokens)
    token_counts_s = sorted(token_counts.items())
    return "".join(f"{token}{count}" for token, count in token_counts_s)


print("--- Reading dishes ---")
data_dir = Path("data")
df = pd.read_parquet(data_dir / "dishes.parquet")

print("--- Hashing dishes by token count ---")
df["token_hash"] = df["tokens"].map(hash_tokens)
n_unique_dishes = len(df["token_hash"].unique())
print(f"--- {n_unique_dishes:,} unique hashes out of {len(df):,} dishes")

print("--- Assigning canonical ids ---")
canonical_dish_ids = df["dish_id"].copy()
for _, df_g in df.groupby("token_hash"):
    canonical_dish_id = df_g["dish_id"].min()
    select = canonical_dish_ids.isin(df_g["dish_id"])
    canonical_dish_ids[select] = canonical_dish_id

df["canonical_dish_id"] = canonical_dish_ids

print("--- Writing to disk ---")
df.to_parquet(data_dir / "dishes_deduped.parquet", index=False)
