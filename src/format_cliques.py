import json
from pathlib import Path

import pandas as pd


def find_exemplar(dishes: list[dict]):
    counts = {}
    for idx, dish in enumerate(dishes):
        tokens = tuple(dish["tokens"])
        if tokens not in counts:
            counts[tokens] = {"count": 0, "idx": idx, "dish_id": dish["dish_id"]}

        counts[tokens]["count"] += 1

    # Exemplar is chosen as the most frequently occurring dish, based on
    # tokenized representation. If there is a tie, the most simple dish, using
    # number of ingredients as a proxy, is selected.
    # Finally, `dish_id` to help use the same dishes consistently, when
    # possible.
    _, max_freq_item = max(
        counts.items(),
        key=lambda kv: (kv[1]["count"], -len(kv[0]), kv[1]["dish_id"]),
    )
    return dishes[max_freq_item["idx"]]


data_dir = Path("data")

df_cliques = pd.read_parquet(data_dir / "cliques.parquet")
df_dishes = pd.read_parquet(data_dir / "dishes.parquet")

# Take an exemplar from each clique.
cliques = []
for clique_id, df_s in df_cliques.groupby("id"):
    dish_ids = list(df_s.dish_id)
    dishes = df_dishes[df_dishes["dish_id"].isin(dish_ids)]
    dishes = dishes.to_dict(orient="records")
    exemplar = find_exemplar(dishes)

    cliques.append(
        {
            "id": clique_id,
            "dish_ids": dish_ids,
            "exemplar": {
                "dish_id": exemplar["dish_id"],
                "raw_text": exemplar["raw_text"],
                "cleaned_text": exemplar["cleaned_text"],
                "tokens": exemplar["tokens"],
            },
        }
    )


print("--- Writing to disk ---")
with (data_dir / "clique_exemplars.json").open("w") as fh:
    json.dump(cliques, fh, indent=4)
