from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms.clique import find_cliques

MIN_DISTANCE = 0.6


def read_distances(input_file: Path, min_distance: float) -> pd.DataFrame:
    # Run filtering immediately to reduce memory pressure.
    return pd.read_parquet(input_file, filters=[("distance", ">=", min_distance)])


def make_graph(records: list[dict]) -> nx.Graph:
    graph = nx.Graph()
    for record in records:
        graph.add_node(record["dish_a_id"])
        graph.add_node(record["dish_b_id"])
        graph.add_edge(
            record["dish_a_id"], record["dish_b_id"], weight=record["distance"]
        )

    return graph


print("--- Reading distances ---")
data_dir = Path("data")

dfs = []
for input_file in data_dir.glob("distances_*.parquet"):
    df_p = read_distances(input_file, min_distance=MIN_DISTANCE)
    print(f"--- Read {len(df_p):,} rows after filtering ----")
    dfs.append(df_p)

df = pd.concat(dfs)
records = df.to_dict(orient="records")

graph = make_graph(records)
clique_records = []
for clique_id, dish_ids in enumerate(find_cliques(graph)):
    for dish_id in dish_ids:
        clique_records.append({"id": clique_id, "dish_id": dish_id})

print("--- Writing to disk ---")
output_file = data_dir / "cliques.parquet"
df_cliques = pd.DataFrame.from_records(clique_records)
df_cliques.to_parquet(output_file, index=False)
