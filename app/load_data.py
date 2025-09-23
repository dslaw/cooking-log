import pandas as pd


def load_data():
    df_dishes = pd.read_parquet("dishes_deduped.parquet")
    df_entries = pd.read_parquet("entries.parquet")

    df_entries = df_entries.rename(columns={
        "date": "entry_date",
        "id": "entry_id",
        "meal": "meal_type",
    })
    df_entries["entry_date"] = pd.to_datetime(df_entries["entry_date"]).dt.date

    # TODO: There's an upstream issue where some of the dishes have an entry id
    # of 0.
    df_merged = df_dishes.merge(df_entries, how="inner", on="entry_id")
    return df_merged
