from collections import Counter

import pandas as pd

from src.config import COOKING_LOG_FILE, DISHES_FILE, ENTRIES_FILE
from src.parser import CookingLogEntry, CookingLogParser, read_cooking_log
from src.text_processor import TextProcessor, make_text_processor


def hash_tokens(tokens: list[str]) -> str:
    token_counts = Counter(tokens)
    token_counts_s = sorted(token_counts.items())
    return "".join(f"{token}{count}" for token, count in token_counts_s)


def make_entries_dataframe(entries: list[CookingLogEntry]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(
        [
            {
                "date": entry.date,
                "meal": str(entry.meal),
                "dishes": entry.dishes,
                "notes": entry.notes,
            }
            for entry in entries
        ]
    )
    df["date"] = pd.to_datetime(df.date)
    df["id"] = range(1, len(df) + 1)
    return df


def make_dishes(
    df_entries: pd.DataFrame, text_processor: TextProcessor
) -> pd.DataFrame:
    dish_id = 1
    dish_records = []
    text_processor = make_text_processor()
    for _, entry in df_entries.iterrows():
        entry_id = entry["id"]
        for dish in entry["dishes"]:
            result = text_processor.process(dish)
            if result is None:
                continue

            cleaned, tokens, language, language_confidence = result

            dish_records.append(
                {
                    "dish_id": dish_id,
                    "entry_id": entry_id,
                    "raw_text": dish,
                    "language": language,
                    "language_confidence": language_confidence,
                    "cleaned_text": cleaned,
                    "tokens": tokens,
                }
            )
            dish_id += 1

    return pd.DataFrame.from_records(dish_records)


def deduplicate(tokenized_dishes: pd.Series, dish_ids: pd.Series) -> pd.Series:
    token_hashes = tokenized_dishes.map(hash_tokens)
    df = pd.DataFrame({"token_hash": token_hashes, "dish_id": dish_ids})
    canonical_dish_ids = df["dish_id"].copy()
    for _, df_g in df.groupby("token_hash"):
        canonical_dish_id = df_g["dish_id"].min()
        select = canonical_dish_ids.isin(df_g["dish_id"])
        canonical_dish_ids[select] = canonical_dish_id

    return canonical_dish_ids


def main():
    parser = CookingLogParser()
    text_processor = make_text_processor()

    cooking_log = read_cooking_log(COOKING_LOG_FILE)
    entries = parser.parse(cooking_log)
    df_entries = make_entries_dataframe(entries)
    print(f"Read {len(df_entries):,} date/meal entries")

    df_dishes = make_dishes(df_entries, text_processor)
    df_entries.drop(columns=["dishes"], inplace=True)
    print(f"Read {len(df_dishes):,} dishes")

    canonical_dish_ids = deduplicate(df_dishes["tokens"], df_dishes["dish_id"])
    df_dishes["canonical_dish_id"] = canonical_dish_ids
    n_unique_dishes = len(canonical_dish_ids.unique())
    print(f"{n_unique_dishes:,} / {len(df_dishes):,} unique dishes")

    df_entries.to_parquet(ENTRIES_FILE, index=False)
    print(f'"Wrote entries to "{ENTRIES_FILE}"')
    df_dishes.to_parquet(DISHES_FILE, index=False)
    print(f'Wrote dishes to "{DISHES_FILE}"')
    return


if __name__ == "__main__":
    main()
