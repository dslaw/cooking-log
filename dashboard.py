import altair as alt
import pandas as pd
import streamlit as st

from src.config import DISHES_FILE, ENTRIES_FILE


def load_data() -> pd.DataFrame:
    df_entries = pd.read_parquet(ENTRIES_FILE)
    df_dishes = pd.read_parquet(DISHES_FILE)

    df_entries = df_entries.rename(
        columns={
            "date": "entry_date",
            "id": "entry_id",
            "meal": "meal_type",
        }
    )
    df_entries["entry_date"] = pd.to_datetime(df_entries["entry_date"]).dt.date

    df_merged = df_dishes.merge(df_entries, how="inner", on="entry_id")
    return df_merged


def with_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    frequencies = df["canonical_dish_id"].value_counts()
    if "frequency" in df.columns:
        df = df.drop(columns=["frequency"])

    return df.merge(frequencies, how="inner", on="canonical_dish_id").rename(
        columns={"count": "frequency"}
    )


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["dish_id"] == df["canonical_dish_id"]]


def filter_df(
    df: pd.DataFrame, start_date, end_date, meal_type: str, raw_text_substr: str
) -> pd.DataFrame:
    out = df.copy()
    if start_date is not None and end_date is not None:
        # inclusive
        out = out[(out["entry_date"] >= start_date) & (out["entry_date"] <= end_date)]

    if meal_type and meal_type != "All":
        out = out[out["meal_type"] == meal_type]

    if raw_text_substr:
        s = raw_text_substr.strip().lower()
        out = out[out["raw_text"].str.lower().str.contains(s, na=False)]

    return out


def main():
    st.set_page_config(page_title="Cooking Log", layout="wide")

    st.sidebar.header("Filters")

    df = with_frequencies(load_data())

    # Date filter is optional - show toggle
    use_date = st.sidebar.checkbox("Filter by date range")
    if use_date:
        min_date = df["entry_date"].min()
        max_date = df["entry_date"].max()
        start_date = st.sidebar.date_input("Start date", value=min_date)
        end_date = st.sidebar.date_input("End date", value=max_date)
    else:
        start_date = None
        end_date = None

    meal_type = st.sidebar.selectbox(
        "Meal type", options=["All", "Lunch", "Dinner"], index=0
    )

    raw_text_substr = st.sidebar.text_input("Raw text contains (substring)")

    # Search button to apply filters and refresh right pane
    if st.sidebar.button("Search"):
        # Save filtered in session state so display updates only when Search pressed
        filtered = filter_df(df, start_date, end_date, meal_type, raw_text_substr)
        st.session_state["filtered_df"] = with_frequencies(filtered)
    # Initialize session state on first run
    if "filtered_df" not in st.session_state:
        st.session_state["filtered_df"] = df

    # Right pane: results
    st.title("Cooked Dishes")

    result_df = st.session_state["filtered_df"]

    # Table view (sorting options removed from UI)
    st.subheader("Dishes (deduplicated)")
    st.write(
        "Showing deduplicated dishes with representative fields and frequency counts."
    )

    # Table column display order
    cols = [
        "entry_date",
        "meal_type",
        "raw_text",
        "frequency",
        "ingredients",
        "dish_id",
        "canonical_dish_id",
    ]

    # Default directions (assumptions): entry_date most-recent first (desc), frequency desc, others asc
    default_directions = [
        ("entry_date", False),
        ("meal_type", True),
        ("raw_text", True),
        ("frequency", False),
        ("canonical_dish_id", True),
        ("dish_id", True),
    ]

    # Checkbox to toggle showing duplicate rows. Default: not showing duplicates.
    show_duplicates = st.checkbox("Show duplicate rows", value=False)

    # Prepare display DataFrame depending on checkbox
    if show_duplicates:
        display_df = result_df
    else:
        # Deduplicate
        display_df = deduplicate(result_df)

    # Apply default sorting only (no interactive sort controls)
    sort_by, sort_ascending = zip(*default_directions)
    display_df = display_df.sort_values(
        by=list(sort_by), ascending=list(sort_ascending), axis=0
    )

    st.dataframe(display_df, column_order=cols, hide_index=True, width="stretch")

    # Entry date vs Frequency scatter plot (aggregated, deduplicated by entry_date)
    st.subheader("Entry date vs Frequency")

    # Always use deduplicated records for this plot
    dedup_now = (
        with_frequencies(deduplicate(result_df))
        if not result_df.empty
        else pd.DataFrame()
    )

    if (
        not dedup_now.empty
        and "entry_date" in dedup_now.columns
        and "frequency" in dedup_now.columns
    ):
        # Ensure datetime for Altair
        dedup_now["entry_date"] = pd.to_datetime(dedup_now["entry_date"])

        # Aggregate by date: sum frequency, and collect up to 3 sample raw_text values
        def sample_texts(x):
            vals = pd.unique(x.dropna().astype(str))
            return "; ".join(vals[:3]) if len(vals) > 0 else ""

        if "raw_text" in dedup_now.columns:
            agg = (
                dedup_now.groupby("entry_date")
                .agg({"frequency": "sum", "raw_text": sample_texts})
                .reset_index()
                .rename(columns={"raw_text": "raw_samples"})
            )
        else:
            agg = (
                dedup_now.groupby("entry_date").agg({"frequency": "sum"}).reset_index()
            )
            agg["raw_samples"] = ""

        # Build tooltip: include raw_samples
        tooltip = [
            alt.Tooltip("raw_samples:N", title="Sample raw_texts"),
            alt.Tooltip("frequency:Q", title="Frequency"),
            alt.Tooltip("entry_date:T", title="Entry date", format="%Y-%m-%d"),
        ]

        scatter = (
            alt.Chart(agg)
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X("entry_date:T", title="Entry date"),
                y=alt.Y("frequency:Q", title="Frequency"),
                tooltip=tooltip,
            )
            .properties(height=320)
        )

        st.altair_chart(scatter, use_container_width=True)
    else:
        st.write("No data available to plot entry date vs frequency.")

    # Frequency plot
    st.subheader("Frequency plot")
    dedup = deduplicate(result_df)
    if not dedup.empty:
        chart_df = dedup.copy()
        chart_df["label"] = chart_df["raw_text"]
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("frequency:Q", title="Frequency"),
                y=alt.Y(
                    "label:N",
                    sort=alt.EncodingSortField(field="frequency", order="descending"),
                    title="Dish",
                ),
                tooltip=["label", "frequency"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No dishes match the current filters.")


if __name__ == "__main__":
    main()
