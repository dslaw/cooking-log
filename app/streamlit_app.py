"""
Streamlit single-page app to browse cooked dishes.

Placeholder data is used below; replace the placeholder `df` with your own DataFrame
that contains the columns: dish_id, raw_text, cleaned_text, canonical_dish_id,
entry_date (datetime.date), meal_type (str).

Run with:
    streamlit run streamlit_app.py

"""
import pandas as pd
import streamlit as st
import altair as alt

from utils import deduplicate_and_count
from load_data import load_data


def filter_df(df: pd.DataFrame, start_date, end_date, meal_type: str, raw_text_substr: str) -> pd.DataFrame:
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

    df = load_data()

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

    meal_type = st.sidebar.selectbox("Meal type", options=["All", "Lunch", "Dinner"], index=0)

    raw_text_substr = st.sidebar.text_input("Raw text contains (substring)")

    # Search button to apply filters and refresh right pane
    if st.sidebar.button("Search"):
        # Save filtered in session state so display updates only when Search pressed
        filtered = filter_df(df, start_date, end_date, meal_type, raw_text_substr)
        st.session_state["filtered_df"] = filtered
    # Initialize session state on first run
    if "filtered_df" not in st.session_state:
        st.session_state["filtered_df"] = df

    # Right pane: results
    st.title("Cooked Dishes")

    result_df = st.session_state["filtered_df"]

    # Deduplicate and compute frequency
    dedup = deduplicate_and_count(result_df)

    # Table view (sorting options removed from UI)
    st.subheader("Dishes (deduplicated)")
    st.write("Showing deduplicated dishes with representative fields and frequency counts.")

    # Table column display order
    cols = ["entry_date", "meal_type", "raw_text", "frequency", "dish_id", "canonical_dish_id", "cleaned_text"]

    # Default directions (assumptions): entry_date most-recent first (desc), frequency desc, others asc
    default_directions = {
        "entry_date": False,
        "meal_type": True,
        "raw_text": True,
        "frequency": False,
        "cleaned_text": True,
        "canonical_dish_id": True,
        "dish_id": True,
    }

    # Two-column layout: left for controls, right for table/chart
    left_col, right_col = st.columns([1, 6])

    # Checkbox to toggle showing duplicate rows. Default: not showing duplicates.
    with left_col:
        show_duplicates = st.checkbox("Show duplicate rows", value=False)

    # Prepare display DataFrame depending on checkbox
    if show_duplicates:
        # original rows view: need to compute frequency per original row
        display_original = result_df.copy()
        if not dedup.empty:
            freq_map = dict(zip(dedup['canonical_dish_id'], dedup['frequency']))
            display_original['frequency'] = display_original['canonical_dish_id'].map(freq_map).fillna(0).astype(int)
        else:
            display_original['frequency'] = 0
        display_df = display_original.copy()
    else:
        # deduplicated view
        display_df = dedup.copy()

    # Apply default sorting only (no interactive sort controls)
    asc_list = [default_directions.get(col, True) for col in cols]
    display_df = display_df.sort_values(by=cols, ascending=asc_list)

    with right_col:
        st.dataframe(display_df, column_order=cols, hide_index=True, width="stretch")

    # Frequency plot
    st.subheader("Frequency plot")
    if not dedup.empty:
        chart_df = dedup.copy()
        chart_df["label"] = chart_df["cleaned_text"].fillna(chart_df["raw_text"]) if "cleaned_text" in chart_df.columns else chart_df["raw_text"]
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("frequency:Q", title="Frequency"),
                y=alt.Y("label:N", sort=alt.EncodingSortField(field="frequency", order="descending"), title="Dish"),
                tooltip=["label", "frequency"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No dishes match the current filters.")


if __name__ == "__main__":
    main()
