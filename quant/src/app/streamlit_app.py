import streamlit as st
import pandas as pd

st.set_page_config(page_title="Quant Portfolio Constructor")

st.title("Quant Portfolio Constructor")

uploaded_file = st.file_uploader("Upload predictions CSV with columns 'Ticker' and 'Prediction'", type=["csv"])
top_n = st.number_input("Number of top long positions", min_value=1, max_value=50, value=5)
short_n = st.number_input("Number of bottom short positions", min_value=1, max_value=50, value=5)
cap_per_name = st.number_input("Maximum absolute weight per name", min_value=0.0, max_value=1.0, value=0.05, step=0.01)


def construct_from_df(df: pd.DataFrame, top_n: int, short_n: int, cap: float) -> pd.DataFrame:
    df = df.sort_values("Prediction", ascending=False).reset_index(drop=True)
    longs = df.head(top_n).copy()
    shorts = df.tail(short_n).copy()
    longs["Weight"] = 1.0 / top_n
    shorts["Weight"] = -1.0 / short_n
    weights = pd.concat([longs[["Ticker", "Weight"]], shorts[["Ticker", "Weight"]]], ignore_index=True)
    weights["Weight"] = weights["Weight"].clip(-cap, cap)
    total_long = weights.loc[weights["Weight"] > 0, "Weight"].sum()
    total_short = -weights.loc[weights["Weight"] < 0, "Weight"].sum()
    if total_long > 0:
        weights.loc[weights["Weight"] > 0, "Weight"] /= total_long
    if total_short > 0:
        weights.loc[weights["Weight"] < 0, "Weight"] /= total_short
    return weights


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Ticker" not in df.columns or "Prediction" not in df.columns:
        st.error("CSV must contain 'Ticker' and 'Prediction' columns.")
    else:
        weights = construct_from_df(df, int(top_n), int(short_n), float(cap_per_name))
        st.subheader("Constructed Portfolio Weights")
        st.dataframe(weights)
        st.bar_chart(weights.set_index('Ticker')['Weight'])
else:
    st.info("Please upload a predictions CSV to construct a portfolio.")
