import streamlit as st
import pandas as pd
from data import get_data
from model import train, predict

st.title("📊 Stock AI Dashboard (ML Ranking System)")

stocks = ["NVDA", "VRT", "RKLB", "CRDO", "AAOI", "AMD", "AAPL"]

all_data = []

for t in stocks:
    df = get_data(t)
    if df.empty:
        continue
    df["ticker"] = t
    all_data.append(df)

data = pd.concat(all_data).dropna()

model = train(data)

results = []

for t in stocks:
    df = get_data(t)

    if df.empty:
        continue

    df = predict(model, df)

    latest = df.iloc[-1]
    prob = latest["prob"]

    if prob > 0.60:
        signal = "BUY"
    elif prob > 0.50:
        signal = "HOLD"
    else:
        signal = "AVOID"

    results.append([t, round(prob, 3), signal])

results_df = pd.DataFrame(results, columns=["Ticker", "Probability", "Signal"])
results_df = results_df.sort_values("Probability", ascending=False)

st.dataframe(results_df)
