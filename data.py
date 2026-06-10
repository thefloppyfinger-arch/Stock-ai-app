import yfinance as yf
import pandas as pd

def get_data(ticker):
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)

    if df.empty:
        return df

    df = df.dropna()

    df["ret1"] = df["Close"].pct_change()
    df["ret5"] = df["Close"].pct_change(5)
    df["ret10"] = df["Close"].pct_change(10)

    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma_ratio"] = df["ma20"] / df["ma50"]

    df["volatility"] = df["ret1"].rolling(10).std()

    df["volume_change"] = df["Volume"].pct_change()

    # Target: outperform over next 10 days
    df["future_return"] = df["Close"].shift(-10) / df["Close"] - 1
    df["target"] = (df["future_return"] > 0.05).astype(int)

    return df.dropna()
