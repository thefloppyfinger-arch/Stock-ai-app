from xgboost import XGBClassifier

FEATURES = [
    "ret1",
    "ret5",
    "ret10",
    "ma_ratio",
    "volatility",
    "volume_change"
]

model = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

def train(df):
    X = df[FEATURES]
    y = df["target"]
    model.fit(X, y)
    return model

def predict(model, df):
    df = df.copy()
    df["prob"] = model.predict_proba(df[FEATURES])[:, 1]
    return df
