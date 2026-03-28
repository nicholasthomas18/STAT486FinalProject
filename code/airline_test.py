"""
Airline Delay Prediction + Anomaly Detection
Adapted for the Kaggle pre-aggregated dataset:
  https://www.kaggle.com/datasets/sriharshaeedala/airline-delay/data

Each row = one carrier x airport x month.
Supervised: predict arr_del15 rate (regression) or high-delay flag (classification)
Anomaly:    Isolation Forest directly on the feature columns — no aggregation needed.

Download the CSV from Kaggle and set DATA_PATH below.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

DATA_PATH = Path("/Users/nicholasthomas/Desktop/STAT486/STAT486FinalProject/data/Airline_Delay_Cause.csv")   # <- set to your downloaded file path


# ─── 1. LOAD & INSPECT ───────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nYear range: {df['year'].min()} – {df['year'].max()}")
    print(f"Carriers:   {df['carrier'].nunique()}")
    print(f"Airports:   {df['airport'].nunique()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ─── 2. CLEAN & ENGINEER FEATURES ────────────────────────────────────────────

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with no flights (division guard)
    df = df[df["arr_flights"] > 0].reset_index(drop=True)

    # --- Derived rate features (normalise counts by flight volume) ---
    df["delay_rate"]      = df["arr_del15"]       / df["arr_flights"]
    df["cancel_rate"]     = df["arr_cancelled"]    / df["arr_flights"]
    df["divert_rate"]     = df["arr_diverted"]     / df["arr_flights"]
    df["mean_delay_mins"] = df["arr_delay"]        / df["arr_flights"].clip(lower=1)

    # Cause fractions: what share of delayed flights is each cause?
    cause_cols = ["carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct"]
    total_cause = df[cause_cols].sum(axis=1).clip(lower=1)
    for col in cause_cols:
        df[f"pct_{col}"] = df[col] / total_cause

    # --- Time features ---
    df["is_summer"]  = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"]  = df["month"].isin([12, 1, 2]).astype(int)
    df["is_holiday_month"] = df["month"].isin([11, 12]).astype(int)
    # Cyclical encoding for month (preserves Jan ≈ Dec proximity)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # --- Encode categoricals ---
    le_carrier = LabelEncoder()
    le_airport = LabelEncoder()
    df["carrier_code"] = le_carrier.fit_transform(df["carrier"])
    df["airport_code"] = le_airport.fit_transform(df["airport"])

    # --- Binary classification target: high-delay carrier-airport-month ---
    # (top quartile of delay_rate = "high delay period")
    threshold = df["delay_rate"].quantile(0.75)
    df["high_delay"] = (df["delay_rate"] >= threshold).astype(int)
    print(f"High-delay threshold: {threshold:.3f}  "
          f"({df['high_delay'].mean()*100:.1f}% of rows positive)")

    return df, le_carrier, le_airport


# ─── 3. FEATURE SETS ─────────────────────────────────────────────────────────

# Supervised model features — intentionally exclude raw counts to
# prevent leakage (arr_del15 is derived from the target)
SUPERVISED_FEATURES = [
    "carrier_code", "airport_code",
    "month_sin", "month_cos",
    "is_summer", "is_winter", "is_holiday_month",
    "arr_flights",          # log-scale in model, represents route/hub size
    "pct_carrier_ct",
    "pct_weather_ct",
    "pct_nas_ct",
    "pct_late_aircraft_ct",
]

# Anomaly detection features — operational snapshot per row
ANOMALY_FEATURES = [
    "delay_rate", "cancel_rate", "divert_rate", "mean_delay_mins",
    "pct_carrier_ct", "pct_weather_ct", "pct_nas_ct",
    "pct_security_ct", "pct_late_aircraft_ct",
    "arr_flights",
]


# ─── 4. SUPERVISED MODEL ─────────────────────────────────────────────────────

def train_classifier(df: pd.DataFrame):
    """
    Time-based split: train on all years except the last, test on the last year.
    GroupShuffleSplit is NOT used here — we do a hard temporal cutoff to
    avoid leakage from rolling operational patterns.
    """
    last_year = df["year"].max()
    train = df[df["year"] < last_year]
    test  = df[df["year"] == last_year]

    print(f"\nTrain: {len(train):,} rows ({train['year'].min()}–{train['year'].max()-1})")
    print(f"Test:  {len(test):,} rows  ({last_year})")

    X_train = train[SUPERVISED_FEATURES].fillna(0)
    y_train = train["high_delay"]
    X_test  = test[SUPERVISED_FEATURES].fillna(0)
    y_test  = test["high_delay"]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=50)

    y_prob = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    print(f"\nTest AUROC: {auroc:.4f}")

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test,
                      feature_names=SUPERVISED_FEATURES, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()
    print("SHAP plot saved: shap_summary.png")

    test = test.copy()
    test["pred_prob"] = y_prob
    return model, test


# ─── 5. ANOMALY DETECTION ────────────────────────────────────────────────────

def run_anomaly_detection(df: pd.DataFrame,
                          contamination: float = 0.05) -> pd.DataFrame:
    """
    Isolation Forest on the full dataset.
    Note: fits on ALL rows (unsupervised — no labels used).
    Tune contamination to match the expected fraction of genuinely
    anomalous carrier-airport-months (typically 3–7%).
    """
    X = df[ANOMALY_FEATURES].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    raw = iso.decision_function(X_scaled)
    # Normalize to [0, 1]: higher = more anomalous
    scores = 1 - (raw - raw.min()) / (raw.max() - raw.min())

    df = df.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"]    = (iso.predict(X_scaled) == -1).astype(int)

    n = df["is_anomaly"].sum()
    print(f"\nAnomalous rows: {n:,} / {len(df):,} ({n/len(df)*100:.1f}%)")

    return df


# ─── 6. JOINT ANALYSIS ───────────────────────────────────────────────────────

def quadrant_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate supervised predictions with anomaly flags.
    Only meaningful on the test set rows that have pred_prob.
    """
    if "pred_prob" not in df.columns:
        raise ValueError("Run train_classifier first to get pred_prob.")

    high_risk = df["pred_prob"] > 0.5
    anomalous = df["is_anomaly"] == 1

    df = df.copy()
    df["quadrant"] = np.select(
        [high_risk & ~anomalous,
         high_risk &  anomalous,
         ~high_risk & ~anomalous,
         ~high_risk &  anomalous],
        ["high_risk_normal",
         "high_risk_anomalous",
         "low_risk_normal",
         "low_risk_anomalous"],
        default="unknown"
    )

    print("\nQuadrant counts:")
    print(df["quadrant"].value_counts())

    # Mean actual delay rate per quadrant
    print("\nMean delay_rate per quadrant:")
    print(df.groupby("quadrant")["delay_rate"].mean().sort_values(ascending=False))

    return df


def inspect_anomalous_rows(df: pd.DataFrame, top_n: int = 20):
    """
    Print the most anomalous carrier-airport-months.
    These should correspond to known disruption events.
    """
    cols = ["year", "month", "carrier_name", "airport_name",
            "delay_rate", "cancel_rate", "mean_delay_mins", "anomaly_score"]
    top = (df[df["is_anomaly"] == 1]
           .sort_values("anomaly_score", ascending=False)
           .head(top_n)[cols])
    print(f"\nTop {top_n} most anomalous carrier-airport-months:")
    print(top.to_string(index=False))
    return top


def plot_anomaly_heatmap(df: pd.DataFrame):
    """
    Heatmap: average anomaly score by carrier x month.
    Reveals which carriers have systemic seasonal issues.
    """
    pivot = df.pivot_table(
        values="anomaly_score",
        index="carrier_name",
        columns="month",
        aggfunc="mean"
    )
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0.3, vmax=0.7)
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean anomaly score")
    ax.set_title("Mean anomaly score by carrier and month")
    plt.tight_layout()
    plt.savefig("anomaly_heatmap.png", dpi=150)
    plt.close()
    print("Heatmap saved: anomaly_heatmap.png")


def plot_anomaly_timeline(df: pd.DataFrame, carrier: str):
    """
    Line chart of anomaly score over time for one carrier, all airports averaged.
    """
    sub = df[df["carrier"] == carrier].copy()
    sub["date"] = pd.to_datetime(sub[["year", "month"]].assign(day=1))
    monthly = sub.groupby("date")["anomaly_score"].mean()

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(monthly.index, monthly.values, alpha=0.35, color="#E24B4A")
    ax.plot(monthly.index, monthly.values, color="#E24B4A", linewidth=1)
    ax.axhline(0.6, color="#888", linestyle="--", linewidth=0.8, label="Anomaly threshold ≈ 0.6")
    ax.set_ylim(0, 1)
    ax.set_title(f"Anomaly score over time — {carrier}")
    ax.set_ylabel("Mean anomaly score")
    ax.legend()
    plt.tight_layout()
    out = f"anomaly_timeline_{carrier}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Timeline saved: {out}")


# ─── 7. MAIN ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=== Step 1: Load ===")
    raw = load_data(DATA_PATH)

    print("\n=== Step 2: Clean & engineer ===")
    df, le_carrier, le_airport = clean_and_engineer(raw)

    print("\n=== Step 3: Train classifier ===")
    model, test_df = train_classifier(df)

    print("\n=== Step 4: Anomaly detection (full dataset) ===")
    df_with_scores = run_anomaly_detection(df, contamination=0.05)

    # Bring anomaly scores into the test split for joint analysis
    score_cols = ["year", "month", "carrier", "airport", "anomaly_score", "is_anomaly"]
    test_df = test_df.merge(df_with_scores[score_cols],
                            on=["year", "month", "carrier", "airport"],
                            how="left")

    print("\n=== Step 5: Joint quadrant analysis ===")
    test_df = quadrant_analysis(test_df)

    print("\n=== Step 6: Inspect top anomalies ===")
    top_anomalies = inspect_anomalous_rows(df_with_scores)

    print("\n=== Step 7: Visualizations ===")
    plot_anomaly_heatmap(df_with_scores)

    # Timeline for Southwest (WN) — should spike around Dec 2022 if in range
    if "WN" in df_with_scores["carrier"].values:
        plot_anomaly_timeline(df_with_scores, "WN")

    print("\nOutput files:")
    print("  shap_summary.png       — feature importance")
    print("  anomaly_heatmap.png    — carrier x month anomaly heatmap")
    print("  anomaly_timeline_*.png — per-carrier anomaly over time")