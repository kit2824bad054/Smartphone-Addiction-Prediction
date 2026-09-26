"""
Train Model: Smartphone Addiction Risk Predictor
Dataset loader, feature preprocessor, model trainer, joblib model persistence,
and JavaScript constant exporter.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

def locate_data_file():
    """Find the dataset file in either local or parent data/ directories."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "data", "cleaned_smartphone_addiction.csv"),
        os.path.join(os.path.dirname(__file__), "data", "teen_phone_addiction_raw.csv"),
        os.path.join("data", "cleaned_smartphone_addiction.csv"),
        os.path.join("data", "teen_phone_addiction_raw.csv"),
        os.path.join("..", "data", "cleaned_smartphone_addiction.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_smartphone_addiction.csv"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Could not find dataset in /data directory.")

def load_and_clean_data(file_path):
    """
    Loads dataset, handles missing values, and maps columns to:
    - screen_time_hours
    - unlocks_per_day
    - social_media_hours
    - night_usage_ratio (ratio 0.0 - 1.0)
    - sleep_hours
    """
    print(f"[1/5] Loading dataset from: {file_path}")
    raw_df = pd.read_csv(file_path)
    print(f"      Raw data shape: {raw_df.shape}")

    df = pd.DataFrame()

    # Column mapping logic supporting both raw Kaggle format and cleaned format
    if "screen_time_hours" in raw_df.columns:
        df["screen_time_hours"] = pd.to_numeric(raw_df["screen_time_hours"], errors="coerce")
        df["unlocks_per_day"] = pd.to_numeric(raw_df["unlocks_per_day"], errors="coerce")
        df["social_media_hours"] = pd.to_numeric(raw_df["social_media_hours"], errors="coerce")
        night_val = pd.to_numeric(raw_df["night_usage_ratio"], errors="coerce")
        # Standardize night_usage_ratio as a 0.0 - 1.0 fraction
        df["night_usage_ratio"] = np.where(night_val > 1.0, night_val / 100.0, night_val)
        df["night_usage_ratio"] = np.clip(df["night_usage_ratio"], 0.0, 1.0)
        df["sleep_hours"] = pd.to_numeric(raw_df["sleep_hours"], errors="coerce")
    else:
        # Map raw Kaggle dataset columns
        df["screen_time_hours"] = pd.to_numeric(raw_df["Daily_Usage_Hours"], errors="coerce")
        df["unlocks_per_day"] = pd.to_numeric(raw_df["Phone_Checks_Per_Day"], errors="coerce")
        df["social_media_hours"] = pd.to_numeric(raw_df["Time_on_Social_Media"], errors="coerce")
        safe_daily = np.maximum(df["screen_time_hours"].fillna(0.5), 0.5)
        bedtime_hrs = pd.to_numeric(raw_df.get("Screen_Time_Before_Bed", 1.0), errors="coerce").fillna(1.0)
        df["night_usage_ratio"] = np.clip(bedtime_hrs / safe_daily, 0.0, 1.0)
        df["sleep_hours"] = pd.to_numeric(raw_df["Sleep_Hours"], errors="coerce")

    # Clean missing values
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"      Dropped {dropped} rows with missing values.")
    else:
        print("      No missing values found.")

    print(f"      Cleaned dataset shape: {df.shape}")
    return df

def create_target_label(df):
    """
    Creates target label (Low/Moderate/High risk) by binning a composite usage score
    grounded in behavioral smartphone addiction criteria:
    - Higher screen time increases risk (+0.30)
    - Higher phone unlocks increases risk (+0.20)
    - Higher social media hours increases risk (+0.20)
    - Higher night usage share increases risk (+0.15)
    - Fewer sleep hours increases risk (-0.15)
    """
    print("[2/5] Creating target labels (Low / Moderate / High Risk)...")
    z_screen = (df["screen_time_hours"] - df["screen_time_hours"].mean()) / df["screen_time_hours"].std()
    z_unlocks = (df["unlocks_per_day"] - df["unlocks_per_day"].mean()) / df["unlocks_per_day"].std()
    z_social = (df["social_media_hours"] - df["social_media_hours"].mean()) / df["social_media_hours"].std()
    z_night = (df["night_usage_ratio"] - df["night_usage_ratio"].mean()) / df["night_usage_ratio"].std()
    z_sleep = -(df["sleep_hours"] - df["sleep_hours"].mean()) / df["sleep_hours"].std()

    composite_score = (
        0.30 * z_screen +
        0.20 * z_unlocks +
        0.20 * z_social +
        0.15 * z_night +
        0.15 * z_sleep
    )

    labels = ["Low", "Moderate", "High"]
    risk_categorical = pd.qcut(composite_score, q=3, labels=labels)
    class_mapping = {"Low": 0, "Moderate": 1, "High": 2}
    y = risk_categorical.map(class_mapping).astype(int)

    counts = risk_categorical.value_counts()
    for lbl in labels:
        print(f"      {lbl} Risk: {counts[lbl]} samples ({counts[lbl] / len(df) * 100:.1f}%)")

    return y, labels

def train_and_evaluate(X, y, class_names):
    """
    Splits data, standardizes with StandardScaler, trains Logistic Regression and
    Random Forest models, and prints comparison metrics.
    """
    print("[3/5] Standardizing features and splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[4/5] Training models...")
    # Logistic Regression (multinomial softmax)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test_scaled)

    # Metrics
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average="weighted")

    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

    print("\n" + "=" * 65)
    print("               MODEL PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"Logistic Regression: Accuracy = {acc_lr * 100:.2f}% | F1 Score = {f1_lr:.4f}")
    print(f"Random Forest:       Accuracy = {acc_rf * 100:.2f}% | F1 Score = {f1_rf:.4f}")
    print("-" * 65)
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=class_names))

    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))

    return scaler, lr, rf, (acc_lr, f1_lr, acc_rf, f1_rf)

def save_joblib_model(scaler, lr, feature_names, class_names):
    """
    Saves trained model and scaler package to model.joblib in both local and backend/ paths.
    """
    payload = {
        "model": lr,
        "scaler": scaler,
        "features": feature_names,
        "classes": class_names,
        "class_mapping": {0: "Low", 1: "Moderate", 2: "High"},
        "feature_labels": {
            "screen_time_hours": "Daily Screen Time",
            "unlocks_per_day": "Phone Unlocks / Checks",
            "social_media_hours": "Social Media Hours",
            "night_usage_ratio": "Night Usage Share",
            "sleep_hours": "Sleep Duration",
        },
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_paths = [
        os.path.join(base_dir, "model.joblib"),
        os.path.join(base_dir, "backend", "model.joblib"),
        os.path.join(base_dir, "..", "backend", "model.joblib"),
    ]

    for p in save_paths:
        target_dir = os.path.dirname(p)
        if os.path.exists(target_dir):
            joblib.dump(payload, p)
            print(f"[5/5] Saved model artifact to: {os.path.abspath(p)}")

def export_js_constants(scaler, lr, feature_names, class_names):
    """
    Prints JavaScript array constants formatted for index.html.
    """
    print("\n" + "=" * 65)
    print("      JAVASCRIPT CONSTANTS (FOR INDEX.HTML)")
    print("=" * 65)
    mean_list = [round(float(v), 6) for v in scaler.mean_]
    scale_list = [round(float(v), 6) for v in scaler.scale_]
    intercept_list = [round(float(v), 6) for v in lr.intercept_]

    print(f"const FEATURE_NAMES = {feature_names};")
    print(f"const RISK_CLASSES = {class_names};")
    print(f"const SCALER_MEAN = {mean_list};")
    print(f"const SCALER_SCALE = {scale_list};")
    print("const LOGREG_COEF = [")
    for idx, row in enumerate(lr.coef_):
        row_vals = [round(float(v), 6) for v in row]
        class_comment = f"// Class {idx} ({class_names[idx]} Risk)"
        print(f"  {row_vals}, {class_comment}")
    print("];")
    print(f"const LOGREG_INTERCEPT = {intercept_list};")
    print("=" * 65 + "\n")

def main():
    file_path = locate_data_file()
    df = load_and_clean_data(file_path)

    features = [
        "screen_time_hours",
        "unlocks_per_day",
        "social_media_hours",
        "night_usage_ratio",
        "sleep_hours",
    ]
    X = df[features].copy()
    y, class_names = create_target_label(df)

    scaler, lr, rf, metrics = train_and_evaluate(X, y, class_names)
    save_joblib_model(scaler, lr, features, class_names)
    export_js_constants(scaler, lr, features, class_names)

if __name__ == "__main__":
    main()
