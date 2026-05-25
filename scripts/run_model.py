# ─────────────────────────────────────────────
# scripts/run_model.py
# Train Risk Classifier + Mortality Regression
# ─────────────────────────────────────────────
"""
Usage:
    python scripts/run_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    "Poverty_Index", "Skilled_Birth_Attendance_pct", "Clean_Water_Access_pct",
    "Immunization_Coverage_pct", "Stunting_Prevalence_pct", "Wasting_Prevalence_pct",
    "ANC_Visits_4plus_pct", "Facility_Delivery_pct", "Female_Literacy_Rate_pct",
    "ASAL_Flag", "Health_System_Score",
]

def load_data():
    df = pd.read_csv("data/processed/county_mortality_indicators_clean.csv")
    df = df[df["Year"] == 2022].copy()
    return df

def encode_labels(df):
    le = LabelEncoder()
    df["Risk_Tier_Encoded"] = le.fit_transform(df["Risk_Tier"])
    return df, le

def train_classifier(X_train, X_test, y_train, y_test):
    print("\n── Risk Tier Classifier (Random Forest) ──")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(rf, "models/risk_classifier.pkl")
    print("  Saved: models/risk_classifier.pkl")
    return rf

def train_regressor(X_train, X_test, y_train, y_test):
    print("\n── Mortality Rate Regressor (Gradient Boosting) ──")
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                    max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"  RMSE : {rmse:.2f} per 1,000 live births")
    print(f"  R²   : {r2:.3f}")
    joblib.dump(gb, "models/mortality_regressor.pkl")
    print("  Saved: models/mortality_regressor.pkl")
    return gb

def print_feature_importance(model, feature_names, label):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    print(f"\n── Top Features ({label}) ──")
    print(importance.sort_values(ascending=False).head(5).to_string())

def main():
    print("Loading data...")
    df = load_data()
    df, le = encode_labels(df)

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available_features].fillna(0)

    # ── Classifier
    y_cls = df["Risk_Tier_Encoded"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    clf = train_classifier(X_tr, X_te, y_tr, y_te)
    print_feature_importance(clf, available_features, "Risk Classifier")

    # ── Regressor
    y_reg = df["Under5_Mortality_Rate_per1000"]
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = train_regressor(X_tr2, X_te2, y_tr2, y_te2)
    print_feature_importance(reg, available_features, "Mortality Regressor")

    # Save label encoder
    joblib.dump(le, "models/label_encoder.pkl")
    print("\nAll models saved to models/")

if __name__ == "__main__":
    main()
