# ─────────────────────────────────────────────
# Child Mortality Recommendation System
# src/risk_classifier.py
# ─────────────────────────────────────────────

"""
RiskClassifier
==============
Trains, evaluates, and predicts county-level child mortality
risk tiers using a Random Forest classifier.

Risk Tiers:
    High    — Under-5 mortality rate ≥ 60 per 1,000 live births
    Medium  — 35 to 59 per 1,000 live births
    Low     — < 35 per 1,000 live births

Usage:
    from src.risk_classifier import RiskClassifier

    clf = RiskClassifier()
    clf.train(X_train, y_train)
    report = clf.evaluate(X_test, y_test)
    predictions = clf.predict(X_new)
    clf.save("models/risk_classifier.pkl")
    clf.load("models/risk_classifier.pkl")
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project benchmark targets
TARGET_ACCURACY = 0.85
TARGET_F1_HIGH_RISK = 0.85


class RiskClassifier:
    """
    Random Forest classifier for county-level child mortality risk tiers.

    Parameters
    ----------
    n_estimators : int
        Number of trees. Default 200.
    max_depth : int
        Maximum tree depth. Default 8.
    random_state : int
        Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
        self.label_encoder = LabelEncoder()
        self.feature_names: list = []
        self.classes_: list = []
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
    ) -> dict:
        """
        Fit the classifier and run cross-validation.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix (counties × features).
        y_train : pd.Series
            Risk tier labels ('High', 'Medium', 'Low').
        cv_folds : int
            Number of cross-validation folds. Default 5.

        Returns
        -------
        dict
            Cross-validation results: mean accuracy, std, per-fold scores.
        """
        self.feature_names = list(X_train.columns)
        y_encoded = self.label_encoder.fit_transform(y_train.astype(str))
        self.classes_ = list(self.label_encoder.classes_)

        logger.info(
            "Training Random Forest — %d samples, %d features, classes: %s",
            len(X_train), X_train.shape[1], self.classes_,
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train, y_encoded, cv=cv, scoring="accuracy"
        )
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()

        logger.info(
            "CV Accuracy: %.3f ± %.3f  (target ≥ %.2f) %s",
            mean_cv, std_cv, TARGET_ACCURACY,
            "✓" if mean_cv >= TARGET_ACCURACY else "✗ below target",
        )

        # Final fit on full training set
        self.model.fit(X_train, y_encoded)
        self._trained = True

        return {
            "cv_mean_accuracy": round(mean_cv, 4),
            "cv_std": round(std_cv, 4),
            "cv_per_fold": [round(s, 4) for s in cv_scores],
            "meets_target": mean_cv >= TARGET_ACCURACY,
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate the trained classifier on a held-out test set.

        Parameters
        ----------
        X_test : pd.DataFrame
        y_test : pd.Series
            True risk tier labels.

        Returns
        -------
        dict
            Accuracy, F1 scores per tier, confusion matrix, full report.
        """
        self._check_trained()
        y_encoded = self.label_encoder.transform(y_test.astype(str))
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_encoded, y_pred)
        report = classification_report(
            y_encoded, y_pred,
            target_names=self.classes_,
            output_dict=True,
        )
        cm = confusion_matrix(y_encoded, y_pred)

        f1_per_class = {
            cls: round(report[cls]["f1-score"], 4)
            for cls in self.classes_
        }

        high_risk_f1 = f1_per_class.get("High", 0)
        logger.info(
            "Test Accuracy: %.3f | High-Risk F1: %.3f (target ≥ %.2f) %s",
            acc, high_risk_f1, TARGET_F1_HIGH_RISK,
            "✓" if high_risk_f1 >= TARGET_F1_HIGH_RISK else "✗ below target",
        )

        print("\n── Risk Tier Classification Report ──")
        print(classification_report(y_encoded, y_pred, target_names=self.classes_))
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=self.classes_, columns=self.classes_).to_string())

        return {
            "accuracy": round(acc, 4),
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm.tolist(),
            "full_report": report,
            "meets_accuracy_target": acc >= TARGET_ACCURACY,
            "meets_high_risk_f1_target": high_risk_f1 >= TARGET_F1_HIGH_RISK,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict risk tiers for new counties.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with same columns as training data.

        Returns
        -------
        pd.Series
            Predicted risk tier labels ('High', 'Medium', 'Low').
        """
        self._check_trained()
        y_encoded = self.model.predict(X)
        return pd.Series(
            self.label_encoder.inverse_transform(y_encoded),
            index=X.index,
            name="Predicted_Risk_Tier",
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return class probabilities for each county.

        Returns
        -------
        pd.DataFrame
            Columns = class labels, rows = counties, values = probabilities.
        """
        self._check_trained()
        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, index=X.index, columns=self.classes_)

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Return the top N most important features.

        Parameters
        ----------
        top_n : int
            Number of features to return. Default 10.

        Returns
        -------
        pd.Series
            Feature importances sorted descending.
        """
        self._check_trained()
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

        print(f"\n── Top {top_n} Features (Risk Classifier) ──")
        print(importance.head(top_n).round(4).to_string())
        return importance.head(top_n)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the trained model and encoder to disk."""
        self._check_trained()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "classes": self.classes_,
            },
            path,
        )
        logger.info("Risk classifier saved to '%s'", path)

    def load(self, path: str) -> None:
        """Load a previously saved classifier from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        self.classes_ = data["classes"]
        self._trained = True
        logger.info("Risk classifier loaded from '%s'", path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                "Classifier not trained. Call train() or load() first."
            )

    @staticmethod
    def assign_risk_tier(mortality_rate: float) -> str:
        """
        Rule-based risk tier assignment from mortality rate.

        Parameters
        ----------
        mortality_rate : float
            Under-5 mortality rate per 1,000 live births.

        Returns
        -------
        str
            'High', 'Medium', or 'Low'.
        """
        if mortality_rate >= 60:
            return "High"
        elif mortality_rate >= 35:
            return "Medium"
        else:
            return "Low"
