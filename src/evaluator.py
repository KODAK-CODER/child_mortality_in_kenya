# ─────────────────────────────────────────────
# Child Mortality Recommendation System
# src/evaluator.py
# ─────────────────────────────────────────────

"""
Evaluator
=========
Evaluates all three models against the project's benchmark targets
and generates a consolidated results report.

Benchmark Targets (from the project proposal):
    Risk Classifier Accuracy  ≥ 85 %
    High-Risk F1 Score        ≥ 0.85
    Mortality RMSE            ≤ 5.0 per 1,000 live births
    Mortality R²              ≥ 0.88
    Recommendation Precision@3 ≥ 0.70
    ASAL Mortality Reduction   ≥ 20 %
    2025 National Projection   ≤ 30 per 1,000

Usage:
    from src.evaluator import Evaluator

    ev = Evaluator()
    ev.evaluate_classifier(clf, X_test, y_test)
    ev.evaluate_regressor(reg, X_test, y_test)
    ev.evaluate_recommender(recommender, ground_truth_df)
    ev.asal_reduction_summary(projections_df)
    ev.print_benchmark_report()
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Benchmark targets ──────────────────────────────────────────────────────────
BENCHMARKS = {
    "classifier_accuracy":      {"target": 0.85,  "label": "Classifier accuracy",        "gte": True},
    "high_risk_f1":             {"target": 0.85,  "label": "High-risk F1 score",          "gte": True},
    "mortality_rmse":           {"target": 5.0,   "label": "Mortality RMSE (per 1,000)",  "gte": False},
    "mortality_r2":             {"target": 0.88,  "label": "Mortality R²",                "gte": True},
    "precision_at_3":           {"target": 0.70,  "label": "Recommendation Precision@3",  "gte": True},
    "asal_reduction_pct":       {"target": 20.0,  "label": "ASAL mortality reduction %",  "gte": True},
    "national_projection_2025": {"target": 30.0,  "label": "2025 national projection",    "gte": False},
}


class Evaluator:
    """
    Consolidates evaluation results across all three models and
    checks them against project benchmark targets.
    """

    def __init__(self):
        self.results: dict = {}

    # ── Classifier evaluation ─────────────────────────────────────────────────

    def evaluate_classifier(
        self,
        clf,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate the risk tier classifier.

        Parameters
        ----------
        clf : RiskClassifier
            Trained RiskClassifier instance.
        X_test : pd.DataFrame
        y_test : pd.Series
            True risk tier labels.

        Returns
        -------
        dict
            Accuracy, F1 per class, confusion matrix.
        """
        y_pred_labels = clf.predict(X_test)
        y_true_labels = y_test.reset_index(drop=True)

        acc = accuracy_score(y_true_labels, y_pred_labels)
        f1_macro = f1_score(y_true_labels, y_pred_labels, average="macro")
        f1_per_class = f1_score(
            y_true_labels, y_pred_labels,
            average=None,
            labels=clf.classes_,
        )
        f1_dict = dict(zip(clf.classes_, f1_per_class.round(4)))
        high_risk_f1 = f1_dict.get("High", 0.0)

        self.results["classifier_accuracy"] = acc
        self.results["high_risk_f1"] = high_risk_f1
        self.results["f1_per_class"] = f1_dict

        logger.info(
            "Classifier — Accuracy: %.3f | High-Risk F1: %.3f | Macro F1: %.3f",
            acc, high_risk_f1, f1_macro,
        )
        print("\n── Classifier Evaluation ──")
        print(classification_report(y_true_labels, y_pred_labels))

        return {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_per_class": f1_dict,
            "high_risk_f1": round(high_risk_f1, 4),
        }

    # ── Regressor evaluation ──────────────────────────────────────────────────

    def evaluate_regressor(
        self,
        reg,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate the mortality rate regressor.

        Parameters
        ----------
        reg
            Trained sklearn-compatible regressor.
        X_test : pd.DataFrame
        y_test : pd.Series
            True mortality rates.

        Returns
        -------
        dict
            RMSE, MAE, R².
        """
        y_pred = reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        self.results["mortality_rmse"] = rmse
        self.results["mortality_r2"]   = r2

        logger.info(
            "Regressor — RMSE: %.2f | MAE: %.2f | R²: %.3f",
            rmse, mae, r2,
        )
        print("\n── Regressor Evaluation ──")
        print(f"  RMSE : {rmse:.3f} per 1,000 live births")
        print(f"  MAE  : {mae:.3f} per 1,000 live births")
        print(f"  R²   : {r2:.4f}")

        return {
            "rmse": round(rmse, 4),
            "mae":  round(mae, 4),
            "r2":   round(r2, 4),
        }

    # ── Recommender evaluation ────────────────────────────────────────────────

    def evaluate_recommender(
        self,
        recommendations: pd.DataFrame,
        ground_truth: pd.DataFrame,
        k: int = 3,
    ) -> dict:
        """
        Evaluate recommendation quality using Precision@K.

        Precision@K = (# relevant recommendations in top K) / K

        A recommendation is considered relevant if the intervention
        was historically deployed in a peer county with outcome ≥ 10%
        mortality reduction.

        Parameters
        ----------
        recommendations : pd.DataFrame
            Output of HybridRecommender.recommend_all().
            Must contain columns: County, Intervention_ID.
        ground_truth : pd.DataFrame
            Deployment records with columns: County, Intervention_ID,
            Outcome_Mortality_Reduction_pct.
        k : int
            Number of recommendations per county to evaluate. Default 3.

        Returns
        -------
        dict
            Mean Precision@K across all evaluated counties.
        """
        relevant_threshold = 10.0
        relevant = ground_truth[
            ground_truth["Outcome_Mortality_Reduction_pct"] >= relevant_threshold
        ].groupby("County")["Intervention_ID"].apply(set).to_dict()

        precision_scores = []
        counties_evaluated = 0

        for county, group in recommendations.groupby("County"):
            if county not in relevant:
                continue
            top_k = group.head(k)["Intervention_ID"].tolist()
            relevant_set = relevant[county]
            hits = sum(1 for i in top_k if i in relevant_set)
            precision_scores.append(hits / k)
            counties_evaluated += 1

        precision_at_k = np.mean(precision_scores) if precision_scores else 0.0
        self.results["precision_at_3"] = precision_at_k

        logger.info(
            "Recommender — Precision@%d: %.3f (counties evaluated: %d)",
            k, precision_at_k, counties_evaluated,
        )
        print(f"\n── Recommender Evaluation (Precision@{k}) ──")
        print(f"  Counties evaluated : {counties_evaluated}")
        print(f"  Precision@{k}        : {precision_at_k:.4f}")

        return {
            f"precision_at_{k}": round(precision_at_k, 4),
            "counties_evaluated": counties_evaluated,
        }

    # ── ASAL reduction analysis ───────────────────────────────────────────────

    def asal_reduction_summary(
        self,
        current_df: pd.DataFrame,
        projection_df: pd.DataFrame,
    ) -> dict:
        """
        Compute the projected mortality reduction for ASAL counties.

        Parameters
        ----------
        current_df : pd.DataFrame
            County indicators for baseline year (2022).
            Must contain: County, ASAL_Flag, Under5_Mortality_Rate_per1000.
        projection_df : pd.DataFrame
            Projected mortality rates for 2025.
            Must contain: County, Projected_U5_Mortality_2025.

        Returns
        -------
        dict
            Mean ASAL reduction, national projection.
        """
        asal = current_df[current_df["ASAL_Flag"] == 1][
            ["County", "Under5_Mortality_Rate_per1000"]
        ].copy()

        asal = asal.merge(projection_df[["County", "Projected_U5_Mortality_2025"]], on="County", how="inner")
        asal["Reduction_pct"] = (
            (asal["Under5_Mortality_Rate_per1000"] - asal["Projected_U5_Mortality_2025"])
            / asal["Under5_Mortality_Rate_per1000"] * 100
        ).round(2)

        mean_reduction = asal["Reduction_pct"].mean()
        national_avg = projection_df["Projected_U5_Mortality_2025"].mean()

        self.results["asal_reduction_pct"]       = mean_reduction
        self.results["national_projection_2025"]  = national_avg

        print("\n── ASAL Mortality Reduction Summary ──")
        print(asal[["County", "Under5_Mortality_Rate_per1000",
                     "Projected_U5_Mortality_2025", "Reduction_pct"]].to_string(index=False))
        print(f"\n  Mean ASAL reduction   : {mean_reduction:.1f}%")
        print(f"  National avg (2025)   : {national_avg:.1f} per 1,000")

        return {
            "mean_asal_reduction_pct": round(mean_reduction, 2),
            "national_projection_2025": round(national_avg, 2),
        }

    # ── Benchmark report ──────────────────────────────────────────────────────

    def print_benchmark_report(self) -> pd.DataFrame:
        """
        Print a formatted benchmark report comparing all results
        against project targets.

        Returns
        -------
        pd.DataFrame
            Benchmark table with pass/fail status.
        """
        rows = []
        for key, spec in BENCHMARKS.items():
            achieved = self.results.get(key, None)
            if achieved is None:
                status = "—"
                passes = None
            else:
                if spec["gte"]:
                    passes = achieved >= spec["target"]
                else:
                    passes = achieved <= spec["target"]
                status = "✓ PASS" if passes else "✗ FAIL"

            rows.append({
                "Metric":   spec["label"],
                "Target":   spec["target"],
                "Achieved": round(achieved, 3) if achieved is not None else "—",
                "Status":   status,
            })

        report = pd.DataFrame(rows)

        print("\n" + "=" * 62)
        print("  CHILD MORTALITY PROJECT — BENCHMARK REPORT")
        print("=" * 62)
        print(report.to_string(index=False))
        print("=" * 62 + "\n")

        return report

    def save_report(
        self,
        output_path: str = "reports/benchmark_report.csv",
    ) -> None:
        """Save the benchmark report as a CSV file."""
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        report = self.print_benchmark_report()
        report.to_csv(output_path, index=False)
        logger.info("Benchmark report saved to '%s'", output_path)
