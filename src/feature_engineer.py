# ─────────────────────────────────────────────
# Child Mortality Recommendation System
# src/feature_engineer.py
# ─────────────────────────────────────────────

"""
FeatureEngineer
===============
Computes the four composite scores used in risk classification
and the recommendation engine:

    1. Health System Score   (0–100)
    2. Nutrition Risk Score  (0–100)
    3. WASH Score            (0–100)
    4. Deprivation Index     (0–100)

Also handles label encoding for categorical columns
and builds the county feature matrix for collaborative filtering.

Usage:
    from src.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()
    df = fe.build_composite_scores(df)
    df = fe.encode_categoricals(df)
    matrix = fe.build_county_feature_matrix(df)
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds composite scores and feature matrices for the
    child mortality recommendation system.

    Parameters
    ----------
    scale_features : bool
        If True, MinMaxScaler is applied to the feature matrix
        before returning. Default True.
    """

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.label_encoders: dict = {}
        self.scaler = MinMaxScaler()
        self._fitted = False

    # ── Composite score builders ──────────────────────────────────────────────

    def _health_system_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Health System Score (0–100)
        Weights:
            Skilled birth attendance  25 %
            Immunization coverage     25 %
            Facility delivery         20 %
            ANC 4+ visits             20 %
            Distance penalty          10 %  (100 - distance_km clipped to 100)
        """
        distance_penalty = (100 - df.get("Distance_to_Facility_km", pd.Series(0, index=df.index)).clip(0, 100))
        score = (
            df.get("Skilled_Birth_Attendance_pct",   pd.Series(0, index=df.index)) * 0.25
            + df.get("Immunization_Coverage_pct",    pd.Series(0, index=df.index)) * 0.25
            + df.get("Facility_Delivery_pct",        pd.Series(0, index=df.index)) * 0.20
            + df.get("ANC_Visits_4plus_pct",         pd.Series(0, index=df.index)) * 0.20
            + distance_penalty                                                       * 0.10
        )
        return score.clip(0, 100).round(2)

    def _nutrition_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Nutrition Risk Score (0–100)
        Higher = worse nutritional status.
        Weights:
            Stunting prevalence  60 %
            Wasting prevalence   40 %
        """
        score = (
            df.get("Stunting_Prevalence_pct", pd.Series(0, index=df.index)) * 0.60
            + df.get("Wasting_Prevalence_pct",  pd.Series(0, index=df.index)) * 0.40
        )
        return score.clip(0, 100).round(2)

    def _wash_score(self, df: pd.DataFrame) -> pd.Series:
        """
        WASH Score (0–100)
        Higher = better water and sanitation access.
        Weights:
            Clean water access   60 %
            Sanitation access    40 %
        """
        score = (
            df.get("Clean_Water_Access_pct",  pd.Series(0, index=df.index)) * 0.60
            + df.get("Sanitation_Access_pct", pd.Series(0, index=df.index)) * 0.40
        )
        return score.clip(0, 100).round(2)

    def _deprivation_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Deprivation Index (0–1)
        Higher = more deprived.
        Weights:
            Poverty index          70 %
            Education deficit      30 %  (100 - female_literacy) / 100
        """
        education_deficit = (
            (100 - df.get("Female_Literacy_Rate_pct", pd.Series(50, index=df.index)).clip(0, 100)) / 100
        )
        score = (
            df.get("Poverty_Index", pd.Series(0, index=df.index)).clip(0, 1) * 0.70
            + education_deficit * 0.30
        )
        return score.clip(0, 1).round(4)

    # ── Public methods ────────────────────────────────────────────────────────

    def build_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all four composite scores as new columns to the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned county indicators dataframe.

        Returns
        -------
        pd.DataFrame
            Input dataframe with four new composite columns appended.
        """
        df = df.copy()
        df["Health_System_Score"] = self._health_system_score(df)
        df["Nutrition_Risk_Score"] = self._nutrition_risk_score(df)
        df["WASH_Score"] = self._wash_score(df)
        df["Deprivation_Index"] = self._deprivation_index(df)

        logger.info(
            "Composite scores built — Health: %.1f–%.1f | Nutrition: %.1f–%.1f | "
            "WASH: %.1f–%.1f | Deprivation: %.3f–%.3f",
            df["Health_System_Score"].min(),   df["Health_System_Score"].max(),
            df["Nutrition_Risk_Score"].min(),  df["Nutrition_Risk_Score"].max(),
            df["WASH_Score"].min(),            df["WASH_Score"].max(),
            df["Deprivation_Index"].min(),     df["Deprivation_Index"].max(),
        )
        return df

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        columns: list = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Label-encode categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list, optional
            Columns to encode. Defaults to ["Risk_Tier", "Region"].
        fit : bool
            If True, fit new encoders. If False, use already-fitted encoders
            (for transforming test data).

        Returns
        -------
        pd.DataFrame
            Dataframe with encoded columns appended as <col>_Encoded.
        """
        if columns is None:
            columns = ["Risk_Tier", "Region"]

        df = df.copy()
        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found — skipping encoding", col)
                continue
            if fit:
                le = LabelEncoder()
                df[f"{col}_Encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(
                    "Encoded '%s' → classes: %s", col, list(le.classes_)
                )
            else:
                if col not in self.label_encoders:
                    raise ValueError(
                        f"Encoder for '{col}' not fitted. Call encode_categoricals(fit=True) first."
                    )
                le = self.label_encoders[col]
                df[f"{col}_Encoded"] = le.transform(df[col].astype(str))
        return df

    def build_county_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: list = None,
        year: int = 2022,
    ) -> pd.DataFrame:
        """
        Build the county feature matrix used for cosine similarity
        in the collaborative filtering step.

        One row per county (filtered to a single year).
        All numeric features are scaled to [0, 1] if scale_features=True.

        Parameters
        ----------
        df : pd.DataFrame
            County indicators dataframe with composite scores already built.
        feature_cols : list, optional
            Features to include. Defaults to the 11 modelling features.
        year : int
            Year to filter to. Default 2022.

        Returns
        -------
        pd.DataFrame
            Index = County name. Columns = scaled feature values.
        """
        if feature_cols is None:
            feature_cols = [
                "Poverty_Index",
                "Skilled_Birth_Attendance_pct",
                "Clean_Water_Access_pct",
                "Immunization_Coverage_pct",
                "Stunting_Prevalence_pct",
                "Wasting_Prevalence_pct",
                "ANC_Visits_4plus_pct",
                "Facility_Delivery_pct",
                "Female_Literacy_Rate_pct",
                "ASAL_Flag",
                "Health_System_Score",
                "Nutrition_Risk_Score",
                "WASH_Score",
                "Deprivation_Index",
            ]

        df_year = df[df["Year"] == year].copy()
        available = [c for c in feature_cols if c in df_year.columns]
        missing = set(feature_cols) - set(available)
        if missing:
            logger.warning("Feature matrix: missing columns skipped — %s", missing)

        matrix = df_year.set_index("County")[available].fillna(0)

        if self.scale_features:
            if not self._fitted:
                scaled = self.scaler.fit_transform(matrix)
                self._fitted = True
            else:
                scaled = self.scaler.transform(matrix)
            matrix = pd.DataFrame(scaled, index=matrix.index, columns=available)

        logger.info(
            "County feature matrix built — %d counties × %d features (year=%d)",
            *matrix.shape, year,
        )
        return matrix

    def score_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary table of composite scores grouped by Risk_Tier.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with composite scores already computed.

        Returns
        -------
        pd.DataFrame
            Mean composite scores per risk tier.
        """
        score_cols = [
            "Health_System_Score",
            "Nutrition_Risk_Score",
            "WASH_Score",
            "Deprivation_Index",
            "Under5_Mortality_Rate_per1000",
        ]
        available = [c for c in score_cols if c in df.columns]
        summary = df.groupby("Risk_Tier")[available].mean().round(2)
        print("\n── Composite Score Summary by Risk Tier ──")
        print(summary.to_string())
        return summary
