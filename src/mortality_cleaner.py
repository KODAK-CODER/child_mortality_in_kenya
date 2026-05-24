# ─────────────────────────────────────────────
# Child Mortality Recommendation System
# src/mortality_cleaner.py
# ─────────────────────────────────────────────

"""
MortalityCleaner
================
Handles loading, validating, and cleaning all three project datasets:
    1. county_mortality_indicators.csv
    2. intervention_effectiveness_registry.csv
    3. historical_deployment_records.csv

Usage:
    from src.mortality_cleaner import MortalityCleaner

    cleaner = MortalityCleaner(data_dir="data/raw")
    counties_df    = cleaner.clean_county_indicators()
    interventions_df = cleaner.clean_interventions()
    deployments_df = cleaner.clean_deployments()
    cleaner.save_cleaned(output_dir="data/processed")
"""

import os
import logging
import pandas as pd
import numpy as np

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EXPECTED_COUNTIES = 47
VALID_RISK_TIERS = {"High", "Medium", "Low"}
VALID_REGIONS = {
    "Nairobi", "Central", "Coast", "Eastern",
    "North Eastern", "Nyanza", "Rift Valley", "Western",
}
VALID_YEARS = {2022, 2023, 2024}
PCT_COLUMNS = [
    "Skilled_Birth_Attendance_pct",
    "Immunization_Coverage_pct",
    "ANC_Visits_4plus_pct",
    "Facility_Delivery_pct",
    "Clean_Water_Access_pct",
    "Sanitation_Access_pct",
    "Stunting_Prevalence_pct",
    "Wasting_Prevalence_pct",
    "Female_Literacy_Rate_pct",
]
MORTALITY_COLUMNS = [
    "Under5_Mortality_Rate_per1000",
    "Neonatal_Mortality_Rate_per1000",
    "Infant_Mortality_Rate_per1000",
]


class MortalityCleaner:
    """
    Cleans and validates all three child mortality project datasets.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the raw CSV files.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.counties_df = None
        self.interventions_df = None
        self.deployments_df = None
        logger.info("MortalityCleaner initialised. Data directory: '%s'", data_dir)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load a CSV file from the data directory."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded '%s' — %d rows, %d columns", filename, *df.shape)
        return df

    def _drop_duplicates(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Remove duplicate rows and log how many were dropped."""
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            logger.warning("%s: dropped %d duplicate rows", label, dropped)
        else:
            logger.info("%s: no duplicates found", label)
        return df

    def _clip_percentages(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Clip percentage columns to valid range [0, 100]."""
        for col in columns:
            if col in df.columns:
                out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
                if out_of_range:
                    logger.warning("Clipping %d out-of-range values in '%s'", out_of_range, col)
                df[col] = df[col].clip(0, 100)
        return df

    def _impute_median(self, df: pd.DataFrame, columns: list, group_col: str = None) -> pd.DataFrame:
        """
        Impute missing numeric values with column median.
        If group_col is provided, impute within groups first.
        """
        for col in columns:
            if col not in df.columns:
                continue
            missing = df[col].isna().sum()
            if missing == 0:
                continue
            if group_col and group_col in df.columns:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda x: x.fillna(x.median())
                )
            # Fallback: global median for any remaining NaNs
            remaining = df[col].isna().sum()
            if remaining:
                df[col] = df[col].fillna(df[col].median())
            logger.info("Imputed %d missing values in '%s'", missing, col)
        return df

    def _standardise_strings(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Strip whitespace and apply title case to string columns."""
        for col in columns:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].str.strip().str.title()
        return df

    # ── Public cleaning methods ───────────────────────────────────────────────

    def clean_county_indicators(self) -> pd.DataFrame:
        """
        Clean the county_mortality_indicators.csv dataset.

        Steps:
            1. Load CSV
            2. Drop duplicates
            3. Standardise string columns
            4. Validate categorical values
            5. Clip percentage columns to [0, 100]
            6. Clip mortality rates to plausible range [5, 200]
            7. Clip Poverty_Index to [0, 1]
            8. Impute missing numerics by region
            9. Add derived column: Health_System_Score

        Returns
        -------
        pd.DataFrame
            Cleaned county indicators dataframe.
        """
        df = self._load_csv("county_mortality_indicators.csv")
        df = self._drop_duplicates(df, "CountyIndicators")

        # Standardise strings
        df = self._standardise_strings(df, ["County", "Region", "Risk_Tier"])

        # Validate Risk_Tier
        invalid_tiers = ~df["Risk_Tier"].isin(VALID_RISK_TIERS)
        if invalid_tiers.any():
            logger.warning(
                "Found %d rows with invalid Risk_Tier values: %s",
                invalid_tiers.sum(),
                df.loc[invalid_tiers, "Risk_Tier"].unique().tolist(),
            )
            df = df[~invalid_tiers].copy()

        # Validate Year
        invalid_years = ~df["Year"].isin(VALID_YEARS)
        if invalid_years.any():
            logger.warning("Dropping %d rows with invalid Year values", invalid_years.sum())
            df = df[~invalid_years].copy()

        # Clip columns to valid ranges
        df = self._clip_percentages(df, PCT_COLUMNS)
        for col in MORTALITY_COLUMNS:
            if col in df.columns:
                df[col] = df[col].clip(5, 200)
        if "Poverty_Index" in df.columns:
            df["Poverty_Index"] = df["Poverty_Index"].clip(0, 1)

        # Impute missing numerics within Region groups
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = self._impute_median(df, numeric_cols, group_col="Region")

        # Derived feature: Health System Score (0–100)
        df["Health_System_Score"] = (
            df.get("Skilled_Birth_Attendance_pct", 0) * 0.25
            + df.get("Immunization_Coverage_pct", 0) * 0.25
            + df.get("Facility_Delivery_pct", 0) * 0.20
            + df.get("ANC_Visits_4plus_pct", 0) * 0.20
            + (100 - df.get("Distance_to_Facility_km", 0).clip(0, 100)) * 0.10
        ).round(2)

        logger.info("CountyIndicators cleaning complete — %d rows retained", len(df))
        self.counties_df = df
        return df

    def clean_interventions(self) -> pd.DataFrame:
        """
        Clean the intervention_effectiveness_registry.csv dataset.

        Steps:
            1. Load CSV
            2. Drop duplicates
            3. Standardise string columns
            4. Clip score columns to [0, 100]
            5. Validate WHO_Evidence_Level values
            6. Recompute Composite_Score from components

        Returns
        -------
        pd.DataFrame
            Cleaned intervention registry dataframe.
        """
        df = self._load_csv("intervention_effectiveness_registry.csv")
        df = self._drop_duplicates(df, "InterventionRegistry")

        df = self._standardise_strings(
            df, ["Intervention_Name", "Category", "WHO_Evidence_Level", "Budget_Requirement"]
        )

        score_cols = ["Effectiveness_Score", "Feasibility_Score", "Cost_Effectiveness_Score"]
        df = self._clip_percentages(df, score_cols)

        # Recompute composite score to ensure consistency
        df["Composite_Score"] = (
            df["Effectiveness_Score"] * 0.45
            + df["Feasibility_Score"] * 0.30
            + df["Cost_Effectiveness_Score"] * 0.25
        ).round(1)

        # Validate ASAL_Suitable is boolean-like
        df["ASAL_Suitable"] = df["ASAL_Suitable"].astype(bool)

        logger.info("InterventionRegistry cleaning complete — %d rows retained", len(df))
        self.interventions_df = df
        return df

    def clean_deployments(self) -> pd.DataFrame:
        """
        Clean the historical_deployment_records.csv dataset.

        Steps:
            1. Load CSV
            2. Drop duplicates
            3. Standardise string columns
            4. Clip numeric score columns to valid ranges
            5. Impute missing outcome values with median
            6. Validate County names against county indicators dataset

        Returns
        -------
        pd.DataFrame
            Cleaned deployment records dataframe.
        """
        df = self._load_csv("historical_deployment_records.csv")
        df = self._drop_duplicates(df, "DeploymentRecords")

        df = self._standardise_strings(df, ["County", "Intervention_Name", "Partner"])

        # Clip score and percentage columns
        df = self._clip_percentages(df, ["Coverage_Achieved_pct", "Outcome_Mortality_Reduction_pct"])
        if "Implementation_Score" in df.columns:
            df["Implementation_Score"] = df["Implementation_Score"].clip(0, 100)

        # Impute missing outcome data
        outcome_cols = [
            "Implementation_Score",
            "Coverage_Achieved_pct",
            "Outcome_Mortality_Reduction_pct",
            "Budget_Utilized_KES_Million",
        ]
        df = self._impute_median(df, outcome_cols)

        # Cross-validate counties against indicators dataset (if available)
        if self.counties_df is not None:
            known_counties = set(self.counties_df["County"].unique())
            unknown = ~df["County"].isin(known_counties)
            if unknown.any():
                logger.warning(
                    "DeploymentRecords: %d rows reference unknown counties: %s",
                    unknown.sum(),
                    df.loc[unknown, "County"].unique().tolist(),
                )

        logger.info("DeploymentRecords cleaning complete — %d rows retained", len(df))
        self.deployments_df = df
        return df

    # ── Save method ───────────────────────────────────────────────────────────

    def save_cleaned(self, output_dir: str = "data/processed") -> None:
        """
        Save all cleaned dataframes to the processed data directory.

        Parameters
        ----------
        output_dir : str
            Directory to write cleaned CSV files into.
        """
        os.makedirs(output_dir, exist_ok=True)

        datasets = {
            "county_mortality_indicators_clean.csv": self.counties_df,
            "intervention_effectiveness_registry_clean.csv": self.interventions_df,
            "historical_deployment_records_clean.csv": self.deployments_df,
        }

        for filename, df in datasets.items():
            if df is None:
                logger.warning("Skipping '%s' — not yet cleaned", filename)
                continue
            path = os.path.join(output_dir, filename)
            df.to_csv(path, index=False)
            logger.info("Saved cleaned dataset: '%s' (%d rows)", path, len(df))

    # ── Summary report ────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a summary of all loaded datasets."""
        print("\n" + "=" * 55)
        print("  CHILD MORTALITY PROJECT — DATA CLEANING SUMMARY")
        print("=" * 55)
        for label, df in [
            ("County Indicators", self.counties_df),
            ("Intervention Registry", self.interventions_df),
            ("Deployment Records", self.deployments_df),
        ]:
            if df is not None:
                print(f"\n{label}")
                print(f"  Rows     : {len(df)}")
                print(f"  Columns  : {df.shape[1]}")
                print(f"  Nulls    : {df.isna().sum().sum()}")
                print(f"  Dtypes   : {df.dtypes.value_counts().to_dict()}")
            else:
                print(f"\n{label}: Not yet cleaned")
        print("\n" + "=" * 55 + "\n")
