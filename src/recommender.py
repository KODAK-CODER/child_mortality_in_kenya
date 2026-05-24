# ─────────────────────────────────────────────
# Child Mortality Recommendation System
# src/recommender.py
# ─────────────────────────────────────────────

"""
HybridRecommender
=================
Generates ranked, county-specific public health intervention
recommendations using a three-step hybrid approach:

    Step 1 — Collaborative Filtering (CF)
        Cosine similarity on county feature vectors to find
        the top-K peer counties with similar mortality profiles,
        then aggregate their successful deployments.

    Step 2 — Content-Based Filtering (CBF)
        Score each intervention against the county's profile
        using a weighted formula:
            Effectiveness  45%
            Feasibility    30%
            Cost-Efficiency 25%

    Step 3 — Rule-Based Layer
        Apply hard filters:
            • ASAL suitability (ASAL counties only see ASAL-suitable interventions)
            • WHO evidence level (minimum 'Moderate')
            • Budget ceiling
        Then return the top-N ranked recommendations.

Usage:
    from src.recommender import HybridRecommender

    rec = HybridRecommender()
    rec.fit(county_matrix, interventions_df, deployments_df)
    recs = rec.recommend(county_name="Mandera", top_n=3)
    all_recs = rec.recommend_all(top_n=3)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Weighting for content-based scoring
CBF_WEIGHTS = {
    "Effectiveness_Score": 0.45,
    "Feasibility_Score": 0.30,
    "Cost_Effectiveness_Score": 0.25,
}

VALID_EVIDENCE_LEVELS = {"Strong", "Moderate"}


class HybridRecommender:
    """
    Hybrid recommendation engine combining collaborative filtering,
    content-based filtering, and rule-based filtering.

    Parameters
    ----------
    n_peers : int
        Number of peer counties to use in collaborative filtering. Default 5.
    cf_weight : float
        Weight given to CF score in the hybrid score. Default 0.40.
    cbf_weight : float
        Weight given to CBF score in the hybrid score. Default 0.60.
    """

    def __init__(
        self,
        n_peers: int = 5,
        cf_weight: float = 0.40,
        cbf_weight: float = 0.60,
    ):
        if abs(cf_weight + cbf_weight - 1.0) > 1e-6:
            raise ValueError("cf_weight + cbf_weight must equal 1.0")

        self.n_peers = n_peers
        self.cf_weight = cf_weight
        self.cbf_weight = cbf_weight

        self.county_matrix: pd.DataFrame = None
        self.interventions: pd.DataFrame = None
        self.deployments: pd.DataFrame = None
        self.similarity_matrix: pd.DataFrame = None
        self._fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        county_matrix: pd.DataFrame,
        interventions_df: pd.DataFrame,
        deployments_df: pd.DataFrame,
        county_indicators: pd.DataFrame = None,
    ) -> None:
        """
        Fit the recommender on county features, interventions, and
        historical deployments.

        Parameters
        ----------
        county_matrix : pd.DataFrame
            Scaled county feature matrix (index = County name).
            Output of FeatureEngineer.build_county_feature_matrix().
        interventions_df : pd.DataFrame
            Intervention effectiveness registry.
        deployments_df : pd.DataFrame
            Historical deployment records.
        county_indicators : pd.DataFrame, optional
            Full county indicators dataframe (used for ASAL flag lookup).
        """
        self.county_matrix = county_matrix
        self.interventions = interventions_df.copy()
        self.deployments = deployments_df.copy()
        self.county_indicators = county_indicators

        # Compute cosine similarity matrix
        sim_values = cosine_similarity(county_matrix.values)
        self.similarity_matrix = pd.DataFrame(
            sim_values,
            index=county_matrix.index,
            columns=county_matrix.index,
        )
        logger.info(
            "Recommender fitted — %d counties, %d interventions, %d deployment records",
            len(county_matrix), len(interventions_df), len(deployments_df),
        )
        self._fitted = True

    # ── Core recommendation ───────────────────────────────────────────────────

    def recommend(
        self,
        county_name: str,
        top_n: int = 3,
        budget_ceiling: str = None,
        require_asal: bool = None,
    ) -> pd.DataFrame:
        """
        Generate ranked intervention recommendations for one county.

        Parameters
        ----------
        county_name : str
            Name of the target county.
        top_n : int
            Number of recommendations to return. Default 3.
        budget_ceiling : str, optional
            Maximum budget tier ('Low', 'Medium', 'High').
            If None, no budget filter is applied.
        require_asal : bool, optional
            If True, only return ASAL-suitable interventions.
            If None, inferred automatically from the county's ASAL_Flag.

        Returns
        -------
        pd.DataFrame
            Ranked recommendations with scores and justification.
        """
        self._check_fitted()

        if county_name not in self.county_matrix.index:
            available = list(self.county_matrix.index[:10])
            raise ValueError(
                f"County '{county_name}' not found in feature matrix. "
                f"Available (first 10): {available}"
            )

        # Auto-detect ASAL status
        if require_asal is None:
            require_asal = self._is_asal(county_name)

        # Step 1 — Collaborative Filtering
        cf_scores = self._collaborative_filter(county_name)

        # Step 2 — Content-Based Filtering
        cbf_scores = self._content_based_filter()

        # Step 3 — Hybrid merge
        hybrid = self._merge_scores(cf_scores, cbf_scores)

        # Step 4 — Rule-based filters
        hybrid = self._apply_rules(hybrid, require_asal, budget_ceiling)

        # Return top N
        result = hybrid.head(top_n).copy()
        result.insert(0, "County", county_name)
        result.insert(1, "Rank", range(1, len(result) + 1))
        result["ASAL_County"] = require_asal

        logger.info(
            "Recommendations generated for '%s' — top %d returned",
            county_name, len(result),
        )
        return result

    def recommend_all(
        self,
        top_n: int = 3,
        budget_ceiling: str = None,
    ) -> pd.DataFrame:
        """
        Generate recommendations for all counties in the feature matrix.

        Parameters
        ----------
        top_n : int
            Number of recommendations per county. Default 3.
        budget_ceiling : str, optional
            Budget ceiling filter.

        Returns
        -------
        pd.DataFrame
            Stacked recommendations for all counties.
        """
        self._check_fitted()
        all_recs = []

        for county in self.county_matrix.index:
            try:
                recs = self.recommend(county, top_n=top_n, budget_ceiling=budget_ceiling)
                all_recs.append(recs)
            except Exception as e:
                logger.warning("Could not generate recs for '%s': %s", county, e)

        result = pd.concat(all_recs, ignore_index=True)
        logger.info(
            "All-county recommendations complete — %d rows for %d counties",
            len(result), self.county_matrix.shape[0],
        )
        return result

    # ── Step 1: Collaborative Filtering ──────────────────────────────────────

    def _collaborative_filter(self, county_name: str) -> pd.Series:
        """
        Find peer counties and aggregate their successful deployments
        into an intervention CF score.

        Returns
        -------
        pd.Series
            CF score per Intervention_ID.
        """
        # Get top-K similar counties (exclude self)
        sim_row = self.similarity_matrix[county_name].drop(county_name)
        peers = sim_row.nlargest(self.n_peers).index.tolist()

        peer_deployments = self.deployments[
            self.deployments["County"].isin(peers)
        ].copy()

        if peer_deployments.empty:
            logger.warning("No deployment records found for peers of '%s'", county_name)
            return pd.Series(dtype=float)

        # Weight deployment outcomes by similarity to the target county
        peer_deployments = peer_deployments.merge(
            sim_row.rename("Similarity").reset_index().rename(columns={"index": "County"}),
            on="County",
            how="left",
        )
        peer_deployments["Weighted_Outcome"] = (
            peer_deployments["Outcome_Mortality_Reduction_pct"]
            * peer_deployments["Similarity"]
            * peer_deployments.get("Implementation_Score", pd.Series(70, index=peer_deployments.index)) / 100
        )

        cf_scores = (
            peer_deployments.groupby("Intervention_ID")["Weighted_Outcome"]
            .mean()
            .rename("CF_Score")
        )

        # Normalise to [0, 100]
        if cf_scores.max() > cf_scores.min():
            cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min()) * 100

        return cf_scores

    # ── Step 2: Content-Based Filtering ──────────────────────────────────────

    def _content_based_filter(self) -> pd.Series:
        """
        Score every intervention using the weighted CBF formula.

        Returns
        -------
        pd.Series
            CBF score per Intervention_ID.
        """
        df = self.interventions.copy()
        cbf = sum(
            df[col] * weight
            for col, weight in CBF_WEIGHTS.items()
            if col in df.columns
        )
        cbf.index = df["Intervention_ID"]
        cbf.name = "CBF_Score"
        return cbf

    # ── Step 3: Hybrid merge ──────────────────────────────────────────────────

    def _merge_scores(
        self,
        cf_scores: pd.Series,
        cbf_scores: pd.Series,
    ) -> pd.DataFrame:
        """
        Merge CF and CBF scores into a single hybrid score.

        Counties with no CF data fall back to CBF-only scoring.
        """
        merged = cbf_scores.to_frame().join(cf_scores, how="left")
        merged["CF_Score"] = merged["CF_Score"].fillna(0)

        has_cf = merged["CF_Score"] > 0
        merged["Hybrid_Score"] = np.where(
            has_cf,
            merged["CF_Score"] * self.cf_weight + merged["CBF_Score"] * self.cbf_weight,
            merged["CBF_Score"],
        )

        merged = merged.reset_index().rename(columns={"index": "Intervention_ID"})
        merged = merged.merge(
            self.interventions[[
                "Intervention_ID", "Intervention_Name", "Category",
                "ASAL_Suitable", "WHO_Evidence_Level", "Budget_Requirement",
                "Effectiveness_Score", "Feasibility_Score",
                "Cost_Effectiveness_Score", "Composite_Score",
            ]],
            on="Intervention_ID",
            how="left",
        )

        return merged.sort_values("Hybrid_Score", ascending=False)

    # ── Step 4: Rule-Based Filters ────────────────────────────────────────────

    def _apply_rules(
        self,
        df: pd.DataFrame,
        require_asal: bool,
        budget_ceiling: str,
    ) -> pd.DataFrame:
        """
        Apply hard rule-based filters to the ranked intervention list.
        """
        # ASAL filter
        if require_asal and "ASAL_Suitable" in df.columns:
            df = df[df["ASAL_Suitable"] == True].copy()

        # WHO evidence level filter
        if "WHO_Evidence_Level" in df.columns:
            df = df[df["WHO_Evidence_Level"].isin(VALID_EVIDENCE_LEVELS)].copy()

        # Budget ceiling filter
        budget_order = {"Low": 1, "Medium": 2, "High": 3}
        if budget_ceiling and "Budget_Requirement" in df.columns:
            ceiling_val = budget_order.get(budget_ceiling, 3)
            df = df[
                df["Budget_Requirement"].map(budget_order).fillna(3) <= ceiling_val
            ].copy()

        return df.reset_index(drop=True)

    # ── Peer county lookup ────────────────────────────────────────────────────

    def get_peer_counties(self, county_name: str) -> pd.DataFrame:
        """
        Return the top peer counties most similar to the given county.

        Parameters
        ----------
        county_name : str

        Returns
        -------
        pd.DataFrame
            Peer counties with similarity scores.
        """
        self._check_fitted()
        sim_row = self.similarity_matrix[county_name].drop(county_name)
        peers = sim_row.nlargest(self.n_peers).reset_index()
        peers.columns = ["Peer_County", "Cosine_Similarity"]
        peers["Cosine_Similarity"] = peers["Cosine_Similarity"].round(4)
        return peers

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_asal(self, county_name: str) -> bool:
        """Look up whether a county is ASAL from the indicators dataframe."""
        if self.county_indicators is not None and "ASAL_Flag" in self.county_indicators.columns:
            row = self.county_indicators[
                self.county_indicators["County"] == county_name
            ]
            if not row.empty:
                return bool(row["ASAL_Flag"].iloc[0])
        # ASAL counties known from the proposal
        asal_counties = {
            "Mandera", "Wajir", "Turkana", "Garissa", "Marsabit",
            "Samburu", "Tana River", "Isiolo", "West Pokot",
            "Kajiado", "Narok", "Lamu",
        }
        return county_name.strip().title() in asal_counties

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Recommender not fitted. Call fit() first."
            )

    def similarity_report(self, county_name: str) -> None:
        """Print a formatted peer similarity report for one county."""
        peers = self.get_peer_counties(county_name)
        print(f"\n── Peer Counties for '{county_name}' ──")
        print(peers.to_string(index=False))
