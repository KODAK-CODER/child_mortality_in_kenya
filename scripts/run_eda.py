# ─────────────────────────────────────────────
# scripts/run_eda.py
# Run Exploratory Data Analysis and save plots
# ─────────────────────────────────────────────
"""
Usage:
    python scripts/run_eda.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("visualizations", exist_ok=True)

def load_processed():
    counties = pd.read_csv("data/processed/county_mortality_indicators_clean.csv")
    interventions = pd.read_csv("data/processed/intervention_effectiveness_registry_clean.csv")
    deployments = pd.read_csv("data/processed/historical_deployment_records_clean.csv")
    return counties, interventions, deployments

def plot_mortality_by_region(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    region_avg = df.groupby("Region")["Under5_Mortality_Rate_per1000"].mean().sort_values(ascending=False)
    region_avg.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Average Under-5 Mortality Rate by Region (2022–2024)", fontsize=13)
    ax.set_ylabel("Mortality Rate per 1,000 Live Births")
    ax.set_xlabel("Region")
    plt.tight_layout()
    plt.savefig("visualizations/mortality_by_region.png", dpi=150)
    plt.close()
    print("Saved: visualizations/mortality_by_region.png")

def plot_risk_tier_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    tier_colors = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
    counts = df[df["Year"] == 2022]["Risk_Tier"].value_counts()
    counts.plot(kind="bar", ax=ax, color=[tier_colors.get(t, "grey") for t in counts.index])
    ax.set_title("County Risk Tier Distribution (2022)", fontsize=13)
    ax.set_ylabel("Number of Counties")
    ax.set_xlabel("Risk Tier")
    plt.tight_layout()
    plt.savefig("visualizations/risk_tier_distribution.png", dpi=150)
    plt.close()
    print("Saved: visualizations/risk_tier_distribution.png")

def plot_correlation_heatmap(df):
    numeric_cols = [
        "Under5_Mortality_Rate_per1000", "Poverty_Index",
        "Skilled_Birth_Attendance_pct", "Clean_Water_Access_pct",
        "Immunization_Coverage_pct", "Stunting_Prevalence_pct",
        "Female_Literacy_Rate_pct", "Health_System_Score"
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix — Key Indicators vs Mortality Rate", fontsize=13)
    plt.tight_layout()
    plt.savefig("visualizations/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: visualizations/correlation_heatmap.png")

def plot_asal_vs_nonasal(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    df_2022 = df[df["Year"] == 2022].copy()
    df_2022["ASAL"] = df_2022["ASAL_Flag"].map({1: "ASAL Counties", 0: "Non-ASAL Counties"})
    sns.boxplot(data=df_2022, x="ASAL", y="Under5_Mortality_Rate_per1000",
                palette={"ASAL Counties": "#d62728", "Non-ASAL Counties": "#2ca02c"}, ax=ax)
    ax.set_title("Under-5 Mortality Rate: ASAL vs Non-ASAL Counties (2022)", fontsize=12)
    ax.set_ylabel("Mortality Rate per 1,000 Live Births")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("visualizations/asal_vs_nonasal.png", dpi=150)
    plt.close()
    print("Saved: visualizations/asal_vs_nonasal.png")

def plot_intervention_scores(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values("Composite_Score", ascending=True)
    colors = df_sorted["Category"].astype("category").cat.codes
    bars = ax.barh(df_sorted["Intervention_Name"], df_sorted["Composite_Score"],
                   color=plt.cm.tab10(colors / colors.max()))
    ax.set_title("Intervention Composite Scores (Effectiveness × Feasibility × Cost-Effectiveness)", fontsize=11)
    ax.set_xlabel("Composite Score (0–100)")
    plt.tight_layout()
    plt.savefig("visualizations/intervention_scores.png", dpi=150)
    plt.close()
    print("Saved: visualizations/intervention_scores.png")

def main():
    print("Loading processed datasets...")
    counties, interventions, deployments = load_processed()

    print("Generating EDA visualizations...")
    plot_mortality_by_region(counties)
    plot_risk_tier_distribution(counties)
    plot_correlation_heatmap(counties)
    plot_asal_vs_nonasal(counties)
    plot_intervention_scores(interventions)

    print("\nAll EDA visualizations saved to visualizations/")

if __name__ == "__main__":
    main()
