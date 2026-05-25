# Child Mortality Recommendation System — Kenya

> A machine learning system that classifies Kenya's 47 counties by child mortality risk tier and generates ranked, evidence-based public health intervention recommendations tailored to each county's unique profile.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)
- [Models](#models)
- [Results](#results)
- [Visualisations](#visualisations)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Authors](#authors)

---

## Project Overview

Kenya's national under-5 mortality rate of **41 per 1,000 live births (2022)** masks a **3.5× disparity** between best and worst performing counties — ranging from 22.1 in Nairobi to 78.2 in Mandera. With limited health budgets, county health teams need data-driven guidance on which interventions will have the greatest impact in their specific context.

This project develops a **hybrid recommendation system** that:

1. **Classifies** all 47 Kenyan counties into High, Medium, or Low mortality risk tiers
2. **Predicts** county-level under-5 mortality rates from socioeconomic and health indicators
3. **Recommends** the top-3 most effective, feasible, and context-appropriate public health interventions for each county

The system is aligned with Kenya's MOH RMNCH strategy and the **UN Sustainable Development Goal 3.2** (reduce under-5 mortality to ≤25 per 1,000 by 2030).

---

## Business Problem

County Health Management Teams across Kenya face two key challenges:

- **Information overload** — dozens of possible interventions with varying evidence levels
- **Resource constraints** — limited budgets require precise targeting of the highest-impact actions

This system solves both by providing **personalised, ranked, county-specific intervention recommendations** grounded in both peer-county learning (collaborative filtering) and evidence-based intervention scoring (content-based filtering).

**Stakeholders:** MOH Kenya RMNH Division · 47 County Health Management Teams · WHO 
UNICEF

---

## Dataset

Three datasets are used in this project:

| Dataset | Source | Records | Description |
|---|---|---|---|
| `county_mortality_indicators.csv` | KDHS 2022 + DHIS2 | 141 rows | County-level mortality rates, health indicators, WASH, nutrition (2022–2024) |
| `intervention_effectiveness_registry.csv` | WHO/Cochrane + MOH | 20 rows | 20 evidence-based interventions with effectiveness, feasibility, and cost scores |
| `historical_deployment_records.csv` | MOH Kenya / Partners | 119 rows | Past intervention deployments with outcome data per county |

**Data Sources:**
- Kenya DHS 2022: https://dhsprogram.com/data/dataset/Kenya_Standard-DHS_2022.cfm
- MOH DHIS2 (KHIS): https://hiskenya.org
- UNICEF Child Mortality: https://data.unicef.org/topic/child-survival/under-five-mortality/
- WHO Intervention Evidence: https://www.who.int/tools/lives-saved-tool

---

## Project Structure

```
child_mortality_in_kenya/
│
├── data/
│   ├── raw/                              # Original CSV datasets (gitignored)
│   │   ├── county_mortality_indicators.csv
│   │   ├── intervention_effectiveness_registry.csv
│   │   └── historical_deployment_records.csv
│   └── processed/                        # Cleaned & engineered outputs (gitignored)
│       ├── county_mortality_indicators_clean.csv
│       ├── intervention_effectiveness_registry_clean.csv
│       ├── historical_deployment_records_clean.csv
│       ├── county_feature_matrix.csv
│       └── county_recommendations.csv
│
├── src/                                  # Source package
│   ├── __init__.py
│   ├── utils.py                          # Shared utilities, paths, logger
│   ├── mortality_cleaner.py              # Data cleaning and validation
│   ├── feature_engineer.py              # Composite score engineering
│   ├── risk_classifier.py               # Random Forest risk tier classifier
│   ├── recommender.py                   # Hybrid recommendation engine
│   └── evaluator.py                     # Model evaluation and benchmarking
│
├── scripts/                              # Runnable pipeline scripts
│   ├── run_cleaning.py                  # Day 1 AM — clean all datasets
│   ├── run_eda.py                       # Day 1 PM — generate EDA visualisations
│   └── run_model.py                     # Day 2 — train and save all models
│
├── notebooks/                            
│   ├── 01_data_preparation_eda.ipynb    # Data cleaning, feature engineering, EDA
│   ├── 02_modeling.ipynb                # Risk classifier, regressor, recommender
│   └── 03_evaluation_recommendations.ipynb  # Benchmarks, projections, reports
│
├── models/                               # Saved model files (gitignored)
│   ├── risk_classifier.pkl
│   ├── mortality_regressor.pkl
│   └── label_encoders.pkl
│
├── visualizations/                       # Generated charts and figures
├── reports/                              # Final reports and recommendation tables
│
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md                            # This file
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- VS Code 

### 1. Clone the repository

```bash
git clone https://github.com/KODAK-CODER/child_mortality_in_kenya.git
cd child_mortality_in_kenya
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add raw datasets

Place the three CSV files into `data/raw/`:

```
data/raw/county_mortality_indicators.csv
data/raw/intervention_effectiveness_registry.csv
data/raw/historical_deployment_records.csv
data/raw/county_mortality_indicators.csv
```


---

## How to Run

### Jupyter Notebooks

Open VS Code, select your `venv` kernel, and run cells top to bottom:

```
notebooks/01_data_preparation_eda.ipynb     
notebooks/02_modeling.ipynb                 
notebooks/03_evaluation_recommendations.ipynb  
```



---

## Models

### Model 1 — Risk Tier Classifier

| Property | Value |
|---|---|
| Algorithm | Random Forest |
| Parameters | 200 trees, max_depth=8, class_weight=balanced |
| Target variable | Risk_Tier (High / Medium / Low) |
| Validation | 5-fold stratified cross-validation |
| Saved to | `models/risk_classifier.pkl` |

**Risk Tier Thresholds:**

| Tier | Under-5 Mortality Rate | Counties |
|---|---|---|
| 🔴 High Risk | ≥ 60 per 1,000 | Mandera, Wajir, Turkana, Garissa, Marsabit |
| 🟠 Medium Risk | 35–59 per 1,000 | 9 mixed rural/peri-urban counties |
| 🟢 Low Risk | < 35 per 1,000 | Nairobi, Mombasa, Kiambu, Nyeri, Nakuru |

### Model 2 — Mortality Rate Regressor

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting |
| Parameters | 300 estimators, learning_rate=0.05, max_depth=4 |
| Target variable | Under5_Mortality_Rate_per1000 (continuous) |
| Saved to | `models/mortality_regressor.pkl` |

### Model 3 — Hybrid Recommendation Engine

| Step | Method | Weight |
|---|---|---|
| Collaborative Filtering | Cosine similarity on county feature vectors → top-5 peer counties | 40% |
| Content-Based Filtering | Weighted intervention scoring (effectiveness 45%, feasibility 30%, cost-effectiveness 25%) | 60% |
| Rule-Based Layer | ASAL suitability filter · WHO evidence level filter · Budget ceiling | Hard filter |

**Engineered Features (composite scores):**

| Score | Formula |
|---|---|
| Health System Score | Skilled attendance 25% + Immunization 25% + Facility delivery 20% + ANC 20% + Distance 10% |
| Nutrition Risk Score | Stunting 60% + Wasting 40% |
| WASH Score | Clean water 60% + Sanitation 40% |
| Deprivation Index | Poverty 70% + Education deficit 30% |

---

## Results

### Benchmark Targets vs Achieved

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Risk Classifier Accuracy | ≥ 85% | 87.2% | ✅ |
| High-Risk F1 Score | ≥ 0.85 | 0.89 | ✅ |
| Mortality RMSE | ≤ 5.0 | 4.23 | ✅ |
| Mortality R² | ≥ 0.88 | 0.91 | ✅ |
| Recommendation Precision@3 | ≥ 0.70 | 0.74 | ✅ |
| ASAL Mortality Reduction | ≥ 20% | 22.1% | ✅ |
| 2025 National Projection | ≤ 30/1,000 | 28.4 | ✅ |

### Key Findings

- **ASAL counties** average 67.9/1,000 vs 31.8/1,000 for non-ASAL — a **2.1× disparity**
- **Poverty index** (r=+0.92) and **skilled birth attendance** (r=−0.89) are the strongest predictors
- **Oral Rehydration Therapy** and **EPI Immunization Scale-Up** are the most frequently recommended interventions nationally
- All 47 counties are projected to achieve at least **15% mortality reduction** with full intervention implementation by 2025

---

## Visualisations

All charts are saved to `visualizations/` after running the scripts or notebooks:

| File | Description |
|---|---|
| `01_mortality_distribution.png` | Histogram and bar chart of mortality by risk tier |
| `02_mortality_by_region.png` | Regional average mortality with min–max error bars |
| `03_asal_comparison.png` | ASAL vs non-ASAL boxplot and indicator comparison |
| `04_correlation_heatmap.png` | Correlation matrix of socioeconomic indicators |
| `05_mortality_trend.png` | Year-on-year mortality trend (2022–2024) by risk tier |
| `06_intervention_scores.png` | Composite scores for all 20 interventions |
| `07_feature_importance_classifier.png` | Top features for risk tier classifier |
| `08_regressor_evaluation.png` | Predicted vs actual and residual plots |
| `09_feature_importance_regressor.png` | Top features for mortality regressor |
| `10_top_recommended_interventions.png` | Most frequently recommended interventions |
| `11_benchmark_scorecard.png` | Visual pass/fail scorecard for all 7 benchmarks |
| `12_baseline_vs_projected.png` | 2022 baseline vs 2025 projected mortality by county |

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Deployment | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Notebooks | Jupyter |
| Version Control | Git, GitHub |
| IDE | VS Code |

---

---

## Future Work

- **DHIS2 integration** — automated county profile updates via MOH Kenya API
- **FastAPI REST endpoint** — `POST /recommend/{county}` for real-time recommendations
- **Streamlit dashboard** — interactive county-level explorer for health planners
- **Quarterly retraining** — automated pipeline triggered by new DHIS2 data
- **Sub-county granularity** — extend to ward-level targeting where data allows

---

## Author

**Dennis kamuri**  
Client: Ministry of Health Kenya — Reproductive, Maternal & Child Health Division  
Version: 1.0.0 

---

## License

This project is developed for public health purposes under the guidance of MOH Kenya.  
Data sources are publicly available via UNICEF, WHO, World Bank, and KNBS open data portals.

---

*For technical questions, refer to the project notebooks in `notebooks/` or the module docstrings in `src/`.*
