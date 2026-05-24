# ─────────────────────────────────────────────
# Child Mortality Recommendation System
# src/__init__.py
# ─────────────────────────────────────────────

"""
Child Mortality Recommendation System — Kenya (2022–2025)

Package structure:
    src/
    ├── __init__.py            # This file
    ├── mortality_cleaner.py   # Data cleaning and preprocessing
    ├── feature_engineer.py    # Composite score engineering
    ├── risk_classifier.py     # Risk tier classification model
    ├── recommender.py         # Hybrid recommendation engine
    └── evaluator.py           # Model evaluation and benchmarking
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"
__project__ = "Child Mortality Recommendation System — Kenya"

from src.mortality_cleaner import MortalityCleaner
from src.feature_engineer import FeatureEngineer
from src.risk_classifier import RiskClassifier
from src.recommender import HybridRecommender
from src.evaluator import Evaluator

__all__ = [
    "MortalityCleaner",
    "FeatureEngineer",
    "RiskClassifier",
    "HybridRecommender",
    "Evaluator",
]
