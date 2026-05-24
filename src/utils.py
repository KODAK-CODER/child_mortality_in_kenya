# ─────────────────────────────────────────────
# Child Mortality Recommendation System
# src/utils.py
# ─────────────────────────────────────────────

"""
utils.py
========
Shared utilities used across all src modules, notebooks, and scripts.

Provides:
    - Path constants        : DATA_RAW, DATA_PROCESSED, MODELS_DIR, VIZ_DIR, REPORTS_DIR
    - get_logger()          : Consistent logging setup for any module
    - load_datasets()       : Load all three cleaned CSVs in one call
    - save_figure()         : Save matplotlib figures with consistent settings
    - check_benchmark()     : Compare a metric against a target and log pass/fail
    - timer()               : Decorator to time any function
    - ensure_dirs()         : Create all project directories if missing
    - flatten_columns()     : Flatten multi-index DataFrame columns after groupby
    - safe_divide()         : Division that returns 0 instead of ZeroDivisionError
    - summarise_dataframe() : Quick shape/null/dtype summary for any DataFrame

Usage:
    from src.utils import get_logger, load_datasets, save_figure, check_benchmark

    logger = get_logger(__name__)
    counties, interventions, deployments = load_datasets()
"""

import os
import time
import logging
import functools
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

# ── Project root (one level above src/) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Directory constants ───────────────────────────────────────────────────────
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
VIZ_DIR        = PROJECT_ROOT / "visualizations"
REPORTS_DIR    = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR  = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR    = PROJECT_ROOT / "scripts"

# ── Raw dataset filenames ─────────────────────────────────────────────────────
RAW_FILES = {
    "counties":      "county_mortality_indicators.csv",
    "interventions": "intervention_effectiveness_registry.csv",
    "deployments":   "historical_deployment_records.csv",
}

# ── Processed dataset filenames ───────────────────────────────────────────────
PROCESSED_FILES = {
    "counties":      "county_mortality_indicators_clean.csv",
    "interventions": "intervention_effectiveness_registry_clean.csv",
    "deployments":   "historical_deployment_records_clean.csv",
    "matrix":        "county_feature_matrix.csv",
    "recommendations": "county_recommendations.csv",
}

# ── Benchmark targets (single source of truth) ────────────────────────────────
BENCHMARKS = {
    "classifier_accuracy":       {"target": 0.85,  "gte": True,  "label": "Classifier accuracy"},
    "high_risk_f1":              {"target": 0.85,  "gte": True,  "label": "High-risk F1 score"},
    "mortality_rmse":            {"target": 5.0,   "gte": False, "label": "Mortality RMSE"},
    "mortality_r2":              {"target": 0.88,  "gte": True,  "label": "Mortality R²"},
    "precision_at_3":            {"target": 0.70,  "gte": True,  "label": "Recommendation Precision@3"},
    "asal_reduction_pct":        {"target": 20.0,  "gte": True,  "label": "ASAL mortality reduction %"},
    "national_projection_2025":  {"target": 30.0,  "gte": False, "label": "2025 national projection"},
}

# ── Log format ────────────────────────────────────────────────────────────────
_LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently configured logger for any module.

    Parameters
    ----------
    name : str
        Logger name — use __name__ in each module.
    level : int
        Logging level. Default logging.INFO.

    Returns
    -------
    logging.Logger

    Example
    -------
    >>> from src.utils import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Dataset loaded successfully")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


_util_logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DIRECTORY MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def ensure_dirs() -> None:
    """
    Create all required project directories if they do not already exist.
    Safe to call multiple times — will not overwrite existing content.

    Example
    -------
    >>> from src.utils import ensure_dirs
    >>> ensure_dirs()
    """
    dirs = [
        DATA_RAW, DATA_PROCESSED, MODELS_DIR,
        VIZ_DIR, REPORTS_DIR, NOTEBOOKS_DIR, SCRIPTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    _util_logger.info("Project directories verified/created")


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_datasets(
    source: str = "processed",
    raw_dir: str | Path = None,
    processed_dir: str | Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three project datasets in a single call.

    Parameters
    ----------
    source : str
        'processed' (default) loads cleaned CSVs from data/processed/.
        'raw' loads original CSVs from data/raw/.
    raw_dir : str | Path, optional
        Override the raw data directory path.
    processed_dir : str | Path, optional
        Override the processed data directory path.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (counties_df, interventions_df, deployments_df)

    Raises
    ------
    FileNotFoundError
        If any required CSV file is missing.

    Example
    -------
    >>> from src.utils import load_datasets
    >>> counties, interventions, deployments = load_datasets()
    >>> counties.shape
    (141, 21)
    """
    if source == "processed":
        data_dir = Path(processed_dir) if processed_dir else DATA_PROCESSED
        filemap  = PROCESSED_FILES
    else:
        data_dir = Path(raw_dir) if raw_dir else DATA_RAW
        filemap  = RAW_FILES

    results = []
    for key in ("counties", "interventions", "deployments"):
        path = data_dir / filemap[key]
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Run scripts/run_cleaning.py first to generate processed files."
            )
        df = pd.read_csv(path)
        _util_logger.info("Loaded %-45s — %d rows, %d cols", filemap[key], *df.shape)
        results.append(df)

    return tuple(results)


def load_county_matrix(processed_dir: str | Path = None) -> pd.DataFrame:
    """
    Load the county feature matrix used for collaborative filtering.

    Returns
    -------
    pd.DataFrame
        Index = County name, columns = scaled feature values.
    """
    data_dir = Path(processed_dir) if processed_dir else DATA_PROCESSED
    path = data_dir / PROCESSED_FILES["matrix"]
    if not path.exists():
        raise FileNotFoundError(
            f"County feature matrix not found: {path}\n"
            "Run notebooks/02_modeling.ipynb to generate it."
        )
    df = pd.read_csv(path, index_col=0)
    _util_logger.info("County feature matrix loaded — shape: %s", df.shape)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE SAVING
# ═════════════════════════════════════════════════════════════════════════════

def save_figure(
    fig: matplotlib.figure.Figure,
    filename: str,
    viz_dir: str | Path = None,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """
    Save a matplotlib figure to the visualizations directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Output filename. Extension determines format (.png, .pdf, .svg).
        If no extension provided, .png is used.
    viz_dir : str | Path, optional
        Override the output directory.
    dpi : int
        Resolution for raster formats. Default 150.
    close : bool
        Close the figure after saving. Default True.

    Returns
    -------
    Path
        Path to the saved file.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_figure(fig, '01_mortality_trend.png')
    """
    out_dir = Path(viz_dir) if viz_dir else VIZ_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if "." not in filename:
        filename += ".png"

    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _util_logger.info("Figure saved: %s", path)

    if close:
        plt.close(fig)

    return path


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK CHECKING
# ═════════════════════════════════════════════════════════════════════════════

def check_benchmark(
    metric_key: str,
    achieved: float,
    custom_target: float = None,
    custom_gte: bool = None,
) -> bool:
    """
    Check a metric against its project benchmark target and log the result.

    Parameters
    ----------
    metric_key : str
        Key from BENCHMARKS dict (e.g. 'classifier_accuracy').
    achieved : float
        The achieved metric value.
    custom_target : float, optional
        Override the benchmark target value.
    custom_gte : bool, optional
        Override the direction (True = higher is better, False = lower is better).

    Returns
    -------
    bool
        True if benchmark is met, False otherwise.

    Example
    -------
    >>> check_benchmark('mortality_rmse', 4.23)
    True
    >>> check_benchmark('precision_at_3', 0.65)
    False
    """
    if metric_key not in BENCHMARKS and custom_target is None:
        raise ValueError(
            f"Unknown benchmark key '{metric_key}'. "
            f"Valid keys: {list(BENCHMARKS.keys())}"
        )

    if metric_key in BENCHMARKS:
        spec = BENCHMARKS[metric_key]
        target = custom_target if custom_target is not None else spec["target"]
        gte    = custom_gte    if custom_gte    is not None else spec["gte"]
        label  = spec["label"]
    else:
        target = custom_target
        gte    = custom_gte if custom_gte is not None else True
        label  = metric_key

    passes = (achieved >= target) if gte else (achieved <= target)
    symbol = "✓ PASS" if passes else "✗ FAIL"
    direction = "≥" if gte else "≤"

    _util_logger.info(
        "%-35s achieved=%.4f  target%s%.4f  %s",
        label, achieved, direction, target, symbol,
    )
    return passes


# ═════════════════════════════════════════════════════════════════════════════
# TIMER DECORATOR
# ═════════════════════════════════════════════════════════════════════════════

def timer(func):
    """
    Decorator that logs the execution time of any function.

    Example
    -------
    >>> from src.utils import timer
    >>>
    >>> @timer
    ... def train_model():
    ...     ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _util_logger.info(
            "%-30s completed in %.2fs", func.__qualname__, elapsed
        )
        return result
    return wrapper


# ═════════════════════════════════════════════════════════════════════════════
# DATAFRAME HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def summarise_dataframe(df: pd.DataFrame, label: str = "DataFrame") -> dict:
    """
    Print and return a quick summary of any DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    label : str
        Name shown in the summary header.

    Returns
    -------
    dict
        Summary stats: rows, cols, nulls, null_pct, dtypes.

    Example
    -------
    >>> from src.utils import summarise_dataframe
    >>> summarise_dataframe(counties_df, "County Indicators")
    """
    rows, cols = df.shape
    nulls      = int(df.isnull().sum().sum())
    null_pct   = round(nulls / max(rows * cols, 1) * 100, 2)
    dtypes     = df.dtypes.value_counts().to_dict()

    print(f"\n── {label} ──")
    print(f"  Rows     : {rows:,}")
    print(f"  Columns  : {cols}")
    print(f"  Nulls    : {nulls} ({null_pct}%)")
    print(f"  Dtypes   : {dtypes}")

    if rows > 0:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            print(f"  Numeric  : {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")

    return {
        "rows": rows, "cols": cols,
        "nulls": nulls, "null_pct": null_pct,
        "dtypes": {str(k): v for k, v in dtypes.items()},
    }


def flatten_columns(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """
    Flatten a multi-index column DataFrame (e.g. after groupby + agg).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns.
    sep : str
        Separator between column levels. Default '_'.

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened single-level column names.

    Example
    -------
    >>> grouped = df.groupby('Region').agg({'rate': ['mean', 'std']})
    >>> grouped = flatten_columns(grouped)
    >>> grouped.columns
    Index(['rate_mean', 'rate_std'], dtype='object')
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [sep.join(str(c) for c in col).strip(sep) for col in df.columns]
    return df


def safe_divide(
    numerator: float,
    denominator: float,
    fallback: float = 0.0,
) -> float:
    """
    Divide two numbers, returning a fallback value instead of raising ZeroDivisionError.

    Parameters
    ----------
    numerator : float
    denominator : float
    fallback : float
        Value returned if denominator is zero. Default 0.0.

    Returns
    -------
    float

    Example
    -------
    >>> safe_divide(10, 0)
    0.0
    >>> safe_divide(10, 4)
    2.5
    """
    if denominator == 0:
        return fallback
    return numerator / denominator


# ═════════════════════════════════════════════════════════════════════════════
# QUICK PROJECT STATUS
# ═════════════════════════════════════════════════════════════════════════════

def project_status() -> None:
    """
    Print a quick status check showing which project files exist.
    Useful for verifying setup at the start of any notebook or script.

    Example
    -------
    >>> from src.utils import project_status
    >>> project_status()
    """
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    checks = {
        "Raw datasets": [DATA_RAW / f for f in RAW_FILES.values()],
        "Processed datasets": [DATA_PROCESSED / f for f in list(PROCESSED_FILES.values())[:3]],
        "County feature matrix": [DATA_PROCESSED / PROCESSED_FILES["matrix"]],
        "Saved models": [
            MODELS_DIR / "risk_classifier.pkl",
            MODELS_DIR / "mortality_regressor.pkl",
        ],
        "Recommendations": [DATA_PROCESSED / PROCESSED_FILES["recommendations"]],
        "Reports": [
            REPORTS_DIR / "benchmark_report.csv",
            REPORTS_DIR / "final_county_recommendations.csv",
        ],
    }

    print(f"\n{BOLD}── Child Mortality Project — Status ──{RESET}")
    for group, paths in checks.items():
        found   = sum(1 for p in paths if p.exists())
        total   = len(paths)
        symbol  = f"{GREEN}✓{RESET}" if found == total else f"{RED}✗{RESET}"
        missing = [p.name for p in paths if not p.exists()]
        print(f"  {symbol}  {group}: {found}/{total}", end="")
        if missing:
            print(f"  (missing: {', '.join(missing)})", end="")
        print()
    print()
