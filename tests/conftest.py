"""
Shared pytest fixtures for all test modules.

Run tests from the project root:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable when running pytest from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame matching the HR dataset schema (200 rows)."""
    np.random.seed(42)
    n = 200
    # Force class imbalance: ~24% leave (realistic)
    left = np.zeros(n, dtype=int)
    left_indices = np.random.choice(n, size=48, replace=False)
    left[left_indices] = 1

    return pd.DataFrame(
        {
            "satisfaction_level": np.random.uniform(0.09, 1.0, n).round(2),
            "last_evaluation": np.random.uniform(0.36, 1.0, n).round(2),
            "number_project": np.random.randint(2, 8, n),
            "average_montly_hours": np.random.randint(96, 310, n),
            "time_spend_company": np.random.randint(2, 10, n),
            "Work_accident": np.random.randint(0, 2, n),
            "left": left,
            "promotion_last_5years": np.random.randint(0, 2, n),
            "sales": np.random.choice(
                ["sales", "technical", "hr", "IT", "support", "management"], n
            ),
            "salary": np.random.choice(["low", "medium", "high"], n),
        }
    )


@pytest.fixture
def sample_csv_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Write sample_df to a temporary CSV file and return its path."""
    csv_path = tmp_path / "test_hr.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def preprocessed_data(sample_df: pd.DataFrame):
    """Return (X_train, X_test, y_train, y_test) from the sample DataFrame."""
    from src.preprocessing.data_preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor(sample_df)
    return preprocessor.run_preprocessing_pipeline()
