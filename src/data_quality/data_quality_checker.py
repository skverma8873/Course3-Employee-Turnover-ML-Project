"""
Data quality checker for the Employee Turnover Analytics pipeline.

Step 1 of the ML pipeline: loads the HR dataset and validates it
for missing values, data types, duplicates, and value ranges before
any analysis is performed.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.utils.exceptions import DataLoadError, DataQualityError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "left",
    "promotion_last_5years",
    "sales",
    "salary",
}

VALID_SALARY_VALUES = {"low", "medium", "high"}
BINARY_COLUMNS = {"Work_accident", "left", "promotion_last_5years"}
UNIT_RANGE_COLUMNS = {"satisfaction_level", "last_evaluation"}


class DataQualityChecker:
    """Loads the HR dataset and performs comprehensive data quality checks.

    This is the first stage of the pipeline.  All downstream classes receive
    the validated DataFrame that is returned by :meth:`load_data`.

    Args:
        filepath: Path to ``HR_comma_sep.csv``.  Accepts both ``str`` and
            :class:`pathlib.Path`.

    Attributes:
        filepath (Path): Resolved path to the CSV file.
        df (pd.DataFrame | None): Loaded DataFrame; ``None`` until
            :meth:`load_data` is called.

    Example:
        >>> checker = DataQualityChecker("Dataset/HR_comma_sep.csv")
        >>> df = checker.load_data()
        >>> report = checker.generate_quality_report()
    """

    def __init__(self, filepath: str) -> None:
        self.filepath: Path = Path(filepath).resolve()
        self.df: pd.DataFrame | None = None
        logger.debug("DataQualityChecker initialised — filepath=%s", self.filepath)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Read the CSV dataset into a DataFrame.

        Returns:
            The loaded :class:`pandas.DataFrame`.

        Raises:
            DataLoadError: If the file does not exist, is empty, or cannot
                be parsed as CSV.
        """
        logger.debug("load_data() called — path=%s", self.filepath)
        try:
            if not self.filepath.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {self.filepath}.  "
                    "Ensure HR_comma_sep.csv is in the Dataset/ directory."
                )

            self.df = pd.read_csv(self.filepath, encoding="utf-8")

            if self.df.empty:
                raise ValueError("The CSV file is empty.")

            logger.info(
                "Data loaded: %d rows × %d columns from %s",
                len(self.df),
                len(self.df.columns),
                self.filepath.name,
            )
            return self.df

        except FileNotFoundError as exc:
            logger.error("File not found: %s", self.filepath)
            raise DataLoadError(str(exc)) from exc
        except pd.errors.EmptyDataError as exc:
            logger.error("Empty CSV file: %s", self.filepath)
            raise DataLoadError(f"CSV file is empty: {self.filepath}") from exc
        except pd.errors.ParserError as exc:
            logger.error("CSV parse error: %s", exc)
            raise DataLoadError(f"Cannot parse CSV — {exc}") from exc
        except ValueError as exc:
            logger.error("Data load validation error: %s", exc)
            raise DataLoadError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Raise DataQualityError if load_data() has not been called."""
        if self.df is None:
            raise DataQualityError(
                "DataFrame is not loaded.  Call load_data() before running checks."
            )

    def check_missing_values(self) -> pd.DataFrame:
        """Count null values per column.

        Returns:
            DataFrame with columns ``missing_count`` and
            ``missing_percentage``, indexed by column name.

        Raises:
            DataQualityError: If :meth:`load_data` has not been called.
        """
        logger.debug("check_missing_values() called")
        self._require_loaded()
        try:
            missing_count = self.df.isnull().sum()
            missing_pct = (missing_count / len(self.df) * 100).round(2)
            result = pd.DataFrame(
                {"missing_count": missing_count, "missing_percentage": missing_pct}
            )
            total_missing = missing_count.sum()
            if total_missing == 0:
                logger.info("Missing values check: no missing values found")
            else:
                logger.warning(
                    "Missing values found: %d total across %d column(s)",
                    total_missing,
                    (missing_count > 0).sum(),
                )
            return result
        except Exception as exc:
            logger.exception("check_missing_values() failed unexpectedly")
            raise DataQualityError(f"Missing value check failed: {exc}") from exc

    def check_data_types(self) -> pd.DataFrame:
        """Return the data type of each column.

        Returns:
            DataFrame with a single column ``dtype``, indexed by column name.

        Raises:
            DataQualityError: If :meth:`load_data` has not been called.
        """
        logger.debug("check_data_types() called")
        self._require_loaded()
        result = pd.DataFrame({"dtype": self.df.dtypes.astype(str)})
        logger.info("Data types: %s", dict(zip(result.index, result["dtype"])))
        return result

    def check_duplicates(self) -> int:
        """Count exact duplicate rows.

        Returns:
            Number of duplicate rows.

        Raises:
            DataQualityError: If :meth:`load_data` has not been called.
        """
        logger.debug("check_duplicates() called")
        self._require_loaded()
        try:
            dup_count: int = int(self.df.duplicated().sum())
            if dup_count > 0:
                logger.warning(
                    "Found %d duplicate row(s) — %.1f%% of dataset",
                    dup_count,
                    dup_count / len(self.df) * 100,
                )
            else:
                logger.info("Duplicates check: no duplicate rows found")
            return dup_count
        except Exception as exc:
            logger.exception("check_duplicates() failed unexpectedly")
            raise DataQualityError(f"Duplicate check failed: {exc}") from exc

    def check_value_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Validate that column values fall within expected ranges.

        Checks performed:

        - ``satisfaction_level``, ``last_evaluation`` — must be in [0, 1]
        - ``Work_accident``, ``left``, ``promotion_last_5years`` — must be {0, 1}
        - ``salary`` — must be in {low, medium, high}

        Returns:
            Dictionary keyed by column name.  Each value is a dict with keys:
            ``valid`` (bool), ``min``, ``max``, ``invalid_count`` (int), and
            optionally ``unexpected_values`` (for categorical columns).

        Raises:
            DataQualityError: If :meth:`load_data` has not been called.
        """
        logger.debug("check_value_ranges() called")
        self._require_loaded()
        results: Dict[str, Dict[str, Any]] = {}

        try:
            # Numeric unit-range columns
            for col in UNIT_RANGE_COLUMNS:
                if col in self.df.columns:
                    col_min = float(self.df[col].min())
                    col_max = float(self.df[col].max())
                    invalid = int(((self.df[col] < 0) | (self.df[col] > 1)).sum())
                    results[col] = {
                        "valid": invalid == 0,
                        "min": col_min,
                        "max": col_max,
                        "invalid_count": invalid,
                    }

            # Binary columns
            for col in BINARY_COLUMNS:
                if col in self.df.columns:
                    unique_vals = set(self.df[col].unique())
                    invalid = int(self.df[col].apply(lambda v: v not in {0, 1}).sum())
                    results[col] = {
                        "valid": unique_vals.issubset({0, 1}),
                        "unique_values": sorted(unique_vals),
                        "invalid_count": invalid,
                    }

            # Salary categorical
            if "salary" in self.df.columns:
                salary_vals = set(self.df["salary"].unique())
                unexpected = salary_vals - VALID_SALARY_VALUES
                results["salary"] = {
                    "valid": len(unexpected) == 0,
                    "unique_values": sorted(salary_vals),
                    "unexpected_values": sorted(unexpected),
                    "invalid_count": int(
                        self.df["salary"].apply(
                            lambda v: v not in VALID_SALARY_VALUES
                        ).sum()
                    ),
                }

            all_valid = all(v["valid"] for v in results.values())
            if all_valid:
                logger.info("Value range check: all columns passed")
            else:
                failed = [k for k, v in results.items() if not v["valid"]]
                logger.warning("Value range check failed for: %s", failed)

            return results

        except Exception as exc:
            logger.exception("check_value_ranges() failed unexpectedly")
            raise DataQualityError(f"Value range check failed: {exc}") from exc

    def generate_quality_report(self) -> Dict[str, Any]:
        """Aggregate all quality checks into a single report dictionary.

        Returns:
            Dict with keys:
            - ``missing_values`` — DataFrame from :meth:`check_missing_values`
            - ``data_types`` — DataFrame from :meth:`check_data_types`
            - ``duplicate_count`` — int from :meth:`check_duplicates`
            - ``value_ranges`` — dict from :meth:`check_value_ranges`
            - ``total_rows`` — int
            - ``total_columns`` — int
            - ``column_names`` — list[str]

        Raises:
            DataQualityError: If :meth:`load_data` has not been called.
        """
        logger.debug("generate_quality_report() called")
        self._require_loaded()
        try:
            report: Dict[str, Any] = {
                "missing_values": self.check_missing_values(),
                "data_types": self.check_data_types(),
                "duplicate_count": self.check_duplicates(),
                "value_ranges": self.check_value_ranges(),
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "column_names": list(self.df.columns),
            }
            logger.info(
                "Quality report generated — rows=%d, cols=%d, duplicates=%d",
                report["total_rows"],
                report["total_columns"],
                report["duplicate_count"],
            )
            return report
        except DataQualityError:
            raise
        except Exception as exc:
            logger.exception("generate_quality_report() failed unexpectedly")
            raise DataQualityError(
                f"Quality report generation failed: {exc}"
            ) from exc
