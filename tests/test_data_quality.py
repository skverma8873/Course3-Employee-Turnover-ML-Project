"""
Unit tests for DataQualityChecker.

Run:  pytest tests/test_data_quality.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_quality.data_quality_checker import DataQualityChecker
from src.utils.exceptions import DataLoadError, DataQualityError


class TestDataQualityCheckerLoad:
    def test_load_data_success(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        df = checker.load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert "left" in df.columns

    def test_load_data_returns_dataframe_stored_on_instance(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        df = checker.load_data()
        assert checker.df is not None
        assert len(checker.df) == len(df)

    def test_load_data_file_not_found(self, tmp_path):
        checker = DataQualityChecker(str(tmp_path / "nonexistent.csv"))
        with pytest.raises(DataLoadError, match="Dataset not found"):
            checker.load_data()

    def test_load_data_empty_file(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        checker = DataQualityChecker(str(empty))
        with pytest.raises(DataLoadError):
            checker.load_data()


class TestMissingValues:
    def test_no_missing_values(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        result = checker.check_missing_values()
        assert "missing_count" in result.columns
        assert "missing_percentage" in result.columns
        assert result["missing_count"].sum() == 0

    def test_with_missing_values(self, sample_df, tmp_path):
        df_with_nan = sample_df.copy()
        df_with_nan.loc[0:4, "satisfaction_level"] = np.nan
        csv_path = tmp_path / "missing.csv"
        df_with_nan.to_csv(csv_path, index=False)
        checker = DataQualityChecker(str(csv_path))
        checker.load_data()
        result = checker.check_missing_values()
        assert result.loc["satisfaction_level", "missing_count"] == 5

    def test_requires_load_data_first(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        with pytest.raises(DataQualityError, match="load_data"):
            checker.check_missing_values()


class TestDataTypes:
    def test_check_data_types_returns_dataframe(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        result = checker.check_data_types()
        assert isinstance(result, pd.DataFrame)
        assert "dtype" in result.columns

    def test_check_data_types_has_all_columns(self, sample_csv_path, sample_df):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        result = checker.check_data_types()
        assert set(result.index) == set(sample_df.columns)


class TestDuplicates:
    def test_no_duplicates(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        assert checker.check_duplicates() == 0

    def test_with_duplicates(self, sample_df, tmp_path):
        df_dup = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
        csv_path = tmp_path / "dups.csv"
        df_dup.to_csv(csv_path, index=False)
        checker = DataQualityChecker(str(csv_path))
        checker.load_data()
        assert checker.check_duplicates() == 10


class TestValueRanges:
    def test_valid_ranges(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        results = checker.check_value_ranges()
        for col, info in results.items():
            assert info["valid"], f"Column {col} failed range check: {info}"

    def test_salary_valid_values(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        results = checker.check_value_ranges()
        assert results["salary"]["valid"]
        assert results["salary"]["unexpected_values"] == []


class TestQualityReport:
    def test_report_has_all_keys(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        report = checker.generate_quality_report()
        expected_keys = {
            "missing_values",
            "data_types",
            "duplicate_count",
            "value_ranges",
            "total_rows",
            "total_columns",
            "column_names",
        }
        assert expected_keys.issubset(set(report.keys()))

    def test_report_total_rows(self, sample_csv_path):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        report = checker.generate_quality_report()
        assert report["total_rows"] == 200

    def test_report_total_columns(self, sample_csv_path, sample_df):
        checker = DataQualityChecker(str(sample_csv_path))
        checker.load_data()
        report = checker.generate_quality_report()
        assert report["total_columns"] == len(sample_df.columns)
