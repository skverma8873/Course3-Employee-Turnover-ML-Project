"""
Unit tests for DataPreprocessor.

Run:  pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import pytest

from src.preprocessing.data_preprocessor import DataPreprocessor
from src.utils.exceptions import PreprocessingError


class TestSeparateFeaturesTarget:
    def test_target_not_in_X(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, y = preprocessor.separate_features_target()
        assert "left" not in X.columns

    def test_target_in_y(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, y = preprocessor.separate_features_target()
        assert y.name == "left"
        assert set(y.unique()).issubset({0, 1})

    def test_X_has_remaining_columns(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, y = preprocessor.separate_features_target()
        assert len(X.columns) == len(sample_df.columns) - 1

    def test_missing_target_raises(self, sample_df):
        df_no_target = sample_df.drop(columns=["left"])
        preprocessor = DataPreprocessor(df_no_target, target_col="left")
        with pytest.raises(PreprocessingError):
            preprocessor.separate_features_target()


class TestEncodeCategorical:
    def test_no_sales_or_salary_in_output(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, _ = preprocessor.separate_features_target()
        X_enc = preprocessor.encode_categorical(X)
        assert "sales" not in X_enc.columns
        assert "salary" not in X_enc.columns

    def test_dummy_columns_created(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, _ = preprocessor.separate_features_target()
        X_enc = preprocessor.encode_categorical(X)
        # Should have at least some salary_ columns
        salary_cols = [c for c in X_enc.columns if c.startswith("salary_")]
        assert len(salary_cols) >= 2

    def test_output_is_new_dataframe(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, _ = preprocessor.separate_features_target()
        X_enc = preprocessor.encode_categorical(X)
        assert X_enc is not X  # New object returned


class TestStratifiedSplit:
    def test_train_test_sizes(self, sample_df):
        preprocessor = DataPreprocessor(sample_df, test_size=0.2)
        X, y = preprocessor.separate_features_target()
        X_enc = preprocessor.encode_categorical(X)
        X_train, X_test, y_train, y_test = preprocessor.stratified_split(X_enc, y)
        total = len(X_train) + len(X_test)
        assert total == len(sample_df)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_class_distribution_preserved(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, y = preprocessor.separate_features_target()
        X_enc = preprocessor.encode_categorical(X)
        _, _, y_train, y_test = preprocessor.stratified_split(X_enc, y)
        original_rate = y.mean()
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - original_rate) < 0.05
        assert abs(test_rate - original_rate) < 0.05


class TestApplySmote:
    def test_smote_balances_classes(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, y = preprocessor.separate_features_target()
        X_enc = preprocessor.encode_categorical(X)
        X_train, _, y_train, _ = preprocessor.stratified_split(X_enc, y)
        X_res, y_res = preprocessor.apply_smote(X_train, y_train)
        counts = y_res.value_counts()
        # After SMOTE minority should equal majority
        assert counts[0] == counts[1]

    def test_smote_preserves_columns(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X, y = preprocessor.separate_features_target()
        X_enc = preprocessor.encode_categorical(X)
        X_train, _, y_train, _ = preprocessor.stratified_split(X_enc, y)
        X_res, _ = preprocessor.apply_smote(X_train, y_train)
        assert list(X_res.columns) == list(X_train.columns)


class TestRunPreprocessingPipeline:
    def test_returns_four_elements(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        result = preprocessor.run_preprocessing_pipeline()
        assert len(result) == 4

    def test_output_types(self, sample_df):
        preprocessor = DataPreprocessor(sample_df)
        X_train, X_test, y_train, y_test = preprocessor.run_preprocessing_pipeline()
        import pandas as pd
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
