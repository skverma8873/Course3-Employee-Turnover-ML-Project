"""
Unit tests for ExploratoryAnalyzer.

Run:  pytest tests/test_eda.py -v
"""

import pandas as pd
import pytest

from src.eda.exploratory_analyzer import ExploratoryAnalyzer
from src.utils.exceptions import VisualizationError


class TestComputeTurnoverRate:
    def test_returns_dataframe(self, sample_df):
        analyzer = ExploratoryAnalyzer(sample_df)
        result = analyzer.compute_turnover_rate_by_feature("salary")
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_df):
        analyzer = ExploratoryAnalyzer(sample_df)
        result = analyzer.compute_turnover_rate_by_feature("salary")
        assert "count" in result.columns
        assert "left_count" in result.columns
        assert "turnover_rate" in result.columns

    def test_turnover_rates_between_0_and_1(self, sample_df):
        analyzer = ExploratoryAnalyzer(sample_df)
        result = analyzer.compute_turnover_rate_by_feature("salary")
        assert result["turnover_rate"].between(0, 1).all()

    def test_counts_sum_to_total_rows(self, sample_df):
        analyzer = ExploratoryAnalyzer(sample_df)
        result = analyzer.compute_turnover_rate_by_feature("salary")
        assert result["count"].sum() == len(sample_df)

    def test_invalid_feature_raises(self, sample_df):
        analyzer = ExploratoryAnalyzer(sample_df)
        with pytest.raises(ValueError, match="not in DataFrame"):
            analyzer.compute_turnover_rate_by_feature("nonexistent_column")


class TestPlotDistribution:
    def test_invalid_column_raises_value_error(self, sample_df, tmp_path):
        analyzer = ExploratoryAnalyzer(sample_df, output_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            analyzer.plot_distribution(
                column="nonexistent",
                title="Test",
            )

    def test_plot_creates_file(self, sample_df, tmp_path):
        analyzer = ExploratoryAnalyzer(sample_df, output_dir=tmp_path)
        analyzer.plot_distribution(
            column="satisfaction_level",
            title="Satisfaction Distribution",
            save_path=tmp_path / "test_dist.png",
        )
        assert (tmp_path / "test_dist.png").exists()


class TestPlotCorrelationHeatmap:
    def test_heatmap_creates_file(self, sample_df, tmp_path):
        analyzer = ExploratoryAnalyzer(sample_df, output_dir=tmp_path)
        analyzer.plot_correlation_heatmap(
            save_path=tmp_path / "test_heatmap.png"
        )
        assert (tmp_path / "test_heatmap.png").exists()


class TestPlotProjectCountBar:
    def test_bar_chart_creates_file(self, sample_df, tmp_path):
        analyzer = ExploratoryAnalyzer(sample_df, output_dir=tmp_path)
        analyzer.plot_project_count_bar(
            save_path=tmp_path / "test_bar.png"
        )
        assert (tmp_path / "test_bar.png").exists()


class TestInputImmutability:
    def test_original_df_not_mutated(self, sample_df):
        original_copy = sample_df.copy()
        analyzer = ExploratoryAnalyzer(sample_df)
        analyzer.compute_turnover_rate_by_feature("salary")
        pd.testing.assert_frame_equal(sample_df, original_copy)
