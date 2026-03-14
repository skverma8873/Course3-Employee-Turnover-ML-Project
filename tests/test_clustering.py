"""
Unit tests for EmployeeClusterer.

Run:  pytest tests/test_clustering.py -v
"""

import pandas as pd
import pytest

from src.clustering.employee_clusterer import EmployeeClusterer
from src.utils.exceptions import ClusteringError


class TestPrepareClusterData:
    def test_only_leavers_included(self, sample_df):
        clusterer = EmployeeClusterer(sample_df)
        data = clusterer.prepare_cluster_data()
        # All rows must come from employees who left
        leavers_count = (sample_df["left"] == 1).sum()
        assert len(data) == leavers_count

    def test_only_two_feature_columns(self, sample_df):
        clusterer = EmployeeClusterer(sample_df)
        data = clusterer.prepare_cluster_data()
        assert set(data.columns) == {"satisfaction_level", "last_evaluation"}

    def test_no_leavers_raises(self, sample_df):
        df_no_leavers = sample_df.copy()
        df_no_leavers["left"] = 0
        clusterer = EmployeeClusterer(df_no_leavers)
        with pytest.raises(ClusteringError):
            clusterer.prepare_cluster_data()

    def test_returns_dataframe(self, sample_df):
        clusterer = EmployeeClusterer(sample_df)
        result = clusterer.prepare_cluster_data()
        assert isinstance(result, pd.DataFrame)


class TestFitKmeans:
    def test_labels_length_matches_data(self, sample_df):
        clusterer = EmployeeClusterer(sample_df, n_clusters=3)
        clusterer.prepare_cluster_data()
        clusterer.fit_kmeans()
        assert len(clusterer.labels) == len(clusterer.cluster_data)

    def test_correct_number_of_clusters(self, sample_df):
        n = 3
        clusterer = EmployeeClusterer(sample_df, n_clusters=n)
        clusterer.prepare_cluster_data()
        clusterer.fit_kmeans()
        assert len(set(clusterer.labels)) == n

    def test_fit_without_prepare_raises(self, sample_df):
        clusterer = EmployeeClusterer(sample_df)
        with pytest.raises(ClusteringError, match="prepare_cluster_data"):
            clusterer.fit_kmeans()

    def test_model_stored_on_instance(self, sample_df):
        clusterer = EmployeeClusterer(sample_df)
        clusterer.prepare_cluster_data()
        model = clusterer.fit_kmeans()
        assert clusterer.kmeans_model is model


class TestGetClusterSummary:
    def test_summary_shape(self, sample_df):
        clusterer = EmployeeClusterer(sample_df, n_clusters=3)
        clusterer.prepare_cluster_data()
        clusterer.fit_kmeans()
        summary = clusterer.get_cluster_summary()
        assert len(summary) == 3

    def test_summary_columns(self, sample_df):
        clusterer = EmployeeClusterer(sample_df, n_clusters=3)
        clusterer.prepare_cluster_data()
        clusterer.fit_kmeans()
        summary = clusterer.get_cluster_summary()
        assert "satisfaction_level" in summary.columns
        assert "last_evaluation" in summary.columns
        assert "count" in summary.columns

    def test_counts_sum_to_total_leavers(self, sample_df):
        clusterer = EmployeeClusterer(sample_df, n_clusters=3)
        clusterer.prepare_cluster_data()
        clusterer.fit_kmeans()
        summary = clusterer.get_cluster_summary()
        total_leavers = (sample_df["left"] == 1).sum()
        assert summary["count"].sum() == total_leavers


class TestInterpretClusters:
    def test_returns_dict_with_cluster_keys(self, sample_df):
        clusterer = EmployeeClusterer(sample_df, n_clusters=3)
        clusterer.prepare_cluster_data()
        clusterer.fit_kmeans()
        interpretations = clusterer.interpret_clusters()
        assert len(interpretations) == 3

    def test_each_cluster_has_label(self, sample_df):
        clusterer = EmployeeClusterer(sample_df, n_clusters=3)
        clusterer.prepare_cluster_data()
        clusterer.fit_kmeans()
        interpretations = clusterer.interpret_clusters()
        for cluster_id, info in interpretations.items():
            assert "label" in info
            assert "description" in info


class TestRunClusteringPipeline:
    def test_pipeline_returns_dict(self, sample_df):
        clusterer = EmployeeClusterer(sample_df, n_clusters=3)
        result = clusterer.run_clustering_pipeline()
        assert isinstance(result, dict)
        assert "summary" in result
        assert "interpretation" in result
        assert "model" in result
