"""
Unit tests for RetentionAdvisor.

Run:  pytest tests/test_retention.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.retention.retention_advisor import RetentionAdvisor, ZONE_ORDER
from src.utils.exceptions import ModelEvaluationError


@pytest.fixture
def advisor_fixture(preprocessed_data):
    X_train, X_test, y_train, y_test = preprocessed_data
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return RetentionAdvisor(model, X_test, y_test)


class TestPredictTurnoverProbabilities:
    def test_probabilities_in_range(self, advisor_fixture):
        probs = advisor_fixture.predict_turnover_probabilities()
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_probabilities_length_matches_test(self, preprocessed_data):
        X_train, X_test, y_train, y_test = preprocessed_data
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        advisor = RetentionAdvisor(model, X_test, y_test)
        probs = advisor.predict_turnover_probabilities()
        assert len(probs) == len(X_test)

    def test_probabilities_stored_on_instance(self, advisor_fixture):
        probs = advisor_fixture.predict_turnover_probabilities()
        assert advisor_fixture.probabilities is not None
        assert len(advisor_fixture.probabilities) == len(probs)


class TestCategorizeRiskZones:
    def test_all_zones_in_valid_set(self, advisor_fixture):
        probs = advisor_fixture.predict_turnover_probabilities()
        zones = advisor_fixture.categorize_risk_zones(probs)
        valid_zones = set(ZONE_ORDER)
        for zone in zones:
            assert str(zone) in valid_zones

    def test_safe_zone_threshold(self, advisor_fixture):
        low_probs = np.array([0.05, 0.10, 0.19])
        zones = advisor_fixture.categorize_risk_zones(low_probs)
        assert all(z == "Safe" for z in zones)

    def test_low_risk_zone_threshold(self, advisor_fixture):
        probs = np.array([0.25, 0.40, 0.59])
        zones = advisor_fixture.categorize_risk_zones(probs)
        assert all(z == "Low-Risk" for z in zones)

    def test_medium_risk_zone_threshold(self, advisor_fixture):
        probs = np.array([0.65, 0.75, 0.89])
        zones = advisor_fixture.categorize_risk_zones(probs)
        assert all(z == "Medium-Risk" for z in zones)

    def test_high_risk_zone_threshold(self, advisor_fixture):
        high_probs = np.array([0.91, 0.95, 0.99])
        zones = advisor_fixture.categorize_risk_zones(high_probs)
        assert all(z == "High-Risk" for z in zones)


class TestSuggestStrategies:
    def test_all_zones_return_string(self, advisor_fixture):
        for zone in ZONE_ORDER:
            strategy = advisor_fixture.suggest_strategies(zone)
            assert isinstance(strategy, str)
            assert len(strategy) > 0

    def test_invalid_zone_raises(self, advisor_fixture):
        with pytest.raises(ValueError, match="Unknown zone"):
            advisor_fixture.suggest_strategies("InvalidZone")


class TestGenerateRetentionReport:
    def test_report_has_required_columns(self, advisor_fixture):
        report = advisor_fixture.generate_retention_report()
        expected = {
            "employee_index",
            "turnover_probability",
            "risk_zone",
            "actual_left",
            "strategy",
        }
        assert expected.issubset(set(report.columns))

    def test_report_length_matches_test_set(self, preprocessed_data):
        X_train, X_test, y_train, y_test = preprocessed_data
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        advisor = RetentionAdvisor(model, X_test, y_test)
        report = advisor.generate_retention_report()
        assert len(report) == len(X_test)

    def test_probabilities_in_report_range(self, advisor_fixture):
        report = advisor_fixture.generate_retention_report()
        assert report["turnover_probability"].between(0, 1).all()
