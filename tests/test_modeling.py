"""
Unit tests for ModelTrainer and ModelEvaluator.

Run:  pytest tests/test_modeling.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling.model_trainer import ModelTrainer
from src.modeling.model_evaluator import ModelEvaluator
from src.utils.exceptions import ModelTrainingError, ModelEvaluationError


@pytest.fixture
def trainer_fixture(preprocessed_data):
    X_train, X_test, y_train, y_test = preprocessed_data
    return ModelTrainer(X_train, y_train, cv_folds=2), X_test, y_test


class TestModelTrainer:
    def test_train_logistic_regression_returns_model(self, trainer_fixture):
        trainer, _, _ = trainer_fixture
        model = trainer.train_logistic_regression()
        assert isinstance(model, LogisticRegression)

    def test_train_random_forest_returns_model(self, trainer_fixture):
        trainer, _, _ = trainer_fixture
        model = trainer.train_random_forest()
        assert isinstance(model, RandomForestClassifier)

    def test_train_gradient_boosting_returns_model(self, trainer_fixture):
        trainer, _, _ = trainer_fixture
        model = trainer.train_gradient_boosting()
        assert isinstance(model, GradientBoostingClassifier)

    def test_train_all_models_returns_three_models(self, trainer_fixture):
        trainer, _, _ = trainer_fixture
        models = trainer.train_all_models()
        assert len(models) == 3
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "Gradient Boosting" in models

    def test_trained_models_stored_on_instance(self, trainer_fixture):
        trainer, _, _ = trainer_fixture
        trainer.train_all_models()
        assert len(trainer.trained_models) == 3

    def test_get_cv_scores_length(self, trainer_fixture):
        trainer, _, _ = trainer_fixture
        model = LogisticRegression(max_iter=500)
        scores = trainer.get_cv_scores(model)
        assert len(scores) == trainer.cv_folds

    def test_get_cv_scores_in_range(self, trainer_fixture):
        trainer, _, _ = trainer_fixture
        model = LogisticRegression(max_iter=500)
        scores = trainer.get_cv_scores(model)
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestModelEvaluator:
    @pytest.fixture
    def evaluator_fixture(self, preprocessed_data):
        X_train, X_test, y_train, y_test = preprocessed_data
        trainer = ModelTrainer(X_train, y_train, cv_folds=2)
        models = trainer.train_all_models()
        return ModelEvaluator(models, X_test, y_test), models

    def test_compute_roc_auc_in_range(self, evaluator_fixture):
        evaluator, models = evaluator_fixture
        for name in models:
            auc = evaluator.compute_roc_auc(name)
            assert 0.0 <= auc <= 1.0, f"AUC for {name} out of range: {auc}"

    def test_compute_roc_auc_invalid_model(self, evaluator_fixture):
        evaluator, _ = evaluator_fixture
        with pytest.raises(ModelEvaluationError, match="not found"):
            evaluator.compute_roc_auc("NonExistentModel")

    def test_compute_confusion_matrix_shape(self, evaluator_fixture):
        evaluator, models = evaluator_fixture
        name = next(iter(models))
        cm = evaluator.compute_confusion_matrix(name)
        assert cm.shape == (2, 2)

    def test_identify_best_model_returns_tuple(self, evaluator_fixture):
        evaluator, _ = evaluator_fixture
        result = evaluator.identify_best_model()
        assert isinstance(result, tuple)
        assert len(result) == 2
        best_name, best_model = result
        assert isinstance(best_name, str)

    def test_generate_evaluation_report_columns(self, evaluator_fixture):
        evaluator, _ = evaluator_fixture
        report = evaluator.generate_evaluation_report()
        expected_cols = {"Model", "AUC", "Precision", "Recall", "F1", "Accuracy"}
        assert expected_cols.issubset(set(report.columns))

    def test_generate_evaluation_report_rows(self, evaluator_fixture):
        evaluator, models = evaluator_fixture
        report = evaluator.generate_evaluation_report()
        assert len(report) == len(models)

    def test_justify_recall_returns_string(self, evaluator_fixture):
        evaluator, _ = evaluator_fixture
        justification = evaluator.justify_recall_over_precision()
        assert isinstance(justification, str)
        assert "RECALL" in justification
