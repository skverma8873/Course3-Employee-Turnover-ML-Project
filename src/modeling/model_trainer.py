"""
Model training with 5-fold cross-validation.

Step 5 of the ML pipeline: trains three classifiers on SMOTE-balanced
training data and evaluates each with stratified k-fold cross-validation.

Models trained:
- Logistic Regression (interpretable baseline)
- Random Forest Classifier (ensemble, feature importance)
- Gradient Boosting Classifier (sequential boosting, typically highest accuracy)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from src.utils.exceptions import ModelTrainingError, VisualizationError
from src.utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CV_FOLDS: int = 5
RANDOM_STATE: int = 123
CV_SCORING: str = "f1"


class ModelTrainer:
    """Train three classifiers with k-fold cross-validation.

    Args:
        X_train: SMOTE-balanced training feature matrix.
        y_train: Balanced training target vector.
        cv_folds: Number of cross-validation folds.  Defaults to 5.
        random_state: Seed for reproducible models.  Defaults to 123.
        output_dir: Directory for saving classification report plots.

    Attributes:
        trained_models (dict): Populated by train methods;
            keys are model names, values are fitted estimators.

    Example:
        >>> trainer = ModelTrainer(X_train, y_train)
        >>> models = trainer.train_all_models()
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = CV_FOLDS,
        random_state: int = RANDOM_STATE,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.trained_models: Dict[str, object] = {}

        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_dir = project_root / "outputs" / "plots" / "modeling"
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz = VisualizationUtils()
        logger.debug(
            "ModelTrainer initialised — X_train=%s, cv_folds=%d",
            X_train.shape,
            cv_folds,
        )

    # ------------------------------------------------------------------
    # Cross-validation helper
    # ------------------------------------------------------------------

    def get_cv_scores(
        self,
        model: object,
        scoring: str = CV_SCORING,
    ) -> np.ndarray:
        """Compute stratified k-fold cross-validation scores.

        Args:
            model: Unfitted or already fitted sklearn estimator.
            scoring: Scoring metric name.  Defaults to ``'f1'``.

        Returns:
            Array of CV scores with length ``cv_folds``.

        Raises:
            ModelTrainingError: If cross-validation fails.
        """
        logger.debug(
            "get_cv_scores() called — scoring=%s, cv=%d", scoring, self.cv_folds
        )
        try:
            scores = cross_val_score(
                model,
                self.X_train,
                self.y_train,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=-1,
            )
            logger.info(
                "CV scores (%s): %s — mean=%.4f ± %.4f",
                scoring,
                np.round(scores, 4),
                scores.mean(),
                scores.std(),
            )
            return scores
        except Exception as exc:
            logger.exception("get_cv_scores() failed")
            raise ModelTrainingError(
                f"Cross-validation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Individual model trainers
    # ------------------------------------------------------------------

    def train_logistic_regression(self) -> LogisticRegression:
        """Train a Logistic Regression classifier with 5-fold CV.

        Returns:
            Fitted :class:`~sklearn.linear_model.LogisticRegression`.

        Raises:
            ModelTrainingError: If training or CV fails.
        """
        model_name = "Logistic Regression"
        logger.info("Training %s ...", model_name)
        try:
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
            cv_scores = self.get_cv_scores(model)
            model.fit(self.X_train, self.y_train)
            self.trained_models[model_name] = model
            logger.info(
                "%s trained — CV F1: %.4f ± %.4f",
                model_name,
                cv_scores.mean(),
                cv_scores.std(),
            )
            return model
        except ModelTrainingError:
            raise
        except Exception as exc:
            logger.exception("train_logistic_regression() failed")
            raise ModelTrainingError(
                f"Logistic Regression training failed: {exc}"
            ) from exc

    def train_random_forest(self) -> RandomForestClassifier:
        """Train a Random Forest Classifier with 5-fold CV.

        Returns:
            Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.

        Raises:
            ModelTrainingError: If training or CV fails.
        """
        model_name = "Random Forest"
        logger.info("Training %s ...", model_name)
        try:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            )
            cv_scores = self.get_cv_scores(model)
            model.fit(self.X_train, self.y_train)
            self.trained_models[model_name] = model
            logger.info(
                "%s trained — CV F1: %.4f ± %.4f",
                model_name,
                cv_scores.mean(),
                cv_scores.std(),
            )
            return model
        except ModelTrainingError:
            raise
        except Exception as exc:
            logger.exception("train_random_forest() failed")
            raise ModelTrainingError(
                f"Random Forest training failed: {exc}"
            ) from exc

    def train_gradient_boosting(self) -> GradientBoostingClassifier:
        """Train a Gradient Boosting Classifier with 5-fold CV.

        Returns:
            Fitted :class:`~sklearn.ensemble.GradientBoostingClassifier`.

        Raises:
            ModelTrainingError: If training or CV fails.
        """
        model_name = "Gradient Boosting"
        logger.info("Training %s ...", model_name)
        try:
            model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
            )
            cv_scores = self.get_cv_scores(model)
            model.fit(self.X_train, self.y_train)
            self.trained_models[model_name] = model
            logger.info(
                "%s trained — CV F1: %.4f ± %.4f",
                model_name,
                cv_scores.mean(),
                cv_scores.std(),
            )
            return model
        except ModelTrainingError:
            raise
        except Exception as exc:
            logger.exception("train_gradient_boosting() failed")
            raise ModelTrainingError(
                f"Gradient Boosting training failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Classification report plot
    # ------------------------------------------------------------------

    def plot_classification_report(
        self,
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot the classification report as a heatmap.

        Args:
            model: Fitted estimator.
            X_test: Test feature matrix.
            y_test: True test labels.
            model_name: Display name for the model (used in title and filename).
            save_path: Override save path.

        Raises:
            ModelTrainingError: If prediction fails.
            VisualizationError: If plotting/saving fails.
        """
        logger.debug("plot_classification_report() called — model=%s", model_name)
        try:
            y_pred = model.predict(X_test)
            report_dict = classification_report(
                y_test, y_pred, output_dict=True, target_names=["Stayed", "Left"]
            )
            # Build DataFrame — exclude 'accuracy' row
            report_df = pd.DataFrame(report_dict).T
            report_df = report_df.drop(
                index=[i for i in report_df.index if i in ("accuracy",)],
                errors="ignore",
            )
            report_df = report_df[["precision", "recall", "f1-score"]].astype(float)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(
                report_df,
                annot=True,
                fmt=".3f",
                cmap="YlGnBu",
                ax=ax,
                linewidths=0.5,
                vmin=0,
                vmax=1,
            )
            ax.set_title(
                f"Classification Report — {model_name}",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            plt.tight_layout()

            safe_name = model_name.lower().replace(" ", "_")
            dest = save_path or (
                self.output_dir / f"classification_report_{safe_name}.png"
            )
            self.viz.save_figure(fig, dest)
            logger.info("Classification report plot saved: %s", dest)

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception(
                "plot_classification_report() failed for %s", model_name
            )
            raise ModelTrainingError(
                f"Classification report plot failed for {model_name}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------

    def train_all_models(self) -> Dict[str, object]:
        """Train all three classifiers sequentially.

        Returns:
            Dict mapping model name → fitted estimator.

        Raises:
            ModelTrainingError: If any model fails to train.
        """
        logger.info("Training all three models with %d-fold CV ...", self.cv_folds)
        try:
            self.train_logistic_regression()
            self.train_random_forest()
            self.train_gradient_boosting()
            logger.info(
                "All models trained: %s", list(self.trained_models.keys())
            )
            return self.trained_models
        except ModelTrainingError:
            raise
        except Exception as exc:
            logger.exception("train_all_models() failed unexpectedly")
            raise ModelTrainingError(
                f"train_all_models() failed: {exc}"
            ) from exc
