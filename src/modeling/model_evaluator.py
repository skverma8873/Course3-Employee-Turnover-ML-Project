"""
Model evaluation — ROC/AUC, confusion matrices, and best-model selection.

Step 6 of the ML pipeline: evaluates all trained classifiers on the
held-out test set, identifies the best model, and justifies why Recall
is the primary metric in the HR turnover context.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from src.utils.exceptions import ModelEvaluationError, VisualizationError
from src.utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_LINE_STYLES = ["-", "--", "-."]
MODEL_COLORS = ["#2ecc71", "#e74c3c", "#3498db"]


class ModelEvaluator:
    """Evaluate trained classifiers on the held-out test set.

    Args:
        models: Dict mapping model name → fitted sklearn estimator.
        X_test: Test feature matrix.
        y_test: True test labels.
        output_dir: Directory for saving evaluation plots.

    Example:
        >>> evaluator = ModelEvaluator(models, X_test, y_test)
        >>> best_name, best_model = evaluator.identify_best_model()
        >>> evaluator.plot_roc_curves()
    """

    def __init__(
        self,
        models: Dict[str, object],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_dir = project_root / "outputs" / "plots" / "modeling"
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz = VisualizationUtils()
        logger.debug(
            "ModelEvaluator initialised — %d models, X_test=%s",
            len(models),
            X_test.shape,
        )

    # ------------------------------------------------------------------
    # ROC / AUC
    # ------------------------------------------------------------------

    def compute_roc_auc(self, model_name: str) -> float:
        """Compute the ROC-AUC score for a single model.

        Args:
            model_name: Key in ``self.models``.

        Returns:
            Float AUC score in [0, 1].

        Raises:
            ModelEvaluationError: If the model is missing or prediction fails.
        """
        logger.debug("compute_roc_auc() called — model=%s", model_name)
        try:
            model = self._get_model(model_name)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            auc = float(roc_auc_score(self.y_test, y_prob))
            logger.info("AUC — %s: %.4f", model_name, auc)
            return auc
        except ModelEvaluationError:
            raise
        except Exception as exc:
            logger.exception(
                "compute_roc_auc() failed for model '%s'", model_name
            )
            raise ModelEvaluationError(
                f"ROC-AUC computation failed for {model_name}: {exc}"
            ) from exc

    def plot_roc_curves(self, save_path: Optional[Path] = None) -> None:
        """Plot overlaid ROC curves for all models on a single figure.

        Args:
            save_path: Override save path.  Defaults to
                ``output_dir/roc_curves.png``.

        Raises:
            ModelEvaluationError: If prediction fails.
            VisualizationError: If plotting/saving fails.
        """
        logger.debug("plot_roc_curves() called — %d models", len(self.models))
        try:
            fig, ax = plt.subplots(figsize=(9, 7))
            for idx, (name, model) in enumerate(self.models.items()):
                y_prob = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                auc = roc_auc_score(self.y_test, y_prob)
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{name} (AUC = {auc:.4f})",
                    color=MODEL_COLORS[idx % len(MODEL_COLORS)],
                    linestyle=MODEL_LINE_STYLES[idx % len(MODEL_LINE_STYLES)],
                    linewidth=2,
                )
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
            ax.set_title(
                "ROC Curves — Model Comparison", fontsize=14, fontweight="bold"
            )
            ax.legend(fontsize=10, loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()

            dest = save_path or (self.output_dir / "roc_curves.png")
            self.viz.save_figure(fig, dest)
            logger.info("ROC curves saved: %s", dest)

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception("plot_roc_curves() failed unexpectedly")
            raise ModelEvaluationError(
                f"ROC curve plot failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def compute_confusion_matrix(self, model_name: str) -> np.ndarray:
        """Compute the confusion matrix for a single model.

        Args:
            model_name: Key in ``self.models``.

        Returns:
            2×2 numpy array ``[[TN, FP], [FN, TP]]``.

        Raises:
            ModelEvaluationError: If prediction fails.
        """
        logger.debug(
            "compute_confusion_matrix() called — model=%s", model_name
        )
        try:
            model = self._get_model(model_name)
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            logger.info("Confusion matrix — %s:\n%s", model_name, cm)
            return cm
        except ModelEvaluationError:
            raise
        except Exception as exc:
            logger.exception(
                "compute_confusion_matrix() failed for '%s'", model_name
            )
            raise ModelEvaluationError(
                f"Confusion matrix computation failed for {model_name}: {exc}"
            ) from exc

    def plot_confusion_matrix(
        self,
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot a confusion matrix heatmap for a single model.

        Args:
            model_name: Key in ``self.models``.
            save_path: Override save path.

        Raises:
            ModelEvaluationError: If prediction fails.
            VisualizationError: If plotting/saving fails.
        """
        logger.debug("plot_confusion_matrix() called — model=%s", model_name)
        try:
            cm = self.compute_confusion_matrix(model_name)
            labels = ["Stayed (0)", "Left (1)"]
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm_df,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                linewidths=0.5,
            )
            ax.set_title(
                f"Confusion Matrix — {model_name}",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_ylabel("Actual", fontsize=11)
            ax.set_xlabel("Predicted", fontsize=11)
            plt.tight_layout()

            safe_name = model_name.lower().replace(" ", "_")
            dest = save_path or (
                self.output_dir / f"confusion_matrix_{safe_name}.png"
            )
            self.viz.save_figure(fig, dest)
            logger.info("Confusion matrix plot saved: %s", dest)

        except (ModelEvaluationError, VisualizationError):
            raise
        except Exception as exc:
            logger.exception(
                "plot_confusion_matrix() failed for '%s'", model_name
            )
            raise VisualizationError(
                f"Confusion matrix plot failed for {model_name}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Best model selection
    # ------------------------------------------------------------------

    def identify_best_model(self) -> Tuple[str, object]:
        """Identify the model with the highest ROC-AUC score.

        Returns:
            Tuple ``(model_name, fitted_estimator)`` for the best model.

        Raises:
            ModelEvaluationError: If no models are available or AUC fails.
        """
        logger.debug("identify_best_model() called")
        if not self.models:
            raise ModelEvaluationError(
                "No trained models available.  Run ModelTrainer first."
            )
        try:
            auc_scores = {
                name: self.compute_roc_auc(name)
                for name in self.models
            }
            best_name = max(auc_scores, key=auc_scores.__getitem__)
            logger.info(
                "Model ranking by AUC: %s",
                sorted(auc_scores.items(), key=lambda x: x[1], reverse=True),
            )
            logger.info("Best model: %s (AUC=%.4f)", best_name, auc_scores[best_name])
            return best_name, self.models[best_name]
        except ModelEvaluationError:
            raise
        except Exception as exc:
            logger.exception("identify_best_model() failed unexpectedly")
            raise ModelEvaluationError(
                f"Best model identification failed: {exc}"
            ) from exc

    def justify_recall_over_precision(self) -> str:
        """Return the HR-domain justification for using Recall as primary metric.

        Returns:
            Multi-line explanation string.
        """
        justification = (
            "WHY RECALL IS THE PRIMARY METRIC IN EMPLOYEE TURNOVER PREDICTION\n"
            "=================================================================\n\n"
            "In binary classification, the confusion matrix produces two types of errors:\n\n"
            "  • False Negative (FN): Model predicts 'Stay' but employee actually LEAVES.\n"
            "    Cost: 50-200% of annual salary (recruitment, onboarding, lost knowledge).\n\n"
            "  • False Positive (FP): Model predicts 'Leave' but employee actually STAYS.\n"
            "    Cost: Minor overhead of an unnecessary retention check-in or conversation.\n\n"
            "Since FN costs vastly outweigh FP costs, we must minimise False Negatives.\n"
            "RECALL = TP / (TP + FN) — directly measures our ability to catch all leavers.\n\n"
            "Precision measures how many of our 'will leave' predictions are correct,\n"
            "but a false positive in HR is a low-cost mistake. A false negative is not.\n\n"
            "Therefore:\n"
            "  PRIMARY METRIC   → Recall (minimise missed leavers)\n"
            "  SECONDARY METRIC → ROC-AUC (overall discriminatory power)\n"
            "  TERTIARY METRIC  → F1-Score (balance for reporting)"
        )
        logger.info("Metric justification: Recall preferred over Precision for HR turnover")
        return justification

    # ------------------------------------------------------------------
    # Evaluation report
    # ------------------------------------------------------------------

    def generate_evaluation_report(self) -> pd.DataFrame:
        """Build a comparison table of all model metrics.

        Returns:
            DataFrame with columns:
            Model, AUC, Precision, Recall, F1, Accuracy.

        Raises:
            ModelEvaluationError: If any metric computation fails.
        """
        logger.debug("generate_evaluation_report() called")
        try:
            rows = []
            for name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                y_prob = model.predict_proba(self.X_test)[:, 1]
                rows.append(
                    {
                        "Model": name,
                        "AUC": round(roc_auc_score(self.y_test, y_prob), 4),
                        "Precision": round(
                            precision_score(self.y_test, y_pred, zero_division=0), 4
                        ),
                        "Recall": round(
                            recall_score(self.y_test, y_pred, zero_division=0), 4
                        ),
                        "F1": round(
                            f1_score(self.y_test, y_pred, zero_division=0), 4
                        ),
                        "Accuracy": round(accuracy_score(self.y_test, y_pred), 4),
                    }
                )
            report_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
            logger.info("Evaluation report:\n%s", report_df.to_string(index=False))
            return report_df
        except Exception as exc:
            logger.exception("generate_evaluation_report() failed unexpectedly")
            raise ModelEvaluationError(
                f"Evaluation report generation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self, model_name: str) -> object:
        """Retrieve a model by name, raising a clear error if absent."""
        if model_name not in self.models:
            raise ModelEvaluationError(
                f"Model '{model_name}' not found.  "
                f"Available: {list(self.models.keys())}"
            )
        return self.models[model_name]
