"""
Employee Turnover Analytics — Full Pipeline Runner

Orchestrates all 7 ML pipeline stages end-to-end:
  1. Data quality checks
  2. Exploratory data analysis
  3. K-Means clustering of employees who left
  4. Preprocessing and SMOTE oversampling
  5. Model training (Logistic Regression, Random Forest, Gradient Boosting)
  6. Model evaluation (ROC/AUC, confusion matrices, best model selection)
  7. Retention strategy recommendations (risk zones)

Usage (Windows — activate venv first):
    python src/pipeline.py
    python src/pipeline.py --data Dataset/HR_comma_sep.csv --output outputs
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger_config import setup_logging
from src.utils.exceptions import EmployeeTurnoverError
from src.data_quality.data_quality_checker import DataQualityChecker
from src.eda.exploratory_analyzer import ExploratoryAnalyzer
from src.clustering.employee_clusterer import EmployeeClusterer
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.modeling.model_trainer import ModelTrainer
from src.modeling.model_evaluator import ModelEvaluator
from src.retention.retention_advisor import RetentionAdvisor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with ``data`` and ``output``.
    """
    parser = argparse.ArgumentParser(
        description="Employee Turnover Analytics — Full ML Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "Dataset" / "HR_comma_sep.csv"),
        help="Path to HR_comma_sep.csv (default: Dataset/HR_comma_sep.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Root output directory (default: outputs/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_data_quality(data_path: Path) -> "pd.DataFrame":
    """Step 1 — Load data and run quality checks."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Quality Checks")
    logger.info("=" * 60)
    checker = DataQualityChecker(str(data_path))
    df = checker.load_data()
    report = checker.generate_quality_report()
    logger.info(
        "Quality summary — rows=%d, cols=%d, duplicates=%d, "
        "columns_with_missing=%d",
        report["total_rows"],
        report["total_columns"],
        report["duplicate_count"],
        int((report["missing_values"]["missing_count"] > 0).sum()),
    )
    return df


def run_eda(df: "pd.DataFrame", output_dir: Path) -> None:
    """Step 2 — Exploratory data analysis."""
    logger.info("=" * 60)
    logger.info("STEP 2: Exploratory Data Analysis")
    logger.info("=" * 60)
    analyzer = ExploratoryAnalyzer(df, output_dir=output_dir / "plots" / "eda")
    analyzer.run_full_eda()


def run_clustering(df: "pd.DataFrame", output_dir: Path) -> dict:
    """Step 3 — K-Means clustering on employees who left."""
    logger.info("=" * 60)
    logger.info("STEP 3: Employee Clustering (K-Means, k=3)")
    logger.info("=" * 60)
    clusterer = EmployeeClusterer(
        df, n_clusters=3, output_dir=output_dir / "plots" / "clustering"
    )
    return clusterer.run_clustering_pipeline()


def run_preprocessing(df: "pd.DataFrame") -> tuple:
    """Step 4 — Encode, split, and SMOTE."""
    logger.info("=" * 60)
    logger.info("STEP 4: Preprocessing & SMOTE")
    logger.info("=" * 60)
    preprocessor = DataPreprocessor(df)
    return preprocessor.run_preprocessing_pipeline()


def run_model_training(
    X_train: "pd.DataFrame",
    y_train: "pd.Series",
    X_test: "pd.DataFrame",
    y_test: "pd.Series",
    output_dir: Path,
) -> dict:
    """Step 5 — Train all three classifiers with 5-fold CV."""
    logger.info("=" * 60)
    logger.info("STEP 5: Model Training (5-fold CV)")
    logger.info("=" * 60)
    trainer = ModelTrainer(
        X_train, y_train, output_dir=output_dir / "plots" / "modeling"
    )
    models = trainer.train_all_models()

    # ── Save every trained model to outputs/models/ ──────────────────────────
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        model_path = models_dir / f"{safe_name}.joblib"
        joblib.dump(model, model_path)
        logger.info("Model saved: %s → %s", name, model_path)

    logger.info("Generating classification report plots ...")
    for name, model in models.items():
        trainer.plot_classification_report(model, X_test, y_test, name)

    return models


def run_model_evaluation(
    models: dict,
    X_test: "pd.DataFrame",
    y_test: "pd.Series",
    output_dir: Path,
) -> tuple:
    """Step 6 — Evaluate all models and identify best."""
    logger.info("=" * 60)
    logger.info("STEP 6: Model Evaluation")
    logger.info("=" * 60)
    evaluator = ModelEvaluator(
        models, X_test, y_test, output_dir=output_dir / "plots" / "modeling"
    )
    evaluator.plot_roc_curves()
    for name in models:
        evaluator.plot_confusion_matrix(name)

    report_df = evaluator.generate_evaluation_report()
    logger.info("Evaluation report:\n%s", report_df.to_string(index=False))

    best_name, best_model = evaluator.identify_best_model()
    logger.info("Best model selected: %s", best_name)
    logger.info(evaluator.justify_recall_over_precision())

    # ── Save best model separately for easy loading ───────────────────────────
    models_dir = output_dir / "models"
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    logger.info("Best model saved: %s → %s", best_name, best_model_path)

    return best_name, best_model


def run_retention(
    best_model: object,
    X_test: "pd.DataFrame",
    y_test: "pd.Series",
    output_dir: Path,
) -> "pd.DataFrame":
    """Step 7 — Predict probabilities and assign retention strategies."""
    logger.info("=" * 60)
    logger.info("STEP 7: Retention Strategy Recommendations")
    logger.info("=" * 60)
    advisor = RetentionAdvisor(
        best_model, X_test, y_test, output_dir=output_dir / "plots" / "retention"
    )
    report = advisor.generate_retention_report()
    advisor.plot_zone_distribution()

    # Save report to CSV
    report_path = output_dir / "retention_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.drop(columns=["strategy"]).to_csv(report_path, index=False)
    logger.info("Retention report saved: %s", report_path)

    # Print strategies for each zone
    from src.retention.retention_advisor import ZONE_ORDER
    for zone in ZONE_ORDER:
        logger.info("\n--- %s ---\n%s", zone, advisor.suggest_strategies(zone))

    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the complete Employee Turnover Analytics pipeline."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialise logging FIRST
    setup_logging(log_dir=output_dir / "logs")

    logger.info("Employee Turnover Analytics Pipeline starting ...")
    logger.info("Data path : %s", args.data)
    logger.info("Output dir: %s", output_dir)

    # Run all stages
    df = run_data_quality(Path(args.data))
    run_eda(df, output_dir)
    run_clustering(df, output_dir)
    X_train, X_test, y_train, y_test = run_preprocessing(df)
    models = run_model_training(X_train, y_train, X_test, y_test, output_dir)
    best_name, best_model = run_model_evaluation(models, X_test, y_test, output_dir)
    run_retention(best_model, X_test, y_test, output_dir)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("All outputs saved to: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except EmployeeTurnoverError as exc:
        # Domain error — already logged; clean exit with error code
        logging.getLogger(__name__).critical(
            "Pipeline failed: %s", exc, exc_info=True
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Pipeline interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).critical(
            "Unexpected pipeline failure: %s", exc, exc_info=True
        )
        sys.exit(2)
