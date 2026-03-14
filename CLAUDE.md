# Employee Turnover Analytics ML Project

## Project Overview

This project addresses a critical business challenge for **Portobello Tech**: predicting and preventing employee turnover. Employee attrition is costly — recruitment, onboarding, and lost productivity can cost 50–200% of an employee's annual salary. By applying machine learning to HR data, we identify employees at risk of leaving and recommend targeted retention strategies.

The system ingests historical HR data (satisfaction scores, evaluations, project counts, working hours, tenure, accidents, promotions, department, and salary level), performs exploratory analysis and clustering, trains multiple classification models, and produces actionable retention recommendations categorised into four risk zones.

---

## Business Context

- **Problem:** Portobello Tech experiences unpredictable employee turnover, leading to high replacement costs and knowledge loss.
- **Goal:** Build a predictive model that identifies employees likely to leave and assigns them to risk zones with tailored retention strategies.
- **Success Criteria:** High recall (minimising false negatives — employees who leave undetected) with a robust AUC score across multiple model architectures.

---

## GitHub Repository

**URL:** https://github.com/skverma8873/Course3-Employee-Turnover-ML-Project

All source code, tests, architecture documents, and pipeline outputs (plots + analysis files) are committed to this repository. Binary artefacts (`.joblib` model files, log files) are excluded via `.gitignore`.

**Commit convention:** `<type>: <description>` where type is one of `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`.

---

## Project Structure

```
Course3-Employee-Turnover-ML-Project/
│
├── CLAUDE.md                          # This file — project guide and conventions
├── README.md                          # Setup, usage, and output reference
├── requirements.txt                   # Pinned Python dependencies
├── .gitignore                         # Excludes venv, models/, logs/, pycache
│
├── Dataset/
│   └── HR_comma_sep.csv               # Source HR dataset (14,999 rows, 10 columns)
│
├── ProjectRequirements/
│   ├── employee_turnover_problem_statement.pdf
│   └── employee_turnover_problem_statement.docx
│
├── Architecture/
│   ├── architecture_document.md       # Full architecture and class design (11 sections)
│   └── Diagrams/
│       ├── class_diagram.mmd          # Mermaid class diagram (8 classes)
│       └── sequence_diagram.mmd       # Mermaid sequence diagrams (7 flows)
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                    # CLI orchestrator — runs all 7 stages end-to-end
│   │
│   ├── utils/                         # Shared infrastructure
│   │   ├── __init__.py
│   │   ├── exceptions.py              # Custom exception hierarchy (7 typed exceptions)
│   │   ├── logger_config.py           # Centralised rotating-file logging setup
│   │   └── visualization_utils.py     # Shared matplotlib/seaborn helpers
│   │
│   ├── data_quality/                  # Step 1: Data validation
│   │   ├── __init__.py
│   │   └── data_quality_checker.py
│   │
│   ├── eda/                           # Step 2: Exploratory data analysis
│   │   ├── __init__.py
│   │   └── exploratory_analyzer.py
│   │
│   ├── clustering/                    # Step 3: K-Means clustering of leavers
│   │   ├── __init__.py
│   │   └── employee_clusterer.py
│   │
│   ├── preprocessing/                 # Step 4: Encoding, splitting, SMOTE
│   │   ├── __init__.py
│   │   └── data_preprocessor.py
│   │
│   ├── modeling/                      # Steps 5–6: Training and evaluation
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   │
│   └── retention/                     # Step 7: Risk zones and strategies
│       ├── __init__.py
│       └── retention_advisor.py
│
├── notebooks/
│   └── employee_turnover_analysis.ipynb  # Interactive notebook for full pipeline
│
├── outputs/                           # Auto-created; plots/ committed, models/ and logs/ gitignored
│   ├── plots/
│   │   ├── eda/
│   │   │   ├── correlation_heatmap.png     + correlation_heatmap.md
│   │   │   ├── all_distributions.png       + all_distributions.md
│   │   │   └── project_count_bar.png       + project_count_bar.md
│   │   ├── clustering/
│   │   │   └── kmeans_clusters.png         + kmeans_clusters.md
│   │   ├── modeling/
│   │   │   ├── roc_curves.png              + roc_curves.md
│   │   │   ├── confusion_matrix_*.png      + confusion_matrix_*.md  (3 models)
│   │   │   └── classification_report_*.png + classification_report_*.md  (3 models)
│   │   └── retention/
│   │       └── risk_zone_distribution.png  + risk_zone_distribution.md
│   ├── models/                        # .joblib files — gitignored (binary artefacts)
│   ├── logs/                          # Rotating log files — gitignored
│   └── retention_report.csv           # Per-employee probability, zone, actual label
│
└── tests/
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures (sample_df, sample_csv_path)
    ├── test_data_quality.py
    ├── test_eda.py
    ├── test_clustering.py
    ├── test_preprocessing.py
    ├── test_modeling.py
    └── test_retention.py
```

**Convention:** Every PNG in `outputs/plots/` has a co-located `.md` file of the same base name containing a layman-friendly analysis, metric explanations, and HR decision-making implications.

---

## Technology Stack

| Library | Purpose | Pinned Version |
|---|---|---|
| pandas | Data manipulation and analysis | 2.1.4 |
| numpy | Numerical operations | 1.26.4 |
| matplotlib | Base plotting framework | 3.8.2 |
| seaborn | Statistical visualisation (heatmaps, distributions) | 0.13.2 |
| scikit-learn | ML models, metrics, preprocessing, clustering | 1.4.2 |
| imbalanced-learn | SMOTE for class imbalance handling | 0.12.3 |
| joblib | Model serialisation to/from `.joblib` files | 1.4.2 |
| jupyter | Interactive notebook environment | 1.0.0 |
| notebook | Jupyter notebook server | 7.1.3 |
| ipykernel | Jupyter kernel | 6.29.3 |
| pytest | Unit testing framework | 8.1.2 |
| pytest-cov | Coverage reporting | 5.0.0 |

---

## ML Pipeline Summary

The pipeline consists of seven sequential stages, orchestrated by `src/pipeline.py`.

### Stage 1: Data Quality Checks (`data_quality_checker.py`)
Load `HR_comma_sep.csv` and validate: missing values, data types, duplicates, and value ranges (e.g., `satisfaction_level` in [0,1], `left` in {0,1}).

### Stage 2: Exploratory Data Analysis (`exploratory_analyzer.py`)
- Correlation heatmap of all numeric features
- Distribution plots for `satisfaction_level`, `last_evaluation`, `average_montly_hours`
- Bar plot of `number_project` grouped by left/stayed

### Stage 3: Clustering (`employee_clusterer.py`)
Apply K-Means (k=3) on employees who left, using `satisfaction_level` and `last_evaluation`. Identifies three departure archetypes: burned-out high performers, poached high performers, and disengaged low performers.

### Stage 4: Preprocessing and Class Imbalance Handling (`data_preprocessor.py`)
- Encode categorical features (`sales`, `salary`) using `pd.get_dummies`
- Stratified 80:20 train-test split with `random_state=123`
- Apply SMOTE to upsample the minority class (`left=1`) in the training set only

### Stage 5: Model Training (`model_trainer.py`)
Train three classifiers with 5-fold cross-validation:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

All trained models are serialised to `outputs/models/<model_name>.joblib` via `joblib.dump()`.

### Stage 6: Model Evaluation (`model_evaluator.py`)
- ROC/AUC curves (all 3 models overlaid)
- Confusion matrices per model
- Classification report heatmaps
- Best model selected by AUC; saved as `outputs/models/best_model.joblib`
- Justification: Recall prioritised over Precision for this domain

### Stage 7: Retention Strategies (`retention_advisor.py`)
- Predict turnover probabilities using the best model
- Categorise employees into risk zones: Safe (<20%), Low-Risk (20–60%), Medium-Risk (60–90%), High-Risk (>90%)
- Generate targeted retention strategies per zone
- Save per-employee report to `outputs/retention_report.csv`

---

## Actual Model Results (from pipeline execution)

| Model | AUC | Recall (Left) | Precision (Left) | F1 (Left) | False Negatives |
|---|---|---|---|---|---|
| Logistic Regression | 0.8205 | 72.1% | 53.2% | 0.612 | 199 |
| Gradient Boosting | 0.9859 | 93.1% | 92.2% | 0.927 | 49 |
| **Random Forest** | **0.9957** | **97.9%** | **98.0%** | **0.980** | **15** |

**Selected model:** Random Forest (highest AUC and lowest false negatives on 3,000-employee test set).

**Risk zone distribution (test set, 3,000 employees):**

| Zone | Count | % | Probability Threshold |
|---|---|---|---|
| Safe | 2,197 | 73.2% | < 20% |
| Low-Risk | 96 | 3.2% | 20–60% |
| Medium-Risk | 60 | 2.0% | 60–90% |
| High-Risk | 647 | 21.6% | > 90% |

---

## Coding Standards

- **Style:** PEP 8 compliant, enforced via linting
- **Type Hints:** Required on all method signatures
- **Docstrings:** Google style on all public classes and methods
- **Immutability:** Never mutate input DataFrames; always return new copies (`df.copy()`)
- **Constants:** Module-level, `ALL_CAPS` (e.g., `RANDOM_STATE = 123`, `TEST_SIZE = 0.2`)
- **File Length:** Maximum 400 lines per module
- **Error Handling:** Validate inputs at boundaries, raise `ValueError` with descriptive messages
- **Paths:** Use `pathlib.Path`, never hardcoded strings
- **Visualisations:** Always call `matplotlib.use('Agg')` at module top for headless operation; call `plt.tight_layout()` and `plt.close()` after saving
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for constants
- **No Global State:** All state belongs to class instances
- **Model persistence:** Use `joblib.dump(model, path)` and `joblib.load(path)` — never pickle directly

---

## Testing Approach

- **Framework:** pytest
- **Coverage Target:** 80% minimum across all `src/` modules
- **Test Location:** `tests/` directory, one test file per source module
- **Shared fixtures:** `tests/conftest.py` — `sample_df` (200-row synthetic DataFrame), `sample_csv_path`, `preprocessed_data`
- **Strategy:** Unit tests for every public method; integration via `conftest` fixtures
- **TDD Workflow:** Write test first (RED) → implement to pass (GREEN) → refactor (IMPROVE)

```bat
REM Run all tests with coverage
pytest tests\ -v --cov=src --cov-report=term-missing

REM HTML coverage report
pytest tests\ -v --cov=src --cov-report=html
```

---

## How to Run

### Setup (Windows)

```bat
cd "D:\SaurabhVerma\COE\self\simpli\project\Course-03-MachineLearning\Course3-Employee-Turnover-ML-Project"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Option A — Full Pipeline CLI (Recommended)

```bat
venv\Scripts\activate.bat
python src\pipeline.py

REM With explicit paths
python src\pipeline.py --data Dataset\HR_comma_sep.csv --output outputs
```

All outputs written to `outputs\` automatically.

### Option B — Jupyter Notebook (Interactive)

```bat
venv\Scripts\activate.bat
jupyter notebook notebooks\employee_turnover_analysis.ipynb
```

### Option C — Import Individual Modules

```python
import sys
sys.path.insert(0, '.')
from src.data_quality.data_quality_checker import DataQualityChecker
checker = DataQualityChecker('Dataset/HR_comma_sep.csv')
df = checker.load_data()
```

---

## Key Design Decisions

1. **Modular Architecture:** Each pipeline stage is a separate module with a dedicated class, enabling independent testing, reuse, and maintainability.

2. **Recall over Precision:** Failing to identify an at-risk employee (false negative) costs far more than flagging a stable employee (false positive). Retention interventions for false positives have minimal downside; missed departures carry high replacement cost.

3. **Three-Model Comparison:** Logistic Regression (interpretable baseline), Random Forest (ensemble, highest AUC), and Gradient Boosting (sequential error correction) provide a balanced portfolio demonstrating why non-linear models outperform linear ones for this problem.

4. **K-Means on Leavers Only:** Clustering only the employees who left reveals distinct departure archetypes (burned-out stars, poached performers, disengaged underperformers), each requiring a different HR response.

5. **SMOTE over Undersampling:** Oversampling the minority class preserves all majority-class information — critical when the dataset is moderately sized (~15,000 rows). SMOTE is applied to training data only to prevent data leakage into evaluation.

6. **Four Risk Zones:** Safe/Low/Medium/High categorisation gives HR actionable priority tiers, each mapped to specific intervention strategies with clear urgency levels.

7. **Stratified Splitting:** Maintains class distribution across train and test sets, ensuring evaluation metrics reflect real-world class proportions.

8. **Centralised Logging:** All modules use `logging.getLogger(__name__)`. A single `setup_logging()` call in `pipeline.py` configures the root logger with `RotatingFileHandler` (10 MB, 5 backups) writing to `outputs/logs/`.

9. **Domain Exception Hierarchy:** All errors are wrapped in typed domain exceptions so callers can catch predictably without depending on low-level library exceptions.

10. **Co-located Analysis Files:** Every PNG in `outputs/plots/` has a matching `.md` file with the same base name, providing layman-friendly analysis, metric definitions, and HR decision-making implications alongside the chart.

---

## Logging Standards

### Initialise logging once at the entry point

```python
from src.utils.logger_config import setup_logging
from pathlib import Path

setup_logging(log_dir=Path('outputs/logs'))
```

### Per-module logger (every `.py` file)

```python
import logging
logger = logging.getLogger(__name__)
```

### Log level guide

| Level | Use for |
|---|---|
| `DEBUG` | Method entry with parameter values; intermediate results |
| `INFO` | Successful completion of significant steps; shapes, counts, metrics |
| `WARNING` | Duplicate rows found; class imbalance; fallback behaviour |
| `ERROR` | Caught exceptions before re-raising |
| `CRITICAL` | Unrecoverable pipeline failure (`logger.critical(..., exc_info=True)`) |

### Never do

- Never use `print()` for operational messages — use the logger
- Never `except: pass` — always log and re-raise
- Never log raw data rows or model weights (privacy / log size)
- Use `logger.exception(msg)` (not `logger.error`) when you need the full traceback

---

## Exception Handling Standards

### Import custom exceptions

```python
from src.utils.exceptions import (
    DataLoadError, DataQualityError, PreprocessingError,
    ModelTrainingError, ModelEvaluationError, ClusteringError,
    VisualizationError, EmployeeTurnoverError
)
```

### Always wrap and re-raise as domain exceptions

```python
try:
    df = pd.read_csv(filepath)
except FileNotFoundError as exc:
    logger.error("File not found: %s", filepath)
    raise DataLoadError(f"Cannot load data from {filepath}: {exc}") from exc
```

### Pipeline exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Domain error (`EmployeeTurnoverError` subclass) |
| 2 | Unexpected / unhandled error |

---

## Output Directory Convention

All generated outputs go under `outputs/` at project root. Selective git tracking applies:

```
outputs/
├── plots/              ← COMMITTED to git (PNGs + analysis .md files)
│   ├── eda/
│   ├── clustering/
│   ├── modeling/
│   └── retention/
├── retention_report.csv  ← COMMITTED to git
├── models/             ← GITIGNORED (.joblib binary files are large)
└── logs/               ← GITIGNORED (rotating log files)
```

The `outputs/` directory is auto-created by the pipeline at runtime. Never manually delete `outputs/plots/` — it contains committed analysis files.

---

## Git Workflow

```bat
REM Stage specific files (never use git add -A blindly)
git add src/some_module.py tests/test_some_module.py

REM Commit with conventional message
git commit -m "feat: add salary band feature to preprocessor"

REM Push to remote
git push
```

**Branch:** `master`
**Remote:** `https://github.com/skverma8873/Course3-Employee-Turnover-ML-Project`

Do not commit:
- `outputs/models/*.joblib` — binary model artefacts
- `outputs/logs/` — log files
- `venv/` — virtual environment
- `__pycache__/`, `.pytest_cache/`, `htmlcov/` — generated caches
