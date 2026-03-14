# Employee Turnover Analytics ML Project

## Project Overview

This project addresses a critical business challenge for **Portobello Tech**: predicting and preventing employee turnover. Employee attrition is costly — recruitment, onboarding, and lost productivity can cost 50-200% of an employee's annual salary. By applying machine learning to HR data, we identify employees at risk of leaving and recommend targeted retention strategies.

The system ingests historical HR data (satisfaction scores, evaluations, project counts, working hours, tenure, accidents, promotions, department, and salary level), performs exploratory analysis and clustering, trains multiple classification models, and produces actionable retention recommendations categorized into four risk zones.

## Business Context

- **Problem:** Portobello Tech experiences unpredictable employee turnover, leading to high replacement costs and knowledge loss.
- **Goal:** Build a predictive model that identifies employees likely to leave and assigns them to risk zones with tailored retention strategies.
- **Success Criteria:** High recall (minimizing false negatives — employees who leave undetected) with a robust AUC score across multiple model architectures.

## Project Structure

```
Course3-Employee-Turnover-ML-Project/
├── CLAUDE.md                          # This file — project guide and conventions
├── requirements.txt                   # Python dependencies with pinned versions
├── Dataset/
│   └── HR_comma_sep.csv               # Source HR dataset (14,999 rows, 10 columns)
├── ProjectRequirements/
│   ├── employee_turnover_problem_statement.pdf
│   └── employee_turnover_problem_statement.docx
├── architecture/
│   ├── architecture_document.md       # Full architecture and class design reference
│   └── Diagrams/
│       ├── class_diagram.mmd          # Mermaid class diagram (all 8 classes)
│       └── sequence_diagram.mmd       # Mermaid sequence diagrams (7 flows)
├── src/
│   ├── __init__.py
│   ├── data_quality/                  # Step 1: Data quality checks
│   │   ├── __init__.py
│   │   └── data_quality_checker.py
│   ├── eda/                           # Step 2: Exploratory data analysis
│   │   ├── __init__.py
│   │   └── exploratory_analyzer.py
│   ├── clustering/                    # Step 3: K-Means clustering of leavers
│   │   ├── __init__.py
│   │   └── employee_clusterer.py
│   ├── preprocessing/                 # Step 4: Encoding, splitting, SMOTE
│   │   ├── __init__.py
│   │   └── data_preprocessor.py
│   ├── modeling/                      # Steps 5-6: Training and evaluation
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── retention/                     # Step 7: Risk zones and strategies
│   │   ├── __init__.py
│   │   └── retention_advisor.py
│   └── utils/                         # Shared visualization utilities
│       ├── __init__.py
│       └── visualization_utils.py
├── notebooks/
│   └── employee_turnover_analysis.ipynb  # Interactive notebook for full pipeline
├── outputs/
│   ├── plots/                         # Saved figures (heatmaps, ROC curves, etc.)
│   └── models/                        # Serialized trained models
└── tests/
    ├── __init__.py
    ├── test_data_quality.py
    ├── test_eda.py
    ├── test_clustering.py
    ├── test_preprocessing.py
    ├── test_modeling.py
    └── test_retention.py
```

## Technology Stack

| Library | Purpose | Version |
|---------|---------|---------|
| pandas | Data manipulation and analysis | >= 2.0 |
| numpy | Numerical operations | >= 1.24 |
| matplotlib | Base plotting framework | >= 3.7 |
| seaborn | Statistical visualization (heatmaps, distributions) | >= 0.12 |
| scikit-learn | ML models, metrics, preprocessing, clustering | >= 1.3 |
| imbalanced-learn | SMOTE for class imbalance handling | >= 0.11 |
| jupyter | Interactive notebook environment | >= 1.0 |
| pytest | Unit testing framework | >= 7.0 |

## ML Pipeline Summary

The pipeline consists of seven sequential stages:

### Stage 1: Data Quality Checks
Load `HR_comma_sep.csv` and validate: missing values, data types, duplicates, and value ranges (e.g., satisfaction_level in [0,1], left in {0,1}).

### Stage 2: Exploratory Data Analysis (EDA)
- Correlation heatmap of all numeric features
- Distribution plots for satisfaction_level, last_evaluation, average_montly_hours
- Bar plot of number_project grouped by left/stayed

### Stage 3: Clustering
Apply K-Means (k=3) on employees who left, using satisfaction_level and last_evaluation. Identify behavioral clusters (e.g., burned-out high performers, disengaged low performers, overworked mid-performers).

### Stage 4: Preprocessing and Class Imbalance Handling
- Encode categorical features (sales, salary) using `pd.get_dummies`
- Stratified 80:20 train-test split with `random_state=123`
- Apply SMOTE to upsample the minority class (left=1) in the training set

### Stage 5: Model Training
Train three classifiers with 5-fold cross-validation:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Stage 6: Model Evaluation
- ROC/AUC curves (overlay all models)
- Confusion matrices per model
- Classification report heatmaps
- Justify Recall over Precision for this domain

### Stage 7: Retention Strategies
- Predict turnover probabilities using the best model
- Categorize employees into risk zones: Safe (<20%), Low-Risk (20-60%), Medium-Risk (60-90%), High-Risk (>90%)
- Generate targeted retention strategies per zone

## Coding Standards

- **Style:** PEP 8 compliant, enforced via linting
- **Type Hints:** Required on all method signatures
- **Docstrings:** Google style on all public classes and methods
- **Immutability:** Never mutate input DataFrames; always return new copies
- **Constants:** Module-level, ALL_CAPS (e.g., `RANDOM_STATE = 123`, `TEST_SIZE = 0.2`)
- **File Length:** Maximum 400 lines per module
- **Error Handling:** Validate inputs at boundaries, raise `ValueError` with descriptive messages
- **Paths:** Use `pathlib.Path`, never hardcoded strings
- **Visualizations:** Always call `plt.tight_layout()` and close figures after saving
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for constants
- **No Global State:** All state belongs to class instances

## Testing Approach

- **Framework:** pytest
- **Coverage Target:** 80% minimum
- **Test Location:** `tests/` directory, one test file per source module
- **Strategy:** Unit tests for every public method; integration tests for pipeline stages
- **TDD Workflow:** Write tests first (RED), implement to pass (GREEN), refactor (IMPROVE)
- **Fixtures:** Use pytest fixtures for shared test data (sample DataFrames, mock models)

## How to Run

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook notebooks/employee_turnover_analysis.ipynb
```

### Run Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run Individual Pipeline Stages
Each module in `src/` can be imported and used independently. The notebook orchestrates the full pipeline.

## Key Design Decisions

1. **Modular Architecture:** Each pipeline stage is a separate module with a dedicated class, enabling independent testing, reuse, and maintainability.

2. **Recall over Precision:** In employee turnover prediction, failing to identify an at-risk employee (false negative) is costlier than flagging a stable employee (false positive). Retention interventions for false positives have minimal downside; missed departures have high cost.

3. **Three-Model Comparison:** Logistic Regression (interpretable baseline), Random Forest (ensemble with feature importance), and Gradient Boosting (sequential error correction) provide a balanced model portfolio.

4. **K-Means on Leavers Only:** Clustering employees who left reveals distinct behavioral archetypes, informing targeted retention strategies rather than generic interventions.

5. **SMOTE over Undersampling:** Oversampling the minority class preserves all majority-class information, critical when the dataset is moderately sized (approximately 15,000 rows).

6. **Four Risk Zones:** The Safe/Low/Medium/High categorization provides HR teams with actionable priority tiers, each mapped to specific intervention strategies.

7. **Stratified Splitting:** Maintains class distribution across train and test sets, ensuring evaluation metrics reflect real-world class proportions.

8. **Centralised Logging:** All modules use `logging.getLogger(__name__)` and a single `setup_logging()` call configures the root logger with rotating file output to `outputs\logs\`.

9. **Domain Exception Hierarchy:** All errors are wrapped in typed domain exceptions (`DataLoadError`, `ModelTrainingError`, etc.) so callers can catch predictably without depending on low-level library exceptions.

## Logging Standards

### Initialise logging once at entry point

```python
from src.utils.logger_config import setup_logging
from pathlib import Path

logger = setup_logging(log_dir=Path('outputs/logs'))
```

### Per-module logger (every .py file)

```python
import logging
logger = logging.getLogger(__name__)
```

### What to log at each level

| Level | Use for |
|-------|---------|
| `DEBUG` | Method entry with parameter values; intermediate results |
| `INFO` | Successful completion of significant steps; shapes, counts, metrics |
| `WARNING` | Duplicate rows found; class imbalance; fallback behaviour |
| `ERROR` | Caught exceptions before re-raising (use `logger.error()`) |
| `CRITICAL` | Unrecoverable pipeline failure (`logger.critical(..., exc_info=True)`) |

### Never do

- Never use `print()` for operational messages — use the logger
- Never `except: pass` — always log and re-raise
- Never log raw data rows or model weights (privacy / log size)
- Use `logger.exception(msg)` (not `logger.error`) when you need the full traceback

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
|------|---------|
| 0 | Success |
| 1 | Domain error (`EmployeeTurnoverError` subclass) |
| 2 | Unexpected / unhandled error |

## Output Directory Convention (Windows)

All generated outputs go under `outputs\` at project root:

```
outputs\
├── plots\eda\           # EDA visualisations (PNG)
├── plots\clustering\    # K-Means scatter plot
├── plots\modeling\      # ROC curves, confusion matrices, classification reports
├── plots\retention\     # Risk zone distribution bar chart
├── models\              # Serialised model files (.joblib)
└── logs\                # Rotating log files (employee_turnover_YYYYMMDD.log)
```

The `outputs\` directory is auto-created by each module — add it to `.gitignore`.
