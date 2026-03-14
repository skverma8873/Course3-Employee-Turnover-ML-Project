# Employee Turnover Analytics ML Project

**Portobello Tech — HR Department Predictive Analytics**

An end-to-end machine learning pipeline that predicts employee turnover,
identifies at-risk employees, and recommends targeted HR retention strategies
across four risk zones.

**Repository:** https://github.com/skverma8873/Course3-Employee-Turnover-ML-Project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [ML Pipeline — 7 Steps](#3-ml-pipeline--7-steps)
4. [Setup Instructions (Windows)](#4-setup-instructions-windows)
5. [Running the Project](#5-running-the-project)
6. [Running Tests](#6-running-tests)
7. [Output Files](#7-output-files)
8. [Model Results](#8-model-results)
9. [Logging](#9-logging)
10. [Dataset Description](#10-dataset-description)
11. [Risk Zone Strategy](#11-risk-zone-strategy)

---

## 1. Project Overview

Portobello Tech uses historical HR data to predict which employees are likely
to leave the company. The system ingests employee records, performs exploratory
analysis and clustering, trains multiple classification models with cross-
validation, and produces actionable retention recommendations.

**Business goal:** Minimise undetected employee departures (high recall) and
prioritise HR interventions based on risk severity.

**Selected model:** Random Forest — AUC 0.9957, Recall 97.9%, only 15 missed
departures out of 714 in the test set.

---

## 2. Project Structure

```
Course3-Employee-Turnover-ML-Project/
│
├── CLAUDE.md                          # Project guide and coding conventions
├── README.md                          # This file
├── requirements.txt                   # Pinned Python dependencies
├── .gitignore                         # Excludes venv, models/, logs/, pycache
│
├── Dataset/
│   └── HR_comma_sep.csv               # Source HR data (14,999 rows, 10 cols)
│
├── ProjectRequirements/
│   ├── employee_turnover_problem_statement.pdf
│   └── employee_turnover_problem_statement.docx
│
├── Architecture/
│   ├── architecture_document.md       # Full system design (11 sections)
│   └── Diagrams/
│       ├── class_diagram.mmd          # Mermaid class diagram (8 classes)
│       └── sequence_diagram.mmd       # Mermaid sequence diagrams (7 flows)
│
├── src/                               # All source code
│   ├── __init__.py
│   ├── pipeline.py                    # Main CLI orchestrator (runs all 7 steps)
│   │
│   ├── utils/                         # Shared infrastructure
│   │   ├── __init__.py
│   │   ├── exceptions.py              # Custom exception hierarchy (7 types)
│   │   ├── logger_config.py           # Centralised rotating-file logging setup
│   │   └── visualization_utils.py     # Shared matplotlib/seaborn helpers
│   │
│   ├── data_quality/                  # Step 1: Data validation
│   │   ├── __init__.py
│   │   └── data_quality_checker.py
│   │
│   ├── eda/                           # Step 2: Exploratory analysis
│   │   ├── __init__.py
│   │   └── exploratory_analyzer.py
│   │
│   ├── clustering/                    # Step 3: K-Means on leavers
│   │   ├── __init__.py
│   │   └── employee_clusterer.py
│   │
│   ├── preprocessing/                 # Step 4: Encoding + SMOTE
│   │   ├── __init__.py
│   │   └── data_preprocessor.py
│   │
│   ├── modeling/                      # Steps 5-6: Train + Evaluate
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   │
│   └── retention/                     # Step 7: Risk zones + strategies
│       ├── __init__.py
│       └── retention_advisor.py
│
├── notebooks/
│   └── employee_turnover_analysis.ipynb  # Interactive notebook (all 7 steps)
│
├── outputs/                           # Auto-created at runtime
│   ├── plots/                         # ← COMMITTED to git
│   │   ├── eda/                       # PNG + .md analysis file per chart
│   │   ├── clustering/                # PNG + .md analysis file
│   │   ├── modeling/                  # PNG + .md analysis file per chart
│   │   └── retention/                 # PNG + .md analysis file
│   ├── retention_report.csv           # ← COMMITTED — per-employee risk report
│   ├── models/                        # ← GITIGNORED — .joblib binary files
│   └── logs/                          # ← GITIGNORED — rotating log files
│
└── tests/                             # pytest test suite
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures (sample_df, sample_csv_path)
    ├── test_data_quality.py
    ├── test_eda.py
    ├── test_clustering.py
    ├── test_preprocessing.py
    ├── test_modeling.py
    └── test_retention.py
```

> **Convention:** Every PNG in `outputs/plots/` has a co-located `.md` file of
> the same base name containing a layman-friendly analysis, metric explanations,
> and HR decision-making implications.

---

## 3. ML Pipeline — 7 Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_quality_checker.py` | Load CSV; check missing values, data types, duplicates, value ranges |
| 2 | `exploratory_analyzer.py` | Correlation heatmap; distribution plots (satisfaction, evaluation, hours); project count bar chart |
| 3 | `employee_clusterer.py` | K-Means (k=3) on employees who left, using satisfaction_level + last_evaluation |
| 4 | `data_preprocessor.py` | One-hot encode sales/salary; stratified 80:20 split (random_state=123); SMOTE upsampling |
| 5 | `model_trainer.py` | Train Logistic Regression, Random Forest, Gradient Boosting with 5-fold CV; serialise to `outputs/models/` |
| 6 | `model_evaluator.py` | ROC/AUC curves, confusion matrices, classification reports; select best model by AUC; save `best_model.joblib` |
| 7 | `retention_advisor.py` | Predict turnover probabilities; categorise into 4 risk zones; save `retention_report.csv` |

---

## 4. Setup Instructions (Windows)

```bat
REM Step 1: Navigate to project directory
cd "D:\SaurabhVerma\COE\self\simpli\project\Course-03-MachineLearning\Course3-Employee-Turnover-ML-Project"

REM Step 2: Create virtual environment
python -m venv venv

REM Step 3: Activate virtual environment (Windows Command Prompt)
venv\Scripts\activate.bat

REM  OR for Windows PowerShell:
REM  venv\Scripts\Activate.ps1

REM Step 4: Install dependencies
pip install -r requirements.txt
```

---

## 5. Running the Project

### Option A — Full Pipeline (Recommended)

Runs all 7 steps, saves all plots, models, and the retention report.

```bat
REM Activate venv first
venv\Scripts\activate.bat

REM Run with defaults (reads Dataset\HR_comma_sep.csv, writes to outputs\)
python src\pipeline.py

REM Run with explicit paths
python src\pipeline.py --data Dataset\HR_comma_sep.csv --output outputs
```

**All outputs are written to `outputs\`:**
- `outputs\plots\eda\` — EDA visualisations + analysis `.md` files
- `outputs\plots\clustering\` — K-Means cluster plot + analysis `.md`
- `outputs\plots\modeling\` — ROC curves, confusion matrices, classification reports + analysis `.md` files
- `outputs\plots\retention\` — Risk zone distribution chart + analysis `.md`
- `outputs\retention_report.csv` — Per-employee risk zone and probability
- `outputs\models\` — Serialised `.joblib` model files (gitignored)
- `outputs\logs\employee_turnover_YYYYMMDD.log` — Full execution log (gitignored)

### Option B — Jupyter Notebook (Interactive)

Runs each pipeline step cell-by-cell with inline explanations and plots.

```bat
venv\Scripts\activate.bat
jupyter notebook notebooks\employee_turnover_analysis.ipynb
```

### Option C — Import Individual Modules

```python
import sys
sys.path.insert(0, '.')  # Run from project root

from src.data_quality.data_quality_checker import DataQualityChecker

checker = DataQualityChecker('Dataset/HR_comma_sep.csv')
df = checker.load_data()
report = checker.generate_quality_report()
print(f"Rows: {report['total_rows']}, Duplicates: {report['duplicate_count']}")
```

---

## 6. Running Tests

```bat
REM Activate venv first
venv\Scripts\activate.bat

REM Run all tests
pytest tests\ -v

REM Run with coverage report (terminal)
pytest tests\ -v --cov=src --cov-report=term-missing

REM Run with HTML coverage report
pytest tests\ -v --cov=src --cov-report=html
REM  Coverage report opens at: htmlcov\index.html

REM Run a single test file
pytest tests\test_data_quality.py -v
```

**Coverage target: 80%+** across all `src/` modules.

---

## 7. Output Files

### Plots and Analysis Files

Each PNG has a co-located `.md` file with the same base name containing
layman-friendly analysis and HR decision-making guidance.

| PNG | Analysis File | Description |
|-----|--------------|-------------|
| `outputs\plots\eda\correlation_heatmap.png` | `correlation_heatmap.md` | Pearson correlation matrix — satisfaction_level is strongest predictor (−0.39) |
| `outputs\plots\eda\all_distributions.png` | `all_distributions.md` | Satisfaction, evaluation, hours distributions — all bimodal, revealing hidden at-risk sub-groups |
| `outputs\plots\eda\project_count_bar.png` | `project_count_bar.md` | Project count by left/stayed — U-shaped departure curve (2 projects and 6–7 projects both drive exits) |
| `outputs\plots\clustering\kmeans_clusters.png` | `kmeans_clusters.md` | K-Means scatter plot — 3 departure archetypes: Burned-Out Stars, Poached Performers, Disengaged Low Performers |
| `outputs\plots\modeling\roc_curves.png` | `roc_curves.md` | ROC curves all 3 models — RF AUC 0.9957, GB 0.9859, LR 0.8205 |
| `outputs\plots\modeling\confusion_matrix_logistic_regression.png` | `confusion_matrix_logistic_regression.md` | LR confusion matrix — 199 missed leavers, 453 false alarms |
| `outputs\plots\modeling\confusion_matrix_random_forest.png` | `confusion_matrix_random_forest.md` | RF confusion matrix — only 15 missed leavers, 14 false alarms |
| `outputs\plots\modeling\confusion_matrix_gradient_boosting.png` | `confusion_matrix_gradient_boosting.md` | GB confusion matrix — 49 missed leavers, 56 false alarms |
| `outputs\plots\modeling\classification_report_logistic_regression.png` | `classification_report_logistic_regression.md` | LR report — Recall 72.1%, Precision 53.2%, F1 0.612 |
| `outputs\plots\modeling\classification_report_random_forest.png` | `classification_report_random_forest.md` | RF report — Recall 97.9%, Precision 98.0%, F1 0.980 |
| `outputs\plots\modeling\classification_report_gradient_boosting.png` | `classification_report_gradient_boosting.md` | GB report — Recall 93.1%, Precision 92.2%, F1 0.927 |
| `outputs\plots\retention\risk_zone_distribution.png` | `risk_zone_distribution.md` | Risk zone bar chart — 647 High-Risk (21.6%), 2,197 Safe (73.2%) |

### Other Output Files

| File | Description |
|------|-------------|
| `outputs\retention_report.csv` | Per-employee turnover probability, risk zone, and actual label |
| `outputs\logs\employee_turnover_YYYYMMDD.log` | Full pipeline execution log (gitignored) |
| `outputs\models\*.joblib` | Serialised model files — LR, RF, GB, best_model (gitignored) |

---

## 8. Model Results

Results from pipeline execution on 3,000-employee test set (20% of 14,999 rows).

### Model Comparison

| Model | AUC | Recall (Left) | Precision (Left) | F1 (Left) | Missed Leavers | False Alarms |
|-------|-----|--------------|-----------------|-----------|---------------|-------------|
| Logistic Regression | 0.8205 | 72.1% | 53.2% | 0.612 | 199 | 453 |
| Gradient Boosting | 0.9859 | 93.1% | 92.2% | 0.927 | 49 | 56 |
| **Random Forest** ✓ | **0.9957** | **97.9%** | **98.0%** | **0.980** | **15** | **14** |

**Selected model:** Random Forest (highest AUC, lowest missed departures).

> **Why Recall over Precision?** Missing a real leaver (false negative) costs 6–9
> months of salary in replacement. Incorrectly flagging a stayer (false positive)
> costs a brief retention conversation. Random Forest reduces missed leavers by
> 92% vs Logistic Regression.

### Risk Zone Distribution (test set — 3,000 employees)

| Zone | Employees | % | Turnover Probability |
|------|-----------|---|---------------------|
| Safe | 2,197 | 73.2% | < 20% |
| Low-Risk | 96 | 3.2% | 20% – 60% |
| Medium-Risk | 60 | 2.0% | 60% – 90% |
| **High-Risk** | **647** | **21.6%** | **> 90%** |

> **Key insight:** The distribution is bimodal — 73% are safe, 22% are critical,
> with almost nothing in between. Employees are not drifting gradually; they cross
> from stable to near-certain departure quickly, often triggered by a single event
> (missed promotion, heavy workload, competitor offer).

---

## 9. Logging

The project uses Python's standard `logging` module with a centralised
configuration in `src/utils/logger_config.py`.

| Setting | Value |
|---------|-------|
| Log file location | `outputs\logs\employee_turnover_YYYYMMDD.log` |
| File log level | DEBUG (all details) |
| Console log level | INFO (key milestones) |
| Rotation | 10 MB per file, 5 backups retained |
| Format | `YYYY-MM-DD HH:MM:SS \| LEVEL    \| module.name \| message` |

**View logs (Windows PowerShell):**
```powershell
# View last 50 lines
Get-Content outputs\logs\employee_turnover_20260315.log -Tail 50

# Filter for errors only
Select-String "ERROR|CRITICAL" outputs\logs\employee_turnover_20260315.log
```

---

## 10. Dataset Description

Source: `Dataset\HR_comma_sep.csv` — 14,999 employee records

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `satisfaction_level` | float | 0.0 – 1.0 | Employee satisfaction score |
| `last_evaluation` | float | 0.0 – 1.0 | Most recent performance evaluation |
| `number_project` | int | 2 – 7 | Number of projects assigned |
| `average_montly_hours` | int | 96 – 310 | Average monthly working hours |
| `time_spend_company` | int | 2 – 10 | Years at the company |
| `Work_accident` | int | 0 or 1 | Whether a workplace accident occurred |
| `left` | int | 0 or 1 | **TARGET**: 1 = employee left, 0 = stayed |
| `promotion_last_5years` | int | 0 or 1 | Whether promoted in last 5 years |
| `sales` | string | 10 departments | Department name |
| `salary` | string | low/medium/high | Salary bracket |

**Class distribution:** ~76% stayed (left=0), ~24% left (left=1) — handled via SMOTE on training set only.

---

## 11. Risk Zone Strategy

### Strategy Table

| Zone | Score Range | Priority | Key HR Actions |
|------|-------------|----------|----------------|
| **Safe** (Green) | < 20% | Monitor | Annual reviews, recognition, optional growth |
| **Low-Risk** (Yellow) | 20% – 60% | Proactive | Stay interviews, career path clarification, skill development |
| **Medium-Risk** (Orange) | 60% – 90% | Urgent | Manager 1:1, workload review, compensation benchmark, HR escalation |
| **High-Risk** (Red) | > 90% | Critical | Senior leadership meeting, personalised retention package, weekly check-ins |

### K-Means Departure Archetypes

The clustering stage identifies three distinct profiles among employees who left — each requires a different retention response for current at-risk employees:

| Archetype | Satisfaction | Evaluation | Why They Left | Retention Lever |
|-----------|-------------|-----------|---------------|-----------------|
| Burned-Out Stars | Very Low (~0.12) | Very High (~0.87) | Overwork, no recognition | Workload cap, immediate recognition, compensation review |
| Poached Performers | High (~0.80) | Very High (~0.91) | Better external offer | Market salary benchmarking, fast-track promotion, equity |
| Disengaged Low Performers | Moderate (~0.40) | Average (~0.52) | Lack of direction, limited prospects | Performance coaching, clearer role expectations, development plan |

---

*Generated for Portobello Tech HR Analytics — Employee Turnover Prediction Pipeline*
