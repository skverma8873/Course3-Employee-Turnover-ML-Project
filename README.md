# Employee Turnover Analytics ML Project

**Portobello Tech — HR Department Predictive Analytics**

An end-to-end machine learning pipeline that predicts employee turnover,
identifies at-risk employees, and recommends targeted HR retention strategies
across four risk zones.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [ML Pipeline — 7 Steps](#3-ml-pipeline--7-steps)
4. [Setup Instructions (Windows)](#4-setup-instructions-windows)
5. [Running the Project](#5-running-the-project)
6. [Running Tests](#6-running-tests)
7. [Output Files](#7-output-files)
8. [Logging](#8-logging)
9. [Dataset Description](#9-dataset-description)
10. [Risk Zone Strategy](#10-risk-zone-strategy)

---

## 1. Project Overview

Portobello Tech uses historical HR data to predict which employees are likely
to leave the company. The system ingests employee records, performs exploratory
analysis and clustering, trains multiple classification models with cross-
validation, and produces actionable retention recommendations.

**Business goal:** Minimise undetected employee departures (high recall) and
prioritise HR interventions based on risk severity.

---

## 2. Project Structure

```
Course3-Employee-Turnover-ML-Project/
│
├── CLAUDE.md                          # Project guide and coding conventions
├── README.md                          # This file
├── requirements.txt                   # Pinned Python dependencies
├── .gitignore                         # Excludes outputs, venv, cache
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
│   │   ├── exceptions.py              # Custom exception hierarchy
│   │   ├── logger_config.py           # Centralised logging setup
│   │   └── visualization_utils.py    # Shared matplotlib/seaborn helpers
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
├── outputs/                           # Auto-created — all generated artifacts
│   ├── plots/
│   │   ├── eda/                       # correlation_heatmap.png, distributions, bar chart
│   │   ├── clustering/                # kmeans_clusters.png
│   │   ├── modeling/                  # roc_curves.png, confusion matrices, reports
│   │   └── retention/                 # risk_zone_distribution.png
│   ├── models/                        # Serialised model files (.joblib)
│   ├── logs/                          # Rotating log files
│   └── retention_report.csv          # Per-employee risk + strategy report
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

---

## 3. ML Pipeline — 7 Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_quality_checker.py` | Load CSV; check missing values, data types, duplicates, value ranges |
| 2 | `exploratory_analyzer.py` | Correlation heatmap; distribution plots (satisfaction, evaluation, hours); project count bar chart |
| 3 | `employee_clusterer.py` | K-Means (k=3) on employees who left, using satisfaction_level + last_evaluation |
| 4 | `data_preprocessor.py` | One-hot encode sales/salary; stratified 80:20 split (random_state=123); SMOTE upsampling |
| 5 | `model_trainer.py` | Train Logistic Regression, Random Forest, Gradient Boosting with 5-fold CV |
| 6 | `model_evaluator.py` | ROC/AUC curves, confusion matrices, classification reports; select best model by AUC |
| 7 | `retention_advisor.py` | Predict turnover probabilities; categorise into 4 risk zones; generate retention strategies |

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
- `outputs\plots\eda\` — EDA visualisations
- `outputs\plots\clustering\` — K-Means cluster plot
- `outputs\plots\modeling\` — ROC curves, confusion matrices, classification reports
- `outputs\plots\retention\` — Risk zone distribution chart
- `outputs\retention_report.csv` — Per-employee risk zone and strategy
- `outputs\logs\employee_turnover_YYYYMMDD.log` — Full execution log

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

| File / Directory | Description |
|------------------|-------------|
| `outputs\plots\eda\correlation_heatmap.png` | Pearson correlation matrix heatmap |
| `outputs\plots\eda\all_distributions.png` | Satisfaction, evaluation, hours distributions |
| `outputs\plots\eda\project_count_bar.png` | Project count by left/stayed |
| `outputs\plots\clustering\kmeans_clusters.png` | K-Means cluster scatter plot |
| `outputs\plots\modeling\roc_curves.png` | ROC curves (all 3 models overlaid) |
| `outputs\plots\modeling\confusion_matrix_*.png` | Per-model confusion matrix |
| `outputs\plots\modeling\classification_report_*.png` | Per-model classification report |
| `outputs\plots\retention\risk_zone_distribution.png` | Risk zone bar chart |
| `outputs\retention_report.csv` | Per-employee probability, zone, actual label |
| `outputs\logs\employee_turnover_YYYYMMDD.log` | Full pipeline execution log |

---

## 8. Logging

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
Get-Content outputs\logs\employee_turnover_20260314.log -Tail 50

# Filter for errors only
Select-String "ERROR|CRITICAL" outputs\logs\employee_turnover_20260314.log
```

---

## 9. Dataset Description

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

---

## 10. Risk Zone Strategy

| Zone | Score Range | Priority | Key HR Actions |
|------|-------------|----------|----------------|
| **Safe** (Green) | < 20% | Monitor | Annual reviews, recognition, optional growth |
| **Low-Risk** (Yellow) | 20% – 60% | Proactive | Stay interviews, career path clarification, skill development |
| **Medium-Risk** (Orange) | 60% – 90% | Urgent | Manager 1:1, workload review, compensation benchmark, HR escalation |
| **High-Risk** (Red) | > 90% | Critical | Senior leadership meeting, personalised retention package, weekly check-ins |

---

*Generated for Portobello Tech HR Analytics — Employee Turnover Prediction Pipeline*
