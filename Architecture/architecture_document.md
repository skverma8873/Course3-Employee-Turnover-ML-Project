# Architecture Document — Employee Turnover Analytics ML Project

---

## Table of Contents

1. [Project Overview and Business Problem](#1-project-overview-and-business-problem)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Class Design](#4-class-design)
5. [ML Pipeline Architecture](#5-ml-pipeline-architecture)
6. [Coding Standards](#6-coding-standards)
7. [Data Flow Diagram](#7-data-flow-diagram)
8. [Evaluation Metrics Justification](#8-evaluation-metrics-justification)
9. [Retention Strategy Framework](#9-retention-strategy-framework)
10. [Logging Architecture](#10-logging-architecture)
11. [Exception Handling Architecture](#11-exception-handling-architecture)
12. [Notebook Architecture](#12-notebook-architecture)

---

## 1. Project Overview and Business Problem

### Business Context

Portobello Tech faces a recurring challenge: unpredictable employee turnover. When employees leave unexpectedly, the company incurs significant costs:

- **Direct costs:** Recruitment fees, onboarding, training for replacements
- **Indirect costs:** Lost institutional knowledge, reduced team morale, project delays
- **Estimated impact:** 50-200% of an employee's annual salary per departure

### Problem Statement

The HR department needs a data-driven system that can:

1. **Identify patterns** in historical turnover data to understand why employees leave
2. **Predict which current employees** are at risk of leaving
3. **Categorize risk levels** to prioritize retention efforts
4. **Recommend targeted strategies** based on risk severity and employee profiles

### Dataset

The project uses `HR_comma_sep.csv` containing 14,999 employee records with 10 features:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| satisfaction_level | float | 0.0 - 1.0 | Employee satisfaction score |
| last_evaluation | float | 0.0 - 1.0 | Most recent performance evaluation score |
| number_project | int | 2 - 7 | Number of projects assigned |
| average_montly_hours | int | 96 - 310 | Average monthly working hours |
| time_spend_company | int | 2 - 10 | Years at the company |
| Work_accident | int | 0 or 1 | Whether the employee had a workplace accident |
| left | int | 0 or 1 | Whether the employee left (TARGET) |
| promotion_last_5years | int | 0 or 1 | Whether promoted in last 5 years |
| sales | categorical | 10 departments | Department name |
| salary | categorical | low / medium / high | Salary bracket |

### Success Criteria

- High recall on the positive class (left=1) to minimize undetected departures
- ROC-AUC score above 0.90 on the held-out test set
- Actionable retention recommendations mapped to four risk zones

---

## 2. Technology Stack

### Core Libraries

| Library | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **pandas** | >= 2.0 | Data manipulation and analysis | Industry standard for tabular data; provides DataFrame operations, CSV reading, groupby aggregations, and dummy encoding via `pd.get_dummies` |
| **numpy** | >= 1.24 | Numerical operations | Foundation for array computations; used for probability arrays, numerical aggregations, and array manipulations across all pipeline stages |
| **matplotlib** | >= 3.7 | Base plotting framework | Provides fine-grained control over figure creation, subplot layouts, axis customization, and figure saving; backend for seaborn |
| **seaborn** | >= 0.12 | Statistical visualizations | Built on matplotlib; provides heatmaps (`sns.heatmap`), distribution plots (`sns.histplot`, `sns.kdeplot`), and styled bar plots with minimal code |
| **scikit-learn** | >= 1.3 | Machine learning models, metrics, and utilities | Comprehensive ML toolkit providing all required algorithms and evaluation tools |
| **imbalanced-learn** | >= 0.11 | SMOTE oversampling | Purpose-built library for handling class imbalance; integrates seamlessly with scikit-learn's API |
| **jupyter** | >= 1.0 | Interactive notebook environment | Enables step-by-step exploration, inline visualization, and narrative documentation of the analysis |
| **pytest** | >= 7.0 | Unit testing framework | Modern testing with fixtures, parametrize, and coverage integration |

### scikit-learn Components Used

| Component | Module | Purpose |
|-----------|--------|---------|
| `LogisticRegression` | `sklearn.linear_model` | Baseline linear classifier; interpretable coefficients |
| `RandomForestClassifier` | `sklearn.ensemble` | Ensemble of decision trees; captures non-linear patterns, provides feature importance |
| `GradientBoostingClassifier` | `sklearn.ensemble` | Sequential boosting; often achieves highest accuracy on structured data |
| `KMeans` | `sklearn.cluster` | Unsupervised clustering of employees who left into behavioral groups |
| `train_test_split` | `sklearn.model_selection` | Standard train-test splitting utility |
| `StratifiedShuffleSplit` | `sklearn.model_selection` | Stratified splitting preserving class distribution |
| `cross_val_score` | `sklearn.model_selection` | K-fold cross-validation scoring |
| `classification_report` | `sklearn.metrics` | Precision, recall, F1 per class |
| `confusion_matrix` | `sklearn.metrics` | True/false positive/negative counts |
| `roc_auc_score` | `sklearn.metrics` | Area under the ROC curve |
| `roc_curve` | `sklearn.metrics` | False positive rate and true positive rate arrays for plotting |

### imbalanced-learn Components Used

| Component | Module | Purpose |
|-----------|--------|---------|
| `SMOTE` | `imblearn.over_sampling` | Synthetic Minority Over-sampling Technique; generates synthetic samples for the minority class (left=1) to balance the training set |

---

## 3. Project Structure

```
Course3-Employee-Turnover-ML-Project/
├── CLAUDE.md
├── requirements.txt
├── Dataset/
│   └── HR_comma_sep.csv
├── ProjectRequirements/
│   ├── employee_turnover_problem_statement.pdf
│   └── employee_turnover_problem_statement.docx
├── architecture/
│   ├── architecture_document.md
│   └── Diagrams/
│       ├── class_diagram.mmd
│       └── sequence_diagram.mmd
├── src/
│   ├── __init__.py
│   ├── data_quality/
│   │   ├── __init__.py
│   │   └── data_quality_checker.py
│   ├── eda/
│   │   ├── __init__.py
│   │   └── exploratory_analyzer.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   └── employee_clusterer.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_preprocessor.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── retention/
│   │   ├── __init__.py
│   │   └── retention_advisor.py
│   └── utils/
│       ├── __init__.py
│       └── visualization_utils.py
├── notebooks/
│   └── employee_turnover_analysis.ipynb
├── outputs/
│   ├── plots/
│   └── models/
└── tests/
    ├── __init__.py
    ├── test_data_quality.py
    ├── test_eda.py
    ├── test_clustering.py
    ├── test_preprocessing.py
    ├── test_modeling.py
    └── test_retention.py
```

### Directory Descriptions

#### Root Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project guide for AI assistants and developers: overview, conventions, how to run |
| `requirements.txt` | Pinned Python dependencies for reproducible environments |

#### `Dataset/`
Contains the raw HR dataset `HR_comma_sep.csv`. This file is read-only and never modified by the pipeline. Any derived data is stored in memory or in `outputs/`.

#### `ProjectRequirements/`
Original problem statement documents provided by the course. These define the exact requirements: which plots to generate, which models to train, which parameters to use (k=3, random_state=123, 80:20 split, 5-fold CV).

#### `architecture/`
Architecture documentation and design diagrams.

- `architecture_document.md` — This file. Comprehensive reference for the system's design, class APIs, coding standards, and pipeline architecture.
- `Diagrams/class_diagram.mmd` — Mermaid class diagram showing all 8 classes, their attributes, methods, and relationships.
- `Diagrams/sequence_diagram.mmd` — Mermaid sequence diagrams showing the runtime interactions for each pipeline stage and the end-to-end flow.

#### `src/`
Source code organized by pipeline stage. Each subdirectory is a Python package (contains `__init__.py`) with a single-responsibility class.

- `data_quality/` — Step 1: Loads data and performs quality checks (missing values, types, duplicates, ranges).
- `eda/` — Step 2: Generates exploratory visualizations (heatmaps, distributions, bar plots).
- `clustering/` — Step 3: K-Means clustering on employees who left.
- `preprocessing/` — Step 4: Feature encoding, stratified splitting, and SMOTE.
- `modeling/` — Steps 5-6: Model training (3 classifiers, 5-fold CV) and evaluation (ROC, confusion matrices).
- `retention/` — Step 7: Turnover probability prediction, risk zone categorization, and strategy generation.
- `utils/` — Shared visualization utilities used across all plotting modules.

#### `notebooks/`
Jupyter notebook `employee_turnover_analysis.ipynb` orchestrates the full pipeline interactively. Each cell corresponds to a pipeline stage, with markdown narrative explaining the findings.

#### `outputs/`
Generated artifacts, excluded from version control (except the directory structure).

- `plots/` — Saved PNG/SVG figures: heatmaps, distribution plots, bar charts, cluster scatter plots, ROC curves, confusion matrices, zone distribution charts.
- `models/` — Serialized trained models (pickle or joblib format) for deployment or future inference.

#### `tests/`
pytest test suite with one test file per source module. Each test file contains unit tests for every public method of the corresponding class, using fixtures for shared test data.

---

## 4. Class Design

### 4.1 DataQualityChecker

**Module:** `src/data_quality/data_quality_checker.py`

**Purpose:** Loads the HR dataset from CSV and performs comprehensive data quality checks. This is the first step in the pipeline, ensuring the data is clean and valid before any analysis.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | — | Path to the CSV file (e.g., `Dataset/HR_comma_sep.csv`) |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `filepath` | `Path` | Resolved path to the data file |
| `df` | `pd.DataFrame` | Loaded dataset (populated after `load_data()`) |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `load_data()` | — | `pd.DataFrame` | Reads the CSV file at `self.filepath` into a pandas DataFrame, stores it as `self.df`, and returns it. Raises `FileNotFoundError` if the file does not exist. Raises `ValueError` if the file is empty. |
| `check_missing_values()` | — | `pd.DataFrame` | Returns a DataFrame with two columns: `missing_count` (number of null values per column) and `missing_percentage` (percentage of nulls per column). Uses `self.df.isnull().sum()` internally. |
| `check_data_types()` | — | `pd.DataFrame` | Returns a DataFrame with column names as index and a single column `dtype` containing the data type of each column. Useful for verifying that numeric columns are not stored as strings. |
| `check_duplicates()` | — | `int` | Returns the count of exact duplicate rows in the dataset using `self.df.duplicated().sum()`. |
| `check_value_ranges()` | — | `dict` | Validates that values fall within expected ranges. Returns a dictionary with column names as keys and validation results as values. Checks include: `satisfaction_level` in [0, 1], `last_evaluation` in [0, 1], `left` in {0, 1}, `Work_accident` in {0, 1}, `salary` in {low, medium, high}. Each entry includes `valid` (bool), `min`, `max`, and any `invalid_count`. |
| `generate_quality_report()` | — | `dict` | Aggregates all quality checks into a single dictionary with keys: `missing_values` (DataFrame), `data_types` (DataFrame), `duplicate_count` (int), `value_ranges` (dict), `total_rows` (int), `total_columns` (int). Calls all other check methods internally. |

---

### 4.2 ExploratoryAnalyzer

**Module:** `src/eda/exploratory_analyzer.py`

**Purpose:** Performs exploratory data analysis through statistical summaries and visualizations. Generates the plots required by the problem statement: correlation heatmap, distribution plots, and project count bar chart.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | The cleaned dataset to analyze |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Copy of the input DataFrame (never mutated) |
| `viz_utils` | `VisualizationUtils` | Instance of shared visualization utilities |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `plot_correlation_heatmap(save_path)` | `save_path: str = None` | `None` | Computes the Pearson correlation matrix of all numeric columns using `df.corr()` and renders it as a seaborn heatmap with annotated values. If `save_path` is provided, saves the figure to that path. |
| `plot_distribution(column, title, save_path)` | `column: str`, `title: str`, `save_path: str = None` | `None` | Plots a distribution (histogram + KDE) of the specified column using `sns.histplot` with `kde=True`. Sets the provided title. Raises `ValueError` if the column does not exist in the DataFrame. |
| `plot_all_distributions(save_path)` | `save_path: str = None` | `None` | Creates a figure with three subplots, one for each of: `satisfaction_level`, `last_evaluation`, `average_montly_hours`. Calls `plot_distribution` internally for each. If `save_path` is provided, saves the combined figure. |
| `plot_project_count_bar(save_path)` | `save_path: str = None` | `None` | Creates a count bar plot of `number_project` with `hue=left`, showing how project counts differ between employees who stayed (left=0) and those who left (left=1). Uses `sns.countplot`. |
| `compute_turnover_rate_by_feature(feature)` | `feature: str` | `pd.DataFrame` | Groups the data by the specified feature and computes the turnover rate (mean of `left`) for each group. Returns a DataFrame with the feature values as index and columns `count`, `left_count`, and `turnover_rate`. |
| `run_full_eda(output_dir)` | `output_dir: str = None` | `None` | Orchestrates all EDA steps in sequence: correlation heatmap, all distribution plots, project count bar chart. If `output_dir` is provided, saves all plots there. Prints summary statistics to stdout. |

---

### 4.3 EmployeeClusterer

**Module:** `src/clustering/employee_clusterer.py`

**Purpose:** Applies K-Means clustering (k=3) to employees who left the company, using satisfaction_level and last_evaluation as features. Identifies distinct behavioral archetypes among departing employees to inform targeted retention strategies.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Full dataset including both stayed and left employees |
| `n_clusters` | `int` | `3` | Number of K-Means clusters |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Copy of input DataFrame |
| `n_clusters` | `int` | Number of clusters (default 3) |
| `cluster_data` | `pd.DataFrame` | Filtered data (left==1) with selected features (populated after `prepare_cluster_data()`) |
| `kmeans_model` | `KMeans` | Fitted KMeans model (populated after `fit_kmeans()`) |
| `labels` | `np.ndarray` | Cluster assignment labels (populated after `fit_kmeans()`) |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `prepare_cluster_data()` | — | `pd.DataFrame` | Filters the DataFrame to rows where `left == 1`, then selects only `satisfaction_level` and `last_evaluation` columns. Stores result in `self.cluster_data` and returns it. Raises `ValueError` if no employees with `left==1` exist. |
| `fit_kmeans()` | — | `KMeans` | Fits a KMeans model with `n_clusters` clusters on `self.cluster_data`. Stores the fitted model in `self.kmeans_model` and the cluster labels in `self.labels`. Returns the fitted KMeans object. Must call `prepare_cluster_data()` first. |
| `plot_clusters(save_path)` | `save_path: str = None` | `None` | Creates a scatter plot of `satisfaction_level` vs `last_evaluation`, colored by cluster assignment. Marks cluster centroids with distinct markers. If `save_path` is provided, saves the figure. |
| `get_cluster_summary()` | — | `pd.DataFrame` | Returns a DataFrame indexed by cluster label (0, 1, 2) with columns: `count`, `mean_satisfaction`, `mean_evaluation`. Computed by grouping `cluster_data` by `self.labels`. |
| `interpret_clusters()` | — | `dict` | Returns a dictionary mapping cluster labels (0, 1, 2) to human-readable descriptions based on their mean satisfaction and evaluation scores. Example interpretations: "High performers with low satisfaction (burned out)", "Low performers with low satisfaction (disengaged)", "High performers with high satisfaction but overworked". |
| `run_clustering_pipeline()` | — | `dict` | End-to-end pipeline: calls `prepare_cluster_data()`, `fit_kmeans()`, `plot_clusters()`, `get_cluster_summary()`, and `interpret_clusters()`. Returns a dictionary with keys: `summary` (DataFrame), `interpretations` (dict), `model` (KMeans). |

---

### 4.4 DataPreprocessor

**Module:** `src/preprocessing/data_preprocessor.py`

**Purpose:** Prepares the dataset for model training: separates features from target, encodes categorical variables, performs stratified train-test split, and applies SMOTE to handle class imbalance.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | The cleaned dataset |
| `target_col` | `str` | `'left'` | Name of the target variable column |
| `test_size` | `float` | `0.2` | Fraction of data reserved for testing |
| `random_state` | `int` | `123` | Random seed for reproducibility |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Copy of input DataFrame |
| `target_col` | `str` | Target column name |
| `test_size` | `float` | Test set fraction |
| `random_state` | `int` | Random seed |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `separate_features_target()` | — | `Tuple[pd.DataFrame, pd.Series]` | Splits the DataFrame into feature matrix `X` (all columns except `target_col`) and target vector `y` (`target_col`). Returns `(X, y)`. Does not modify the original DataFrame. |
| `encode_categorical(X)` | `X: pd.DataFrame` | `pd.DataFrame` | Applies `pd.get_dummies()` to the `sales` and `salary` columns, converting them to binary indicator columns. Drops the original categorical columns. Returns a new DataFrame with all numeric columns. |
| `stratified_split(X, y)` | `X: pd.DataFrame`, `y: pd.Series` | `Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` | Performs an 80:20 stratified train-test split using `train_test_split` with `stratify=y` and `random_state=123`. Returns `(X_train, X_test, y_train, y_test)`. Stratification ensures the class distribution of `left` is preserved in both sets. |
| `apply_smote(X_train, y_train)` | `X_train: pd.DataFrame`, `y_train: pd.Series` | `Tuple[pd.DataFrame, pd.Series]` | Applies SMOTE oversampling to the training set, generating synthetic samples for the minority class (left=1) until both classes have equal counts. Returns `(X_train_resampled, y_train_resampled)`. Uses `random_state` for reproducibility. |
| `run_preprocessing_pipeline()` | — | `Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` | Full pipeline: `separate_features_target()` -> `encode_categorical()` -> `stratified_split()` -> `apply_smote()`. Returns `(X_train_smote, X_test, y_train_smote, y_test)`. Note: SMOTE is only applied to the training set; the test set remains untouched. |
| `get_class_distribution(y)` | `y: pd.Series` | `dict` | Returns a dictionary with class labels as keys and counts as values. Example: `{0: 9000, 1: 3000}`. Used to verify class balance before and after SMOTE. |

---

### 4.5 ModelTrainer

**Module:** `src/modeling/model_trainer.py`

**Purpose:** Trains three classification models (Logistic Regression, Random Forest, Gradient Boosting) with 5-fold cross-validation, generates cross-validation scores, and plots classification reports.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | `pd.DataFrame` | — | Training feature matrix (post-SMOTE) |
| `y_train` | `pd.Series` | — | Training target vector (post-SMOTE) |
| `cv_folds` | `int` | `5` | Number of cross-validation folds |
| `random_state` | `int` | `123` | Random seed for model initialization |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `X_train` | `pd.DataFrame` | Training features |
| `y_train` | `pd.Series` | Training target |
| `cv_folds` | `int` | Number of CV folds |
| `random_state` | `int` | Random seed |
| `models` | `dict` | Dictionary of trained models {name: fitted_model} |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `train_logistic_regression()` | — | `LogisticRegression` | Instantiates `LogisticRegression(random_state=self.random_state, max_iter=1000)`, fits it on `X_train`/`y_train`, evaluates with 5-fold cross-validation, stores in `self.models`, and returns the fitted model. |
| `train_random_forest()` | — | `RandomForestClassifier` | Instantiates `RandomForestClassifier(random_state=self.random_state, n_estimators=100)`, fits it on `X_train`/`y_train`, evaluates with 5-fold CV, stores in `self.models`, and returns the fitted model. |
| `train_gradient_boosting()` | — | `GradientBoostingClassifier` | Instantiates `GradientBoostingClassifier(random_state=self.random_state, n_estimators=100)`, fits it on `X_train`/`y_train`, evaluates with 5-fold CV, stores in `self.models`, and returns the fitted model. |
| `get_cv_scores(model, scoring)` | `model: object`, `scoring: str = 'f1'` | `np.ndarray` | Runs `cross_val_score(model, self.X_train, self.y_train, cv=self.cv_folds, scoring=scoring)` and returns the array of fold scores. Common scoring values: `'f1'`, `'recall'`, `'precision'`, `'roc_auc'`. |
| `plot_classification_report(model, X_test, y_test, model_name, save_path)` | `model: object`, `X_test: pd.DataFrame`, `y_test: pd.Series`, `model_name: str`, `save_path: str = None` | `None` | Generates predictions on the test set, computes `classification_report` as a dictionary, converts it to a DataFrame, and plots it as a seaborn heatmap. The heatmap shows precision, recall, F1-score for each class. If `save_path` is provided, saves the figure. |
| `train_all_models()` | — | `dict` | Calls `train_logistic_regression()`, `train_random_forest()`, and `train_gradient_boosting()` sequentially. Returns `self.models` dictionary: `{'Logistic Regression': model, 'Random Forest': model, 'Gradient Boosting': model}`. |

---

### 4.6 ModelEvaluator

**Module:** `src/modeling/model_evaluator.py`

**Purpose:** Evaluates trained models on the held-out test set. Computes ROC-AUC scores, plots ROC curves and confusion matrices, identifies the best model, and provides the recall-over-precision justification.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `dict` | — | Dictionary of trained models `{name: fitted_model}` |
| `X_test` | `pd.DataFrame` | — | Test feature matrix |
| `y_test` | `pd.Series` | — | Test target vector |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `models` | `dict` | Dictionary of trained models |
| `X_test` | `pd.DataFrame` | Test features |
| `y_test` | `pd.Series` | Test target |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `compute_roc_auc(model_name)` | `model_name: str` | `float` | Retrieves the model by name, computes `predict_proba` on `X_test`, and returns `roc_auc_score(y_test, probabilities[:, 1])`. Raises `KeyError` if `model_name` is not in `self.models`. |
| `plot_roc_curves(save_path)` | `save_path: str = None` | `None` | Iterates over all models, computes `roc_curve` for each, and plots them on a single figure with a diagonal reference line. Each curve is labeled with the model name and AUC score. Legend is displayed. If `save_path` is provided, saves the figure. |
| `compute_confusion_matrix(model_name)` | `model_name: str` | `np.ndarray` | Generates predictions using the named model and returns `confusion_matrix(y_test, predictions)` as a 2x2 numpy array. |
| `plot_confusion_matrix(model_name, save_path)` | `model_name: str`, `save_path: str = None` | `None` | Computes the confusion matrix for the named model and plots it as a seaborn heatmap with annotations. Labels axes as "Predicted" and "Actual" with class names "Stayed" and "Left". |
| `identify_best_model()` | — | `Tuple[str, object]` | Computes ROC-AUC for all models and returns `(name, model)` of the model with the highest AUC score. |
| `justify_recall_over_precision()` | — | `str` | Returns a text explanation of why recall is prioritized over precision in the employee turnover context. The explanation covers: (1) False negatives mean missing an at-risk employee who then leaves, incurring high replacement costs. (2) False positives mean offering retention incentives to a stable employee, which has minimal downside and may even improve morale. (3) The asymmetric cost structure makes recall the more business-critical metric. |
| `generate_evaluation_report()` | — | `pd.DataFrame` | Creates a summary DataFrame with one row per model and columns: `model_name`, `roc_auc`, `precision_class_0`, `recall_class_0`, `f1_class_0`, `precision_class_1`, `recall_class_1`, `f1_class_1`, `accuracy`. Provides a comprehensive comparison table for model selection. |

---

### 4.7 RetentionAdvisor

**Module:** `src/retention/retention_advisor.py`

**Purpose:** Uses the best-performing model to predict turnover probabilities for test-set employees, assigns them to risk zones, and generates targeted retention strategies for each zone.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `best_model` | `object` | — | The best trained classifier (from `ModelEvaluator.identify_best_model()`) |
| `X_test` | `pd.DataFrame` | — | Test feature matrix |
| `y_test` | `pd.Series` | — | Test target vector |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_model` | `object` | The best-performing trained model |
| `X_test` | `pd.DataFrame` | Test features |
| `y_test` | `pd.Series` | Test target |
| `probabilities` | `np.ndarray` | Predicted turnover probabilities (populated after prediction) |
| `risk_zones` | `pd.Series` | Categorized risk zones (populated after categorization) |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `predict_turnover_probabilities()` | — | `np.ndarray` | Calls `self.best_model.predict_proba(self.X_test)` and extracts the probability of class 1 (left). Stores the array in `self.probabilities` and returns it. |
| `categorize_risk_zones(probabilities)` | `probabilities: np.ndarray` | `pd.Series` | Maps each probability score to a risk zone: **Safe** (< 0.20), **Low-Risk** (0.20 - 0.60), **Medium-Risk** (0.60 - 0.90), **High-Risk** (> 0.90). Returns a pandas Series with zone labels. Stores in `self.risk_zones`. |
| `get_zone_counts()` | — | `dict` | Returns a dictionary with zone names as keys and employee counts as values. Example: `{'Safe': 1800, 'Low-Risk': 500, 'Medium-Risk': 350, 'High-Risk': 150}`. |
| `suggest_strategies(zone)` | `zone: str` | `str` | Returns a detailed retention strategy string for the specified zone. Raises `ValueError` if the zone is not one of Safe, Low-Risk, Medium-Risk, High-Risk. See Section 9 for strategy details. |
| `generate_retention_report()` | — | `pd.DataFrame` | Creates a comprehensive DataFrame with columns: `employee_index`, `turnover_probability`, `risk_zone`, `actual_outcome` (from y_test), and `recommended_strategy`. One row per test-set employee. |
| `plot_zone_distribution(save_path)` | `save_path: str = None` | `None` | Creates a bar chart showing the count of employees in each risk zone, ordered from Safe to High-Risk. Uses distinct colors per zone (green, yellow, orange, red). If `save_path` is provided, saves the figure. |

---

### 4.8 VisualizationUtils

**Module:** `src/utils/visualization_utils.py`

**Purpose:** Shared visualization helper class providing consistent styling, figure saving, and reusable plot creation methods. Used by ExploratoryAnalyzer, EmployeeClusterer, ModelTrainer, ModelEvaluator, and RetentionAdvisor.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `style` | `str` | `'seaborn-v0_8'` | Matplotlib style to apply |
| `figsize` | `Tuple[int, int]` | `(10, 6)` | Default figure size for all plots |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `style` | `str` | Matplotlib style name |
| `figsize` | `Tuple[int, int]` | Default figure dimensions |

**Methods:**

| Method | Parameters | Return Type | Purpose |
|--------|-----------|-------------|---------|
| `set_plot_style()` | — | `None` | Applies `plt.style.use(self.style)` and sets seaborn's color palette. Configures default font sizes for titles, labels, and ticks. Should be called once at initialization. |
| `save_figure(fig, filepath)` | `fig: plt.Figure`, `filepath: str` | `None` | Saves the figure to the specified file path using `fig.savefig()` with `dpi=150`, `bbox_inches='tight'`. Creates parent directories if they do not exist. Closes the figure after saving to free memory. |
| `create_heatmap(data, title, ax)` | `data: pd.DataFrame`, `title: str`, `ax: plt.Axes = None` | `plt.Axes` | Creates a seaborn heatmap with annotations, a diverging color map (`coolwarm`), and the specified title. If `ax` is None, creates a new figure. Returns the Axes object. |
| `create_distribution_plot(data, title, ax)` | `data: pd.Series`, `title: str`, `ax: plt.Axes = None` | `plt.Axes` | Creates a histogram with KDE overlay using `sns.histplot(kde=True)`. Sets the title and axis labels. If `ax` is None, creates a new figure. Returns the Axes object. |
| `create_bar_plot(data, x, hue, title, ax)` | `data: pd.DataFrame`, `x: str`, `hue: str`, `title: str`, `ax: plt.Axes = None` | `plt.Axes` | Creates a count bar plot using `sns.countplot` with the specified x-axis column and hue grouping. Sets the title and legend. If `ax` is None, creates a new figure. Returns the Axes object. |

---

## 5. ML Pipeline Architecture

The pipeline consists of seven sequential stages, each implemented as a dedicated class. Stages are loosely coupled: each accepts standard data structures (DataFrames, Series, arrays) and produces outputs consumable by the next stage.

### Stage 1: Data Quality Checks

**Class:** `DataQualityChecker`

**Input:** File path to `HR_comma_sep.csv`

**Process:**
1. Load the CSV file into a pandas DataFrame
2. Check for missing values across all columns (expect zero nulls)
3. Verify data types match expectations (floats for scores, ints for counts, objects for categoricals)
4. Count duplicate rows
5. Validate value ranges (satisfaction in [0,1], left in {0,1}, salary in {low, medium, high})

**Output:** Quality report dictionary and the loaded DataFrame

**Interaction:** The loaded DataFrame is passed to all subsequent stages.

### Stage 2: Exploratory Data Analysis

**Class:** `ExploratoryAnalyzer`

**Input:** Cleaned DataFrame from Stage 1

**Process:**
1. Compute Pearson correlation matrix of all numeric features
2. Render correlation heatmap (identifies features most correlated with `left`)
3. Plot distributions of `satisfaction_level`, `last_evaluation`, `average_montly_hours`
4. Plot bar chart of `number_project` grouped by `left` status
5. Compute turnover rates by key features (department, salary, project count)

**Output:** Saved visualization files in `outputs/plots/`; printed summary statistics

**Interaction:** Informs feature selection and provides context for model interpretation.

### Stage 3: Clustering

**Class:** `EmployeeClusterer`

**Input:** DataFrame from Stage 1 (uses only rows where `left==1`)

**Process:**
1. Filter to employees who left
2. Select `satisfaction_level` and `last_evaluation` as clustering features
3. Fit KMeans with k=3
4. Assign cluster labels
5. Generate scatter plot colored by cluster
6. Compute cluster summaries (mean satisfaction, mean evaluation per cluster)
7. Interpret clusters into human-readable archetypes

**Output:** Cluster model, labeled data, cluster interpretations, scatter plot

**Interaction:** Cluster interpretations inform the retention strategies in Stage 7.

### Stage 4: Preprocessing and Class Imbalance Handling

**Class:** `DataPreprocessor`

**Input:** DataFrame from Stage 1

**Process:**
1. Separate features (X) from target (y = `left`)
2. Encode categorical columns `sales` and `salary` using `pd.get_dummies`, producing binary indicator columns
3. Perform 80:20 stratified train-test split with `random_state=123`
4. Apply SMOTE to the training set only, upsampling the minority class (left=1)

**Output:** `X_train_smote`, `X_test`, `y_train_smote`, `y_test`

**Interaction:** Training data goes to ModelTrainer (Stage 5); test data goes to ModelEvaluator (Stage 6) and RetentionAdvisor (Stage 7).

### Stage 5: Model Training

**Class:** `ModelTrainer`

**Input:** `X_train_smote`, `y_train_smote` from Stage 4

**Process:**
1. Train Logistic Regression with 5-fold cross-validation
2. Train Random Forest Classifier with 5-fold cross-validation
3. Train Gradient Boosting Classifier with 5-fold cross-validation
4. Compute cross-validation scores (F1, recall, precision) for each model
5. Generate classification report heatmaps for each model on the test set

**Output:** Dictionary of three trained models `{name: fitted_model}`; classification report visualizations

**Interaction:** The models dictionary is passed to ModelEvaluator (Stage 6).

### Stage 6: Model Evaluation

**Class:** `ModelEvaluator`

**Input:** Models dictionary from Stage 5; `X_test`, `y_test` from Stage 4

**Process:**
1. Compute ROC-AUC score for each model
2. Plot overlay ROC curves with AUC annotations
3. Compute and plot confusion matrices for each model
4. Generate comprehensive evaluation report table
5. Identify the best model by highest AUC
6. Provide recall-over-precision justification

**Output:** Evaluation report DataFrame, ROC curve plot, confusion matrix plots, best model identification

**Interaction:** The best model is passed to RetentionAdvisor (Stage 7).

### Stage 7: Retention Strategies

**Class:** `RetentionAdvisor`

**Input:** Best model from Stage 6; `X_test`, `y_test` from Stage 4

**Process:**
1. Predict turnover probabilities using `predict_proba`
2. Categorize employees into four risk zones based on probability thresholds
3. Count employees per zone
4. Generate targeted retention strategies for each zone
5. Compile full retention report with employee indices, probabilities, zones, and strategies
6. Plot zone distribution bar chart

**Output:** Retention report DataFrame, zone distribution visualization

---

## 6. Coding Standards

### Style and Formatting

- **PEP 8 compliance:** All code follows PEP 8 style guidelines. Use a linter (flake8 or ruff) to enforce.
- **Line length:** Maximum 100 characters per line.
- **Imports:** Group in order: standard library, third-party, local. One import per line. Absolute imports only.

### Type Hints

All method signatures must include type hints for parameters and return values:

```
def method_name(self, param_a: str, param_b: int = 5) -> pd.DataFrame:
```

Use `Tuple`, `Dict`, `List`, `Optional` from `typing` module where needed.

### Docstrings

Use Google-style docstrings on all public classes and methods:

```
def method_name(self, param: str) -> pd.DataFrame:
    """Short description of what this method does.

    Args:
        param: Description of the parameter.

    Returns:
        Description of what is returned.

    Raises:
        ValueError: When the parameter is invalid.
    """
```

### Immutability

- Never mutate input DataFrames, Series, or arrays
- Always return new objects using `.copy()`, `.assign()`, or constructor calls
- Store intermediate results in new variables, not by overwriting inputs

### Constants

Define module-level constants in ALL_CAPS:

```
RANDOM_STATE = 123
TEST_SIZE = 0.2
N_CLUSTERS = 3
CV_FOLDS = 5
RISK_ZONE_THRESHOLDS = {
    'Safe': (0.0, 0.2),
    'Low-Risk': (0.2, 0.6),
    'Medium-Risk': (0.6, 0.9),
    'High-Risk': (0.9, 1.0),
}
```

### File Length

- Maximum 400 lines per module
- If a file approaches this limit, extract helper functions or utility classes into separate modules

### Error Handling

- Validate inputs at method entry points (system boundaries)
- Raise `ValueError` with clear, descriptive messages
- Raise `FileNotFoundError` for missing files
- Raise `KeyError` for invalid model names or feature names
- Never silently swallow exceptions

### Path Handling

- Use `pathlib.Path` for all file operations
- Accept paths as parameters; never hardcode absolute paths
- Create parent directories with `Path.mkdir(parents=True, exist_ok=True)` before writing

### Visualization Standards

- Always call `plt.tight_layout()` before saving or displaying
- Always close figures after saving with `plt.close(fig)`
- Use consistent figure sizes via `VisualizationUtils.figsize`
- Use descriptive titles and axis labels on every plot
- Include legends where multiple series are plotted

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Variables, functions | snake_case | `turnover_rate`, `compute_roc_auc` |
| Classes | PascalCase | `DataQualityChecker`, `ModelTrainer` |
| Constants | UPPER_CASE | `RANDOM_STATE`, `CV_FOLDS` |
| Module files | snake_case | `data_quality_checker.py` |
| Test files | test_ prefix | `test_data_quality.py` |

### State Management

- No global mutable state
- All state belongs to class instances
- Configuration values are passed via constructor parameters or module-level constants

### Testing

- Unit tests required for every public method
- Use pytest framework with fixtures for shared test data
- Target 80% code coverage minimum
- Test file naming: `test_<module_name>.py`
- Use parametrize for testing multiple inputs
- Mock external dependencies (file I/O, model fitting) where appropriate

---

## 7. Data Flow Diagram

The following describes the data flow through the pipeline as a textual diagram:

```
                              HR_comma_sep.csv
                                     |
                                     v
                         +-------------------------+
                         |   DataQualityChecker    |
                         |   - load_data()         |
                         |   - check_missing()     |
                         |   - check_types()       |
                         |   - check_duplicates()  |
                         |   - check_ranges()      |
                         +-------------------------+
                                     |
                              pd.DataFrame (raw)
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
        +---------------------+          +------------------------+
        | ExploratoryAnalyzer |          |  EmployeeClusterer     |
        | - heatmap           |          |  - filter left==1      |
        | - distributions     |          |  - KMeans k=3          |
        | - bar plots         |          |  - cluster scatter     |
        | - turnover rates    |          |  - interpret clusters  |
        +---------------------+          +------------------------+
                    |                                 |
              plots saved                    cluster insights
              to outputs/plots/              (inform Stage 7)
                    |                                 |
                    +----------------+----------------+
                                     |
                              pd.DataFrame (raw)
                                     |
                                     v
                         +-------------------------+
                         |   DataPreprocessor      |
                         |   - separate X, y       |
                         |   - get_dummies()       |
                         |   - stratified split    |
                         |   - SMOTE (train only)  |
                         +-------------------------+
                                     |
                    +----------------+----------------+
                    |                                 |
         X_train_smote, y_train_smote         X_test, y_test
                    |                                 |
                    v                                 |
         +---------------------+                     |
         |   ModelTrainer      |                     |
         |   - LogReg + 5-CV   |                     |
         |   - RF + 5-CV       |                     |
         |   - GBC + 5-CV      |                     |
         |   - classif reports |                     |
         +---------------------+                     |
                    |                                 |
            dict {name: model}                        |
                    |                                 |
                    v                                 v
         +-----------------------------------------------+
         |            ModelEvaluator                      |
         |   - ROC-AUC per model                         |
         |   - ROC curves overlay                        |
         |   - confusion matrices                        |
         |   - identify best model                       |
         |   - recall > precision justification          |
         +-----------------------------------------------+
                    |
             best model + X_test + y_test
                    |
                    v
         +---------------------+
         |  RetentionAdvisor   |
         |  - predict_proba    |
         |  - risk zones       |
         |  - zone counts      |
         |  - strategies       |
         |  - retention report |
         +---------------------+
                    |
                    v
         Retention Report (pd.DataFrame)
         Zone Distribution (bar chart)
```

### Data Transformations Summary

| Stage | Input Shape | Output Shape | Key Transformation |
|-------|-------------|--------------|-------------------|
| 1. Quality | CSV file | (14999, 10) DataFrame | File I/O |
| 2. EDA | (14999, 10) | Plots (no shape change) | Statistical analysis |
| 3. Clustering | (3571, 2) subset | (3571, 2) + labels | KMeans assignment |
| 4. Preprocessing | (14999, 10) | Train: (~12000+, 20+), Test: (~3000, 20+) | Encoding + split + SMOTE |
| 5. Training | Train set | 3 fitted models | Model fitting |
| 6. Evaluation | Models + test set | Metrics + plots | Prediction + scoring |
| 7. Retention | Best model + test set | Report DataFrame | Probability + categorization |

---

## 8. Evaluation Metrics Justification

### Why Recall is More Important than Precision

In the context of employee turnover prediction, the cost of errors is asymmetric:

**False Negative (Type II Error): Predicting an employee will stay when they actually leave**
- The employee departs without any retention intervention
- The company incurs full replacement cost (50-200% of annual salary)
- Team disruption, knowledge loss, and project delays occur
- This is the **high-cost error** that must be minimized

**False Positive (Type I Error): Predicting an employee will leave when they actually stay**
- HR offers retention incentives (raise, promotion, flexible hours) to a stable employee
- The employee receives a positive gesture that may improve satisfaction and loyalty
- Cost is limited to the retention incentive itself, which is far less than a replacement
- This is the **low-cost error** that is acceptable

**Conclusion:** Because false negatives are far more costly than false positives, **recall** (the ability to correctly identify employees who will leave) is the priority metric. A model with high recall and moderate precision is preferred over one with high precision and moderate recall.

### ROC-AUC as Primary Ranking Metric

ROC-AUC is chosen as the primary metric for comparing models because:

1. **Threshold-independent:** AUC evaluates the model's ability to rank positive cases higher than negative cases across all possible classification thresholds, not just the default 0.5.
2. **Handles class imbalance:** Unlike accuracy, AUC is not inflated by the majority class. An AUC of 0.50 indicates random guessing regardless of class distribution.
3. **Interpretable:** AUC represents the probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative example.
4. **Enables fair comparison:** All three models can be compared on the same scale, and the ROC curve visualization shows performance trade-offs at every threshold.

### Confusion Matrix Interpretation for HR

For the HR use case, the 2x2 confusion matrix maps to business outcomes:

| | Predicted: Stay | Predicted: Leave |
|---|---|---|
| **Actual: Stay** | True Negative: Correctly identified stable employee. No action needed. | False Positive: Stable employee flagged for retention. Low cost: offer incentives. |
| **Actual: Leave** | False Negative: At-risk employee missed. High cost: full replacement. | True Positive: At-risk employee correctly identified. Retention intervention triggered. |

The goal is to maximize the bottom-right cell (True Positives) and minimize the bottom-left cell (False Negatives), which corresponds to maximizing recall.

---

## 9. Retention Strategy Framework

### Risk Zone Definitions

| Zone | Probability Range | Color Code | Priority |
|------|------------------|------------|----------|
| **Safe** | < 20% | Green | Low — Monitor |
| **Low-Risk** | 20% - 60% | Yellow | Medium — Engage |
| **Medium-Risk** | 60% - 90% | Orange | High — Intervene |
| **High-Risk** | > 90% | Red | Critical — Immediate Action |

### Zone-Specific Strategies

#### Safe Zone (< 20% turnover probability)

**Assessment:** These employees show strong indicators of engagement and satisfaction. They are unlikely to leave in the near term.

**Recommended Actions:**
- Continue regular check-ins at standard cadence (quarterly)
- Recognize contributions through existing recognition programs
- Ensure career development plans are up to date
- Include in mentorship programs as mentors for newer employees
- Monitor for sudden changes in satisfaction or workload indicators

#### Low-Risk Zone (20% - 60% turnover probability)

**Assessment:** These employees show some early warning signs. They may have slightly reduced satisfaction, high workloads, or stagnant career progression. Proactive engagement can prevent escalation.

**Recommended Actions:**
- Schedule one-on-one meetings with direct managers within 2 weeks
- Review current workload and project assignments for balance
- Discuss career aspirations and create or update individual development plans
- Consider skill-building opportunities (training, conferences, stretch assignments)
- Evaluate compensation competitiveness against market rates
- Assess team dynamics and address any interpersonal friction
- Increase check-in frequency to monthly

#### Medium-Risk Zone (60% - 90% turnover probability)

**Assessment:** These employees are showing significant disengagement signals. Without intervention, departure is likely within the next quarter. Immediate, targeted action is required.

**Recommended Actions:**
- Arrange skip-level meeting with senior leadership within 1 week
- Conduct a stay interview to identify specific pain points and unmet needs
- Offer concrete retention incentives: salary adjustment, role change, or promotion path
- Review and reduce excessive project loads or overtime hours
- Explore lateral moves to teams or projects aligned with the employee's interests
- Assign a senior mentor or sponsor for career advocacy
- If satisfaction is low and evaluation is high, investigate burnout and consider sabbatical or reduced hours
- Provide access to employee assistance programs (EAP) if personal factors are suspected
- Set bi-weekly follow-up check-ins to track improvement

#### High-Risk Zone (> 90% turnover probability)

**Assessment:** These employees are very likely to leave imminently. They may already be disengaged or actively searching for other opportunities. Maximum-intensity retention efforts are warranted, along with contingency planning.

**Recommended Actions:**
- Immediate meeting with HR Business Partner and direct manager within 48 hours
- Direct conversation about the employee's concerns, needs, and future plans
- Present a compelling counter-offer if the employee is a high performer: significant raise, promotion, role redesign, flexible work arrangement
- Fast-track any pending promotions or role changes
- Address root causes identified in the conversation (e.g., toxic manager, lack of growth, compensation gap)
- If the employee has decided to leave, begin knowledge transfer planning immediately
- Initiate succession planning and identify potential internal or external replacements
- Conduct an exit interview (or pre-exit interview) to capture feedback
- Analyze whether the factors driving this employee's departure affect others in the same team or department
- Document lessons learned for systemic HR improvements

### Cross-Zone Considerations

- **Cluster-Informed Strategies:** Use insights from Stage 3 (clustering) to tailor strategies. For example, if a high-risk employee falls into the "burned-out high performer" cluster, prioritize workload reduction and recognition over compensation changes.
- **Department-Level Patterns:** If a department shows disproportionate medium/high-risk employees, investigate systemic issues (management quality, workload distribution, growth opportunities).
- **Periodic Reassessment:** Risk zones should be recomputed periodically (monthly or quarterly) as employee circumstances change. An employee in the safe zone today could shift to low-risk if conditions change.
- **Cost-Benefit Analysis:** For high-cost retention interventions (significant raises, promotions), compare the cost against the estimated replacement cost for that role and seniority level.

---

## 10. Logging Architecture

### Design Principles

The logging system follows Python's standard `logging` module hierarchy with two key principles:
1. **Structured output** — every log line includes timestamp, level, module name, and message
2. **Dual destination** — simultaneous file logging (DEBUG+) and console logging (INFO+)

### Logger Configuration (`src/utils/logger_config.py`)

| Setting | Value |
|---------|-------|
| File handler level | DEBUG |
| Console handler level | INFO |
| Log format | `%(asctime)s \| %(levelname)-8s \| %(name)s \| %(message)s` |
| Date format | `%Y-%m-%d %H:%M:%S` |
| Log file name | `employee_turnover_YYYYMMDD.log` |
| Log file location | `outputs\logs\` |
| Rotation | RotatingFileHandler — 10 MB max, 5 backups |
| Encoding | UTF-8 (Windows compatible) |

### Logger Hierarchy

Each module obtains its own named logger using `logging.getLogger(__name__)`:

```
root
└── src
    ├── src.data_quality.data_quality_checker
    ├── src.eda.exploratory_analyzer
    ├── src.clustering.employee_clusterer
    ├── src.preprocessing.data_preprocessor
    ├── src.modeling.model_trainer
    ├── src.modeling.model_evaluator
    ├── src.retention.retention_advisor
    └── src.utils.visualization_utils
```

### Logging Conventions per Level

| Level | When to Use | Example |
|-------|-------------|---------|
| DEBUG | Method entry, intermediate values | `logger.debug("check_missing_values() called on %d rows", len(self.df))` |
| INFO | Key milestones, results, metrics | `logger.info("Data loaded: %d rows × %d columns", rows, cols)` |
| WARNING | Recoverable anomalies | `logger.warning("Found %d duplicate rows", n)` |
| ERROR | Caught exceptions before re-raise | `logger.error("Failed to load data: %s", exc)` |
| CRITICAL | Unrecoverable pipeline failure | `logger.critical("Pipeline aborted: %s", exc, exc_info=True)` |

### Log File Management (Windows)

Log files are written to `outputs\logs\employee_turnover_YYYYMMDD.log`. The
`RotatingFileHandler` ensures files do not exceed 10 MB, retaining 5 backup
files (`.log.1` through `.log.5`). UTF-8 encoding ensures Windows compatibility.

**View logs (PowerShell):**
```powershell
Get-Content outputs\logs\employee_turnover_20260314.log -Tail 50
Select-String "ERROR|CRITICAL" outputs\logs\employee_turnover_20260314.log
```

---

## 11. Exception Handling Architecture

### Exception Hierarchy (`src/utils/exceptions.py`)

```
EmployeeTurnoverError          # Base — catch all project errors
├── DataLoadError              # File not found, empty, unparseable CSV
├── DataQualityError           # Validation failures, check before load
├── PreprocessingError         # Encoding, split, or SMOTE failures
├── ModelTrainingError         # Model fit or cross-validation failures
├── ModelEvaluationError       # Prediction, metrics, or best-model failures
├── ClusteringError            # KMeans failures, empty leaver subset
└── VisualizationError         # Matplotlib/Seaborn rendering or save failures
```

### Standard Exception Handling Pattern

Every public method follows this pattern:

```python
def method_name(self, param: type) -> ReturnType:
    logger.debug("method_name() called with param=%s", param)
    try:
        # ... implementation
        logger.info("method_name() completed successfully")
        return result
    except SpecificKnownError as exc:
        logger.error("method_name() failed: %s", exc)
        raise DomainException(f"Descriptive message: {exc}") from exc
    except Exception as exc:
        logger.exception("method_name() encountered unexpected error")
        raise DomainException(f"Unexpected error in method_name: {exc}") from exc
```

Key rules:
- Use `logger.exception()` (not `logger.error()`) when you need the full traceback in the log
- Always chain exceptions with `raise X from original_exc` to preserve root cause
- Never use bare `except: pass` — always log and re-raise

### Exception-to-Module Mapping

| Module | Raises | Wraps |
|--------|--------|-------|
| `data_quality_checker.py` | `DataLoadError`, `DataQualityError` | `FileNotFoundError`, `EmptyDataError`, `ValueError` |
| `exploratory_analyzer.py` | `VisualizationError` | `KeyError`, `ValueError`, matplotlib errors |
| `employee_clusterer.py` | `ClusteringError`, `VisualizationError` | `ValueError` (empty data), sklearn errors |
| `data_preprocessor.py` | `PreprocessingError` | `ValueError`, imblearn errors |
| `model_trainer.py` | `ModelTrainingError`, `VisualizationError` | sklearn fit errors |
| `model_evaluator.py` | `ModelEvaluationError`, `VisualizationError` | sklearn predict errors |
| `retention_advisor.py` | `ModelEvaluationError`, `VisualizationError` | `AttributeError`, `ValueError` |

### Pipeline-Level Error Handling (`src/pipeline.py`)

```python
try:
    main()
except EmployeeTurnoverError as exc:
    logger.critical("Pipeline failed: %s", exc, exc_info=True)
    sys.exit(1)
except KeyboardInterrupt:
    logger.warning("Pipeline interrupted by user")
    sys.exit(0)
except Exception as exc:
    logger.critical("Unexpected failure: %s", exc, exc_info=True)
    sys.exit(2)
```

**Exit codes:** 0 = success, 1 = domain error, 2 = unexpected error

---

## 12. Notebook Architecture

This section documents the two-tier notebook system: the **interactive analysis
notebook** in `notebooks/` and the **module-level reference notebooks** co-located
with every Python source file in `src/`.

---

### 12.1 Overview

The project exposes all pipeline logic through two complementary notebook layers:

| Layer | Location | Count | Purpose |
|-------|----------|-------|---------|
| Interactive pipeline notebook | `notebooks/` | 1 | End-to-end walkthrough; runs the full 7-step ML pipeline with inline outputs |
| Module reference notebooks | `src/**/*.ipynb` | 19 | One notebook per `.py` file; documents every class and method with Javadoc-style cells |

Both layers share the same Jupyter kernel (`employee-turnover-venv`) and are kept
in sync with their corresponding Python source files.

---

### 12.2 Jupyter Kernel

A dedicated kernel is registered so that all notebooks resolve imports from the
project's virtual environment without manual path manipulation.

| Property | Value |
|----------|-------|
| Kernel name | `employee-turnover-venv` |
| Display name | `Employee Turnover (venv)` |
| Registration path | `C:\Users\<user>\AppData\Roaming\jupyter\kernels\employee-turnover-venv\` |
| Python binary | `venv\Scripts\python.exe` |

**Register the kernel (run once after creating the venv):**

```bat
venv\Scripts\python.exe -m ipykernel install --user ^
    --name employee-turnover-venv ^
    --display-name "Employee Turnover (venv)"
```

**Verify registration:**

```bat
venv\Scripts\python.exe -m jupyter kernelspec list
```

---

### 12.3 Interactive Pipeline Notebook

**File:** `notebooks/employee_turnover_analysis.ipynb`

This is the primary deliverable notebook. It executes the complete 7-step ML
pipeline interactively, displaying all outputs (plots, tables, metrics) inline.

#### Cell Structure

The notebook uses a three-layer cell hierarchy:

```
[Step header markdown]      ← Section title, context, and expected outputs
    [Cell documentation]    ← Per-cell markdown: Purpose, Inputs, Outputs, Step-by-step
    [Code cell]             ← Executable implementation
```

After enhancement, the notebook contains **67 cells** broken down as:

| Cell type | Count | Role |
|-----------|-------|------|
| Step header markdown | 7 | One per pipeline stage (## Step N headings) |
| Cell documentation markdown | 29 | Javadoc-style documentation before every code cell |
| Code cells | 30 | Executable pipeline implementation |
| Summary markdown | 1 | Key findings and conclusions |

#### Cell Documentation Standard

Every code cell is preceded by a markdown cell with the following sections:

| Section | Content |
|---------|---------|
| **Purpose** | Plain-English explanation of why this cell exists |
| **What this cell does — step by step** | Numbered walkthrough of each action in layman terms |
| **Inputs table** | Variable name, type, and description of every input consumed |
| **Outputs / Variables created table** | Variable name, type, and description of every output produced |
| **Key insight / Expected outcome** | What to look for in the cell output |

Additional context sections are added where applicable:

- **Model training cells** — rationale for each algorithm choice
- **Confusion matrix cell** — labelled 2×2 table explaining TN / FP / FN / TP
- **Evaluation report cell** — metric formula definitions in plain English
- **Risk zone cell** — threshold boundaries and HR priority per zone
- **Retention strategies cell** — intervention playbook summary table

#### Notebook Cell Map

| Cell range | Pipeline step | Key variables produced |
|-----------|--------------|----------------------|
| 00 | Overview markdown | — |
| 01–02 | Environment setup & imports | `df`, `logger`, `DATA_PATH`, `OUTPUT_DIR` |
| 03–13 | Step 1–2: Data quality + EDA | `checker`, `df`, `analyzer`, `missing`, `ranges` |
| 14–24 | Step 3–4: Clustering + preprocessing | `clusterer`, `results`, `X_train`, `X_test`, `y_train`, `y_test` |
| 25–41 | Step 5: Model training | `trainer`, `models` + 3 `.joblib` files |
| 42–54 | Step 6: Model evaluation | `evaluator`, `eval_report`, `best_model`, `best_name` |
| 55–65 | Step 7: Retention strategies | `advisor`, `retention_report`, `probs` + CSV |
| 66 | Summary & conclusions | — |

---

### 12.4 Module Reference Notebooks (`src/**/*.ipynb`)

One `.ipynb` file is generated alongside every `.py` file in `src/`. These
notebooks serve as interactive class and method reference documentation — the
equivalent of Javadoc rendered in a readable, runnable format.

#### Full Notebook Inventory

| # | Notebook | Folder | Pipeline role |
|---|----------|--------|---------------|
| 1 | `pipeline.ipynb` | `src/` | End-to-end CLI orchestrator |
| 2 | `__init__.ipynb` | `src/` | Package init — version metadata |
| 3 | `data_quality_checker.ipynb` | `src/data_quality/` | Stage 1 — data validation |
| 4 | `__init__.ipynb` | `src/data_quality/` | Package init |
| 5 | `exploratory_analyzer.ipynb` | `src/eda/` | Stage 2 — EDA visualisations |
| 6 | `__init__.ipynb` | `src/eda/` | Package init |
| 7 | `employee_clusterer.ipynb` | `src/clustering/` | Stage 3 — K-Means clustering |
| 8 | `__init__.ipynb` | `src/clustering/` | Package init |
| 9 | `data_preprocessor.ipynb` | `src/preprocessing/` | Stage 4 — encoding, split, SMOTE |
| 10 | `__init__.ipynb` | `src/preprocessing/` | Package init |
| 11 | `model_trainer.ipynb` | `src/modeling/` | Stage 5 — model training |
| 12 | `model_evaluator.ipynb` | `src/modeling/` | Stage 6 — evaluation & best model |
| 13 | `__init__.ipynb` | `src/modeling/` | Package init |
| 14 | `retention_advisor.ipynb` | `src/retention/` | Stage 7 — risk zones & strategies |
| 15 | `__init__.ipynb` | `src/retention/` | Package init |
| 16 | `exceptions.ipynb` | `src/utils/` | Custom exception hierarchy |
| 17 | `logger_config.ipynb` | `src/utils/` | Centralised logging setup |
| 18 | `visualization_utils.ipynb` | `src/utils/` | Shared matplotlib/seaborn helpers |
| 19 | `__init__.ipynb` | `src/utils/` | Package init |

#### Cell Structure per Module Notebook

Each module notebook follows this structure:

```
[Module title + file path markdown]
[Imports & Module-Level Constants markdown + code cell]
    [Class markdown]            ← Rich class-level documentation
    [Class header code cell]    ← class Foo: signature + __init__
        [Method markdown]       ← Per-method Javadoc cell
        [Method code cell]      ← def method(): implementation
        ... (repeated for each method)
```

For `exceptions.py`, each exception class gets its own markdown + code cell pair
since exception classes have no public methods.

For `pipeline.py`, each top-level function gets its own markdown + code cell pair
since there is no class wrapper.

#### Class-Level Documentation Standard

Every class markdown cell includes:

| Section | Content |
|---------|---------|
| **Class name heading** | `## Class: \`ClassName\`` |
| **Purpose description** | What the class is responsible for in the pipeline |
| **Constructor Arguments table** | Argument name, type, description |
| **Instance Attributes table** | Attribute name, type, what state it holds |
| **Public Methods at a Glance table** | Every public method with a one-line summary |
| **Usage Example** | `>>> obj = ClassName(...)` code snippet from docstring |

#### Method-Level Documentation Standard

Every method markdown cell includes:

| Section | Content |
|---------|---------|
| **Method heading** | `### \`ClassName.\`\`method_name()\`` |
| **Purpose** | What the method does in plain English |
| **Arguments table** | Parameter name, type, full description |
| **Returns** | What the method gives back (type + description) |
| **Raises** | Exception class and the condition that triggers it |
| **Example** | Code sample (where present in the docstring) |

---

### 12.5 Notebook Generation and Maintenance

The module reference notebooks are generated programmatically from the Python
source files using AST (Abstract Syntax Tree) parsing. This ensures the notebook
documentation is always structurally consistent with the source code.

#### Generation Process

1. The AST parser traverses each `.py` file to find module-level items:
   imports, constants, class definitions, and top-level functions.
2. For each class, the `__init__` docstring is parsed using a Google-style
   docstring parser to extract `Args:`, `Attributes:`, `Returns:`, `Raises:`,
   and `Example:` sections.
3. The class header (signature up to the first method) is placed in a code cell
   immediately following the class markdown cell.
4. Each method body is placed in its own code cell, preceded by its method
   markdown cell.
5. The notebook JSON is written with kernel metadata pointing to
   `employee-turnover-venv`.

#### Keeping Notebooks in Sync

When a `.py` source file is modified (new method added, docstring updated,
class restructured), re-run the generation script to regenerate all notebooks:

```bat
venv\Scripts\activate.bat
python _enhance_notebooks.py
```

> The generator is idempotent — regenerating overwrites existing notebooks
> without duplicating cells.

#### Git Tracking Policy

Module reference notebooks (`.ipynb` files in `src/`) are **committed to git**
because they serve as living documentation. They follow the same commit
conventions as source files:

```
docs: regenerate src notebooks after DataPreprocessor refactor
feat: add src/modeling/model_evaluator.ipynb for new evaluation methods
```

The main notebook (`notebooks/employee_turnover_analysis.ipynb`) is also
committed to git as it is a primary project deliverable.

**Not committed:** Notebook checkpoint files (`.ipynb_checkpoints/`) are
excluded via `.gitignore`.
