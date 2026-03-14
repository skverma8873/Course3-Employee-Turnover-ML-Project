"""
Custom exception hierarchy for the Employee Turnover Analytics project.

All project-specific exceptions inherit from EmployeeTurnoverError,
allowing callers to catch domain errors predictably while still
distinguishing between specific failure modes.
"""


class EmployeeTurnoverError(Exception):
    """Base exception for all Employee Turnover Analytics errors.

    Args:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}] {self.message}"


class DataLoadError(EmployeeTurnoverError):
    """Raised when the dataset cannot be loaded.

    Typical causes:
    - File not found at the specified path
    - File is empty or cannot be parsed as CSV
    - Insufficient read permissions (Windows)
    """


class DataQualityError(EmployeeTurnoverError):
    """Raised when data fails quality validation checks.

    Typical causes:
    - Unexpected column names or missing required columns
    - Values outside expected ranges (e.g., satisfaction_level > 1)
    - DataFrame is None or empty when a check is attempted before load_data()
    """


class PreprocessingError(EmployeeTurnoverError):
    """Raised when data preprocessing fails.

    Typical causes:
    - Categorical encoding errors (unexpected category values)
    - SMOTE failure (insufficient minority class samples)
    - Train-test split produces empty partitions
    """


class ModelTrainingError(EmployeeTurnoverError):
    """Raised when model training or cross-validation fails.

    Typical causes:
    - Estimator fit errors (e.g., convergence failure)
    - Invalid cross-validation parameters
    - Feature matrix/target vector shape mismatch
    """


class ModelEvaluationError(EmployeeTurnoverError):
    """Raised when model evaluation or prediction fails.

    Typical causes:
    - Model has not been trained (predict called before fit)
    - predict_proba not available on the given estimator
    - Mismatched feature columns between train and test
    """


class ClusteringError(EmployeeTurnoverError):
    """Raised when K-Means clustering fails.

    Typical causes:
    - No employees with left==1 found in the dataset
    - n_clusters exceeds number of available samples
    - Clustering performed before prepare_cluster_data() is called
    """


class VisualizationError(EmployeeTurnoverError):
    """Raised when a plot cannot be created or saved.

    Typical causes:
    - Output directory is not writable (Windows permissions)
    - Invalid column name passed to a plot method
    - Matplotlib/Seaborn rendering errors
    """
