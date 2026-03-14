"""
Data preprocessing and class-imbalance handling.

Step 4 of the ML pipeline:
- Encodes categorical features (sales, salary) using pd.get_dummies
- Stratified 80:20 train-test split (random_state=123)
- SMOTE oversampling of the minority class (left=1) in the training set
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.utils.exceptions import PreprocessingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 123
TEST_SIZE: float = 0.2
TARGET_COL: str = "left"
CATEGORICAL_COLS: List[str] = ["sales", "salary"]


class DataPreprocessor:
    """Encodes features, splits data, and applies SMOTE oversampling.

    All transformations return new DataFrames — the original ``df`` passed
    to the constructor is never mutated.

    Args:
        df: Cleaned HR DataFrame containing all original columns.
        target_col: Name of the binary target column.  Defaults to ``'left'``.
        test_size: Fraction of data for the test set.  Defaults to 0.20.
        random_state: Seed for reproducibility.  Defaults to 123.

    Example:
        >>> preprocessor = DataPreprocessor(df)
        >>> X_train, X_test, y_train, y_test = preprocessor.run_preprocessing_pipeline()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COL,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.df: pd.DataFrame = df.copy()
        self.target_col: str = target_col
        self.test_size: float = test_size
        self.random_state: int = random_state
        logger.debug(
            "DataPreprocessor initialised — rows=%d, target=%s, "
            "test_size=%.2f, random_state=%d",
            len(df),
            target_col,
            test_size,
            random_state,
        )

    # ------------------------------------------------------------------
    # Feature / target separation
    # ------------------------------------------------------------------

    def separate_features_target(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split the DataFrame into feature matrix X and target vector y.

        Returns:
            Tuple ``(X, y)`` where X is all columns except ``target_col``
            and y is the target column as a Series.

        Raises:
            PreprocessingError: If ``target_col`` is not in the DataFrame.
        """
        logger.debug("separate_features_target() called")
        try:
            if self.target_col not in self.df.columns:
                raise ValueError(
                    f"Target column '{self.target_col}' not found in DataFrame."
                )
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col].copy()
            logger.info(
                "Features/target separated — X shape=%s, y shape=%s, "
                "class distribution: {0: %d, 1: %d}",
                X.shape,
                y.shape,
                (y == 0).sum(),
                (y == 1).sum(),
            )
            return X, y
        except ValueError as exc:
            logger.error("separate_features_target() failed: %s", exc)
            raise PreprocessingError(str(exc)) from exc
        except Exception as exc:
            logger.exception("separate_features_target() failed unexpectedly")
            raise PreprocessingError(
                f"Feature/target separation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------

    def encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns using pd.get_dummies.

        Identifies which of the standard categorical columns (sales, salary)
        are present in X, applies ``pd.get_dummies`` with ``drop_first=False``,
        and returns the transformed DataFrame.

        Args:
            X: Feature matrix (should not contain the target column).

        Returns:
            New DataFrame with categorical columns replaced by dummy variables.

        Raises:
            PreprocessingError: If encoding fails.
        """
        logger.debug("encode_categorical() called — input shape=%s", X.shape)
        try:
            present_cats = [c for c in CATEGORICAL_COLS if c in X.columns]
            if not present_cats:
                logger.warning(
                    "No standard categorical columns found in X — "
                    "returning X unchanged"
                )
                return X.copy()

            X_encoded = pd.get_dummies(X, columns=present_cats, drop_first=False)
            new_cols = set(X_encoded.columns) - set(X.columns)
            logger.info(
                "Categorical encoding complete — encoded %s, "
                "new columns added: %d, output shape=%s",
                present_cats,
                len(new_cols),
                X_encoded.shape,
            )
            return X_encoded
        except Exception as exc:
            logger.exception("encode_categorical() failed unexpectedly")
            raise PreprocessingError(
                f"Categorical encoding failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Stratified train-test split
    # ------------------------------------------------------------------

    def stratified_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified 80:20 train-test split preserving class proportions.

        Args:
            X: Encoded feature matrix.
            y: Binary target vector.

        Returns:
            Tuple ``(X_train, X_test, y_train, y_test)``.

        Raises:
            PreprocessingError: If the split fails (e.g., too few samples).
        """
        logger.debug(
            "stratified_split() called — X shape=%s, test_size=%.2f",
            X.shape,
            self.test_size,
        )
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
            logger.info(
                "Stratified split: train=%d (left=%d, stayed=%d), "
                "test=%d (left=%d, stayed=%d)",
                len(y_train),
                (y_train == 1).sum(),
                (y_train == 0).sum(),
                len(y_test),
                (y_test == 1).sum(),
                (y_test == 0).sum(),
            )
            return X_train, X_test, y_train, y_test
        except Exception as exc:
            logger.exception("stratified_split() failed unexpectedly")
            raise PreprocessingError(
                f"Stratified split failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # SMOTE oversampling
    # ------------------------------------------------------------------

    def apply_smote(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Oversample the minority class using SMOTE.

        Applied to the **training set only**.  The test set is never
        resampled so evaluation reflects real-world class proportions.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector (may be imbalanced).

        Returns:
            Tuple ``(X_resampled, y_resampled)`` with balanced classes.

        Raises:
            PreprocessingError: If SMOTE fails.
        """
        logger.debug(
            "apply_smote() called — before: {0: %d, 1: %d}",
            (y_train == 0).sum(),
            (y_train == 1).sum(),
        )
        try:
            before_dist = self.get_class_distribution(y_train)
            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            # Preserve column names
            X_res = pd.DataFrame(X_res, columns=X_train.columns)
            y_res = pd.Series(y_res, name=y_train.name)
            after_dist = self.get_class_distribution(y_res)
            logger.info(
                "SMOTE applied — before: %s → after: %s",
                before_dist,
                after_dist,
            )
            return X_res, y_res
        except Exception as exc:
            logger.exception("apply_smote() failed unexpectedly")
            raise PreprocessingError(f"SMOTE oversampling failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Class distribution helper
    # ------------------------------------------------------------------

    def get_class_distribution(self, y: pd.Series) -> Dict[str, int]:
        """Compute class counts for a target vector.

        Args:
            y: Binary target Series.

        Returns:
            Dict ``{0: count_stayed, 1: count_left, 'total': n}``.
        """
        counts = y.value_counts().to_dict()
        counts["total"] = len(y)
        return {str(k): v for k, v in counts.items()}

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def run_preprocessing_pipeline(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Execute the full preprocessing pipeline.

        Steps:
        1. Separate features and target
        2. Encode categorical columns
        3. Stratified train-test split
        4. Apply SMOTE to training data

        Returns:
            Tuple ``(X_train_smote, X_test, y_train_smote, y_test)``.

        Raises:
            PreprocessingError: On any step failure.
        """
        logger.info("Starting preprocessing pipeline")
        try:
            X, y = self.separate_features_target()
            X_enc = self.encode_categorical(X)
            X_train, X_test, y_train, y_test = self.stratified_split(X_enc, y)
            X_train_sm, y_train_sm = self.apply_smote(X_train, y_train)
            logger.info(
                "Preprocessing complete — X_train_smote=%s, X_test=%s",
                X_train_sm.shape,
                X_test.shape,
            )
            return X_train_sm, X_test, y_train_sm, y_test
        except PreprocessingError:
            raise
        except Exception as exc:
            logger.exception("run_preprocessing_pipeline() failed unexpectedly")
            raise PreprocessingError(
                f"Preprocessing pipeline failed: {exc}"
            ) from exc
