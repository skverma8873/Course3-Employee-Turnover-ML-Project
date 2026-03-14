"""
Exploratory Data Analysis for the Employee Turnover Analytics pipeline.

Step 2 of the ML pipeline: generates the four visualisations required by
the problem statement and computes turnover-rate summaries per feature.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.exceptions import VisualizationError
from src.utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISTRIBUTION_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "average_montly_hours",
]
DISTRIBUTION_LABELS = {
    "satisfaction_level": ("Employee Satisfaction Level", "Satisfaction Score"),
    "last_evaluation": ("Employee Last Evaluation Score", "Evaluation Score"),
    "average_montly_hours": ("Employee Average Monthly Hours", "Monthly Hours"),
}


class ExploratoryAnalyzer:
    """Performs EDA via statistical summaries and visualisations.

    All plots are saved to ``output_dir`` when provided, and always closed
    after saving to avoid memory leaks on Windows.

    Args:
        df: The cleaned HR DataFrame (copy is stored — input is never mutated).
        output_dir: Directory for saving plots.  Defaults to
            ``outputs/plots/eda/`` relative to the project root.

    Attributes:
        df (pd.DataFrame): Immutable copy of the input DataFrame.
        output_dir (Path): Resolved output directory for plots.
        viz (VisualizationUtils): Shared plot helper.

    Example:
        >>> analyzer = ExploratoryAnalyzer(df, output_dir=Path("outputs/plots/eda"))
        >>> analyzer.run_full_eda()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.df: pd.DataFrame = df.copy()
        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_dir = project_root / "outputs" / "plots" / "eda"
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz = VisualizationUtils()
        logger.debug(
            "ExploratoryAnalyzer initialised — rows=%d, output_dir=%s",
            len(self.df),
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self, save_path: Optional[Path] = None
    ) -> None:
        """Plot a Pearson correlation heatmap for all numeric features.

        Args:
            save_path: Override save path.  Defaults to
                ``output_dir/correlation_heatmap.png``.

        Raises:
            VisualizationError: If the plot cannot be created or saved.
        """
        logger.debug("plot_correlation_heatmap() called")
        try:
            numeric_df = self.df.select_dtypes(include="number")
            corr_matrix = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(12, 9))
            self.viz.create_heatmap(
                corr_matrix,
                title="Correlation Matrix of Employee Features",
                ax=ax,
                fmt=".2f",
                cmap="coolwarm",
            )
            plt.tight_layout()

            dest = save_path or (self.output_dir / "correlation_heatmap.png")
            self.viz.save_figure(fig, dest)
            logger.info("Correlation heatmap saved: %s", dest)

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception("plot_correlation_heatmap() failed unexpectedly")
            raise VisualizationError(
                f"Correlation heatmap failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Distribution plots
    # ------------------------------------------------------------------

    def plot_distribution(
        self,
        column: str,
        title: str,
        xlabel: str = "",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot a histogram + KDE for a single numeric column.

        Args:
            column: Column name to plot.
            title: Plot title.
            xlabel: X-axis label.  Defaults to ``column``.
            save_path: Override save path.

        Raises:
            ValueError: If ``column`` is not in the DataFrame.
            VisualizationError: If the plot cannot be created or saved.
        """
        logger.debug("plot_distribution() called — column=%s", column)
        if column not in self.df.columns:
            raise ValueError(
                f"Column '{column}' not found in DataFrame.  "
                f"Available columns: {list(self.df.columns)}"
            )
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            self.viz.create_distribution_plot(
                self.df[column], title=title, xlabel=xlabel or column, ax=ax
            )
            plt.tight_layout()

            dest = save_path or (
                self.output_dir / f"distribution_{column}.png"
            )
            self.viz.save_figure(fig, dest)
            logger.info("Distribution plot saved: %s", dest)

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception(
                "plot_distribution() failed for column '%s'", column
            )
            raise VisualizationError(
                f"Distribution plot for '{column}' failed: {exc}"
            ) from exc

    def plot_all_distributions(
        self, save_path: Optional[Path] = None
    ) -> None:
        """Plot distributions for satisfaction_level, last_evaluation, and
        average_montly_hours side-by-side in a single figure.

        Args:
            save_path: Override save path.  Defaults to
                ``output_dir/all_distributions.png``.

        Raises:
            VisualizationError: If the plot cannot be created or saved.
        """
        logger.debug("plot_all_distributions() called")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, col in zip(axes, DISTRIBUTION_COLUMNS):
                title, xlabel = DISTRIBUTION_LABELS.get(col, (col, col))
                self.viz.create_distribution_plot(
                    self.df[col], title=title, xlabel=xlabel, ax=ax
                )
            fig.suptitle(
                "Distribution of Key Employee Metrics",
                fontsize=15,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()

            dest = save_path or (self.output_dir / "all_distributions.png")
            self.viz.save_figure(fig, dest)
            logger.info("All distributions plot saved: %s", dest)

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception("plot_all_distributions() failed unexpectedly")
            raise VisualizationError(
                f"All-distributions plot failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Project count bar chart
    # ------------------------------------------------------------------

    def plot_project_count_bar(
        self, save_path: Optional[Path] = None
    ) -> None:
        """Bar chart of project count segmented by employee turnover.

        Uses ``sns.countplot(x='number_project', hue='left')`` to show
        how project load differs between employees who stayed and left.

        Inference logged automatically:
        - Employees with very few (2) or very many (6-7) projects have
          higher turnover rates, suggesting under-utilisation or burn-out.

        Args:
            save_path: Override save path.  Defaults to
                ``output_dir/project_count_bar.png``.

        Raises:
            VisualizationError: If the plot cannot be created or saved.
        """
        logger.debug("plot_project_count_bar() called")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.viz.create_bar_plot(
                data=self.df,
                x="number_project",
                hue="left",
                title="Employee Project Count: Stayed vs Left",
                xlabel="Number of Projects",
                ylabel="Employee Count",
                ax=ax,
            )
            plt.tight_layout()

            dest = save_path or (self.output_dir / "project_count_bar.png")
            self.viz.save_figure(fig, dest)
            logger.info("Project count bar chart saved: %s", dest)
            logger.info(
                "Inference: Employees with 2 or 6-7 projects show highest "
                "turnover — indicative of under-utilisation or burn-out."
            )

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception("plot_project_count_bar() failed unexpectedly")
            raise VisualizationError(
                f"Project count bar chart failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Turnover rate by feature
    # ------------------------------------------------------------------

    def compute_turnover_rate_by_feature(
        self, feature: str
    ) -> pd.DataFrame:
        """Compute turnover rate grouped by a categorical or numeric feature.

        Args:
            feature: Column name to group by.

        Returns:
            DataFrame with columns ``count``, ``left_count``, and
            ``turnover_rate``, indexed by unique values of ``feature``.

        Raises:
            ValueError: If ``feature`` is not in the DataFrame.
        """
        logger.debug(
            "compute_turnover_rate_by_feature() called — feature=%s", feature
        )
        if feature not in self.df.columns:
            raise ValueError(
                f"Feature '{feature}' not in DataFrame.  "
                f"Available: {list(self.df.columns)}"
            )
        grouped = (
            self.df.groupby(feature)["left"]
            .agg(count="count", left_count="sum")
            .assign(turnover_rate=lambda d: (d["left_count"] / d["count"]).round(4))
        )
        logger.info(
            "Turnover rate by '%s':\n%s", feature, grouped.to_string()
        )
        return grouped

    # ------------------------------------------------------------------
    # Full EDA orchestration
    # ------------------------------------------------------------------

    def run_full_eda(self, output_dir: Optional[Path] = None) -> None:
        """Run all EDA steps sequentially and save all plots.

        Steps:
        1. Correlation heatmap
        2. Distribution plots (satisfaction, evaluation, hours)
        3. Project count bar chart
        4. Turnover rate by department and salary

        Args:
            output_dir: Override output directory for all plots.
        """
        logger.info("Starting full EDA — output_dir=%s", output_dir or self.output_dir)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("EDA Step 2.1: Correlation heatmap")
            self.plot_correlation_heatmap()

            logger.info("EDA Step 2.2: Distribution plots")
            self.plot_all_distributions()

            logger.info("EDA Step 2.3: Project count bar chart")
            self.plot_project_count_bar()

            logger.info("EDA: Turnover rate summaries")
            self.compute_turnover_rate_by_feature("salary")
            self.compute_turnover_rate_by_feature("sales")
            self.compute_turnover_rate_by_feature("number_project")

            logger.info("Full EDA completed successfully")

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception("run_full_eda() encountered an unexpected error")
            raise VisualizationError(f"EDA pipeline failed: {exc}") from exc
