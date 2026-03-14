"""
K-Means clustering of employees who left the company.

Step 3 of the ML pipeline: identifies distinct behavioural archetypes
among departing employees using satisfaction_level and last_evaluation,
helping HR design targeted retention interventions.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.utils.exceptions import ClusteringError, VisualizationError
from src.utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLUSTERS: int = 3
RANDOM_STATE: int = 42
CLUSTER_FEATURES = ["satisfaction_level", "last_evaluation"]
CLUSTER_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]


class EmployeeClusterer:
    """K-Means clustering on employees who left the company.

    Only employees with ``left == 1`` are used for clustering so that the
    centroids reflect distinct departure archetypes rather than the mixed
    population.

    Args:
        df: Full HR DataFrame (containing both stayers and leavers).
        n_clusters: Number of K-Means clusters.  Defaults to 3.
        random_state: Random seed for reproducibility.
        output_dir: Directory for saving cluster plots.

    Attributes:
        cluster_data (pd.DataFrame | None): Filtered data (left==1) with
            satisfaction_level + last_evaluation columns.
        kmeans_model (KMeans | None): Fitted KMeans instance.
        labels (np.ndarray | None): Cluster assignments for cluster_data.

    Example:
        >>> clusterer = EmployeeClusterer(df, n_clusters=3)
        >>> results = clusterer.run_clustering_pipeline()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_clusters: int = N_CLUSTERS,
        random_state: int = RANDOM_STATE,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.df: pd.DataFrame = df.copy()
        self.n_clusters: int = n_clusters
        self.random_state: int = random_state
        self.cluster_data: Optional[pd.DataFrame] = None
        self.kmeans_model: Optional[KMeans] = None
        self.labels: Optional[np.ndarray] = None

        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_dir = project_root / "outputs" / "plots" / "clustering"
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz = VisualizationUtils()
        logger.debug(
            "EmployeeClusterer initialised — n_clusters=%d, rows=%d",
            n_clusters,
            len(df),
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_cluster_data(self) -> pd.DataFrame:
        """Filter to employees who left and select clustering features.

        Returns:
            DataFrame with columns ``satisfaction_level`` and
            ``last_evaluation``, containing only rows where ``left == 1``.

        Raises:
            ClusteringError: If no employees with ``left == 1`` are found.
        """
        logger.debug("prepare_cluster_data() called")
        try:
            leavers = self.df[self.df["left"] == 1][CLUSTER_FEATURES].copy()
            if leavers.empty:
                raise ValueError(
                    "No employees with left==1 found in the dataset."
                )
            self.cluster_data = leavers.reset_index(drop=True)
            logger.info(
                "Cluster data prepared: %d employees who left", len(self.cluster_data)
            )
            return self.cluster_data
        except ValueError as exc:
            logger.error("prepare_cluster_data() failed: %s", exc)
            raise ClusteringError(str(exc)) from exc
        except Exception as exc:
            logger.exception("prepare_cluster_data() failed unexpectedly")
            raise ClusteringError(
                f"Cluster data preparation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit_kmeans(self) -> KMeans:
        """Fit K-Means on the prepared cluster data.

        :meth:`prepare_cluster_data` must be called first.

        Returns:
            The fitted :class:`~sklearn.cluster.KMeans` instance.

        Raises:
            ClusteringError: If clustering data is not prepared or fitting fails.
        """
        logger.debug("fit_kmeans() called — n_clusters=%d", self.n_clusters)
        if self.cluster_data is None:
            raise ClusteringError(
                "Cluster data not prepared.  Call prepare_cluster_data() first."
            )
        try:
            self.kmeans_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            self.labels = self.kmeans_model.fit_predict(self.cluster_data)
            logger.info(
                "K-Means fitted — inertia=%.4f, n_clusters=%d",
                self.kmeans_model.inertia_,
                self.n_clusters,
            )
            return self.kmeans_model
        except Exception as exc:
            logger.exception("fit_kmeans() failed unexpectedly")
            raise ClusteringError(f"K-Means fitting failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_clusters(self, save_path: Optional[Path] = None) -> None:
        """Scatter plot of clusters coloured by K-Means label.

        Args:
            save_path: Override save path.  Defaults to
                ``output_dir/kmeans_clusters.png``.

        Raises:
            ClusteringError: If the model has not been fitted yet.
            VisualizationError: If the plot cannot be saved.
        """
        logger.debug("plot_clusters() called")
        if self.labels is None or self.cluster_data is None:
            raise ClusteringError(
                "Model not fitted.  Call fit_kmeans() before plot_clusters()."
            )
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            for cluster_id in range(self.n_clusters):
                mask = self.labels == cluster_id
                ax.scatter(
                    self.cluster_data.loc[mask, "satisfaction_level"],
                    self.cluster_data.loc[mask, "last_evaluation"],
                    c=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                    label=f"Cluster {cluster_id}",
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=0.4,
                    s=40,
                )
            # Plot centroids
            centroids = self.kmeans_model.cluster_centers_
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker="X",
                s=200,
                c="black",
                zorder=5,
                label="Centroids",
            )
            ax.set_title(
                "K-Means Clusters of Employees Who Left (k=3)",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Satisfaction Level", fontsize=12)
            ax.set_ylabel("Last Evaluation Score", fontsize=12)
            ax.legend(fontsize=10)
            plt.tight_layout()

            dest = save_path or (self.output_dir / "kmeans_clusters.png")
            self.viz.save_figure(fig, dest)
            logger.info("Cluster scatter plot saved: %s", dest)

        except VisualizationError:
            raise
        except Exception as exc:
            logger.exception("plot_clusters() failed unexpectedly")
            raise VisualizationError(f"Cluster plot failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Summary & interpretation
    # ------------------------------------------------------------------

    def get_cluster_summary(self) -> pd.DataFrame:
        """Compute mean features and size per cluster.

        Returns:
            DataFrame indexed by cluster ID with columns
            ``satisfaction_level``, ``last_evaluation``, and ``count``.

        Raises:
            ClusteringError: If the model has not been fitted.
        """
        logger.debug("get_cluster_summary() called")
        if self.labels is None or self.cluster_data is None:
            raise ClusteringError(
                "Model not fitted.  Call fit_kmeans() before get_cluster_summary()."
            )
        summary = (
            self.cluster_data.copy()
            .assign(cluster=self.labels)
            .groupby("cluster")
            .agg(
                satisfaction_level=("satisfaction_level", "mean"),
                last_evaluation=("last_evaluation", "mean"),
                count=("satisfaction_level", "count"),
            )
            .round(4)
        )
        logger.info("Cluster summary:\n%s", summary.to_string())
        return summary

    def interpret_clusters(self) -> Dict[int, Dict[str, str]]:
        """Label each cluster based on centroid satisfaction and evaluation.

        Heuristic rules:
        - High satisfaction + High evaluation → ``"Poached / Better Opportunity"``
        - Low satisfaction + High evaluation → ``"Burned-Out High Performers"``
        - Low satisfaction + Low evaluation → ``"Disengaged Low Performers"``

        Returns:
            Dict mapping cluster ID → ``{'label': str, 'description': str}``.

        Raises:
            ClusteringError: If the model has not been fitted.
        """
        logger.debug("interpret_clusters() called")
        summary = self.get_cluster_summary()
        interpretations: Dict[int, Dict[str, str]] = {}

        sat_median = summary["satisfaction_level"].median()
        eval_median = summary["last_evaluation"].median()

        for cluster_id, row in summary.iterrows():
            high_sat = row["satisfaction_level"] >= sat_median
            high_eval = row["last_evaluation"] >= eval_median

            if high_sat and high_eval:
                label = "Poached / Better Opportunity"
                description = (
                    "High satisfaction and strong evaluation — these employees "
                    "likely left for external opportunities (higher pay, "
                    "better role).  Competitive compensation and career growth "
                    "programmes are key retention levers."
                )
            elif not high_sat and high_eval:
                label = "Burned-Out High Performers"
                description = (
                    "Low satisfaction despite strong performance — these "
                    "employees are overworked or under-recognised.  Workload "
                    "reduction, meaningful recognition, and autonomy "
                    "improvements are recommended."
                )
            else:
                label = "Disengaged Low Performers"
                description = (
                    "Low satisfaction and low evaluation — these employees are "
                    "already disengaged.  Performance improvement plans, "
                    "role-fit assessment, or managed exits should be considered."
                )

            interpretations[int(cluster_id)] = {
                "label": label,
                "description": description,
                "satisfaction_mean": round(float(row["satisfaction_level"]), 4),
                "evaluation_mean": round(float(row["last_evaluation"]), 4),
                "count": int(row["count"]),
            }
            logger.info(
                "Cluster %d — %s (n=%d)", cluster_id, label, row["count"]
            )

        return interpretations

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def run_clustering_pipeline(self) -> Dict[str, Any]:
        """Execute the full clustering pipeline end-to-end.

        Steps: prepare → fit → summary → interpret → plot.

        Returns:
            Dict with keys ``summary`` (DataFrame),
            ``interpretation`` (dict), and ``model`` (KMeans).

        Raises:
            ClusteringError: On any clustering failure.
            VisualizationError: On plot failure.
        """
        logger.info("Starting clustering pipeline — n_clusters=%d", self.n_clusters)
        try:
            self.prepare_cluster_data()
            self.fit_kmeans()
            summary = self.get_cluster_summary()
            interpretation = self.interpret_clusters()
            self.plot_clusters()
            logger.info("Clustering pipeline completed successfully")
            return {
                "summary": summary,
                "interpretation": interpretation,
                "model": self.kmeans_model,
            }
        except (ClusteringError, VisualizationError):
            raise
        except Exception as exc:
            logger.exception("run_clustering_pipeline() failed unexpectedly")
            raise ClusteringError(f"Clustering pipeline failed: {exc}") from exc
