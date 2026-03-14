"""
Retention strategy advisor — risk zone categorisation and targeted recommendations.

Step 7 of the ML pipeline: uses the best trained model to predict the
probability of each employee leaving, assigns employees to one of four
risk zones, and maps each zone to a specific HR retention strategy.

Risk Zones:
- Safe (Green)        : score < 20%
- Low-Risk (Yellow)   : 20% <= score < 60%
- Medium-Risk (Orange): 60% <= score < 90%
- High-Risk (Red)     : score >= 90%
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.exceptions import ModelEvaluationError, VisualizationError
from src.utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAFE_THRESHOLD: float = 0.20
LOW_RISK_THRESHOLD: float = 0.60
MEDIUM_RISK_THRESHOLD: float = 0.90

ZONE_ORDER = ["Safe", "Low-Risk", "Medium-Risk", "High-Risk"]
ZONE_COLORS: Dict[str, str] = {
    "Safe": "#27ae60",
    "Low-Risk": "#f1c40f",
    "Medium-Risk": "#e67e22",
    "High-Risk": "#e74c3c",
}

RETENTION_STRATEGIES: Dict[str, str] = {
    "Safe": (
        "Employee is stable. Continue current engagement practices:\n"
        "  • Schedule annual performance and satisfaction reviews\n"
        "  • Recognise and celebrate contributions publicly\n"
        "  • Offer learning budgets and optional stretch projects\n"
        "  • No urgent HR intervention required"
    ),
    "Low-Risk": (
        "Employee shows early warning signals. Proactive engagement recommended:\n"
        "  • Conduct a quarterly 'stay interview' to surface concerns early\n"
        "  • Clarify career growth path and promotion timeline\n"
        "  • Offer skill development programmes or mentoring\n"
        "  • Ensure workload is balanced across the team\n"
        "  • Monitor satisfaction monthly"
    ),
    "Medium-Risk": (
        "Employee is at significant risk. Immediate manager action required:\n"
        "  • Schedule an immediate 1:1 between manager and employee\n"
        "  • Review workload — investigate signs of overwork or underutilisation\n"
        "  • Benchmark and review compensation against market rates\n"
        "  • Offer flexible working arrangements or project rotation\n"
        "  • Escalate to HR Business Partner for structured support\n"
        "  • Bi-weekly check-ins until risk score improves"
    ),
    "High-Risk": (
        "Employee departure is imminent. Urgent senior-level intervention required:\n"
        "  • Immediate meeting with senior leadership and HR\n"
        "  • Develop a personalised retention package:\n"
        "      – Salary revision aligned to market benchmarks\n"
        "      – Accelerated promotion or title change\n"
        "      – Role redesign or lateral move to a preferred team\n"
        "      – Enhanced benefits (remote work, training budget, bonus)\n"
        "  • Weekly check-ins with direct manager\n"
        "  • Begin knowledge-transfer planning as contingency"
    ),
}


class RetentionAdvisor:
    """Predict turnover probability, assign risk zones, and recommend strategies.

    Args:
        best_model: The best fitted sklearn estimator (must support
            ``predict_proba``).
        X_test: Test feature matrix.
        y_test: True test labels.
        output_dir: Directory for saving retention plots.

    Attributes:
        probabilities (np.ndarray | None): Per-employee turnover probabilities
            (class 1).  Populated after :meth:`predict_turnover_probabilities`.
        risk_zones (pd.Series | None): Zone label per employee.  Populated
            after :meth:`categorize_risk_zones`.

    Example:
        >>> advisor = RetentionAdvisor(best_model, X_test, y_test)
        >>> report = advisor.generate_retention_report()
    """

    def __init__(
        self,
        best_model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.best_model = best_model
        self.X_test = X_test
        self.y_test = y_test
        self.probabilities: Optional[np.ndarray] = None
        self.risk_zones: Optional[pd.Series] = None

        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_dir = project_root / "outputs" / "plots" / "retention"
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz = VisualizationUtils()
        logger.debug(
            "RetentionAdvisor initialised — X_test=%s", X_test.shape
        )

    # ------------------------------------------------------------------
    # Probability prediction
    # ------------------------------------------------------------------

    def predict_turnover_probabilities(self) -> np.ndarray:
        """Predict the probability of each employee leaving.

        Returns:
            1-D numpy array of turnover probabilities (class 1) in [0, 1],
            length equals ``len(X_test)``.

        Raises:
            ModelEvaluationError: If the model lacks ``predict_proba`` or
                prediction fails.
        """
        logger.debug("predict_turnover_probabilities() called")
        try:
            if not hasattr(self.best_model, "predict_proba"):
                raise AttributeError(
                    f"{type(self.best_model).__name__} does not support "
                    "predict_proba.  Use a probabilistic classifier."
                )
            self.probabilities = self.best_model.predict_proba(self.X_test)[:, 1]
            logger.info(
                "Turnover probabilities computed — min=%.4f, mean=%.4f, max=%.4f",
                float(self.probabilities.min()),
                float(self.probabilities.mean()),
                float(self.probabilities.max()),
            )
            return self.probabilities
        except AttributeError as exc:
            logger.error("predict_turnover_probabilities() failed: %s", exc)
            raise ModelEvaluationError(str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "predict_turnover_probabilities() failed unexpectedly"
            )
            raise ModelEvaluationError(
                f"Probability prediction failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Risk zone categorisation
    # ------------------------------------------------------------------

    def categorize_risk_zones(self, probabilities: np.ndarray) -> pd.Series:
        """Assign each employee to a risk zone based on turnover probability.

        Args:
            probabilities: Array of turnover probabilities in [0, 1].

        Returns:
            Categorical Series with values in
            ``['Safe', 'Low-Risk', 'Medium-Risk', 'High-Risk']``.

        Raises:
            ModelEvaluationError: If categorisation fails.
        """
        logger.debug("categorize_risk_zones() called — n=%d", len(probabilities))
        try:
            zones = pd.cut(
                probabilities,
                bins=[-0.001, SAFE_THRESHOLD, LOW_RISK_THRESHOLD,
                      MEDIUM_RISK_THRESHOLD, 1.001],
                labels=["Safe", "Low-Risk", "Medium-Risk", "High-Risk"],
            )
            self.risk_zones = pd.Series(zones, name="risk_zone")
            counts = self.risk_zones.value_counts()
            logger.info(
                "Risk zone distribution: Safe=%d, Low-Risk=%d, "
                "Medium-Risk=%d, High-Risk=%d",
                counts.get("Safe", 0),
                counts.get("Low-Risk", 0),
                counts.get("Medium-Risk", 0),
                counts.get("High-Risk", 0),
            )
            return self.risk_zones
        except Exception as exc:
            logger.exception("categorize_risk_zones() failed unexpectedly")
            raise ModelEvaluationError(
                f"Risk zone categorisation failed: {exc}"
            ) from exc

    def get_zone_counts(self) -> Dict[str, int]:
        """Return employee count per risk zone.

        Returns:
            Dict mapping zone name → count, ordered by ZONE_ORDER.

        Raises:
            ModelEvaluationError: If risk zones have not been computed.
        """
        if self.risk_zones is None:
            raise ModelEvaluationError(
                "Risk zones not computed.  Call categorize_risk_zones() first."
            )
        counts = self.risk_zones.value_counts().to_dict()
        return {zone: int(counts.get(zone, 0)) for zone in ZONE_ORDER}

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def suggest_strategies(self, zone: str) -> str:
        """Return the retention strategy for a given risk zone.

        Args:
            zone: One of ``'Safe'``, ``'Low-Risk'``, ``'Medium-Risk'``,
                ``'High-Risk'``.

        Returns:
            Multi-line strategy string.

        Raises:
            ValueError: If ``zone`` is not a recognised zone name.
        """
        if zone not in RETENTION_STRATEGIES:
            raise ValueError(
                f"Unknown zone '{zone}'.  Valid zones: {list(RETENTION_STRATEGIES)}"
            )
        return RETENTION_STRATEGIES[zone]

    # ------------------------------------------------------------------
    # Retention report
    # ------------------------------------------------------------------

    def generate_retention_report(self) -> pd.DataFrame:
        """Build the full per-employee retention report.

        Calls :meth:`predict_turnover_probabilities` and
        :meth:`categorize_risk_zones` if not already done.

        Returns:
            DataFrame with columns:
            ``employee_index``, ``turnover_probability``, ``risk_zone``,
            ``actual_left``, ``strategy``.

        Raises:
            ModelEvaluationError: On any computation failure.
        """
        logger.debug("generate_retention_report() called")
        try:
            if self.probabilities is None:
                self.predict_turnover_probabilities()
            if self.risk_zones is None:
                self.categorize_risk_zones(self.probabilities)

            report = pd.DataFrame(
                {
                    "employee_index": self.X_test.index,
                    "turnover_probability": np.round(self.probabilities, 4),
                    "risk_zone": self.risk_zones.values,
                    "actual_left": self.y_test.values,
                    "strategy": [
                        self.suggest_strategies(str(z))
                        for z in self.risk_zones.values
                    ],
                }
            ).reset_index(drop=True)

            logger.info(
                "Retention report generated — %d employees, zone distribution: %s",
                len(report),
                self.get_zone_counts(),
            )
            return report
        except ModelEvaluationError:
            raise
        except Exception as exc:
            logger.exception("generate_retention_report() failed unexpectedly")
            raise ModelEvaluationError(
                f"Retention report generation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Zone distribution plot
    # ------------------------------------------------------------------

    def plot_zone_distribution(
        self, save_path: Optional[Path] = None
    ) -> None:
        """Bar chart of employee counts per risk zone.

        Args:
            save_path: Override save path.  Defaults to
                ``output_dir/risk_zone_distribution.png``.

        Raises:
            ModelEvaluationError: If risk zones have not been computed.
            VisualizationError: If plotting/saving fails.
        """
        logger.debug("plot_zone_distribution() called")
        if self.risk_zones is None:
            raise ModelEvaluationError(
                "Risk zones not computed.  Call generate_retention_report() first."
            )
        try:
            zone_counts = self.get_zone_counts()
            colors = [ZONE_COLORS[z] for z in ZONE_ORDER]
            counts = [zone_counts[z] for z in ZONE_ORDER]

            fig, ax = plt.subplots(figsize=(9, 6))
            bars = ax.bar(ZONE_ORDER, counts, color=colors, edgecolor="white", width=0.6)

            # Annotate bars with counts
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

            ax.set_title(
                "Employee Risk Zone Distribution",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Risk Zone", fontsize=12)
            ax.set_ylabel("Number of Employees", fontsize=12)
            ax.set_ylim(0, max(counts) * 1.15)
            plt.tight_layout()

            dest = save_path or (self.output_dir / "risk_zone_distribution.png")
            self.viz.save_figure(fig, dest)
            logger.info("Zone distribution plot saved: %s", dest)

        except (ModelEvaluationError, VisualizationError):
            raise
        except Exception as exc:
            logger.exception("plot_zone_distribution() failed unexpectedly")
            raise VisualizationError(
                f"Zone distribution plot failed: {exc}"
            ) from exc
