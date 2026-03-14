"""
Shared visualisation utilities for the Employee Turnover Analytics project.

All plot-generating modules import :class:`VisualizationUtils` to ensure
consistent styling, figure management, and file-saving behaviour across
the pipeline.

Windows note: ``matplotlib.use('Agg')`` is set at module import time so
figures can be saved without a display server (headless execution, CI, etc.).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.exceptions import VisualizationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_STYLE: str = "seaborn-v0_8"
DEFAULT_FIGSIZE: Tuple[int, int] = (10, 6)
DEFAULT_PALETTE: str = "husl"
DPI: int = 150


class VisualizationUtils:
    """Shared utilities for creating, styling, and saving matplotlib figures.

    All plot-generating classes in the pipeline instantiate one
    ``VisualizationUtils`` and delegate figure management to it.

    Args:
        style: Matplotlib/Seaborn style name.  Defaults to ``'seaborn-v0_8'``.
        figsize: Default (width, height) in inches for new figures.
    """

    def __init__(
        self,
        style: str = DEFAULT_STYLE,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    ) -> None:
        self.style = style
        self.figsize = figsize
        self.set_plot_style()
        logger.debug(
            "VisualizationUtils initialised — style=%s, figsize=%s",
            style,
            figsize,
        )

    # ------------------------------------------------------------------
    # Style management
    # ------------------------------------------------------------------

    def set_plot_style(self) -> None:
        """Apply the configured matplotlib/seaborn style and colour palette.

        Falls back to the default style if the requested style is unavailable.
        """
        try:
            plt.style.use(self.style)
        except OSError:
            logger.warning(
                "Style '%s' not available — using default style", self.style
            )
            plt.style.use("default")

        sns.set_palette(DEFAULT_PALETTE)
        logger.debug("Plot style applied: %s", self.style)

    # ------------------------------------------------------------------
    # Figure saving
    # ------------------------------------------------------------------

    def save_figure(self, fig: plt.Figure, filepath: Path) -> None:
        """Save a matplotlib figure to disk and close it to free memory.

        The parent directory is created automatically if it does not exist.

        Args:
            fig: The matplotlib Figure to save.
            filepath: Destination path (should end in ``.png`` or ``.svg``).

        Raises:
            VisualizationError: If the figure cannot be saved.
        """
        logger.debug("Saving figure to %s", filepath)
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info("Figure saved: %s", filepath)
        except OSError as exc:
            logger.error("Failed to save figure to %s: %s", filepath, exc)
            raise VisualizationError(
                f"Cannot save figure to {filepath}: {exc}"
            ) from exc
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Plot factory helpers
    # ------------------------------------------------------------------

    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        ax: Optional[plt.Axes] = None,
        fmt: str = ".2f",
        cmap: str = "coolwarm",
    ) -> plt.Axes:
        """Render a annotated seaborn heatmap.

        Args:
            data: Square DataFrame suitable for ``sns.heatmap``.
            title: Plot title.
            ax: Existing Axes to draw into.  If ``None`` a new figure is
                created using ``self.figsize``.
            fmt: Format string for cell annotations.
            cmap: Colour map name.

        Returns:
            The populated :class:`~matplotlib.axes.Axes`.

        Raises:
            VisualizationError: On any rendering error.
        """
        logger.debug("Creating heatmap: %s", title)
        try:
            if ax is None:
                _, ax = plt.subplots(figsize=self.figsize)
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                ax=ax,
                linewidths=0.5,
                square=False,
            )
            ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
            return ax
        except Exception as exc:
            logger.exception("Heatmap creation failed for '%s'", title)
            raise VisualizationError(
                f"Failed to create heatmap '{title}': {exc}"
            ) from exc

    def create_distribution_plot(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ax: Optional[plt.Axes] = None,
        color: str = "steelblue",
    ) -> plt.Axes:
        """Render a histogram + KDE distribution plot.

        Args:
            data: 1-D numeric Series to plot.
            title: Plot title.
            xlabel: X-axis label.
            ax: Existing Axes.  If ``None`` a new figure is created.
            color: Bar / line colour.

        Returns:
            The populated :class:`~matplotlib.axes.Axes`.

        Raises:
            VisualizationError: On any rendering error.
        """
        logger.debug("Creating distribution plot: %s", title)
        try:
            if ax is None:
                _, ax = plt.subplots(figsize=self.figsize)
            sns.histplot(data, kde=True, ax=ax, color=color, edgecolor="white")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            return ax
        except Exception as exc:
            logger.exception("Distribution plot creation failed for '%s'", title)
            raise VisualizationError(
                f"Failed to create distribution plot '{title}': {exc}"
            ) from exc

    def create_bar_plot(
        self,
        data: pd.DataFrame,
        x: str,
        hue: str,
        title: str,
        ax: Optional[plt.Axes] = None,
        xlabel: str = "",
        ylabel: str = "Count",
    ) -> plt.Axes:
        """Render a seaborn count bar plot with a hue grouping.

        Args:
            data: Source DataFrame.
            x: Column name for the x-axis categories.
            hue: Column name for the hue (colour) grouping.
            title: Plot title.
            ax: Existing Axes.  If ``None`` a new figure is created.
            xlabel: X-axis label (defaults to ``x``).
            ylabel: Y-axis label.

        Returns:
            The populated :class:`~matplotlib.axes.Axes`.

        Raises:
            VisualizationError: On any rendering error.
        """
        logger.debug("Creating bar plot: %s (x=%s, hue=%s)", title, x, hue)
        try:
            if ax is None:
                _, ax = plt.subplots(figsize=self.figsize)
            sns.countplot(data=data, x=x, hue=hue, ax=ax, palette="Set2")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(xlabel or x, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(title=hue, labels=["Stayed (0)", "Left (1)"])
            return ax
        except Exception as exc:
            logger.exception("Bar plot creation failed for '%s'", title)
            raise VisualizationError(
                f"Failed to create bar plot '{title}': {exc}"
            ) from exc
