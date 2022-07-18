from typing import Any, Dict

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from src.correspondence.base import OneDimensionResults


def plot_profile_coordiantes(
    row_profiles: OneDimensionResults,
    column_profiles: OneDimensionResults,
    splot: mpl.axes.Subplot = None,
) -> mpl.axes.Subplot:
    rows = row_profiles.factor_score[:, :2]
    columns = column_profiles.factor_score[:, :2]
    row_names = row_profiles.name.ravel()
    column_names = column_profiles.name.ravel()

    if splot is None:
        splot = plt.subplot(111)

    splot.scatter(
        rows[:, 0].ravel(),
        rows[:, 1].ravel(),
        s=1000 * row_profiles.inertia.ravel(),
        label="Row profiles",
        color="C0",
    )
    for i, profile_name in enumerate(row_names):
        f1, f2 = rows[i, :2]
        splot.annotate(
            profile_name,
            (f1, f2),
            xytext=(-15, 15),
            xycoords="data",
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    splot.scatter(
        columns[:, 0].ravel(),
        columns[:, 1].ravel(),
        s=1000 * column_profiles.inertia.ravel(),
        label="Column profiles",
        color="C1",
    )

    for i, profile_name in enumerate(column_names):
        g1, g2 = columns[i, :2]
        splot.annotate(
            profile_name,
            (g1, g2),
            xytext=(-15, -15),
            xycoords="data",
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    splot.spines["left"].set_position("center")
    splot.spines["bottom"].set_position("center")
    splot.spines["right"].set_visible(False)
    splot.spines["top"].set_visible(False)
    splot.get_xaxis().set_ticks([])
    splot.get_yaxis().set_ticks([])
    splot.axes.set_aspect("equal", adjustable="datalim")
    splot.figure.suptitle("Correspondence analysis plot", fontsize=16, weight="bold")
    splot.legend(bbox_to_anchor=(0.7, 1.1), ncol=2, frameon=False)

    return splot


def multidimensional_scaling_2d_plot(
    x: np.ndarray,
    y: np.ndarray,
    annot: np.ndarray = None,
    splot: mpl.axes.Subplot = None,
    fig_kws: Dict[str, Any] = None,
    display_: bool = True,
) -> mpl.axes.Subplot:
    """
    Two-dimensional graphic representation of data after performing multidimensional
    scaling.
    """
    if fig_kws is None:
        fig_kws = {"figsize": (12, 6)}

    if splot is None:
        _ = plt.figure(**fig_kws)
        splot = plt.subplot(111)

    splot.scatter(x, y)

    if annot is not None:
        for i in range(x.size):
            splot.annotate(
                annot[i],
                (x[i], y[i]),
                xytext=(-10, 10),
                xycoords="data",
                textcoords="offset points",
            )

    splot.axes.set_aspect("equal", adjustable="datalim")
    splot.spines["right"].set_visible(False)
    splot.spines["top"].set_visible(False)
    splot.figure.suptitle(
        "Multidimensional Scaling 2D Representiation", fontsize=16, weight="bold"
    )

    if display_:
        plt.show()

    return splot
