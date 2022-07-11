import numpy as np
from matplotlib import pyplot as plt

from src.base import OneDimensionResults


def plot_profile_coordiantes(
    row_profiles: OneDimensionResults,
    column_profiles: OneDimensionResults,
    row_profile_names: np.ndarray,
    column_profile_names: np.ndarray,
    splot: plt.subplot = None,
):
    row_coordinates = row_profiles.factor_scores[:, :2]
    column_coordinates = column_profiles.factor_scores[:, :2]

    if splot is None:
        splot = plt.subplot(111)

    splot.scatter(
        row_coordinates[:, 0].ravel(),
        row_coordinates[:, 1].ravel(),
        s=1000 * row_profiles.inertia.ravel(),
        label="Row profiles",
        color="C0",
    )
    for i, profile_name in enumerate(row_profile_names.ravel()):
        f1, f2 = row_coordinates[i, :2]
        splot.annotate(
            profile_name,
            (f1, f2),
            xytext=(f1, f2 * 0.99),
            xycoords="data",
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    splot.scatter(
        column_coordinates[:, 0].ravel(),
        column_coordinates[:, 1].ravel(),
        s=1000 * column_profiles.inertia.ravel(),
        label="Column profiles",
        color="C1",
    )

    for i, profile_name in enumerate(column_profile_names.ravel()):
        g1, g2 = column_coordinates[i, :2]
        splot.annotate(
            profile_name,
            (g1, g2),
            xytext=(g1, g2 * 0.99),
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
    return splot
