from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.base import BaseCorrespondenceAnalysis, OneDimensionResults


def destructure_dictionary(dictionary: Dict[Any, Any], keys: List[str] = None):
    """
    Given a dictionaty object it returns a list of dictionary values in the
    order of `keys`. If `keys` is None then dictionary values are returned according to
    tha alphabetical order of its keys.

    :param dictionary: dictionary to be destructured
    :param keys: list of key values in `dictionary` to define the order in which values
    in `dictionary` to be returned.

    :return values: List of values in `dictionary` ordered as `keys`.
    """
    if keys is None:
        keys = sorted(list(dictionary.keys()))

    values = list(map(lambda k: dictionary[k], keys))
    return values


@dataclass
class CorrespondenceAnalysisResults:
    """Container class for correspondence analysis results."""

    row: OneDimensionResults
    column: OneDimensionResults


class CorrespondenceAnalysis(BaseCorrespondenceAnalysis):
    """Correspondence analysis class."""

    def __init__(self, table):
        """
        Loads contingency table and computes correspondence analysis.

        :param table:
            Contingency table for which correspondence analysis will be performed.
        """
        super(CorrespondenceAnalysis, self).__init__(table)

    def _fit(self):
        """
        Runs correspondence analysis for the inputed contingency table.
        """
        key_order = ["weights", "mass", "distance", "inertia"]
        self.profiles = CorrespondenceAnalysisResults(
            OneDimensionResults(
                *destructure_dictionary(self.row_profiles.__dict__, key_order),
                None,
                None,
                None,
                None
            ),
            OneDimensionResults(
                *destructure_dictionary(self.column_profiles.__dict__, key_order),
                None,
                None,
                None,
                None
            ),
        )

    def _standardized_residuals_matrix(self):
        """
        Computes the matrix to be decomposed using singular value decomposition
        such that the factor scores for row/column profiles can be obtained with
        its resulting projection matrices.
        """
        return (
            np.diag(self.row_profiles.weights)
            @ (self.table - self.fittedvalues)
            @ np.diag(self.column_profiles.weights)
        )

    def _profile_correlation(self):
        """
        Correlation between row/column profile and their axis. Morover it is the
        proportion of variance in a profile explained by the axis.
        """

    def _profile_contribution(self):
        """
        The contribution of the row/column profile to the inertia of its axis.
        """

    def _angle(self):
        """Angle between the axis and the profile."""
