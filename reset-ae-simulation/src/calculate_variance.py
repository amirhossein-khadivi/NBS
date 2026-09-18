import numpy as np


def calculate_variance(values):
    """
    Calculate sample variance.
    """

    return np.var(
        values,
        ddof=1
    )


def calculate_shock_variance(shocks):
    """
    Calculate variance of reset shocks.
    """

    if len(shocks) < 2:
        return np.nan

    return np.var(
        shocks,
        ddof=1
    )


def calculate_mean_variance(
    results,
    key
):
    """
    Calculate mean variance across trajectories.
    """

    variances = [
        result[key]
        for result in results
    ]

    return np.mean(variances)