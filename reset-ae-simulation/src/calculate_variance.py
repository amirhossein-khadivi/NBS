import numpy as np


def calculate_variance(values):
    """
    Calculate sample variance.
    """

    if len(values) < 2:
        return np.nan

    return np.var(
        values,
        ddof=1
    )


def calculate_shock_statistics(
    shocks,
    reset_indicator,
    start=0
):
    """
    Calculate the mean and variance of D_t
    at actual reset timesteps after the
    stabilization point.

    D_t = R_t^reset - R_t^normal
    """

    mask = (
        (reset_indicator == 1)
        & (np.arange(len(shocks)) >= start)
    )

    reset_shocks = shocks[mask]

    if len(reset_shocks) == 0:
        return np.nan, np.nan, 0

    mu_d = np.mean(reset_shocks)

    if len(reset_shocks) < 2:
        sigma_d_squared = np.nan
    else:
        sigma_d_squared = np.var(
            reset_shocks,
            ddof=1
        )

    return (
        mu_d,
        sigma_d_squared,
        len(reset_shocks)
    )


def find_stabilization_time(
    loss,
    window,
    epsilon
):
    """
    Find the earliest timestep T* at which
    the difference between two consecutive
    moving-window averages is below epsilon.

    Based on:

        |mean[L(t-W:t)] - mean[L(t-2W:t-W)]| < epsilon
    """

    T = len(loss)

    if T < 2 * window:
        return None

    for t in range(
        2 * window,
        T + 1
    ):

        recent_mean = np.mean(
            loss[t - window:t]
        )

        previous_mean = np.mean(
            loss[t - 2 * window:t - window]
        )

        difference = abs(
            recent_mean - previous_mean
        )

        if difference < epsilon:
            return t

    return None


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
        if not np.isnan(result[key])
    ]

    if len(variances) == 0:
        return np.nan

    return np.mean(variances)