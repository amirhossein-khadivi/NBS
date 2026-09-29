import numpy as np

from src.generate_trajectory import (
    generate_shared_noise,
    generate_normal_trajectory
)

from src.generate_reset import (
    generate_reset_indicator,
    generate_reset_trajectory
)

from src.calculate_variance import (
    calculate_variance,
    calculate_shock_statistics,
    find_stabilization_time
)

from src.calculate_theoretical import (
    theoretical_var_indicator_shock,
    theoretical_mean_shift,
    theoretical_v_reset,
    theoretical_v_reset_p_zero
)


def run_single_trajectory(config, seed=None):

    if seed is None:
        seed = config.seed

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Shared stochastic noise
    # --------------------------------------------------

    ae_noise, predictor_noise = generate_shared_noise(
        config,
        rng
    )

    # --------------------------------------------------
    # Normal trajectory
    # --------------------------------------------------

    (
        normal_ae_loss,
        normal_predictor_loss,
        normal_reward
    ) = generate_normal_trajectory(
        config,
        ae_noise,
        predictor_noise
    )

    # --------------------------------------------------
    # Stabilization point T*
    # --------------------------------------------------

    stabilization_time = find_stabilization_time(
        normal_ae_loss,
        config.stabilization_window,
        config.stabilization_epsilon
    )

    if stabilization_time is None:
        stabilization_time = 0

    # --------------------------------------------------
    # Reset indicator
    # --------------------------------------------------

    reset_indicator = generate_reset_indicator(
        config.T,
        config.p,
        rng
    )

    # --------------------------------------------------
    # Reset trajectory
    # --------------------------------------------------

    (
        reset_ae_loss,
        reset_predictor_loss,
        reset_reward
    ) = generate_reset_trajectory(
        config,
        reset_indicator,
        ae_noise,
        predictor_noise
    )

    # --------------------------------------------------
    # Reward shock
    # --------------------------------------------------

    shock = reset_reward - normal_reward

    # --------------------------------------------------
    # Stable-regime data
    # --------------------------------------------------

    normal_reward_stable = normal_reward[
        stabilization_time:
    ]

    reset_reward_stable = reset_reward[
        stabilization_time:
    ]

    # --------------------------------------------------
    # Empirical variances
    # --------------------------------------------------

    normal_variance = calculate_variance(
        normal_reward_stable
    )

    reset_variance = calculate_variance(
        reset_reward_stable
    )

    # --------------------------------------------------
    # Shock statistics
    # --------------------------------------------------

    (
        mu_d,
        sigma_d_squared,
        n_reset_events
    ) = calculate_shock_statistics(
        shock,
        reset_indicator,
        start=stabilization_time
    )

    # --------------------------------------------------
    # Theoretical quantities
    # --------------------------------------------------

    theoretical_mean_shift_value = (
        theoretical_mean_shift(
            config.p,
            mu_d
        )
    )

    theoretical_shock_variance = (
        theoretical_var_indicator_shock(
            config.p,
            mu_d,
            sigma_d_squared
        )
    )

    theoretical_reset_variance = (
        theoretical_v_reset(
            config.p,
            normal_variance,
            mu_d,
            sigma_d_squared
        )
    )

    theoretical_reset_variance_p_zero = (
        theoretical_v_reset_p_zero(
            normal_variance
        )
    )

    # --------------------------------------------------
    # Return everything
    # --------------------------------------------------

    return {

        # Normal trajectory
        "normal_ae_loss": normal_ae_loss,
        "normal_predictor_loss": normal_predictor_loss,
        "normal_reward": normal_reward,

        # Reset trajectory
        "reset_ae_loss": reset_ae_loss,
        "reset_predictor_loss": reset_predictor_loss,
        "reset_reward": reset_reward,

        # Reset information
        "reset_indicator": reset_indicator,

        # Stabilization
        "stabilization_time": stabilization_time,

        # Shock
        "shock": shock,

        # Stable-regime shock statistics
        "mu_d": mu_d,
        "sigma_d_squared": sigma_d_squared,
        "n_reset_events": n_reset_events,

        # Empirical results
        "normal_variance": normal_variance,
        "reset_variance": reset_variance,

        # Theoretical results
        "theoretical_mean_shift":
            theoretical_mean_shift_value,

        "theoretical_shock_variance":
            theoretical_shock_variance,

        "theoretical_reset_variance":
            theoretical_reset_variance,

        "theoretical_reset_variance_p_zero":
            theoretical_reset_variance_p_zero
    }


def run_multiple_trajectories(config):

    results = []

    for i in range(config.n_trajectories):

        result = run_single_trajectory(
            config,
            seed=config.seed + i
        )

        results.append(result)

    return results