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
    calculate_shock_variance
)


def run_single_trajectory(config, seed=None):

    if seed is None:
        seed = config.seed

    rng = np.random.default_rng(seed)

    # ==================================================
    # 1. Generate shared noise
    # ==================================================

    ae_noise, predictor_noise = generate_shared_noise(
        config,
        rng
    )

    # ==================================================
    # 2. Generate NORMAL trajectory
    # ==================================================

    (
        normal_ae_loss,
        normal_predictor_loss,
        normal_reward
    ) = generate_normal_trajectory(
        config,
        ae_noise,
        predictor_noise
    )

    # ==================================================
    # 3. Generate reset indicators
    # ==================================================

    reset_indicator = generate_reset_indicator(
        config.T,
        config.p,
        rng
    )

    # ==================================================
    # 4. Generate RESET trajectory
    # ==================================================

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

    # ==================================================
    # 5. Calculate shock
    # ==================================================

    shock = (
        reset_reward
        - normal_reward
    )

    # Shock is meaningful at reset points
    reset_shocks = shock[
        reset_indicator == 1
    ]

    # ==================================================
    # 6. Variance
    # ==================================================

    normal_variance = calculate_variance(
        normal_reward
    )

    reset_variance = calculate_variance(
        reset_reward
    )

    shock_variance = calculate_shock_variance(
        reset_shocks
    )

    return {

        # ------------------------------
        # Normal trajectory
        # ------------------------------

        "normal_ae_loss": normal_ae_loss,
        "normal_predictor_loss": normal_predictor_loss,
        "normal_reward": normal_reward,

        # ------------------------------
        # Reset trajectory
        # ------------------------------

        "reset_ae_loss": reset_ae_loss,
        "reset_predictor_loss": reset_predictor_loss,
        "reset_reward": reset_reward,

        # ------------------------------
        # Reset information
        # ------------------------------

        "reset_indicator": reset_indicator,
        "shock": shock,
        "reset_shocks": reset_shocks,

        # ------------------------------
        # Statistics
        # ------------------------------

        "normal_variance": normal_variance,
        "reset_variance": reset_variance,
        "shock_variance": shock_variance,
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