import numpy as np


def generate_shared_noise(config, rng):
    """
    Generate the random noise that will be shared
    between the normal and reset trajectories.

    This ensures a fair paired comparison.
    """

    ae_noise = rng.normal(
        0,
        config.ae_noise_std,
        config.T
    )

    predictor_noise = rng.normal(
        0,
        config.predictor_noise_std,
        config.T
    )

    return ae_noise, predictor_noise


def calculate_ae_loss(
    learning_age,
    config,
    noise
):
    """
    Calculate Autoencoder loss at a given learning age.
    """

    deterministic_loss = (
        config.minimum_ae_loss
        + (
            config.initial_ae_loss
            - config.minimum_ae_loss
        )
        * np.exp(
            -config.learning_rate * learning_age
        )
    )

    loss = deterministic_loss + noise

    return max(
        config.minimum_ae_loss,
        loss
    )


def generate_normal_trajectory(
    config,
    ae_noise,
    predictor_noise
):
    """
    Generate the normal trajectory without reset.

    The AE continuously learns from the beginning
    until the end of the trajectory.
    """

    T = config.T

    ae_loss = np.zeros(T)
    predictor_loss = np.zeros(T)
    reward = np.zeros(T)

    for t in range(T):

        # In the normal trajectory, learning age
        # is simply the current timestep.
        learning_age = t

        ae_loss[t] = calculate_ae_loss(
            learning_age,
            config,
            ae_noise[t]
        )

        predictor_loss[t] = (
            config.beta * ae_loss[t]
            + predictor_noise[t]
        )

        reward[t] = -predictor_loss[t]

    return (
        ae_loss,
        predictor_loss,
        reward
    )