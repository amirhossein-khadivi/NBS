import numpy as np


def generate_reset_indicator(T, p, rng):
    """
    Generate Bernoulli reset indicators.

    I_t ~ Bernoulli(p)
    """

    return rng.binomial(
        n=1,
        p=p,
        size=T
    )


def generate_reset_trajectory(
    config,
    reset_indicator,
    ae_noise,
    predictor_noise
):
    """
    Generate the trajectory with Autoencoder resets.

    The same noise as the normal trajectory is used.
    The only difference is the reset of the learning age.
    """

    T = config.T

    ae_loss = np.zeros(T)
    predictor_loss = np.zeros(T)
    reward = np.zeros(T)

    # Number of steps since the latest reset
    learning_age = 0

    for t in range(T):

        # ------------------------------------------------
        # Check whether reset occurs
        # ------------------------------------------------

        if reset_indicator[t] == 1:

            # Autoencoder goes back to its initial
            # learning state.
            learning_age = 0

        # ------------------------------------------------
        # Calculate AE loss
        # ------------------------------------------------

        ae_loss[t] = (
            config.minimum_ae_loss
            + (
                config.initial_ae_loss
                - config.minimum_ae_loss
            )
            * np.exp(
                -config.learning_rate * learning_age
            )
            + ae_noise[t]
        )

        ae_loss[t] = max(
            config.minimum_ae_loss,
            ae_loss[t]
        )

        # ------------------------------------------------
        # Predictor loss
        # ------------------------------------------------

        predictor_loss[t] = (
            config.beta * ae_loss[t]
            + predictor_noise[t]
        )

        # ------------------------------------------------
        # Reward
        # ------------------------------------------------

        reward[t] = -predictor_loss[t]

        # Move one step forward in AE training
        learning_age += 1

    return (
        ae_loss,
        predictor_loss,
        reward
    )