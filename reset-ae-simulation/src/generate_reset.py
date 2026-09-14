import numpy as np

def generate_reset_indicator(
        T: int, 
        p: float,
        rng: np.random.Generator
) -> np.ndarray:

    I = rng.binomial(
        n=1,
        p=p,
        size=T
    )

    return I

def generat_reset_reward_path(
        normal_reward: np.ndarray,
        shock: np.ndarray,
        reset_indicator: np.ndarray
) -> np.ndarray:

    reset_reward_path = (
        normal_reward + 
        (reset_indicator * shock)
    )

    return reset_reward_path