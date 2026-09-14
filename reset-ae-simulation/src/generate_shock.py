import numpy as np

def generate_reset_reward(
        T: int,
        mean: float,
        std: float,
        rng: np.random.Generator
) -> np.ndarray:
    
    reset_reward = rng.normal(
        loc=mean,
        scale=std,
        size=T
    )

    return reset_reward

def generate_shok(
        normal_reward: np.ndarray,
        reset_reward: np.ndarray
) -> np.ndarray:

    d = reset_reward - normal_reward

    return d