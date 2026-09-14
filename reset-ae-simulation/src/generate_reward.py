import numpy as np

def generate_normal_reward(
        T: int,
        mean: float, 
        std: float,
        rng: np.random.Generator
) -> np.ndarray:
    reward = rng.normal(
        loc=mean,
        scale=std,
        size=T
    )

    return reward