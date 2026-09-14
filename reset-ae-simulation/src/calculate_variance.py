import numpy as np

def calculate_variance(values: np.ndarray) -> float:

    return float(np.var(values, ddof=1))


def calculate_sigma_d_squared(
        shock: np.ndarray
) -> float:

    return calculate_variance(shock)

def calculate_var_indicator_shock(
        reset_indicator: np.ndarray,
        shock: np.ndarray
) -> float:

    shock_component = reset_indicator * shock

    return calculate_variance(shock_component)

def calculate_v_noreset(
        normal_reward: np.ndarray
) -> float:
    
    return calculate_variance(normal_reward)

def calculate_v_reset(
        reset_reward: np.ndarray, 
) -> float:
    return calculate_variance(reset_reward)
