def theoretical_var_indicator_shock(
        p: float,
        sigma_d_squared: float
) -> float:

    return p * sigma_d_squared


def theoretical_v_reset(
        p: float,
        v_noreset: float,
        sigma_d_squared: float
) -> float:

    return (
        (1.0 - 2.0 * p) * v_noreset
        + p * sigma_d_squared
    )


def theoretical_v_reset_p_zero(
        v_noreset: float
) -> float:

    p = 0.0

    return (
        (1.0 - 2.0 * p) * v_noreset
        + p * 0.0
    )