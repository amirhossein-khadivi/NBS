def theoretical_var_indicator_shock(
    p: float,
    mu_d: float,
    sigma_d_squared: float
) -> float:
    """
    Theoretical variance of the composite shock I_t D_t.

    Var(I_t D_t)
        = p * sigma_D^2
        + p * (1-p) * mu_D^2
    """

    return (
        p * sigma_d_squared
        + p * (1.0 - p) * mu_d ** 2
    )


def theoretical_mean_shift(
    p: float,
    mu_d: float
) -> float:
    """
    Theoretical mean shift caused by reset.

    E[I_t D_t] = p * mu_D
    """

    return p * mu_d


def theoretical_v_reset(
    p: float,
    v_noreset: float,
    mu_d: float,
    sigma_d_squared: float
) -> float:
    """
    Theoretical reset reward variance.

    V_reset =
        (1 - 2p) V_noreset
        + p sigma_D^2
        + p(1-p) mu_D^2
    """

    return (
        (1.0 - 2.0 * p) * v_noreset
        + p * sigma_d_squared
        + p * (1.0 - p) * mu_d ** 2
    )


def theoretical_v_reset_p_zero(
    v_noreset: float
) -> float:
    """
    Theoretical reset variance when p = 0.

    V_reset = V_noreset
    """

    return v_noreset