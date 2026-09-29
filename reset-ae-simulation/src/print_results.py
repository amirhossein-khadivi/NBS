def print_trajectory_results(result, config):

    print("=" * 70)
    print("SINGLE TRAJECTORY RESULTS")
    print("=" * 70)

    print(f"T                 : {config.T}")
    print(f"K                 : {config.K}")
    print(f"p = 1/K           : {config.p:.4f}")
    print(
        f"Number of resets  : "
        f"{result['reset_indicator'].sum()}"
    )
    print(
        f"Stabilization T*  : "
        f"{result['stabilization_time']}"
    )
    print(
        f"Stable reset events : "
        f"{result['n_reset_events']}"
    )

    print()

    print("-" * 70)
    print("EMPIRICAL RESULTS")
    print("-" * 70)

    print(
        f"Normal reward variance : "
        f"{result['normal_variance']:.6f}"
    )

    print(
        f"Reset reward variance  : "
        f"{result['reset_variance']:.6f}"
    )

    print(
        f"Shock mean (mu_D)      : "
        f"{result['mu_d']:.6f}"
    )

    print(
        f"Shock variance "
        f"(sigma_D^2)           : "
        f"{result['sigma_d_squared']:.6f}"
    )

    print()

    print("-" * 70)
    print("THEORETICAL RESULTS")
    print("-" * 70)

    print(
        f"Theoretical mean shift "
        f"(p * mu_D)            : "
        f"{result['theoretical_mean_shift']:.6f}"
    )

    print(
        f"Theoretical shock "
        f"variance Var(I_t D_t) : "
        f"{result['theoretical_shock_variance']:.6f}"
    )

    print(
        f"Theoretical reset "
        f"variance               : "
        f"{result['theoretical_reset_variance']:.6f}"
    )

    print(
        f"Theoretical reset "
        f"variance (p = 0)       : "
        f"{result['theoretical_reset_variance_p_zero']:.6f}"
    )

    print()

    print("-" * 70)
    print("EMPIRICAL vs THEORETICAL")
    print("-" * 70)

    print(
        f"Empirical reset variance   : "
        f"{result['reset_variance']:.6f}"
    )

    print(
        f"Theoretical reset variance : "
        f"{result['theoretical_reset_variance']:.6f}"
    )

    print("=" * 70)


def print_multiple_results(results, config):

    normal_variances = [
        r["normal_variance"]
        for r in results
        if r["normal_variance"] == r["normal_variance"]
    ]

    reset_variances = [
        r["reset_variance"]
        for r in results
        if r["reset_variance"] == r["reset_variance"]
    ]

    mu_values = [
        r["mu_d"]
        for r in results
        if r["mu_d"] == r["mu_d"]
    ]

    sigma_values = [
        r["sigma_d_squared"]
        for r in results
        if r["sigma_d_squared"] == r["sigma_d_squared"]
    ]

    theoretical_shock_variances = [
        r["theoretical_shock_variance"]
        for r in results
        if r["theoretical_shock_variance"]
        == r["theoretical_shock_variance"]
    ]

    theoretical_reset_variances = [
        r["theoretical_reset_variance"]
        for r in results
        if r["theoretical_reset_variance"]
        == r["theoretical_reset_variance"]
    ]

    stabilization_times = [
        r["stabilization_time"]
        for r in results
        if r["stabilization_time"] is not None
    ]

    mean_normal_variance = (
        sum(normal_variances)
        / len(normal_variances)
    )

    mean_reset_variance = (
        sum(reset_variances)
        / len(reset_variances)
    )

    mean_mu_d = (
        sum(mu_values)
        / len(mu_values)
        if mu_values
        else float("nan")
    )

    mean_sigma_d_squared = (
        sum(sigma_values)
        / len(sigma_values)
        if sigma_values
        else float("nan")
    )

    mean_theoretical_shock_variance = (
        sum(theoretical_shock_variances)
        / len(theoretical_shock_variances)
    )

    mean_theoretical_reset_variance = (
        sum(theoretical_reset_variances)
        / len(theoretical_reset_variances)
    )

    mean_stabilization_time = (
        sum(stabilization_times)
        / len(stabilization_times)
        if stabilization_times
        else float("nan")
    )

    print()
    print("=" * 70)
    print("MULTIPLE TRAJECTORIES RESULTS")
    print("=" * 70)

    print(f"Trajectories : {len(results)}")
    print(f"T            : {config.T}")
    print(f"K            : {config.K}")
    print(f"p            : {config.p:.4f}")

    print(
        f"Mean T*      : "
        f"{mean_stabilization_time:.2f}"
    )

    print()

    print("-" * 70)
    print("EMPIRICAL RESULTS")
    print("-" * 70)

    print(
        f"Mean normal variance : "
        f"{mean_normal_variance:.6f}"
    )

    print(
        f"Mean reset variance  : "
        f"{mean_reset_variance:.6f}"
    )

    print(
        f"Mean shock "
        f"(mu_D)              : "
        f"{mean_mu_d:.6f}"
    )

    print(
        f"Mean shock variance "
        f"(sigma_D^2)         : "
        f"{mean_sigma_d_squared:.6f}"
    )

    print()

    print("-" * 70)
    print("THEORETICAL RESULTS")
    print("-" * 70)

    print(
        f"Mean theoretical shock "
        f"variance              : "
        f"{mean_theoretical_shock_variance:.6f}"
    )

    print(
        f"Mean theoretical reset "
        f"variance              : "
        f"{mean_theoretical_reset_variance:.6f}"
    )

    print()

    print("-" * 70)
    print("EMPIRICAL vs THEORETICAL")
    print("-" * 70)

    print(
        f"Empirical reset variance   : "
        f"{mean_reset_variance:.6f}"
    )

    print(
        f"Theoretical reset variance : "
        f"{mean_theoretical_reset_variance:.6f}"
    )

    print("=" * 70)