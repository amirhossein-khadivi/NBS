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
        f"Shock variance         : "
        f"{result['shock_variance']:.6f}"
    )

    print()

    print("-" * 70)
    print("THEORETICAL RESULTS")
    print("-" * 70)

    print(
        f"Theoretical shock variance : "
        f"{result['theoretical_shock_variance']:.6f}"
    )

    print(
        f"Theoretical reset variance : "
        f"{result['theoretical_reset_variance']:.6f}"
    )

    print(
        f"Theoretical reset variance "
        f"(p = 0)                 : "
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
    ]

    reset_variances = [
        r["reset_variance"]
        for r in results
    ]

    shock_variances = [
        r["shock_variance"]
        for r in results
        if not (
            r["shock_variance"] != r["shock_variance"]
        )
    ]

    theoretical_shock_variances = [
        r["theoretical_shock_variance"]
        for r in results
    ]

    theoretical_reset_variances = [
        r["theoretical_reset_variance"]
        for r in results
    ]

    mean_normal_variance = (
        sum(normal_variances)
        / len(normal_variances)
    )

    mean_reset_variance = (
        sum(reset_variances)
        / len(reset_variances)
    )

    mean_shock_variance = (
        sum(shock_variances)
        / len(shock_variances)
        if shock_variances
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

    print()
    print("=" * 70)
    print("MULTIPLE TRAJECTORIES RESULTS")
    print("=" * 70)

    print(f"Trajectories : {len(results)}")
    print(f"T            : {config.T}")
    print(f"K            : {config.K}")
    print(f"p            : {config.p:.4f}")

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
        f"Mean shock variance  : "
        f"{mean_shock_variance:.6f}"
    )

    print()

    print("-" * 70)
    print("THEORETICAL RESULTS")
    print("-" * 70)

    print(
        f"Mean theoretical shock variance : "
        f"{mean_theoretical_shock_variance:.6f}"
    )

    print(
        f"Mean theoretical reset variance : "
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