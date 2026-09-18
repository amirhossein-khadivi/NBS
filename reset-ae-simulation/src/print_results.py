def print_trajectory_results(result, config):
    print("=" * 60)
    print("SINGLE TRAJECTORY RESULTS")
    print("=" * 60)

    print(f"T                 : {config.T}")
    print(f"K                 : {config.K}")
    print(f"p = 1/K           : {config.p:.4f}")
    print(f"Number of resets  : {result['reset_indicator'].sum()}")

    print()

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

    print("=" * 60)


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

    print("=" * 60)
    print("MULTIPLE TRAJECTORIES")
    print("=" * 60)

    print(f"Trajectories : {len(results)}")
    print(f"T            : {config.T}")
    print(f"K            : {config.K}")
    print(f"p            : {config.p:.4f}")

    print()

    print(
        f"Mean normal variance : "
        f"{sum(normal_variances) / len(normal_variances):.6f}"
    )

    print(
        f"Mean reset variance  : "
        f"{sum(reset_variances) / len(reset_variances):.6f}"
    )

    if shock_variances:
        print(
            f"Mean shock variance  : "
            f"{sum(shock_variances) / len(shock_variances):.6f}"
        )

    print("=" * 60)