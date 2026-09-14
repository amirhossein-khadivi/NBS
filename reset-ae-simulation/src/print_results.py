def print_results(results: dict) -> None:

    print('=' * 70)
    print('Results of Reward Reset Simulation')
    print('=' * 70)

    print(f"T                = {results['T']}")
    print(f"K                = {results['K']}")
    print(f"p = 1/K          = {results['p']:.6f}")

    print('-' * 70)

    print(
        f"sigma_d^2           = "
        f"{results['sigma_d_squared']:.8f}"

        f"\nLemma 3"
    )
    print(
        f"Var(I_t d_t) empirical= "
        f"{results['var_id_empirical']:.8f}"
    )

    print(
        f"p * sigma_d^2          = "
        f"{results['var_id_theoritical']}"
    )

    print('-' * 70)

    print('Theorem 2')

    print(
        f"V_noreset              = "
        f"{results['v_noreset']:.8f}"
    )

    print(
        f"V_reset empirical        = "
        f"{results['v_reset_empirical']:.8f}"
    )

    print(
        f"V_reset theoritical       = "
        f"{results['v_reset_theoritical']}"
    )

    print('-' * 70)

    print(
        f"Number of resets         = "
        f"{results['number_of_resets']}"
    )

    print(
        f"V_reset / V_noreset         = "
        f"{results['variance_ratio']:6f}"
    )

    print('-' * 70)

    if results['v_reset_empirical'] > results['v_noreset']:
        print(
            'Result: Reset increases empirical reward'
        )
    elif results['v_reset_empirical'] < results['v_noreset']:
        print(
            f'Result: Reset decreases empirical reward variance'
        )
    else:
        print(
            f'Result: Both variance are approximately equal.'
        )

    print('=' * 70)