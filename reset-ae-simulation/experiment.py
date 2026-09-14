import numpy as np

from src.config import simulationconfig

from src.generate_reward import (
    generate_normal_reward
)

from src.generate_shock import (
    generate_reset_reward,
    generate_shok
)

from src.generate_reset import (
    generate_reset_indicator,
    generat_reset_reward_path
)

from src.calculate_variance import (
    calculate_sigma_d_squared,
    calculate_var_indicator_shock,
    calculate_v_reset,
    calculate_v_noreset
)

from src.calculate_theoritical import (
    theoritical_v_reset,
    theoritical_var_indicator_shock
)

from src.print_results import print_results

from src.plot_results import (
    plot_reward_trajectories,
    plot_reset_shocks,
    plot_variance_comparison
)

def run_single_expriment(
        config: simulationconfig,
        show_plots: bool = True
) -> dict:

    rng = np.random.default_rng(
        config.seed
    )

    normal_reward = generate_normal_reward(
        T=config.T,
        mean=config.reward_mean,
        std=config.reward_std,
        rng=rng
    )

    reset_reward = generate_reset_reward(
        T=config.T,
        mean=config.reset_reward_mean,
        std=config.reset_reward_std,
        rng=rng
    )

    shock = generate_shok(
        normal_reward=normal_reward,
        reset_reward=reset_reward
    )

    reset_indicator = generate_reset_indicator(
        T=config.T,
        p=config.p,
        rng=rng
    )

    reset_reward_path = generat_reset_reward_path(
        normal_reward=normal_reward,
        shock=shock,
        reset_indicator=reset_indicator
    )

    sigma_d_squared = calculate_sigma_d_squared(
        shock
    )

    var_id_empirical = calculate_var_indicator_shock(
        reset_indicator,
        shock
    )

    v_noreset = calculate_v_noreset(
        normal_reward
    )

    v_reset_empirical = calculate_v_reset(
        reset_reward_path
    )

    var_id_theorotical = theoritical_var_indicator_shock(
        p=config.p,
        sigma_d_squared=sigma_d_squared, 
    )

    v_reset_theoritical = theoritical_v_reset(
        p=config.p,
        v_noreset=v_noreset,
        sigma_d_squared=sigma_d_squared
    )

    results = {
        'T': config.T,
        'K': config.K,
        'p': config.p,

        'sigma_d_squared': sigma_d_squared,

        'var_id_empirical': var_id_empirical,
        'var_id_theoritical': var_id_theorotical,

        'v_noreset': v_noreset,

        'v_reset_empirical': v_reset_empirical,
        'v_reset_theoritical': v_reset_theoritical,

        'variance_ratio': (
            v_reset_empirical / v_noreset
        ),

        'number_of_resets': int(
            np.sum(reset_indicator)
        ),

        'normal_reward': normal_reward,
        'reset_reward': reset_reward_path,
        'shock': shock,
        'reset_indicator': reset_indicator
    }


    print_results(results=results)

    if show_plots:

        plot_reward_trajectories(
            normal_reward,
            reset_reward_path
        )

        plot_reset_shocks(
            reset_indicator,
            shock
        )

        plot_variance_comparison(
            v_noreset,
            v_reset_empirical,
            v_reset_theoritical
        )

    return results

def run_multiple_expriments(
        K_values=(10, 20, 50, 100),
        repetitions=100,
        T=1000,
        reward_std=0.02,
        reset_reward_std=0.50,
        seed=42
) -> list:

    all_results = []

    for k in K_values:

        for repetition in range(repetitions):

            config = simulationconfig(
                T=T,
                K=k,
                reward_std=reward_std,
                reset_reward_std=reset_reward_std,
                seed=seed + repetition
            )

            results = run_single_expriment(
                config=config,
                show_plots=False
            )

            all_results.append({
                'K': k,
                'p': config.p,
                'repetition': repetition,
                'V_noreset': results['v_noreset'],
                'V_reset': results['v_reset_empirical'],
                'Sigma_d_squared': results['sigma_d_squared'],
                'Var_id_empirical': results['var_id_empirical'],
                'Var_id_theoritical':results['var_id_theoritical']
            })

    return all_results
