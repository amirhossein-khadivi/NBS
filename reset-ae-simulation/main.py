from src.config import simulationconfig
from experiment import run_single_expriment

def main():

    config = simulationconfig(
        T=1000,
        K=50,
        reward_mean=0.0,
        reward_std=0.02,
        reset_reward_mean=0.0,
        reset_reward_std=0.50,
        seed=42
    )


    run_single_expriment(
        config=config,
        show_plots=True
    )


if __name__ == '__main__':
    main()