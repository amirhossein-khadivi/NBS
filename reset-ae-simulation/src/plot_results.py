import numpy as np
import matplotlib.pyplot as plt

def plot_reward_trajectories(
        normal_reward: np.ndarray,
        reset_reward: np.ndarray
) -> None:

    plt.figure(figsize=(12, 5))
    
    plt.plot(
        reset_reward,
        label='With Reset'
    )

    plt.plot(
        normal_reward, 
        label='No Reset'
    )

    plt.xlabel('Training Step')
    plt.ylabel('Reward')

    plt.title(
        'Reward Trajectory: No Reset vs. Reset'
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_reset_shocks(
        reset_indicator: np.ndarray,
        shock: np.ndarray
) -> None:

    reset_indices = np.where(
        reset_indicator == 1
    )[0]

    reset_shocks = shock[
        reset_indicator == 1
    ]

    plt.figure(figsize=(12, 4))

    plt.stem(
        reset_indices,
        reset_shocks,
        basefmt=" "
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel("Training Step")
    plt.ylabel("d_t")

    plt.title(
        'Reward Shocks Caused by Reset'
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_variance_comparison(
        v_noreset: float,
        v_reset_empirical: float,
        v_reset_theoritical: float
) -> None:

    labels = [
        'No Reset',
        'Reset\n(Empirical)',
        'Reset\n(Theoritical)'
    ]

    values = [
        v_noreset,
        v_reset_empirical,
        v_reset_theoritical
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        values
    )

    plt.ylabel('Reward Variance')

    plt.title(
        'Comparison of Reward Variance'
    )

    plt.grid(
        axis='y',
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


