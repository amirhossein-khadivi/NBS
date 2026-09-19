import matplotlib.pyplot as plt
import numpy as np


def plot_ae_loss(result):
    """
    Plot Autoencoder Loss for normal and reset trajectories.
    """

    T = len(result["normal_ae_loss"])
    steps = np.arange(T)

    plt.figure(figsize=(12, 5))

    plt.plot(
        steps,
        result["normal_ae_loss"],
        label="Normal"
    )

    plt.plot(
        steps,
        result["reset_ae_loss"],
        label="With Reset"
    )

    reset_points = np.where(
        result["reset_indicator"] == 1
    )[0]

    for point in reset_points:
        plt.axvline(
            point,
            linestyle="--",
            alpha=0.25
        )

    plt.xlabel("Timestep")
    plt.ylabel("Autoencoder Loss")
    plt.title("Autoencoder Loss: Normal vs Reset")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_predictor_loss(result):
    """
    Plot Predictor Loss for normal and reset trajectories.
    """

    T = len(result["normal_predictor_loss"])
    steps = np.arange(T)

    plt.figure(figsize=(12, 5))

    plt.plot(
        steps,
        result["normal_predictor_loss"],
        label="Normal"
    )

    plt.plot(
        steps,
        result["reset_predictor_loss"],
        label="With Reset"
    )

    reset_points = np.where(
        result["reset_indicator"] == 1
    )[0]

    for point in reset_points:
        plt.axvline(
            point,
            linestyle="--",
            alpha=0.25
        )

    plt.xlabel("Timestep")
    plt.ylabel("Predictor Loss")
    plt.title("Predictor Loss: Normal vs Reset")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_reward(result):
    """
    Plot Reward for normal and reset trajectories.
    """

    T = len(result["normal_reward"])
    steps = np.arange(T)

    plt.figure(figsize=(12, 5))

    plt.plot(
        steps,
        result["normal_reward"],
        label="Normal"
    )

    plt.plot(
        steps,
        result["reset_reward"],
        label="With Reset"
    )

    reset_points = np.where(
        result["reset_indicator"] == 1
    )[0]

    for point in reset_points:
        plt.axvline(
            point,
            linestyle="--",
            alpha=0.25
        )

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward: Normal vs Reset")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_reward_shock(result):
    """
    Plot reward shock:

        D_t = R_t^reset - R_t^normal
    """

    T = len(result["shock"])
    steps = np.arange(T)

    shock = result["shock"]

    plt.figure(figsize=(12, 5))

    plt.plot(
        steps,
        shock,
        label=r"$D_t = R_t^{reset} - R_t^{normal}$"
    )

    reset_points = np.where(
        result["reset_indicator"] == 1
    )[0]

    for point in reset_points:
        plt.axvline(
            point,
            linestyle="--",
            alpha=0.25
        )

    plt.axhline(
        0.0,
        linestyle="-",
        alpha=0.5
    )

    plt.xlabel("Timestep")
    plt.ylabel("Reward Shock")
    plt.title("Reward Shock Caused by Reset")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_variance_comparison(result):
    """
    Compare empirical and theoretical variance quantities.
    """

    labels = [
        "Normal Variance",
        "Reset Variance\n(Empirical)",
        "Reset Variance\n(Theoretical)",
        "Shock Variance\n(Empirical)",
        "Shock Variance\n(Theoretical)"
    ]

    values = [
        result["normal_variance"],
        result["reset_variance"],
        result["theoretical_reset_variance"],
        result["shock_variance"],
        result["theoretical_shock_variance"]
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(11, 6))

    plt.bar(
        x,
        values
    )

    plt.xticks(
        x,
        labels
    )

    plt.ylabel("Variance")
    plt.title("Empirical vs Theoretical Variance")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


def plot_all_results(result):
    """
    Generate all five plots.
    """

    plot_ae_loss(result)

    plot_predictor_loss(result)

    plot_reward(result)

    plot_reward_shock(result)

    plot_variance_comparison(result)