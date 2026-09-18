import matplotlib.pyplot as plt


def plot_single_trajectory(result):
    T = len(result["normal_reward"])
    steps = range(T)

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

    reset_points = [
        i
        for i, value in enumerate(result["reset_indicator"])
        if value == 1
    ]

    for point in reset_points:
        plt.axvline(
            point,
            linestyle="--",
            alpha=0.4
        )

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Normal vs Reset Reward Trajectory")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_losses(result):
    T = len(result["normal_ae_loss"])
    steps = range(T)

    plt.figure(figsize=(12, 5))

    plt.plot(
        steps,
        result["normal_ae_loss"],
        label="Normal AE Loss"
    )

    plt.plot(
        steps,
        result["reset_ae_loss"],
        label="Reset AE Loss"
    )

    plt.xlabel("Timestep")
    plt.ylabel("Autoencoder Loss")
    plt.title("Autoencoder Loss Trajectory")
    plt.legend()
    plt.grid(True)

    plt.show()