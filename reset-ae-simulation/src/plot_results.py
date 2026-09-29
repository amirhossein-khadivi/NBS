import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Helper functions
# ============================================================

def calculate_multiple_mean(results, key):
    """
    Calculate point-wise mean across multiple trajectories.
    """

    values = np.array([
        result[key]
        for result in results
    ])

    return np.mean(values, axis=0)


def calculate_multiple_scalar_mean(results, key):
    """
    Calculate mean of a scalar quantity across trajectories.
    """

    return np.mean([
        result[key]
        for result in results
    ])


def calculate_multiple_stabilization_time(results):
    """
    Calculate mean stabilization time T* across trajectories.

    Only trajectories with a valid stabilization_time
    are included.
    """

    stabilization_times = [
        result["stabilization_time"]
        for result in results
        if result["stabilization_time"] is not None
    ]

    if not stabilization_times:
        return np.nan

    return np.mean(stabilization_times)


# ============================================================
# Original single-trajectory plots
# ============================================================

def plot_single_trajectory(result):
    """
    Plot the original single-trajectory results.

    A vertical line is added at the single-trajectory
    stabilization time T*.
    """

    single_T_star = result["stabilization_time"]

    # --------------------------------------------------------
    # 1. Autoencoder Loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        result["normal_ae_loss"],
        label="Normal AE Loss"
    )

    plt.plot(
        result["reset_ae_loss"],
        label="Reset AE Loss"
    )

    # T* vertical line
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"$T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("AE Loss")
    plt.title("Autoencoder Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 2. Predictor Loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        result["normal_predictor_loss"],
        label="Normal Predictor Loss"
    )

    plt.plot(
        result["reset_predictor_loss"],
        label="Reset Predictor Loss"
    )

    # T* vertical line
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"$T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Predictor Loss")
    plt.title("Predictor Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 3. Reward
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        result["normal_reward"],
        label="Normal Reward"
    )

    plt.plot(
        result["reset_reward"],
        label="Reset Reward"
    )

    # T* vertical line
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"$T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 4. Shock
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        result["shock"],
        label="Reset Shock"
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    # T* vertical line
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"$T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Reward Difference")
    plt.title("Reset Shock")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 5. Variance Comparison
    # --------------------------------------------------------

    labels = [
        "Normal Variance",
        "Reset Variance (Empirical)",
        "Reset Variance (Theoretical)",
        "Shock Variance (Empirical)",
        "Shock Variance (Theoretical)"
    ]

    values = [
        result["normal_variance"],
        result["reset_variance"],
        result["theoretical_reset_variance"],
        result["sigma_d_squared"],
        result["theoretical_shock_variance"]
    ]

    plt.figure(figsize=(11, 5))

    plt.bar(
        labels,
        values
    )

    plt.ylabel("Variance")
    plt.title("Variance Comparison")
    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Original plots for multiple trajectories
# ============================================================

def plot_multiple_trajectories(results):
    """
    Plot point-wise mean of multiple trajectories.

    A vertical line is added at the mean stabilization
    time T* across trajectories.
    """

    # --------------------------------------------------------
    # Calculate T*
    # --------------------------------------------------------

    multiple_T_star = calculate_multiple_stabilization_time(
        results
    )

    # --------------------------------------------------------
    # Calculate point-wise means
    # --------------------------------------------------------

    normal_ae_mean = calculate_multiple_mean(
        results,
        "normal_ae_loss"
    )

    reset_ae_mean = calculate_multiple_mean(
        results,
        "reset_ae_loss"
    )

    normal_predictor_mean = calculate_multiple_mean(
        results,
        "normal_predictor_loss"
    )

    reset_predictor_mean = calculate_multiple_mean(
        results,
        "reset_predictor_loss"
    )

    normal_reward_mean = calculate_multiple_mean(
        results,
        "normal_reward"
    )

    reset_reward_mean = calculate_multiple_mean(
        results,
        "reset_reward"
    )

    shock_mean = calculate_multiple_mean(
        results,
        "shock"
    )

    # --------------------------------------------------------
    # 1. Autoencoder Loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        normal_ae_mean,
        label="Mean Normal AE Loss"
    )

    plt.plot(
        reset_ae_mean,
        label="Mean Reset AE Loss"
    )

    # Mean T* vertical line
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("AE Loss")
    plt.title("Autoencoder Loss - Multiple Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 2. Predictor Loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        normal_predictor_mean,
        label="Mean Normal Predictor Loss"
    )

    plt.plot(
        reset_predictor_mean,
        label="Mean Reset Predictor Loss"
    )

    # Mean T* vertical line
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Predictor Loss")
    plt.title("Predictor Loss - Multiple Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 3. Reward
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        normal_reward_mean,
        label="Mean Normal Reward"
    )

    plt.plot(
        reset_reward_mean,
        label="Mean Reset Reward"
    )

    # Mean T* vertical line
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Reward - Multiple Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 4. Shock
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        shock_mean,
        label="Mean Reset Shock"
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    # Mean T* vertical line
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Reward Difference")
    plt.title("Reset Shock - Multiple Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 5. Variance Comparison
    # --------------------------------------------------------

    normal_variance_mean = calculate_multiple_scalar_mean(
        results,
        "normal_variance"
    )

    reset_variance_mean = calculate_multiple_scalar_mean(
        results,
        "reset_variance"
    )

    theoretical_reset_variance_mean = calculate_multiple_scalar_mean(
        results,
        "theoretical_reset_variance"
    )

    shock_variance_mean = calculate_multiple_scalar_mean(
        results,
        "sigma_d_squared"
    )

    theoretical_shock_variance_mean = calculate_multiple_scalar_mean(
        results,
        "theoretical_shock_variance"
    )

    labels = [
        "Normal Variance",
        "Reset Variance (Empirical)",
        "Reset Variance (Theoretical)",
        "Shock Variance (Empirical)",
        "Shock Variance (Theoretical)"
    ]

    values = [
        normal_variance_mean,
        reset_variance_mean,
        theoretical_reset_variance_mean,
        shock_variance_mean,
        theoretical_shock_variance_mean
    ]

    plt.figure(figsize=(11, 5))

    plt.bar(
        labels,
        values
    )

    plt.ylabel("Variance")
    plt.title("Variance Comparison - Multiple Trajectories")
    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Single vs Multiple comparison plots
# ============================================================

def plot_single_vs_multiple(
    single_result,
    multiple_results
):
    """
    Compare a single trajectory with the
    point-wise mean of multiple trajectories.

    Vertical lines:
        - Single trajectory T*
        - Mean T* across multiple trajectories
    """

    # --------------------------------------------------------
    # Calculate T*
    # --------------------------------------------------------

    single_T_star = single_result["stabilization_time"]

    multiple_T_star = calculate_multiple_stabilization_time(
        multiple_results
    )

    # --------------------------------------------------------
    # Print T* values
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STABILIZATION TIMES USED IN PLOTS")
    print("=" * 70)

    print(
        f"Single T*    : "
        f"{single_T_star}"
    )

    print(
        f"Multiple T* : "
        f"{multiple_T_star:.2f}"
    )

    print("=" * 70)

    # ========================================================
    # 1. AE Loss
    # ========================================================

    normal_ae_mean = calculate_multiple_mean(
        multiple_results,
        "normal_ae_loss"
    )

    reset_ae_mean = calculate_multiple_mean(
        multiple_results,
        "reset_ae_loss"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        single_result["normal_ae_loss"],
        label="Single Normal AE Loss"
    )

    plt.plot(
        normal_ae_mean,
        label="Multiple Mean Normal AE Loss",
        linewidth=2
    )

    plt.plot(
        single_result["reset_ae_loss"],
        label="Single Reset AE Loss"
    )

    plt.plot(
        reset_ae_mean,
        label="Multiple Mean Reset AE Loss",
        linewidth=2
    )

    # Single T*
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"Single $T^*$"
        )

    # Multiple Mean T*
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Multiple Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("AE Loss")
    plt.title("Single vs Multiple - Autoencoder Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # 2. Predictor Loss
    # ========================================================

    normal_predictor_mean = calculate_multiple_mean(
        multiple_results,
        "normal_predictor_loss"
    )

    reset_predictor_mean = calculate_multiple_mean(
        multiple_results,
        "reset_predictor_loss"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        single_result["normal_predictor_loss"],
        label="Single Normal Predictor Loss"
    )

    plt.plot(
        normal_predictor_mean,
        label="Multiple Mean Normal Predictor Loss",
        linewidth=2
    )

    plt.plot(
        single_result["reset_predictor_loss"],
        label="Single Reset Predictor Loss"
    )

    plt.plot(
        reset_predictor_mean,
        label="Multiple Mean Reset Predictor Loss",
        linewidth=2
    )

    # Single T*
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"Single $T^*$"
        )

    # Multiple Mean T*
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Multiple Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Predictor Loss")
    plt.title("Single vs Multiple - Predictor Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # 3. Reward
    # ========================================================

    normal_reward_mean = calculate_multiple_mean(
        multiple_results,
        "normal_reward"
    )

    reset_reward_mean = calculate_multiple_mean(
        multiple_results,
        "reset_reward"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        single_result["normal_reward"],
        label="Single Normal Reward"
    )

    plt.plot(
        normal_reward_mean,
        label="Multiple Mean Normal Reward",
        linewidth=2
    )

    plt.plot(
        single_result["reset_reward"],
        label="Single Reset Reward"
    )

    plt.plot(
        reset_reward_mean,
        label="Multiple Mean Reset Reward",
        linewidth=2
    )

    # Single T*
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"Single $T^*$"
        )

    # Multiple Mean T*
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Multiple Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Single vs Multiple - Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # 4. Shock
    # ========================================================

    shock_mean = calculate_multiple_mean(
        multiple_results,
        "shock"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        single_result["shock"],
        label="Single Shock"
    )

    plt.plot(
        shock_mean,
        label="Multiple Mean Shock",
        linewidth=2
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    # Single T*
    if single_T_star is not None:
        plt.axvline(
            x=single_T_star,
            linestyle="--",
            linewidth=1.5,
            label=r"Single $T^*$"
        )

    # Multiple Mean T*
    if not np.isnan(multiple_T_star):
        plt.axvline(
            x=multiple_T_star,
            linestyle=":",
            linewidth=2,
            label=r"Multiple Mean $T^*$"
        )

    plt.xlabel("Time step")
    plt.ylabel("Reward Difference")
    plt.title("Single vs Multiple - Reset Shock")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main plotting function
# ============================================================

def plot_all_results(
    single_result,
    multiple_results
):
    """
    Generate all plots:

    1. Original single-trajectory plots
       + Single T* vertical line

    2. Original multiple-trajectory mean plots
       + Mean T* vertical line

    3. Single-vs-multiple comparison plots
       + Both T* vertical lines
    """

    # --------------------------------------------------------
    # Original single plots
    # --------------------------------------------------------

    plot_single_trajectory(
        single_result
    )

    # --------------------------------------------------------
    # Original multiple plots
    # --------------------------------------------------------

    plot_multiple_trajectories(
        multiple_results
    )

    # --------------------------------------------------------
    # Single vs Multiple comparison plots
    # --------------------------------------------------------

    plot_single_vs_multiple(
        single_result,
        multiple_results
    )