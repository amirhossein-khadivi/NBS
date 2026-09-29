from src.config import SimulationConfig

from experiment import (
    run_single_trajectory,
    run_multiple_trajectories
)

from src.print_results import (
    print_trajectory_results,
    print_multiple_results
)

from src.plot_results import (
    plot_all_results
)


def main():

    config = SimulationConfig()

    # --------------------------------------------------
    # Single trajectory
    # --------------------------------------------------

    result = run_single_trajectory(
        config
    )

    print_trajectory_results(
        result,
        config
    )

    # --------------------------------------------------
    # Multiple trajectories
    # --------------------------------------------------

    results = run_multiple_trajectories(
        config
    )

    print_multiple_results(
        results,
        config
    )

    # --------------------------------------------------
    # Plot single + multiple
    # --------------------------------------------------

    plot_all_results(
        result,
        results
    )


if __name__ == "__main__":
    main()