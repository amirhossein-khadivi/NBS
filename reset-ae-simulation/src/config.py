from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Trajectory
    T: int = 1000
    K: int = 50
    n_trajectories: int = 100

    # Autoencoder loss
    initial_ae_loss: float = 1.0
    minimum_ae_loss: float = 0.01
    learning_rate: float = 0.005

    # Noise
    ae_noise_std: float = 0.01
    predictor_noise_std: float = 0.01

    # Predictor
    beta: float = 0.8

    # Stabilization criterion
    stabilization_window: int = 50
    stabilization_epsilon: float = 0.01

    # Reproducibility
    seed: int = 42

    @property
    def p(self):
        return 1.0 / self.K