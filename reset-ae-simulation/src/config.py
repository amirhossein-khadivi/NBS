from dataclasses import dataclass

@dataclass
class simulationconfig:

    T: int = 1000

    K: int = 50

    reward_mean: float = 0.0

    reward_std: float = 0.02

    reset_reward_mean: float = 0.0

    reset_reward_std: float = 0.50

    seed: int = 42

    @property
    def p(self) -> float:
        return 1.0 / self.K