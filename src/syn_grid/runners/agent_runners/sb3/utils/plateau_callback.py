import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PlateauCallback(BaseCallback):
    def __init__(
        self, terminate_threshold: int, plateau_threshold: int, min_delta: float = 0.2
    ):
        super().__init__()
        self._terminate_threshold = terminate_threshold
        self._plateau_threshold = plateau_threshold
        self._min_delta = min_delta
        self._best_mean_reward = -np.inf
        self._last_improvement_step = 0

    def _on_step(self) -> bool:
        if self.model.ep_info_buffer is not None and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if mean_reward > self._best_mean_reward + self._min_delta:
                self._best_mean_reward = mean_reward
                self._last_improvement_step = self.num_timesteps

            terminate_threshold = (
                self._terminate_threshold
                if self._best_mean_reward > 1
                else self._plateau_threshold
            )
            if self.num_timesteps - self._last_improvement_step >= terminate_threshold:
                return False
        return True
