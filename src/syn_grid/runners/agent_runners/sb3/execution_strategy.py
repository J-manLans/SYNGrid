"""
Strategies for how a trained SB3 model selects actions during evaluation.

Training is identical across the current SB3 agents -- they all just call
model.learn(...) -- so BaseSB3Runner.train() needs no strategy seam. Eval
differs only in whether the model needs episode-scoped state carried between
predict() calls (recurrent policies) or not (everything else). That's the
one axis this module isolates.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm

# ================= #
#      Protocol     #
# ================= #


class ExecutionStrategy(Protocol):
    def reset(self, num_envs: int) -> None:
        """Called once before each evaluation episode."""
        ...

    def predict(self, model: BaseAlgorithm, obs) -> NDArray:
        """Return the next action for obs, given the model."""
        ...

    def on_step(self, dones: NDArray) -> None:
        """Called after each env.step(), with the per-env done flags."""
        ...


# ================= #
#  Strategies  #
# ================= #


class StatelessExecutionStrategy:
    """Default strategy: no state carried between predict() calls."""

    def reset(self, num_envs: int) -> None:
        pass

    def predict(self, model: BaseAlgorithm, obs) -> NDArray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    def on_step(self, dones: NDArray) -> None:
        pass


class RecurrentExecutionStrategy:
    """Carries LSTM hidden state and episode-start flags across predict() calls."""

    def __init__(self):
        self._lstm_states = None
        self._episode_starts: NDArray | None = None

    def reset(self, num_envs: int) -> None:
        self._lstm_states = None
        self._episode_starts = np.ones((num_envs,), dtype=bool)

    def predict(self, model: BaseAlgorithm, obs) -> NDArray:
        action, self._lstm_states = model.predict(
            obs,
            state=self._lstm_states,
            episode_start=self._episode_starts,
            deterministic=True,
        )
        return action

    def on_step(self, dones: NDArray) -> None:
        self._episode_starts = dones
