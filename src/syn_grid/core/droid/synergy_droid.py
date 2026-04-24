from syn_grid.config.models import DroidConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.gymnasium.action_space import DroidAction
from syn_grid.core.droid.digestion_engine import DigestionEngine

from typing import Final


class SynergyDroid:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: DroidConf):
        """
        Initializes the droid.

        Defines the game world so the droid know its bounds, set its starting score and store it for later resetting.
        """

        self._GRID_ROWS: Final[int] = conf.grid_rows
        self._GRID_COLS: Final[int] = conf.grid_cols
        self._STARTING_SCORE: Final[float] = conf.starting_score
        self._STEP_PENALTY: Final[float] = conf.step_penalty
        self.DIGESTION_ENGINE: Final[DigestionEngine] = DigestionEngine()


    def reset(self) -> None:
        """
        Initialize Droids starting position at the center of the grid and reset its score and the digestion engine.
        """

        self.position: list[int] = [self._GRID_ROWS // 2, self._GRID_COLS // 2]
        self.score: float = self._STARTING_SCORE
        self.DIGESTION_ENGINE.reset()

    # ================= #
    #        API        #
    # ================= #

    def perform_action(self, agent_action: DroidAction) -> float:
        """Performs current action"""

        # Move droid to the next cell
        match agent_action:
            case DroidAction.LEFT:
                self._moveTowardsMinBound(1)
            case DroidAction.RIGHT:
                self._moveTowardsMaxBound(1, self._GRID_COLS - 1)
            case DroidAction.UP:
                self._moveTowardsMinBound(0)
            case DroidAction.DOWN:
                self._moveTowardsMaxBound(0, self._GRID_ROWS - 1)
            case _:
                raise TypeError("This action isn't implemented")

        return self._apply_reward(self._STEP_PENALTY)

    def consume_orb(self, orb: BaseOrb) -> float:
        """Consumes the orb, add its reward to its score and returns the reward"""

        reward = self.DIGESTION_ENGINE.digest(orb.consume())
        return self._apply_reward(reward)

    # ================= #
    #      Helpers      #
    # ================= #

    def _moveTowardsMinBound(self, axis: int) -> None:
        self.position[axis] = max(self.position[axis] - 1, 0)

    def _moveTowardsMaxBound(self, axis: int, bound: int) -> None:
        self.position[axis] = min(self.position[axis] + 1, bound)

    def _apply_reward(self, reward: float):
        self.score += reward
        return reward
