from typing import Final

from syn_grid.core.droid.digestion.digester import OrbDigester
from syn_grid.core.droid.digestion.engine import D, DigestionEngine
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.gymnasium.action_space import DroidAction


class SynergyDroid:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        digesters: list[OrbDigester],
        grid_dimensions: tuple[int, int],
        max_energy: float,
        step_penalty: float,
        boundary_penalty: float,
    ):
        """
        Initializes the droid.

        Defines the game world so the droid know its bounds, and store its max energy for later
        resetting.
        """

        self._rows: Final[int] = grid_dimensions[0]
        self._cols: Final[int] = grid_dimensions[1]
        self._max_energy: Final[float] = max_energy
        self._step_penalty: Final[float] = step_penalty
        self._boundary_penalty: Final[float] = boundary_penalty

        self.digestion_engine: Final[DigestionEngine] = DigestionEngine(digesters)

    def reset(self) -> None:
        """
        Initialize Droids starting position at the center of the grid and reset its energy, score
        and the digestion engine.
        """

        self.position: list[int] = [
            self._rows // 2,
            self._cols // 2,
        ]
        self.energy: float = self._max_energy
        self.score: float = 0.0
        self.digestion_engine.reset()

    # ================= #
    #        API        #
    # ================= #

    def perform_action(self, agent_action: DroidAction) -> float:
        """Performs current action"""

        # Move droid to the next cell
        match agent_action:
            case DroidAction.LEFT:
                boundary_penalty = self._move_towards_min_bound(1)
            case DroidAction.RIGHT:
                boundary_penalty = self._move_towards_max_bound(
                    1, self._cols - 1
                )
            case DroidAction.UP:
                boundary_penalty = self._move_towards_min_bound(0)
            case DroidAction.DOWN:
                boundary_penalty = self._move_towards_max_bound(
                    0, self._rows - 1
                )
            case _:
                raise TypeError("This action isn't implemented")

        return self._apply_reward(self._step_penalty + boundary_penalty)

    def consume_orb(self, orb: BaseOrb) -> float:
        """Consumes the orb, add its reward to its energy and score and returns the reward"""

        reward = self.digestion_engine.digest(orb.consume())
        return self._apply_reward(reward)

    def get_digester(self, digester_class: type[D]) -> D:
        """Return the droid's digester that is an instance of this class.

        This is how termination and the scenario's readers reach what a digester knows. Asking for a
        base class finds the subclass the droid holds. It raises when the droid holds none.
        """

        return self.digestion_engine.get(digester_class)

    def on_termination(self, ending_reward: float) -> float:
        """
        Settles the droid when the episode is terminated, and returns what that adds to the last
        step's reward.

        The droid takes in what its digestion has gathered and not paid, together with what the
        ending costs or pays. Their sum goes to the score and leaves energy as the episode
        ended it, so a droid that ran out of energy still has none. It is called once per episode.
        """

        settled = self.digestion_engine.undigested() + ending_reward
        self.score += settled

        return settled

    # ================= #
    #      Helpers      #
    # ================= #

    def _move_towards_min_bound(self, axis: int) -> float:
        if self.position[axis] - 1 < 0:
            self.position[axis] = 0
            return self._boundary_penalty

        self.position[axis] = self.position[axis] - 1
        return 0.0

    def _move_towards_max_bound(self, axis: int, bound: int) -> float:
        if self.position[axis] + 1 > bound:
            self.position[axis] = bound
            return self._boundary_penalty

        self.position[axis] = self.position[axis] + 1
        return 0.0

    def _apply_reward(self, reward: float) -> float:
        # Clamp the energy to a minimum of 0 and a max of max_energy to not overload the poor droid
        # (this also has positive implications for the observation)
        self.energy = min(max(self.energy + reward, 0.0), self._max_energy)
        self.score += reward

        return reward
