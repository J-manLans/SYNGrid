from typing import Final

from syn_grid.core.droid.digestion.digester import OrbDigester
from syn_grid.core.droid.digestion.engine import D, DigestionEngine
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.droid.droid_action import DroidAction


class SynergyDroid:
    """The agent's body in the world: it moves on the grid, eats orbs and keeps energy and score.

    Every reward the agent is paid goes through the droid: what its moves cost, what its digestion
    pays for an eaten orb, and what is settled when the episode is terminated. Energy is what the
    droid has left. It starts at the max energy, stays between 0 and it, and takes in the rewards
    paid during the episode. Score is the episode's return. It starts at 0, has no bounds, and also
    takes in what is settled at the end.

    The droid digests through its own engine, built from the digesters it is handed. Nothing
    outside reaches the engine; a digester is read through ``get_digester``.

    Attributes:
        position: the droid's cell as ``[row, col]``.
        energy: what the droid has left, between 0 and the max energy.
        score: the sum of every reward the droid has taken in this episode.
    """

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
        """Build the droid and its digestion engine.

        Args:
            digesters: one per orb kind the world holds. The droid builds its engine from them.
            grid_dimensions: the grid's rows and columns, which bound the droid's moves.
            max_energy: the energy the droid starts each episode with, and the most it can hold.
            step_penalty: what each move costs.
            boundary_penalty: what a move into the edge of the grid costs, on top of the step.
        """

        self._rows: Final[int] = grid_dimensions[0]
        self._cols: Final[int] = grid_dimensions[1]
        self._max_energy: Final[float] = max_energy
        self._step_penalty: Final[float] = step_penalty
        self._boundary_penalty: Final[float] = boundary_penalty

        self._digestion_engine: Final[DigestionEngine] = DigestionEngine(digesters)

    def reset(self) -> None:
        """Start an episode: the droid at the centre of the grid, at full energy, with a score of 0.

        It also clears the episode state of every digester.
        """

        self.position: list[int] = [
            self._rows // 2,
            self._cols // 2,
        ]
        self.energy: float = self._max_energy
        self.score: float = 0.0
        self._digestion_engine.reset()

    # ================= #
    #        API        #
    # ================= #

    def perform_action(self, agent_action: DroidAction) -> float:
        """Move the droid one cell and return what the move cost.

        A move costs the step penalty. A move into the edge of the grid leaves the droid where it
        is and also costs the boundary penalty. The cost goes to energy and score.
        """

        # Move droid to the next cell
        match agent_action:
            case DroidAction.LEFT:
                boundary_penalty = self._move_towards_min_bound(1)
            case DroidAction.RIGHT:
                boundary_penalty = self._move_towards_max_bound(1, self._cols - 1)
            case DroidAction.UP:
                boundary_penalty = self._move_towards_min_bound(0)
            case DroidAction.DOWN:
                boundary_penalty = self._move_towards_max_bound(0, self._rows - 1)
            case _:
                raise TypeError("This action isn't implemented")

        return self._apply_reward(self._step_penalty + boundary_penalty)

    def consume_orb(self, orb: BaseOrb) -> float:
        """Eat an orb and return what its digestion pays.

        The orb is digested by the digester of its kind, and what that pays goes to energy and
        score.
        """

        reward = self._digestion_engine.digest(orb.consume())
        return self._apply_reward(reward)

    def get_digester(self, digester_class: type[D]) -> D:
        """Return the droid's digester that is an instance of this class.

        This is how termination and the scenario's readers reach what a digester knows.
        """

        return self._digestion_engine.get(digester_class)

    def on_termination(self, ending_reward: float) -> float:
        """Settle the droid when the episode is terminated, and return what that adds to the last
        step's reward.

        The droid takes in what its digestion has gathered and not paid, together with what the
        ending costs or pays. Their sum goes to the score and leaves energy as the episode
        ended it, so a droid that ran out of energy still has none. It is called once per episode.
        """

        settled = self._digestion_engine.undigested() + ending_reward
        self.score += settled

        return settled

    # ================= #
    #      Helpers      #
    # ================= #

    def _move_towards_min_bound(self, axis: int) -> float:
        """Step one cell towards 0 on this axis; return the boundary penalty if blocked."""

        if self.position[axis] - 1 < 0:
            self.position[axis] = 0
            return self._boundary_penalty

        self.position[axis] = self.position[axis] - 1
        return 0.0

    def _move_towards_max_bound(self, axis: int, bound: int) -> float:
        """Step one cell towards the bound on this axis; return the boundary penalty if blocked."""

        if self.position[axis] + 1 > bound:
            self.position[axis] = bound
            return self._boundary_penalty

        self.position[axis] = self.position[axis] + 1
        return 0.0

    def _apply_reward(self, reward: float) -> float:
        """Add a reward to energy and score, and return it unchanged."""

        # Clamp the energy to a minimum of 0 and a max of max_energy to not overload the poor droid
        # (this also has positive implications for the observation)
        self.energy = min(max(self.energy + reward, 0.0), self._max_energy)
        self.score += reward

        return reward
