from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.gymnasium.action_space import AgentAction
from synergygrid.core.agent.digestion_engine import DigestionEngine


class SynergyAgent:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows: int, grid_cols: int, starting_score: int = 40):
        """
        Initializes the agent.

        Defines the game world so the agent know its bounds, set its starting score and store it for later resetting.
        """

        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self.score = starting_score
        self._starting_score = starting_score
        self.digestion_engine = DigestionEngine()

    def reset(self) -> None:
        """Initialize Agents starting position at the center of the grid and reset its score"""

        self.position = [self._grid_rows // 2, self._grid_cols // 2]
        self.score = self._starting_score

    # ================= #
    #        API        #
    # ================= #

    def perform_action(self, agent_action: AgentAction) -> int:
        """Performs current action"""

        # Move Agent to the next cell
        match agent_action:
            case AgentAction.LEFT:
                self._moveTowardsMinBound(1)
            case AgentAction.RIGHT:
                self._moveTowardsMaxBound(1, self._grid_cols - 1)
            case AgentAction.UP:
                self._moveTowardsMinBound(0)
            case AgentAction.DOWN:
                self._moveTowardsMaxBound(0, self._grid_rows - 1)
            case _:
                raise TypeError("This action isn't implemented")

        self.score -= 1
        return -1

    def consume_resource(self, resource: BaseResource) -> int:
        """Consumes the resource, add its reward to its score and returns the reward"""

        reward = self.digestion_engine.digest(resource.consume())
        self.score += reward
        return reward

    # ================= #
    #      Helpers      #
    # ================= #

    def _moveTowardsMinBound(self, axis: int) -> None:
        self.position[axis] = max(self.position[axis] - 1, 0)

    def _moveTowardsMaxBound(self, axis: int, bound: int) -> None:
        self.position[axis] = min(self.position[axis] + 1, bound)
