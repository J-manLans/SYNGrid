from enum import Enum
from synergygrid.core.resources import BaseResource


class AgentAction(Enum):
    """Actions the Agent is capable of performing i.e. go in a certain direction"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class SynergyAgent:

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows: int, grid_cols: int, starting_score: int = 20):
        """
        Initializes the agent.

        Defines the game world so the agent know its bounds, set its starting score and store it for later resetting.
        """

        if grid_cols <= 1 or grid_rows <= 1:
            raise ValueError("grid_cols and grid_rows should be larger than 0")

        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self.score = starting_score
        self._starting_score = starting_score

    def reset(self) -> None:
        """Initialize Agents starting position at the center of the grid and reset its score"""

        self.position = [self._grid_rows // 2, self._grid_cols // 2]
        self.score = self._starting_score

    # ================= #
    #        API        #
    # ================= #

    def perform_action(self, agent_action: AgentAction) -> None:
        """Performs current action"""

        # Move Agent to the next cell
        if agent_action == AgentAction.LEFT:
            self._moveTowardsMinBound(1)
        elif agent_action == AgentAction.RIGHT:
            self._moveTowardsMaxBound(1, self._grid_cols - 1)
        elif agent_action == AgentAction.UP:
            self._moveTowardsMinBound(0)
        elif agent_action == AgentAction.DOWN:
            self._moveTowardsMaxBound(0, self._grid_rows - 1)

        self.score -= 1

    def consume_resource(self, resource: BaseResource) -> int:
        """Consumes the resource, add its reward to its score and returns the reward"""

        reward = resource.consume()
        self.score += reward
        return reward

    # ================= #
    #      Helpers      #
    # ================= #

    def _moveTowardsMinBound(self, axis: int) -> None:
        self.position[axis] = max(self.position[axis] - 1, 0)

    def _moveTowardsMaxBound(self, axis: int, bound: int) -> None:
        self.position[axis] = min(self.position[axis] + 1, bound)
