from syn_grid.config.models import ObservationConf
from syn_grid.core.grid_world import GridWorld

from abc import ABC, abstractmethod
from gymnasium import spaces
from numpy.typing import NDArray

class BaseObservationHandler(ABC):
    def __init__(self, world: GridWorld, obs_conf: ObservationConf):
        """
        Initialize the observation handler.

        Args:
            world: The environment instance (e.g., GridWorld).
                Provides access to agent state, objects, etc.
            obs_conf: Configuration object containing observation parameters
                (e.g., grid size, normalization limits).
        """
        self._world = world
        self._grid_rows = obs_conf.grid_rows
        self._grid_cols = obs_conf.grid_cols

    @abstractmethod
    def setup_obs_space(self) -> spaces.Space:
        """
        Define the Gymnasium observation space.

        Returns:
            spaces.Box:
                A Box space with shape (H, W, C), matching the output of
                `get_observation()` after normalization.

        Requirements:
        - Bounds must match the normalized output range
        - Typically [0, 1] for active values
        - Must be consistent across episodes
        """

    @abstractmethod
    def get_observation(self) -> NDArray:
        """
        Extract the raw observation from the environment.

        This method should:
        - Read the current world state
        - Construct a tensor of shape (H, W, C)
        - Populate channels according to the handler's design

        Returns:
            NDArray:
                A tensor of shape (H, W, C) representing the current state.
                Values may be unnormalized.

        Notes:
        - No normalization should happen here
        - No randomness should be introduced (deterministic mapping)
        """

    @abstractmethod
    def normalize_obs(self, obs: NDArray) -> NDArray:
        """
        Normalize the observation tensor.

        Args:
            obs (NDArray):
                Raw observation from `get_observation()`.

        Returns:
            NDArray:
                Normalized observation tensor with same shape (H, W, C).

        Requirements:
        - Must not change shape
        - Should map values into the bounds defined in `setup_obs_space()`
        - Should preserve semantic meaning (e.g., no mixing channels)

        Notes:
        - If observations are already normalized, this can be a no-op
        """
