from syn_grid.config.models import ObservationConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.orb_meta import OrbCategory
from syn_grid.core.orbs.orb_meta import DirectType
from syn_grid.core.orbs.orb_meta import SynergyType

from numpy.typing import NDArray
import numpy as np
from typing import Any
from gymnasium import spaces
from gymnasium.spaces import Dict


class ObservationHandler:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world: GridWorld, obs_conf: ObservationConf):
        self._world = world
        self._grid_rows = obs_conf.grid_rows
        self._grid_cols = obs_conf.grid_cols
        self._max_tier = obs_conf.max_tier
        self._max_steps = obs_conf.max_steps

    def reset(self):
        self.step_count_down = self._max_steps

    # ================== #
    #       API      #
    # ================== #

    def setup_obs_space(self) -> Dict:
        """
        Gymnasium requires an observation space definition. Here we represent the state as a flat
        vector. The space is used by Gymnasium to validate observations returned by reset() and step().

        Set up:
        - self._raw_low / self._raw_high: the original raw ranges (used for normalization)
        - self.*_mask: boolean mask that indicates "absent" orbs
        - self.observation_space: the normalized observation space that agents will see
        (0..1 for active features, -1 for absent fields)
        """
        # original raw bounds — match _get_observation()
        agent_raw_low, self._agent_raw_high = self._build_agent_box_bounds(
            False, np.float16
        )
        orb_raw_low, self._orb_raw_high = self._build_orb_box_bounds(False, np.float16)

        # reusable buffers for get_observation() to avoid per-step allocations
        # TODO: think I need to switch this to something else, because reward is now float
        self._agent_data = np.zeros_like(agent_raw_low, dtype=np.int16)
        self._orb_data = np.zeros_like(orb_raw_low, dtype=np.int16)

        # normalized bounds — match _normalize_obs()
        # inactive orbs keep -1 as a valid "low" value; active features map to 0..1
        agent_low_norm, agent_high_norm = self._build_agent_box_bounds(True, np.float16)
        orb_low_norm, orb_high_norm = self._build_orb_box_bounds(True, np.float16)

        self.observation_space: dict[str, spaces.Space] = {
            "agent data": spaces.Box(
                # steps, row, col, current tier chain
                low=agent_low_norm,
                high=agent_high_norm,
                dtype=np.float16,
            ),
            "orbs data": spaces.Box(
                # row, col, timer, tier
                low=orb_low_norm,
                high=orb_high_norm,
                dtype=np.float16,
            ),
        }

        return spaces.Dict(self.observation_space)

    def get_observation(self) -> dict[str, NDArray]:
        agent_row, agent_col = self._world.droid.position

        # NOTE: change here
        # ---- Agent ---- #
        self._agent_data[0] = agent_row
        self._agent_data[1] = agent_col
        self._agent_data[2] = self.step_count_down
        self._agent_data[3] = self._world.droid.score
        self._agent_data[4] = self._world.droid.digestion_engine.chained_tiers

        # NOTE: change here
        # ---- Orbs ---- #
        active = self._world.get_orb_is_active_status(False)
        positions = self._world.get_orb_positions(False)
        remaining = self._world.get_orb_life()
        categories = self._world.get_orb_categories()
        types = self._world.get_orb_types()
        tiers = self._world.get_orb_tiers()

        for i in range(len(self._world.ALL_ORBS)):
            if active[i]:
                # NOTE: change here
                pos = positions[i]
                r_timer = remaining[i]
                r_cat = int(categories[i])
                r_type = int(types[i])
                r_tier = tiers[i]

                # NOTE: change here
                self._orb_data[i] = [
                    pos[0],
                    pos[1],
                    r_timer,
                    r_cat,
                    r_type,
                    r_tier,
                ]
            else:
                # NOTE: change here
                self._orb_data[i] = [-1, -1, -1, -1, -1, -1]

        return {"agent data": self._agent_data, "orbs data": self._orb_data}

    def normalize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        # --- Agent --- #
        agent = obs["agent data"]
        absent_mask = agent == -1.0

        norm_agent = self._normalize_obs_fields(
            agent, absent_mask, self._agent_raw_high
        )

        # --- Orbs --- #
        norm_res = obs["orbs data"].astype(np.float16)
        for i in range(norm_res.shape[0]):
            row = norm_res[i]
            absent_mask = row == -1.0

            norm_row = self._normalize_obs_fields(
                row, absent_mask, self._orb_raw_high[i]
            )
            norm_res[i] = norm_row

        return {
            "agent data": norm_agent,
            "orbs data": norm_res,
        }

    # ================== #
    #       Helpers      #
    # ================== #

    # TODO: Go over these. They work well for direct orbs. Thinking a subclass for different
    # orb types would be needed so constant flipping back and forth when testing different
    # things which introduce bugs can be avoided.
    def _build_agent_box_bounds(
        self, normalized: bool, arr_type
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        if normalized:
            min_row = min_col = min_steps = min_score = 0.0
            min_tier_chain = -1.0

            max_steps = max_row = max_col = max_score = max_tier_chain = 1.0
        else:
            min_row = min_col = min_steps = min_score = 0
            min_tier_chain = -1

            max_row = self._grid_rows - 1
            max_col = self._grid_cols - 1
            max_steps = self._max_steps
            max_score = 50
            # guards against div / 0 when just using direct rewards
            max_tier_chain = max(1, self._max_tier)

        # NOTE: change here
        low = [min_row, min_col, min_steps, min_score, min_tier_chain]
        high = [max_row, max_col, max_steps, max_score, max_tier_chain]

        low_arr = np.asarray(low, dtype=arr_type)
        high_arr = np.asarray(high, dtype=arr_type)

        self._control_arrays(low_arr, high_arr)

        return low_arr, high_arr

    def _build_orb_box_bounds(
        self, normalized: bool, arr_type
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        if normalized:
            no_orb_yx = -1.0
            min_r_life_span = -1.0
            min_r_tier = -1.0
            min_r_cat = -1.0
            min_r_type = -1.0

            max_row = 1.0
            max_col = 1.0
            max_r_life_span = 1.0
            max_r_tier = 1.0
            max_r_cat = 1.0
            max_r_type = 1.0
        else:
            no_orb_yx = -1
            min_r_life_span = -1
            min_r_cat = -1
            min_r_type = -1
            min_r_tier = -1

            max_row = self._grid_rows - 1
            max_col = self._grid_cols - 1
            max_r_life_span = (self._grid_rows - 1) + (self._grid_cols - 1)
            max_r_cat = len(OrbCategory) - 1
            max_r_type = max(1, max(len(DirectType) - 1, len(SynergyType) - 1))
            # guards against div / 0 when just using direct rewards
            max_r_tier = max(1, self._max_tier)

        N = len(self._world.ALL_ORBS)
        # NOTE: change here
        low = np.tile(
            [
                no_orb_yx,
                no_orb_yx,
                min_r_life_span,
                min_r_cat,
                min_r_type,
                min_r_tier,
            ],
            (N, 1),
        )
        # NOTE: change here
        high = np.tile(
            [max_row, max_col, max_r_life_span, max_r_cat, max_r_type, max_r_tier],
            (N, 1),
        )

        low_arr = np.asarray(low, dtype=arr_type)
        high_arr = np.asarray(high, dtype=arr_type)

        self._control_arrays(low_arr, high_arr)

        return low_arr, high_arr

    def _control_arrays(self, low_arr, high_arr):
        if low_arr.shape != high_arr.shape:
            raise ValueError(
                f"low/high shape mismatch: {low_arr.shape} != {high_arr.shape}"
            )
        if np.any(high_arr <= low_arr):
            raise ValueError(
                "All high bounds must be greater than low bounds for raw ranges"
            )

    def _normalize_obs_fields(
        self, array: NDArray[np.float16], mask, scale: NDArray[np.float16]
    ) -> NDArray[np.float16]:
        """
        Normalize observation to 0..1 while preserving sentinel values (-1) for absent orbs.
        """

        # Prepare output array
        norm_array = np.empty_like(array, dtype=np.float16)

        # Indices for "regular" entries and "masked/special" entries
        masked_idx = np.where(mask)[0]
        regular_idx = np.where(~mask)[0]

        # Normalize regular entries element wise
        norm_array[regular_idx] = array[regular_idx] / scale[regular_idx]

        if mask.any():
            # Handle masked/special entries
            special_values = array[masked_idx]

            absent_mask = special_values == -1.0
            present_mask = ~absent_mask

            # Keep sentinel values
            norm_array[masked_idx[absent_mask]] = -1.0

            # Normalize only present values
            norm_array[masked_idx[present_mask]] = (
                special_values[present_mask] / scale[masked_idx[present_mask]]
            )

        return norm_array
