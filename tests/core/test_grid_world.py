from syn_grid.config.models import (
    GridWorldConf,
    OrbConf,
    OrbManagerConf,
    DroidConf,
    NegativeConf,
    TierConf,
)
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.orb_meta import DirectType, SynergyType

import pytest
import numpy as np


class TestGridWorld:
    @pytest.fixture
    def grid_world(self):
        """
        Fixture to create and initialize a GridWorld instance with 1 active orb.
        Resets the world before each test to ensure a clean state.

        def __init__(
            self,
            conf: GridWorldConf,
            orb_manager_conf: OrbManagerConf,
            droid_conf: DroidConf,
            negative_orb_conf: NegativeConf,
            tier_orb_conf: TierConf,
        ):

        """

        grid_world_conf = GridWorldConf(
            grid_rows=5, grid_cols=5, max_active_orbs=1, max_tier=1
        )
        orb_manager_conf = OrbManagerConf(
            negative=OrbConf(enabled=True, weight=1),
            tier=OrbConf(enabled=True, weight=2)
        )
        droid_conf = DroidConf(grid_rows=5, grid_cols=5, starting_score=40)
        negative_orb_conf = NegativeConf(reward=-3, cool_down=7)
        tier_orb_conf = TierConf(
            linear_reward_growth=True,
            step_wise_scoring=True,
            growth_factor=1.5,
            base_reward=3,
            cool_down=10,
        )

        gw = GridWorld(
            grid_world_conf,
            orb_manager_conf,
            droid_conf,
            negative_orb_conf,
            tier_orb_conf
        )
        gw.reset()

        return gw

    def test_initialization(self, grid_world: GridWorld):
        """
        Test that verifies the initialization of the GridWorld object.
        Checks if the grid's rows and columns are correctly set and that the grid contains the expected number of orbs.
        """
        active_orbs = grid_world.get_orb_is_active_status(False)
        active_cnt = sum(active_orbs)

        assert (
            active_cnt == 1
        )  # There should be exactly one active orb after initialization.
        assert grid_world.grid_rows == 1  # The grid should have 1 row.
        assert grid_world.grid_cols == 2  # The grid should have 2 columns.

    def test_orb_positions(self, grid_world: GridWorld):
        """
        Test to verify that the orb positions are correctly returned.
        Ensures each orb position is a list with exactly two elements (representing x, y coordinates) and that they are of type np.int64.
        """
        positions = grid_world.get_orb_positions(False)

        for pos in positions:
            assert isinstance(pos, list)  # Each position should be a list.
            assert len(pos) == 2  # Each list should contain two elements (x, y).
            assert isinstance(pos[0], np.integer) and isinstance(
                pos[1], np.integer
            )  # Each coordinate should be an np.integer.

    def test_get_orb_is_active_status(self, grid_world: GridWorld):
        """
        Test that checks if the orb's active status is returned as a boolean.
        Ensures that each orb's status is either True or False.
        """
        statuses = grid_world.get_orb_is_active_status(False)

        for status in statuses:
            assert isinstance(status, bool)  # Each status should be a boolean.

    def test_get_orb_types(self, grid_world: GridWorld):
        """
        Verify that get_orb_types() returns valid indices corresponding to enums.

        Each integer returned should not exceed the length of the largest type enum
        (DirectType or SynergyType). This ensures that the orb integers are
        valid indices for the respective orb types in the world.
        """
        types = grid_world.get_orb_types()

        for t in types:
            max_type = max(len(DirectType), len(SynergyType))
            assert (
                t <= max_type
            )  # Integer should not exceed the max number of enum members.

    def test_get_orb_timers(self, grid_world: GridWorld):
        """
        Test to ensure that each orb's timer is correctly returned and is an integer signaling remaining life.
        This confirms that the orbs have valid timers.
        """
        timers = grid_world.get_orb_life()

        for timer in timers:
            # Each timer should be an integer signaling remaining life.
            assert isinstance(timer, int)
