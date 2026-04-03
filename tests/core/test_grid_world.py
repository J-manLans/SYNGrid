from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.orb_meta import DirectType, SynergyType
from syn_grid.config.config_manager import ConfigManager

from tests.utils.config_helpers import get_test_config

import pytest
import numpy as np


class TestGridWorld:
    @pytest.fixture
    def grid_world(self):
        """
        Fixture to create and initialize a GridWorld instance with 1 active orb.
        Resets the world before each test to ensure a clean state.
        """

        config_manager = ConfigManager("test_configs.yaml")

        run_conf = get_test_config().world
        grid_world_conf = run_conf.grid_world_conf
        orb_manager_conf = run_conf.orb_factory_conf
        droid_conf = run_conf.droid_conf
        negative_orb_conf = run_conf.negative_orb_conf
        tier_orb_conf = run_conf.tier_orb_conf

        gw = GridWorld(
            grid_world_conf,
            orb_manager_conf,
            droid_conf,
            negative_orb_conf,
            tier_orb_conf,
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
        assert grid_world._grid_rows == 5  # The grid should have 5 row.
        assert grid_world._grid_cols == 5  # The grid should have 5 columns.

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
