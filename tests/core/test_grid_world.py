import pytest
from synergygrid.core import (
    GridWorld,
    AgentAction,
    ResourceMeta,
    DirectType,
    SynergyType,
)
import numpy as np


class TestGridWorld:
    @pytest.fixture
    def grid_world(self):
        """
        Fixture to create and initialize a GridWorld instance with 1 active resource.
        Resets the world before each test to ensure a clean state.
        """
        gw = GridWorld(max_active_resources=1, grid_rows=1, grid_cols=2)
        gw.reset()

        return gw

    def test_initialization(self, grid_world: GridWorld):
        """
        Test that verifies the initialization of the GridWorld object.
        Checks if the grid's rows and columns are correctly set and that the grid contains the expected number of resources.
        """
        active_resources = grid_world.get_resource_is_active_status(False)
        active_cnt = sum(active_resources)

        assert (
            active_cnt == 1
        )  # There should be exactly one active resource after initialization.
        assert grid_world.grid_rows == 1  # The grid should have 1 row.
        assert grid_world.grid_cols == 2  # The grid should have 2 columns.

    def test_resource_positions(self, grid_world: GridWorld):
        """
        Test to verify that the resource positions are correctly returned.
        Ensures each resource position is a list with exactly two elements (representing x, y coordinates) and that they are of type np.int64.
        """
        positions = grid_world.get_resource_positions(False)

        for pos in positions:
            assert isinstance(pos, list)  # Each position should be a list.
            assert len(pos) == 2  # Each list should contain two elements (x, y).
            assert isinstance(pos[0], np.integer) and isinstance(
                pos[1], np.integer
            )  # Each coordinate should be an np.integer.

    def test_get_resource_is_active_status(self, grid_world: GridWorld):
        """
        Test that checks if the resource's active status is returned as a boolean.
        Ensures that each resource's status is either True or False.
        """
        statuses = grid_world.get_resource_is_active_status(False)

        for status in statuses:
            assert isinstance(status, bool)  # Each status should be a boolean.

    def test_get_resource_types(self, grid_world: GridWorld):
        """
        Verify that get_resource_types() returns valid indices corresponding to enums.

        Each integer returned should not exceed the length of the largest type enum
        (DirectType or SynergyType). This ensures that the resource integers are
        valid indices for the respective resource types in the world.
        """
        types = grid_world.get_resource_types()

        for t in types:
            max_type = max(len(DirectType), len(SynergyType))
            assert (
                t <= max_type
            )  # Integer should not exceed the max number of enum members.

    def test_get_resource_timers(self, grid_world: GridWorld):
        """
        Test to ensure that each resource's timer is correctly returned and is an integer signaling remaining life.
        This confirms that the resources have valid timers.
        """
        timers = grid_world.get_resource_life()

        for timer in timers:
            # Each timer should be an integer signaling remaining life.
            assert isinstance(timer, int)
