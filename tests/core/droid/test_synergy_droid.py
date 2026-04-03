from syn_grid.core.droid.synergy_droid import DroidAction, SynergyDroid
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    OrbMeta,
    OrbCategory,
    DirectType,
    SynergyType,
)

from tests.utils.config_helpers import get_test_config

import pytest


class DummyPositiveOrb(BaseOrb):
    """
    Minimal stub orb for testing.

    Simulates a orb that always returns a fixed positive reward
    when consumed. Used to isolate droid logic from orb logic.
    """

    def consume(self):
        super()._consume()
        return self


class DummyNegativeOrb(BaseOrb):
    """
    Minimal stub orb for testing.

    Simulates a orb that always returns a fixed negative reward
    when consumed. Used to test score deduction logic.
    """

    def consume(self):
        super()._consume()
        return self


class TestSynergyDroid:
    """
    Unit tests for the SynergyDroid class.

    Tests cover:
    - Droid initialization and score
    - Movement in all directions
    - Boundary handling
    - Reset behavior
    - Orb consumption (positive and negative)
    - Edge cases (minimal grid, invalid inputs)
    """

    @pytest.fixture
    def droid(self):
        """
        Returns a SynergyDroid instance reset to its initial state. Used as a reusable fixture for most tests.
        """

        droid = SynergyDroid(get_test_config().world.droid_conf)
        droid.reset()

        return droid

    @pytest.mark.parametrize(
        "y, x, expected_position",
        [
            (6, 6, [3, 3]),
            (5, 5, [2, 2]),
            (1, 1, [0, 0]),
        ],
    )
    def test_initial_positions(self, y, x, expected_position):
        """
        Check that the droid starts at the center of the grid when reset,
        for various grid sizes.
        """

        conf = get_test_config().world.droid_conf.model_copy(update={"grid_rows": y, "grid_cols": x})

        droid = SynergyDroid(conf)
        droid.reset()

        assert droid.position == expected_position

    @pytest.mark.parametrize("score", [100, 0, -100])
    def test_initial_score(self, score):
        """
        Ensure that the droid's score is initialized to the starting value
        after creation.
        """

        conf = get_test_config().world.droid_conf.model_copy(update={"starting_score": score})

        droid = SynergyDroid(conf)

        assert droid.score == score

    @pytest.mark.parametrize(
        "action, expected_position",
        [
            (DroidAction.RIGHT, [2, 3]),
            (DroidAction.LEFT, [2, 1]),
            (DroidAction.UP, [1, 2]),
            (DroidAction.DOWN, [3, 2]),
        ],
    )
    def test_movement(
        self, droid: SynergyDroid, action: DroidAction, expected_position: list[int]
    ):
        """
        Verify that each movement action correctly updates the droid's position
        when not at a boundary.
        """

        initial_score = droid.score
        droid.perform_action(action)

        assert droid.position == expected_position
        assert initial_score == droid.score + 1

    @pytest.mark.parametrize("action", [None, "LEFT"])
    def test_invalid_movement(self, droid: SynergyDroid, action: DroidAction):
        """
        Ensure that passing an invalid value to perform_action
        (anything other than an DroidAction enum) raises a TypeError.
        """

        with pytest.raises(TypeError):
            droid.perform_action(action)

    @pytest.mark.parametrize(
        "position, action, expected_position",
        [
            ([0, 0], DroidAction.LEFT, [0, 0]),
            ([0, 4], DroidAction.RIGHT, [0, 4]),
            ([0, 0], DroidAction.UP, [0, 0]),
            ([4, 0], DroidAction.DOWN, [4, 0]),
        ],
    )
    def test_boundaries(
        self,
        droid: SynergyDroid,
        position: list[int],
        action: DroidAction,
        expected_position: list[int],
    ):
        """
        Ensure the droid does not move outside the grid boundaries
        when attempting moves at edges.
        """

        droid.position = position.copy()
        droid.perform_action(action)

        assert droid.position == expected_position

    @pytest.mark.parametrize(
        "action",
        [DroidAction.LEFT, DroidAction.RIGHT, DroidAction.UP, DroidAction.DOWN],
    )
    def test_edge_case_boundaries(self, action: DroidAction):
        """
        Verify that on a minimal 1x1 grid, the droid cannot move
        in any direction and remains at [0, 0].
        """

        conf = get_test_config().world.droid_conf.model_copy(update={"grid_rows": 1, "grid_cols": 1})

        droid = SynergyDroid(conf)
        droid.reset()
        droid.perform_action(action)

        assert droid.position == [0, 0]

    def test_reset_restores_position(self, droid: SynergyDroid):
        """
        Ensure that calling reset() restores the droid's position
        to the center of the grid.
        """

        starting_position = droid.position.copy()
        droid.position = [0, 0]
        droid.reset()

        assert droid.position == starting_position

    def test_reset_restores_score(self, droid: SynergyDroid):
        """
        Ensure that calling reset() restores the droid's score
        to its original starting value.
        """

        starting_score = droid.score
        droid.score += 10
        droid.reset()

        assert droid.score == starting_score

    def test_consume_orb_adds_score(self, droid: SynergyDroid):
        """
        Verify that consuming a positive orb increases the droid's score
        and returns the correct reward value.
        """

        orb = DummyPositiveOrb(3, 10, OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, 0))
        reward = droid.consume_orb(orb)

        assert reward == 3
        assert droid.score == 43

    def test_consume_negative_orb_reduces_score(self, droid: SynergyDroid):
        """
        Verify that consuming a negative orb decreases the droid's score
        and returns the correct negative reward value.
        """

        orb = DummyNegativeOrb(-3, 5, OrbMeta(OrbCategory.DIRECT, DirectType.NEGATIVE))
        reward = droid.consume_orb(orb)

        assert reward == -3
        assert droid.score == 37
