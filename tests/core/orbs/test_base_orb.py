from syn_grid.core.utils.timer import Timer
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    SynergyType,
    OrbMeta,
    OrbCategory,
)

import pytest
import numpy as np


class DummyOrb(BaseOrb):
    """
    Minimal concrete implementation of BaseOrb
    used to isolate and test BaseOrb behavior.
    """

    def consume(self):
        super().consume()
        return self


class TestBaseOrb:
    """
    Unit tests for BaseOrb.

    Tests cover:
    - Initialization validation
    - Spawn lifecycle
    - Depletion behavior
    - Reset behavior
    - Consume flow
    - Timer behavior
    - Edge cases
    """

    @pytest.fixture
    def meta(self):
        return OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, 1)

    @pytest.fixture
    def orb(self, meta):
        BaseOrb.set_life_span(5, 5)
        return DummyOrb(3, 10, meta)

    @pytest.mark.parametrize(
        "rows, cols",
        [
            (5, 5),
            (10, 3),
            (2, 2),
        ],
    )
    def test_valid_initialization(self, meta, rows, cols):
        """
        Verify that BaseOrb initializes correctly for valid grid sizes.

        Ensures that:
        - The orb starts inactive.
        - The internal timer starts at zero.
        - No exception is raised for valid grid dimensions.
        """

        BaseOrb.set_life_span(rows, cols)
        orb = DummyOrb(3.0, 10, meta)
        orb.TIMER.reset()

        assert orb.is_active is False
        assert orb.TIMER.remaining == 0

    def test_spawn_activates_orb(self, orb: DummyOrb):
        """
        Verify that calling spawn():

        - Activates the orb.
        - Sets the orb position correctly.
        - Initializes the timer with a positive lifespan value.
        """

        orb.spawn([np.int64(2), np.int64(3)])

        assert orb.is_active is True
        assert orb.position == [np.int64(2), np.int64(3)]
        assert orb.TIMER.remaining > 0

    def test_deplete_deactivates_and_sets_cool_down(self, orb: DummyOrb):
        """
        Ensure that deplete_orb():

        - Deactivates the orb.
        - Sets the timer to the configured cooldown value.
        - Does not leave the orb active.
        """

        orb.spawn([np.int64(1), np.int64(1)])

        orb.de_spawn()

        assert orb.is_active is False
        assert orb.TIMER.remaining == 10

    def test_reset_restores_default_state(self, orb: DummyOrb):
        """
        Verify that reset():

        - Deactivates the orb.
        - Clears the timer (remaining == 0).
        - Restores the orb to its initial inactive state.
        """

        orb.spawn([np.int64(1), np.int64(1)])

        orb.reset()

        assert orb.is_active is False
        assert orb.TIMER.remaining == 0

    def test_consume_returns_reward_and_sets_cool_down(self, orb: DummyOrb):
        """
        Ensure that consuming a orb:

        - Returns the expected reward value.
        - Deactivates the orb.
        - Sets the timer to the configured cool-down.
        """

        orb.spawn([np.int64(0), np.int64(0)])

        reward = orb.consume().REWARD

        assert reward == 3
        assert orb.is_active is False
        assert orb.TIMER.remaining == 10

    def test_timer_set_and_tick(self):
        """
        Verify Timer lifecycle behavior:

        - set() initializes the remaining duration.
        - tick() decrements the timer correctly.
        - is_completed() returns True when remaining reaches zero.
        """

        timer = Timer()

        timer.set(3)
        assert timer.remaining == 3

        timer.tick()
        assert timer.remaining == 2

        timer.tick()
        timer.tick()

        assert timer.remaining == 0
        assert timer.is_completed() is True

    def test_timer_never_goes_negative(self):
        """
        Ensure that repeated calls to tick() do not cause
        the timer to drop below zero.
        """

        timer = Timer()

        timer.set(1)

        timer.tick()
        timer.tick()
        timer.tick()

        assert timer.remaining == 0
