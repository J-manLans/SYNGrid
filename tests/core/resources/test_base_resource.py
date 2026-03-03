import pytest
import numpy as np
from synergygrid.core.resources import (
    BaseResource,
    DirectType,
    ResourceMeta,
    ResourceCategory,
)


class DummyResource(BaseResource):
    """
    Minimal concrete implementation of BaseResource
    used to isolate and test BaseResource behavior.
    """
    def consume(self):
        self._consume()
        return 1


class TestBaseResource:
    """
    Unit tests for BaseResource.

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
        return ResourceMeta(ResourceCategory.DIRECT, DirectType.POSITIVE, 0)

    @pytest.fixture
    def resource(self, meta):
        return DummyResource((5, 5), 3, meta)

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
        Verify that BaseResource initializes correctly for valid grid sizes.

        Ensures that:
        - The resource starts inactive.
        - The internal timer starts at zero.
        - No exception is raised for valid grid dimensions.
        """
        resource = DummyResource((rows, cols), 3, meta)

        assert resource.is_active is False
        assert resource.timer.remaining == 0

    @pytest.mark.parametrize(
        "rows, cols",
        [
            (1, 5),
            (5, 1),
            (0, 5),
            (5, 0),
        ],
    )
    def test_invalid_grid_raises(self, meta, rows, cols):
        """
        Ensure that creating a resource with invalid grid dimensions
        (rows or columns <= 1) raises a ValueError.

        This validates constructor boundary checks.
        """
        with pytest.raises(ValueError):
            DummyResource((rows, cols), 3, meta)

    def test_spawn_activates_resource(self, resource):
        """
        Verify that calling spawn():

        - Activates the resource.
        - Sets the resource position correctly.
        - Initializes the timer with a positive lifespan value.
        """
        resource.spawn([np.int64(2), np.int64(3)])

        assert resource.is_active is True
        assert resource.position == [np.int64(2), np.int64(3)]
        assert resource.timer.remaining > 0

    def test_deplete_deactivates_and_sets_cooldown(self, resource):
        """
        Ensure that deplete_resource():

        - Deactivates the resource.
        - Sets the timer to the configured cooldown value.
        - Does not leave the resource active.
        """
        resource.spawn([np.int64(1), np.int64(1)])

        resource.deplete_resource()

        assert resource.is_active is False
        assert resource.timer.remaining == 3

    def test_reset_restores_default_state(self, resource):
        """
        Verify that reset():

        - Deactivates the resource.
        - Clears the timer (remaining == 0).
        - Restores the resource to its initial inactive state.
        """
        resource.spawn([np.int64(1), np.int64(1)])

        resource.reset()

        assert resource.is_active is False
        assert resource.timer.remaining == 0

    def test_consume_returns_reward_and_sets_cooldown(self, resource):
        """
        Ensure that consuming a resource:

        - Returns the expected reward value.
        - Deactivates the resource.
        - Sets the timer to the configured cooldown.
        """
        resource.spawn([np.int64(0), np.int64(0)])

        reward = resource.consume()

        assert reward == 1
        assert resource.is_active is False
        assert resource.timer.remaining == 3

    def test_timer_set_and_tick(self):
        """
        Verify Timer lifecycle behavior:

        - set() initializes the remaining duration.
        - tick() decrements the timer correctly.
        - is_completed() returns True when remaining reaches zero.
        """
        timer = BaseResource.Timer()

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
        timer = BaseResource.Timer()

        timer.set(1)

        timer.tick()
        timer.tick()
        timer.tick()

        assert timer.remaining == 0