"""How a timer counts a set number of steps down, or never completes when set to None."""

from collections.abc import Callable

from syn_grid.core.utils.timer import Timer

# ================= #
#     Constants     #
# ================= #


type Tick = Callable[[Timer, int], Timer]


# ================= #
#      Helpers      #
# ================= #


def _set(duration: int | None) -> Timer:
    """A new timer set to `duration`."""

    timer = Timer()
    timer.set(duration)

    return timer


# ================= #
#       Tests       #
# ================= #


class TestCountdown:
    """A timer set to a number of steps."""

    def test_a_new_timer_is_completed(self):
        assert Timer().is_completed()

    def test_it_completes_on_its_last_step_and_not_before(self, tick: Tick):
        timer = _set(3)

        assert not tick(timer, 2).is_completed()
        assert tick(timer, 1).is_completed()

    def test_a_timer_set_to_0_is_completed_at_once(self):
        assert _set(0).is_completed()

    def test_a_completed_timer_stays_completed_and_counts_a_new_duration_in_full(
        self, tick: Tick
    ):
        timer = tick(_set(1), 5)

        assert timer.is_completed()

        # Ticks past the end leave nothing owed, so a new duration counts from its start
        timer.set(2)

        assert not tick(timer, 1).is_completed()
        assert tick(timer, 1).is_completed()


class TestNone:
    """A timer set to None, as an orb that does not come back or never expires is."""

    def test_it_never_completes(self, tick: Tick):
        assert not tick(_set(None), 100).is_completed()


class TestReset:
    """A reset, at the start of an episode."""

    def test_it_completes_a_timer_mid_countdown_or_set_to_none(self):
        counting, never = _set(5), _set(None)

        counting.reset()
        never.reset()

        assert counting.is_completed()
        assert never.is_completed()
