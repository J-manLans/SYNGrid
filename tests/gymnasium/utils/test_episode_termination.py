from unittest.mock import MagicMock

import pytest

from syn_grid.legacy.gymnasium_legacy.utils.episode_logging.keys import LogKey
from syn_grid.legacy.gymnasium_legacy.utils.episode_termination import check_episode_end

# The value shipped in configs.yaml. Deliberately NOT used as the primary
# assertion below: -1.0 is exactly what the old hardcode was, so a test that
# only checks this value would have passed against the bug it is meant to catch.
SHIPPED_TIMEOUT = -1.0


def _world(
    *,
    single_chain_mode: bool = True,
    max_tier_scoring: bool = True,
    curriculum_training: bool = False,
    chains_broken: int = 0,
    chains_completed: int = 0,
    score: int = 10,
    active_orbs: int = 3,
    pending_reward: float = 0.0,
) -> MagicMock:
    world = MagicMock()
    world._conf.single_chain_mode = single_chain_mode
    world._conf.max_tier_scoring = max_tier_scoring
    world._conf.curriculum_training = curriculum_training
    world.droid.score = score
    world.droid.digestion_engine.stats = {
        LogKey.CHAINS_BROKEN: chains_broken,
        LogKey.CHAINS_COMPLETED: chains_completed,
    }
    world.droid.digestion_engine._pending_reward = pending_reward
    world._active_orbs = list(range(active_orbs))
    return world


def _end(world, steps_left, delay_mode=False, timeout=SHIPPED_TIMEOUT, reward=0.0):
    return check_episode_end(
        world=world,
        steps_left=steps_left,
        delay_mode=delay_mode,
        timeout_penalty=timeout,
        reward=reward,
    )


class TestTimeout:
    def test_timeout_uses_the_supplied_penalty_not_a_hardcoded_one(self):
        """The regression: this used to be a hardcoded -1. Uses -2.5 on purpose,
        because -1.0 would also have satisfied a hardcoded -1."""

        terminated, truncated, reward = _end(_world(), steps_left=0, timeout=-2.5)

        assert terminated
        assert not truncated
        assert reward == -2.5

    @pytest.mark.parametrize("timeout", [-0.1, -0.5, -1.0, -2.5, -10.0])
    def test_timeout_magnitude_is_configurable(self, timeout: float):
        _, _, reward = _end(_world(), steps_left=0, timeout=timeout)

        assert reward == timeout

    def test_timeout_terminates_and_never_truncates(self):
        """Timeouts are terminations by design, so no value function bootstraps
        at the horizon. Pinned here so it is not later 'fixed' by mistake."""

        _, truncated, _ = _end(_world(), steps_left=0)

        assert truncated is False

    def test_score_depletion_also_terminates(self):
        terminated, truncated, _ = _end(_world(score=0), steps_left=5)

        assert terminated
        assert not truncated

    def test_shipped_value_is_applied(self):
        _, _, reward = _end(_world(), steps_left=0, timeout=SHIPPED_TIMEOUT)

        assert reward == SHIPPED_TIMEOUT

    def test_a_live_reward_is_passed_through_untouched(self):
        """Nothing has terminated, so the step's reward must survive unchanged
        rather than being clobbered with a penalty."""

        terminated, truncated, reward = _end(
            _world(), steps_left=5, timeout=SHIPPED_TIMEOUT, reward=0.75
        )

        assert not terminated
        assert not truncated
        assert reward == 0.75


class TestChainBreak:
    def test_chain_break_terminates(self):
        terminated, truncated, _ = _end(_world(chains_broken=1), steps_left=5)

        assert terminated
        assert not truncated

    def test_chain_break_leaves_the_reward_alone(self):
        """A chain break settles its reward in DigestionEngine, not here. Guards
        against a timeout penalty being wired into this branch later."""

        world = _world(chains_broken=1)
        _, _, at_shipped = _end(
            world, steps_left=5, reward=0.4, timeout=SHIPPED_TIMEOUT
        )
        _, _, at_other = _end(world, steps_left=5, reward=0.4, timeout=-99.0)

        assert at_shipped == at_other == 0.4


class TestMaxTierReached:
    def test_curriculum_training_off_uses_the_ceiling_reward(self):
        world = _world(chains_completed=1, curriculum_training=False)
        _, _, reward = _end(world, steps_left=5)

        assert reward == 10.0

    def test_curriculum_training_on_leaves_the_reward_to_the_engine(self):
        """With curriculum on, the consumption reward stands. This is the branch
        the spatial scenario actually runs, so the ceiling is not applied."""

        world = _world(chains_completed=1, curriculum_training=True)
        _, _, reward = _end(world, steps_left=5)

        assert reward == 0.0

    @pytest.mark.parametrize("timeout", [-0.1, -1.0, -10.0])
    def test_completion_is_independent_of_the_timeout_penalty(self, timeout: float):
        world = _world(chains_completed=1, curriculum_training=False)
        _, _, reward = _end(world, steps_left=5, timeout=timeout)

        assert reward == 10.0


class TestDelayMode:
    def test_delay_mode_timeout_uses_the_supplied_penalty(self):
        terminated, _, reward = _end(
            _world(), steps_left=0, delay_mode=True, timeout=-2.5
        )

        assert terminated
        assert reward == -2.5

    def test_last_orb_in_delay_mode_halves_the_penalty(self):
        """Regression: this was floor division, so -1.0 // 2 == -1.0 and any small
        negative penalty collapsed to -1.0 -- a ~100x amplification."""

        world = _world(active_orbs=0)
        terminated, _, reward = _end(world, steps_left=5, delay_mode=True, timeout=-2.5)

        assert terminated
        assert reward == -2.5 / 2

    @pytest.mark.parametrize(
        ("timeout", "expected"), [(-0.1, -0.05), (-0.01, -0.005), (-2.5, -1.25)]
    )
    def test_half_penalty_is_exact(self, timeout: float, expected: float):
        world = _world(active_orbs=0)
        _, _, reward = _end(world, steps_left=5, delay_mode=True, timeout=timeout)

        assert reward == pytest.approx(expected)


class TestContinuousMode:
    def test_continuous_timeout_settles_pending_reward(self):
        world = _world(
            single_chain_mode=False, max_tier_scoring=False, pending_reward=3.5
        )
        terminated, truncated, reward = _end(world, steps_left=0)

        assert terminated
        assert not truncated
        assert reward == 3.5

    def test_continuous_mode_does_not_substitute_the_timeout_penalty(self):
        """Continuous mode pays out the pending reward instead. Asymmetric with
        single-chain mode, and deliberately so -- pinned to keep it a decision."""

        world = _world(
            single_chain_mode=False, max_tier_scoring=False, pending_reward=3.5
        )
        _, _, reward = _end(world, steps_left=0, timeout=-99.0)

        assert reward == 3.5
