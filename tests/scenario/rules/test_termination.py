"""Termination rules, tested without a world, a config, or a scenario selector.

These used to poke ``world._conf.single_chain_mode`` and friends on a mock and
call a free function. That tested the old structure as much as the behaviour:
the flags were the only way to reach these branches, so the test had to know
they existed. They are constructor arguments now, so each case says which rules
it is exercising and nothing else.
"""

from unittest.mock import MagicMock

import pytest

from syn_grid.config.models import ScoringMode
from syn_grid.scenario.rules.termination import (
    ContinuousTermination,
    GoalTermination,
)

# The value shipped in configs.yaml. Deliberately NOT the primary assertion in
# the first test below: -1.0 is exactly what the old hardcode was, so a test
# that only checked this value would have passed against the bug it exists to
# catch.
SHIPPED_TIMEOUT = -1.0


def _world(
    *,
    chains_broken: int = 0,
    chains_completed: int = 0,
    score: float = 10,
    active_orbs: int = 3,
    pending_reward: float = 0.0,
) -> MagicMock:
    world = MagicMock()
    world.droid.score = score
    world.droid.digestion_engine.chains_broken = chains_broken
    world.droid.digestion_engine.chains_completed = chains_completed
    world.droid.digestion_engine.pending_reward = pending_reward
    world.active_orbs = list(range(active_orbs))
    return world


def _goal(
    *,
    delay: bool = False,
    curriculum: bool = False,
    scoring: ScoringMode = ScoringMode.MAX_TIER,
    timeout_penalty: float = SHIPPED_TIMEOUT,
) -> GoalTermination:
    return GoalTermination(
        timeout_penalty=timeout_penalty,
        delay=delay,
        scoring=scoring,
        curriculum=curriculum,
    )


def _continuous(
    *, scoring: ScoringMode = ScoringMode.MAX_TIER
) -> ContinuousTermination:
    return ContinuousTermination(scoring=scoring)


def _end(rules, world, steps_left, reward=0.0):
    return rules.evaluate(world, steps_left, reward)


class TestTimeout:
    def test_timeout_uses_the_supplied_penalty_not_a_hardcoded_one(self):
        """The regression: this used to be a hardcoded -1. Uses -2.5 on purpose,
        because -1.0 would also have satisfied a hardcoded -1."""

        outcome = _end(_goal(timeout_penalty=-2.5), _world(), steps_left=0)

        assert outcome.terminated
        assert not outcome.truncated
        assert outcome.reward == -2.5

    @pytest.mark.parametrize("timeout", [-0.1, -0.5, -1.0, -2.5, -10.0])
    def test_timeout_magnitude_is_configurable(self, timeout: float):
        outcome = _end(_goal(timeout_penalty=timeout), _world(), steps_left=0)

        assert outcome.reward == timeout

    def test_timeout_terminates_and_never_truncates(self):
        """Timeouts are terminations by design, so no value function bootstraps
        at the horizon. Pinned here so it is not later 'fixed' by mistake."""

        outcome = _end(_goal(), _world(), steps_left=0)

        assert outcome.truncated is False

    def test_score_depletion_also_terminates(self):
        outcome = _end(_goal(), _world(score=0), steps_left=5)

        assert outcome.terminated
        assert not outcome.truncated

    def test_shipped_value_is_applied(self):
        outcome = _end(_goal(), _world(), steps_left=0)

        assert outcome.reward == SHIPPED_TIMEOUT

    def test_a_live_reward_is_passed_through_untouched(self):
        """Nothing has terminated, so the step's reward must survive unchanged
        rather than being clobbered with a penalty."""

        outcome = _end(_goal(), _world(), steps_left=5, reward=0.75)

        assert not outcome.terminated
        assert not outcome.truncated
        assert outcome.reward == 0.75

    def test_a_timeout_under_a_non_max_tier_mode_settles_the_pending_reward(self):
        """Max-tier scoring never accumulates, so it has nothing to settle. Any
        other mode does, and hands it over when the clock runs out."""

        outcome = _end(
            _goal(scoring=ScoringMode.THRESHOLD, timeout_penalty=-2.5),
            _world(pending_reward=3.5),
            steps_left=0,
        )

        assert outcome.reward == 3.5


class TestChainBreak:
    def test_chain_break_terminates(self):
        outcome = _end(_goal(), _world(chains_broken=1), steps_left=5)

        assert outcome.terminated
        assert not outcome.truncated

    def test_chain_break_leaves_the_reward_alone(self):
        """A chain break settles its reward in DigestionEngine, not here. Guards
        against a timeout penalty being wired into this branch later."""

        world = _world(chains_broken=1)
        at_shipped = _end(_goal(), world, steps_left=5, reward=0.4)
        at_other = _end(_goal(timeout_penalty=-99.0), world, steps_left=5, reward=0.4)

        assert at_shipped.reward == at_other.reward == 0.4

    def test_a_delay_scenario_lets_a_broken_chain_be_recovered_from(self):
        """The whole point of delay: a break costs the cooldown, not the episode."""

        outcome = _end(_goal(delay=True), _world(chains_broken=1), steps_left=5)

        assert not outcome.terminated


class TestMaxTierReached:
    def test_curriculum_off_uses_the_ceiling_reward(self):
        outcome = _end(
            _goal(curriculum=False), _world(chains_completed=1), steps_left=5
        )

        assert outcome.reward == 10.0

    def test_curriculum_on_leaves_the_reward_to_the_engine(self):
        """With curriculum on, the consumption reward stands. This is the branch
        the spatial scenario actually runs, so the ceiling is not applied."""

        outcome = _end(_goal(curriculum=True), _world(chains_completed=1), steps_left=5)

        assert outcome.reward == 0.0

    @pytest.mark.parametrize("timeout", [-0.1, -1.0, -10.0])
    def test_completion_is_independent_of_the_timeout_penalty(self, timeout: float):
        outcome = _end(
            _goal(curriculum=False, timeout_penalty=timeout),
            _world(chains_completed=1),
            steps_left=5,
        )

        assert outcome.reward == 10.0

    def test_the_ceiling_only_applies_under_max_tier_scoring(self):
        """Threshold scoring already pays on completion, so replacing that with a
        flat ceiling would throw away the chain's length."""

        outcome = _end(
            _goal(curriculum=False, scoring=ScoringMode.THRESHOLD),
            _world(chains_completed=1),
            steps_left=5,
            reward=7.5,
        )

        assert outcome.reward == 7.5


class TestDelayMode:
    def test_delay_mode_timeout_uses_the_supplied_penalty(self):
        """Delay overrides the pending-reward settlement: with the field on
        cooldown, a partial reward is not progress towards the objective."""

        outcome = _end(
            _goal(delay=True, timeout_penalty=-2.5),
            _world(pending_reward=99.0),
            steps_left=0,
        )

        assert outcome.terminated
        assert outcome.reward == -2.5

    def test_delay_mode_ignores_the_scoring_mode_for_its_timeout(self):
        outcome = _end(
            _goal(delay=True, scoring=ScoringMode.STEP_WISE, timeout_penalty=-0.4),
            _world(pending_reward=99.0),
            steps_left=0,
        )

        assert outcome.reward == -0.4

    def test_an_exhausted_delay_field_halves_the_penalty(self):
        """Regression: this was floor division, so -1.0 // 2 == -1.0 and any small
        negative penalty collapsed to -1.0 -- a ~100x amplification."""

        outcome = _end(
            _goal(delay=True, timeout_penalty=-2.5),
            _world(active_orbs=0),
            steps_left=5,
        )

        assert outcome.terminated
        assert outcome.reward == -2.5 / 2

    @pytest.mark.parametrize(
        ("timeout", "expected"), [(-0.1, -0.05), (-0.01, -0.005), (-2.5, -1.25)]
    )
    def test_half_penalty_is_exact(self, timeout: float, expected: float):
        outcome = _end(
            _goal(delay=True, timeout_penalty=timeout),
            _world(active_orbs=0),
            steps_left=5,
        )

        assert outcome.reward == pytest.approx(expected)

    def test_a_non_delay_scenario_ignores_an_empty_field(self):
        """Only delay can run the field empty. Without a cooldown there is always
        something to spawn, so the branch is unreachable rather than merely false."""

        outcome = _end(_goal(delay=False), _world(active_orbs=0), steps_left=5)

        assert not outcome.terminated


class TestContinuous:
    def test_timeout_settles_pending_reward(self):
        outcome = _end(
            _continuous(scoring=ScoringMode.STEP_WISE),
            _world(pending_reward=3.5),
            steps_left=0,
        )

        assert outcome.terminated
        assert not outcome.truncated
        assert outcome.reward == 3.5

    def test_does_not_substitute_the_timeout_penalty(self):
        """Continuous mode pays out the pending reward instead, and has no timeout
        penalty to substitute. Asymmetric with a goal scenario, deliberately --
        pinned to keep it a decision."""

        outcome = _end(
            _continuous(scoring=ScoringMode.THRESHOLD),
            _world(pending_reward=3.5),
            steps_left=0,
        )

        assert outcome.reward == 3.5

    def test_max_tier_scoring_settles_nothing(self):
        """It never accumulated a partial reward, so a timeout has nothing to hand
        over and the step's own reward stands."""

        outcome = _end(
            _continuous(scoring=ScoringMode.MAX_TIER),
            _world(pending_reward=3.5),
            steps_left=0,
            reward=0.25,
        )

        assert outcome.terminated
        assert outcome.reward == 0.25

    def test_there_is_no_chain_and_nothing_to_complete(self):
        """A continuous episode has no chain, so neither a break nor a completion
        ends it -- only the clock or the score."""

        outcome = _end(
            _continuous(),
            _world(chains_broken=4, chains_completed=4),
            steps_left=5,
        )

        assert not outcome.terminated

    def test_score_depletion_terminates(self):
        outcome = _end(_continuous(), _world(score=0), steps_left=5)

        assert outcome.terminated
