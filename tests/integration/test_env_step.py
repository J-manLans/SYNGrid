"""How the environment's step hands off between the world, the watcher and termination, and what
holds for any locked scenario, seed and actions."""

from dataclasses import dataclass, field, replace
from functools import cache

import pytest
from hypothesis import given
from hypothesis import strategies as st

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import ScenarioConf
from syn_grid.core.droid.synergy_droid import DroidAction
from syn_grid.core.grid_world import GridWorld, StepResult
from syn_grid.gymnasium.environment import SYNGridEnv
from syn_grid.scenario.blocks.termination import EpisodeOutcome, TerminationRules
from syn_grid.scenario.registry import SCENARIOS, build_scenario
from syn_grid.scenario.scenario import Scenario

# ================= #
#     Stand-ins     #
# ================= #


@dataclass
class CallLog:
    """What the recording watcher and termination saw, in the order they saw it."""

    events: list[str] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    worlds: list[GridWorld] = field(default_factory=list)


class RecordingWatcher:
    def __init__(self, log: CallLog):
        self._log = log

    def reset(self) -> None:
        self._log.events.append("reset")

    def react(self, world: GridWorld, result: StepResult) -> None:
        self._log.events.append("react")
        self._log.results.append(result)
        self._log.worlds.append(world)


@dataclass(frozen=True)
class RecordingTermination:
    inner: TerminationRules
    log: CallLog

    def evaluate(self, world: GridWorld) -> EpisodeOutcome:
        self.log.events.append("evaluate")
        return self.inner.evaluate(world)


# ================= #
#      Helpers      #
# ================= #

# === Invariants === #


@cache
def _locked_env(name: ScenarioName) -> SYNGridEnv:
    """A locked scenario's env from its base config, built once and reset by each example."""

    conf = SCENARIOS[name].base_conf()
    return SYNGridEnv(build_scenario(name, conf), render_mode=None)


_TOWARD_NEXT_TIER = 4


def _toward_next_tier(env: SYNGridEnv) -> int:
    """The action that moves the droid towards the lowest tier orb on the field, rows first, so
    that some episodes complete their chain and are paid. Left when no tier orb is on the field."""

    tiers = [o for o in env.world.orbs_on_field() if o.META.TIER is not None]
    if not tiers:
        return DroidAction.LEFT.value

    target = min(tiers, key=lambda o: o.META.TIER or 0).position
    row, col = env.world.droid.position
    if target[0] != row:
        return (DroidAction.DOWN if target[0] > row else DroidAction.UP).value
    return (DroidAction.RIGHT if target[1] > col else DroidAction.LEFT).value


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def log() -> CallLog:
    return CallLog()


@pytest.fixture
def env(log: CallLog) -> SYNGridEnv:
    name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
    conf: ScenarioConf = SCENARIOS[name].base_conf()
    scenario: Scenario = build_scenario(name, conf)

    scenario = replace(
        scenario,
        make_watcher=lambda: RecordingWatcher(log),
        termination=RecordingTermination(scenario.termination, log),
    )

    return SYNGridEnv(scenario, render_mode=None)


# ================= #
#       Tests       #
# ================= #


class TestWatcherInTheStep:
    """The environment calls its scenario's watcher once a step, between the world and
    termination, and resets it with each episode."""

    def test_reset_resets_the_watcher(self, env: SYNGridEnv, log: CallLog):
        env.reset(seed=0)
        env.reset(seed=1)

        assert log.events.count("reset") == 2

    def test_each_step_reacts_before_termination_with_the_world_and_its_result(
        self, env: SYNGridEnv, log: CallLog
    ):
        env.reset(seed=0)
        log.events.clear()

        _, reward, terminated, _, _ = env.step(0)
        env.step(0)

        assert log.events == ["react", "evaluate", "react", "evaluate"]
        assert log.worlds[0] is env.world
        # A step left from the centre of the grid cannot end the episode, so the env's reward is
        # the world's
        assert not terminated
        assert reward == pytest.approx(log.results[0].reward)


class TestInvariants:
    """What holds for any locked scenario, seed and actions."""

    @given(
        name=st.sampled_from([n for n, e in SCENARIOS.items() if e.lock is not None]),
        seed=st.integers(0, 2**32 - 1),
        # A short pattern, repeated until the episode ends, so some episodes run out the clock.
        # 4 steps towards the next tier orb, and a walk of only those completes most chains.
        pattern=st.just([_TOWARD_NEXT_TIER])
        | st.lists(st.integers(0, _TOWARD_NEXT_TIER), min_size=1, max_size=8),
    )
    def test_an_episode_ends_by_max_steps_and_pays_no_more_than_its_chain(
        self, name: ScenarioName, seed: int, pattern: list[int]
    ):
        env = _locked_env(name)
        env.reset(seed=seed)
        max_steps = env.world.steps_left
        chain_pays = sum(
            o.REWARD for o in env.world.ALL_ORBS if o.META.TIER is not None
        )

        episode_return = 0.0
        terminated = truncated = False
        for step in range(1, max_steps + 1):
            action = pattern[step % len(pattern)]
            if action == _TOWARD_NEXT_TIER:
                action = _toward_next_tier(env)
            _, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break

        assert terminated or truncated
        assert episode_return <= chain_pays
