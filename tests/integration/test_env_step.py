"""How the environment's step hands off between the world, the watcher and termination."""

from dataclasses import dataclass, field, replace

import pytest

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import ScenarioConf
from syn_grid.core.grid_world import GridWorld, StepResult
from syn_grid.gymnasium.environment import SYNGridEnv
from syn_grid.scenario.blocks.termination import EpisodeOutcome, TerminationRules
from syn_grid.scenario.registry import SCENARIOS, build_scenario
from syn_grid.scenario.scenario import Scenario

# ================= #
#     Fixtures      #
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


@pytest.fixture
def log() -> CallLog:
    return CallLog()


@pytest.fixture
def env(log: CallLog) -> SYNGridEnv:
    name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
    manager = ConfigManager()
    conf: ScenarioConf = SCENARIOS[name].full_conf(
        manager.load_raw(manager.scenario_config_path(name))
    )
    scenario: Scenario = build_scenario(name, conf)

    scenario = replace(
        scenario,
        make_watcher=lambda: RecordingWatcher(log),
        termination=RecordingTermination(scenario.termination, log),
    )

    return SYNGridEnv(scenario, render_mode=None)


# ================= #
#      Tests        #
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
