from pathlib import Path

from gymnasium.wrappers import RecordEpisodeStatistics

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.runner_models import RunnerConf
from syn_grid.gymnasium.utils.env_factory import register_env
from syn_grid.gymnasium.utils.episode_logging.csv_episode_logger import (
    CSVEpisodeLogger,
)
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Fixtures      #
# ================= #


class _OwnRunner(BaseAgentRunner):
    """A runner of a user's own, as the base expects one."""

    def train(self) -> None: ...

    def eval(self) -> None: ...


def _runner(log_dir: Path) -> _OwnRunner:
    register_env()
    name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
    manager = ConfigManager()
    entry = SCENARIOS[name]
    scenario = entry.build(
        name.value,
        entry.full_conf(manager.load_raw(manager.scenario_config_path(name))),
    )
    runner_conf = manager.load_config("runner_config.yaml", RunnerConf)
    runner = _OwnRunner(scenario, runner_conf)
    runner._log_dir = log_dir

    return runner


# ================= #
#      Tests        #
# ================= #


class TestWrapEpisodeStats:
    """Episode statistics, with the CSV logger on top when a folder is given."""

    def test_the_csv_logger_sits_on_the_statistics_it_reads(self, tmp_path: Path):
        runner = _runner(tmp_path)

        plain = runner._wrap_episode_stats(runner._make_raw_env(None))
        logged = runner._wrap_episode_stats(
            runner._make_raw_env(None), csv_sub_dir="train"
        )

        assert isinstance(plain, RecordEpisodeStatistics)
        assert isinstance(logged, CSVEpisodeLogger)
        assert isinstance(logged.env, RecordEpisodeStatistics)
