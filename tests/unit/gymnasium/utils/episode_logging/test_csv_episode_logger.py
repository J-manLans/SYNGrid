"""What the CSV episode logger writes: one row per finished episode, the standard statistics and then the
scenario's metrics, in a file of its own per env."""

import csv
from pathlib import Path
from typing import Any, override

import gymnasium as gym
import pytest
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics

from syn_grid.gymnasium.utils.episode_logging.csv_episode_logger import (
    CSVEpisodeLogger,
)
from syn_grid.gymnasium.utils.episode_logging.keys import SYN_STATS_KEY, LogKey

# ================= #
#     Constants     #
# ================= #


RUN_ID = "a_run"
METRICS = ("chains_broken", "negatives_eaten")


# ================= #
#     Stand-ins     #
# ================= #


class _Episodes(gym.Env):
    """Stands in for SYNGrid's env: each episode lasts the steps it is given and pays 1 a step, and the
    step that ends it puts the scenario's metrics in `info`, as the env does."""

    observation_space = spaces.Discrete(1)
    action_space = spaces.Discrete(1)

    def __init__(
        self,
        lengths: list[int],
        metrics: list[dict[str, float]],
        *,
        truncates: bool = False,
    ):
        self._lengths = iter(lengths)
        self._metrics = iter(metrics)
        self._truncates = truncates
        self._left = 0

    @override
    def reset(self, *, seed=None, options=None) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self._left = next(self._lengths)
        return 0, {}

    @override
    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self._left -= 1
        ended = self._left == 0
        info = {SYN_STATS_KEY: next(self._metrics)} if ended else {}
        # A goal scenario's episodes are terminated; a continuous scenario's clock truncates them
        return 0, 1.0, ended and not self._truncates, ended and self._truncates, info


# ================= #
#      Helpers      #
# ================= #


def _logger(tmp_path: Path, env: gym.Env, *, env_idx: int = 0) -> CSVEpisodeLogger:
    return CSVEpisodeLogger(
        RecordEpisodeStatistics(env), tmp_path, RUN_ID, METRICS, env_idx=env_idx
    )


def _play(logger: CSVEpisodeLogger, episodes: int) -> None:
    for _ in range(episodes):
        logger.reset()
        ended = False
        while not ended:
            _, _, terminated, truncated, _ = logger.step(0)
            ended = terminated or truncated


def _rows(tmp_path: Path, env_idx: int = 0) -> list[dict[str, str]]:
    with (tmp_path / RUN_ID / f"_env{env_idx}.csv").open(newline="") as f:
        return list(csv.DictReader(f))


# ================= #
#       Tests       #
# ================= #


class TestRows:
    """What each finished episode writes."""

    def test_one_row_per_episode_with_its_return_length_and_metrics(
        self, tmp_path: Path
    ):
        env = _Episodes(
            lengths=[3, 5],
            metrics=[
                {"chains_broken": 1, "negatives_eaten": 0},
                {"chains_broken": 0, "negatives_eaten": 2},
            ],
        )
        logger = _logger(tmp_path, env)

        _play(logger, episodes=2)
        logger.close()

        rows = _rows(tmp_path)
        assert [
            (
                row[LogKey.REWARD],
                row[LogKey.LENGTH],
                row["chains_broken"],
                row["negatives_eaten"],
            )
            for row in rows
        ] == [("3.0", "3", "1", "0"), ("5.0", "5", "0", "2")]

    def test_an_episode_the_clock_truncates_is_written_too(self, tmp_path: Path):
        env = _Episodes(
            lengths=[4],
            metrics=[{"chains_broken": 0, "negatives_eaten": 1}],
            truncates=True,
        )
        logger = _logger(tmp_path, env)

        _play(logger, episodes=1)
        logger.close()

        assert [row[LogKey.LENGTH] for row in _rows(tmp_path)] == ["4"]

    def test_the_columns_are_the_standard_statistics_then_the_metrics_in_order(
        self, tmp_path: Path
    ):
        logger = _logger(tmp_path, _Episodes(lengths=[], metrics=[]))
        logger.close()

        with (tmp_path / RUN_ID / "_env0.csv").open(newline="") as f:
            header = next(csv.reader(f))

        assert header == [*LogKey, *METRICS]

    def test_a_row_is_on_disk_as_soon_as_its_episode_ends(self, tmp_path: Path):
        # A run that crashes or is killed keeps the episodes it finished
        logger = _logger(
            tmp_path,
            _Episodes(
                lengths=[2], metrics=[{"chains_broken": 0, "negatives_eaten": 0}]
            ),
        )

        _play(logger, episodes=1)

        assert len(_rows(tmp_path)) == 1
        logger.close()


class TestFiles:
    """Where the rows go."""

    def test_each_env_writes_a_file_of_its_own_in_the_runs_folder(self, tmp_path: Path):
        first = _logger(
            tmp_path,
            _Episodes(
                lengths=[1], metrics=[{"chains_broken": 1, "negatives_eaten": 0}]
            ),
            env_idx=0,
        )
        second = _logger(
            tmp_path,
            _Episodes(
                lengths=[2], metrics=[{"chains_broken": 0, "negatives_eaten": 1}]
            ),
            env_idx=1,
        )

        _play(first, episodes=1)
        _play(second, episodes=1)
        first.close()
        second.close()

        assert [row[LogKey.LENGTH] for row in _rows(tmp_path, 0)] == ["1"]
        assert [row[LogKey.LENGTH] for row in _rows(tmp_path, 1)] == ["2"]


class TestWhatItWraps:
    """What the logger has to be put on."""

    def test_an_env_without_episode_statistics_is_refused(self, tmp_path: Path):
        with pytest.raises(TypeError, match="RecordEpisodeStatistics"):
            CSVEpisodeLogger(
                _Episodes(lengths=[], metrics=[]), tmp_path, RUN_ID, METRICS
            )
