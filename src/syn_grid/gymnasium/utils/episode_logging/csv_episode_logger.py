import csv
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import RecordEpisodeStatistics

from syn_grid.gymnasium.utils.episode_logging.keys import (
    STATS_KEY,
    SYN_STATS_KEY,
    LogKey,
)


class CSVEpisodeLogger(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    Logs episode statistics to a CSV file.

    Records one row for each completed episode: the standard episode
    statistics produced by ``RecordEpisodeStatistics``, followed by the
    scenario's metrics, which the environment puts in ``info`` under
    ``SYN_STATS_KEY``. The columns are the standard keys and the metric
    names it is given.

    The environment must be wrapped with ``RecordEpisodeStatistics``
    before being passed to this wrapper.

    The wrapper is primarily intended for evaluation runs, where it can
    be used to persist episode-level data for analysis and plotting.
    It can also be used during training. When multiple environments are
    used during training, episodes from all wrapped environments are logged.

    The wrapper does not distinguish between training and evaluation;
    it records the same episode-level data in either case.

    Args:
        env: Environment to wrap.
        log_dir: Directory where the CSV file is written.
        model_id: Identifier used as the CSV filename.
        metric_names: The names of the scenario's metrics, in order. They
            are the CSV's columns after the standard ones.
        env_idx: Index of this environment among parallel envs, appended to the
            filename so each parallel env writes to its own file.

    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        log_dir: Path,
        model_id: str,
        metric_names: tuple[str, ...],
        env_idx: int = 0,
    ):
        if not self._has_wrapper(env, RecordEpisodeStatistics):
            raise TypeError("CSVEpisodeLogger requires RecordEpisodeStatistics.")

        super().__init__(env)

        csv_path = log_dir / f"{model_id}" / f"_env{env_idx}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # The file is kept open for the wrapper lifetime; closed in the close() override. A `with`
        # block would close it right after __init__, so ruff's SIM115 is silenced on purpose.
        self._csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
        # The standard keys first, then the scenario's metrics; the header is written once.
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=[*LogKey, *metric_names]
        )
        self._csv_writer.writeheader()

    # ================= #
    #       API         #
    # ================= #

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            row = {**info[STATS_KEY], **info[SYN_STATS_KEY]}
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        super().close()
        self._csv_file.close()

    # ================= #
    #      Helpers      #
    # ================= #

    def _has_wrapper(self, env: gym.Env, wrapper_type: type[gym.Wrapper]) -> bool:
        while isinstance(env, gym.Wrapper):
            if isinstance(env, wrapper_type):
                return True
            env = env.env

        return False
