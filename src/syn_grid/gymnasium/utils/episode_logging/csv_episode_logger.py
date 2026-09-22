import csv
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from syn_grid.gymnasium.utils.episode_logging.keys import STATS_KEY, LogKey


class CSVEpisodeLogger(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    Logs episode statistics to a CSV file.

    Records one row for each completed episode, including the standard
    episode statistics produced by ``RecordEpisodeStatistics`` and
    SYNGrid-specific episode statistics.

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
    """

    # TODO: this class will change as more scenarios are added and different labels will get used.
    # I think the LogKey setup will grow a little and become composable...but we'll see, as it is
    # for now works.

    def __init__(self, env: gym.Env[ObsType, ActType], log_dir: Path, model_id: str):
        super().__init__(env)

        csv_path = log_dir / f"{model_id}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # The file is kept open for the wrapper lifetime; closed in the close() override. A `with`
        # block would close it right after __init__, so ruff's SIM115 is silenced on purpose.
        self._csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
        # Creates the writer with the field names defined by the LogKey enum,
        # then writes the header.
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(LogKey))
        self._csv_writer.writeheader()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            row = {**info["episode"], **info[STATS_KEY]}
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        super().close()
        self._csv_file.close()
