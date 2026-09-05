import csv
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import RecordEpisodeStatistics

from syn_grid.gymnasium.utils.episode_logging.log_keys import LogKey


class EpisodeStatsWrapper(RecordEpisodeStatistics[ObsType, ActType]):
    def __init__(self, env: gym.Env[ObsType, ActType], log_dir: Path, model_id: str):
        super().__init__(env)

        self._broken_chains = 0
        self._chain_progress_step = 0
        self._completed_chains = 0

        csv_path = log_dir / f"{model_id}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(csv_path, "w", newline="")  # noqa: SIM115 - Kept open for the wrapper lifetime; closed in close()
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(LogKey))
        self._csv_writer.writeheader()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if info[LogKey.CHAINS_BROKEN]:
            self._broken_chains += 1
        elif info[LogKey.CHAIN_PROGRESSED]:
            self._chain_progress_step += 1
        elif info[LogKey.CHAINS_COMPLETED]:
            self._completed_chains += 1

        if terminated or truncated:
            # Parent's episode stats dict
            info[self._stats_key][LogKey.CHAINS_BROKEN] = self._broken_chains
            info[self._stats_key][LogKey.CHAIN_PROGRESSED] = self._chain_progress_step
            info[self._stats_key][LogKey.CHAINS_COMPLETED] = self._completed_chains

            self._csv_writer.writerow(
                {
                    LogKey.EPISODE: self.episode_count,
                    LogKey.REWARD: info[self._stats_key]["r"],
                    LogKey.LENGTH: info[self._stats_key]["l"],
                    LogKey.CHAINS_BROKEN: self._broken_chains,
                    LogKey.CHAIN_PROGRESSED: self._chain_progress_step,
                    LogKey.CHAINS_COMPLETED: self._completed_chains,
                }
            )
            self._csv_file.flush()

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._broken_chains = 0
        self._chain_progress_step = 0
        self._completed_chains = 0
        return obs, info

    def close(self) -> None:
        self._csv_file.close()
        super().close()
