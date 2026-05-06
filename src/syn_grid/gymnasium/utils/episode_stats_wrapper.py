import csv
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.core import ActType, ObsType
from typing import Any, SupportsFloat
from pathlib import Path


class EpisodeStatsWrapper(RecordEpisodeStatistics[ObsType, ActType]):
    def __init__(self, env: gym.Env[ObsType, ActType], log_dir: Path, model_id: str):
        super().__init__(env)

        self._completed_chains = 0
        self._broken_chains = 0

        csv_path = log_dir / f"{model_id}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "episode",
                "reward",
                "length",
                "chains_completed",
                "chains_broken",
            ],
        )
        self._csv_writer.writeheader()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if info["tier_chain_broken"]:
            self._broken_chains += 1
        elif info["max_tier_reached"]:
            self._completed_chains += 1

        if terminated or truncated:
            # Parent's episode stats dict
            info[self._stats_key]["chains_completed"] = self._completed_chains
            info[self._stats_key]["chains_broken"] = self._broken_chains

            self._csv_writer.writerow(
                {
                    "episode": self.episode_count,
                    "reward": info[self._stats_key]["r"],
                    "length": info[self._stats_key]["l"],
                    "chains_completed": self._completed_chains,
                    "chains_broken": self._broken_chains,
                }
            )
            self._csv_file.flush()

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._completed_chains = 0
        self._broken_chains = 0
        return obs, info

    def close(self) -> None:
        self._csv_file.close()
        super().close()
