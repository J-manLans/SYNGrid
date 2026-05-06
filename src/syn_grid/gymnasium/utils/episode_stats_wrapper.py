import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.core import ActType, ObsType
from typing import Any, SupportsFloat


class EpisodeStatsWrapper(RecordEpisodeStatistics[ObsType, ActType]):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType]
    ):
        super().__init__(env)
        self._completed_chains = 0
        self._broken_chains = 0

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if info['tier_chain_broken']:
            self._broken_chains += 1
        elif info["max_tier_reached"]:
            self._completed_chains += 1

        if terminated or truncated:
            # Parent's episode stats dict
            info[self._stats_key]["chains_completed"] = self._completed_chains
            info[self._stats_key]["chains_broken"] = self._broken_chains

            # Also add top-level for Monitor compatibility
            info["chains_completed"] = self._completed_chains
            info["chains_broken"] = self._broken_chains

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._completed_chains = 0
        self._broken_chains = 0

        return obs, info
