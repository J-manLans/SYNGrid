from syn_grid.config.models import WorldConfig, ObsConfig

import gymnasium as gym
from gymnasium import Env
from gymnasium.envs.registration import registry, register
from gymnasium.utils.env_checker import check_env


def register_env() -> None:
    """Register the SynergyGrid Gym environment. Once registered, the id is usable in gym.make()."""

    if "syn_grid-v0" not in registry:
        register(
            id="syn_grid-v0",
            entry_point="syn_grid.gymnasium.environment:SYNGridEnv",
        )


def make(render_mode: str | None, run_conf: WorldConfig, obs_conf: ObsConfig) -> Env:
    """
    Creates the registered environment and check it for correctness, used when training or evaluating the agent.
    """

    env = gym.make(
        "syn_grid-v0", render_mode=render_mode, run_conf=run_conf, obs_conf=obs_conf
    )

    return env


def check_my_env(env: Env):
    check_env(env.unwrapped)
