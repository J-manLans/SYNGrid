import gymnasium as gym
from gymnasium import Env
from gymnasium.envs.registration import register, registry
from gymnasium.utils.env_checker import check_env

from syn_grid.scenario.scenario import Scenario


def register_env() -> None:
    """Register the SynergyGrid Gym environment. Once registered, the id is usable in gym.make()."""

    if "syn_grid-v0" not in registry:
        register(
            id="syn_grid-v0",
            entry_point="syn_grid.gymnasium.environment:SYNGridEnv",
        )


def make(scenario: Scenario, render_mode: str | None) -> Env:
    """
    Creates the registered environment and check it for correctness, used when training or evaluating the agent.
    """

    return gym.make("syn_grid-v0", scenario=scenario, render_mode=render_mode)


def check_my_env(env: Env):
    check_env(env.unwrapped)
