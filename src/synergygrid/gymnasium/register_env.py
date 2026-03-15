from gymnasium.envs.registration import registry, register


def register_env():
    """Register the SynergyGrid Gym environment. Once registered, the id is usable in gym.make()."""

    if "synergy_grid-v0" not in registry:
        register(
            id="synergy_grid-v0",
            entry_point="synergygrid.gymnasium.environment:SYNGridEnv",
        )
