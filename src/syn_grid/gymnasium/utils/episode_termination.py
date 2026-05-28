from syn_grid.core.grid_world import GridWorld


def check_episode_end(
    world: GridWorld, steps_left: int, delay_mode: bool, reward: float
) -> tuple[bool, bool, float]:
    terminated = False
    truncated = False

    if world.droid.score <= 0:
        # always terminate when agent is out of score
        terminated = True

    if world._conf.single_chain_mode:
        terminated, truncated, reward = _check_single_chain_end(
            world, steps_left, delay_mode, terminated, truncated, reward
        )
    else:
        terminated, truncated, reward = _check_continuous_mode_end(
            world, steps_left, terminated, truncated, reward
        )

    return terminated, truncated, reward


# ================== #
#       Helpers      #
# ================== #


def _check_single_chain_end(
    world: GridWorld, steps_left: int, delay_mode: bool, terminated: bool, truncated: bool, reward: float
) -> tuple[bool, bool, float]:
    # === tier chain broken ===#
    if world.droid.digestion_engine.tier_chain_broken:
        if not delay_mode:
            terminated = True

    # === max steps reached === #
    elif steps_left <= 0:
        if not world._conf.max_tier_scoring:
            reward = world.droid.digestion_engine._pending_reward
        else:
            reward = -1

        terminated = True

    # === max tier reached ===#
    elif world.droid.digestion_engine.max_tier_reached:
        if not world._conf.curriculum_training and world._conf.max_tier_scoring:
            reward = 10

        terminated = True

    return terminated, truncated, reward


def _check_continuous_mode_end(
    world: GridWorld, steps_left: int, terminated: bool, truncated: bool, reward: float
) -> tuple[bool, bool, float]:
    # === max steps reached === #
    if steps_left <= 0:
        if not world._conf.max_tier_scoring:
            reward = world.droid.digestion_engine._pending_reward
        terminated = True

    return terminated, truncated, reward
