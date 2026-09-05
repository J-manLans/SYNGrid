from syn_grid.core.grid_world import GridWorld


def check_episode_end(
    world: GridWorld,
    steps_left: int,
    delay_mode: bool,
    timeout_penalty: float,
    reward: float,
) -> tuple[bool, bool, float]:
    terminated = False
    truncated = False

    if world.droid.score <= 0:
        # always terminate when agent is out of score
        terminated = True

    if world._conf.single_chain_mode:
        terminated, truncated, reward = _single_chain_mode_termination(
            world,
            steps_left,
            delay_mode,
            timeout_penalty,
            terminated,
            truncated,
            reward,
        )
    else:
        terminated, truncated, reward = _continuous_mode_termination(
            world, steps_left, terminated, truncated, reward
        )

    return terminated, truncated, reward


# ================== #
#       Helpers      #
# ================== #


def _single_chain_mode_termination(
    world: GridWorld,
    steps_left: int,
    delay_mode: bool,
    timeout_penalty: float,
    terminated: bool,
    truncated: bool,
    reward: float,
) -> tuple[bool, bool, float]:
    # === tier chain broken ===#
    if world.droid.digestion_engine.tier_chain_broken and not delay_mode:
        terminated = True

    # === max steps reached === #
    if steps_left <= 0:
        if not world._conf.max_tier_scoring:
            reward = world.droid.digestion_engine._pending_reward

        if delay_mode:
            reward = timeout_penalty

        terminated = True

    # === max tier reached ===#
    elif world.droid.digestion_engine.max_tier_reached:
        if not world._conf.curriculum_training and world._conf.max_tier_scoring:
            # Overrides the reward from the consumption to a fixed ceiling
            reward = 10.0

        terminated = True

    # === last orb consumed in delay mode ===#
    elif delay_mode and len(world._active_orbs) == 0:
        terminated = True
        reward = timeout_penalty // 2

    return terminated, truncated, reward


def _continuous_mode_termination(
    world: GridWorld, steps_left: int, terminated: bool, truncated: bool, reward: float
) -> tuple[bool, bool, float]:
    # === max steps reached === #
    if steps_left <= 0:
        if not world._conf.max_tier_scoring:
            reward = world.droid.digestion_engine._pending_reward
        terminated = True

    return terminated, truncated, reward
