from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.utils.episode_logging.keys import LogKey


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
    # TODO: will be changed moving forward...somehow, not sure yet into what, think this whole
    # class needs rework
    if world.droid.digestion_engine.stats[LogKey.CHAINS_BROKEN] > 0 and not delay_mode:
        terminated = True

    # === max steps reached === #
    elif steps_left <= 0:
        if not world._conf.max_tier_scoring:
            reward = world.droid.digestion_engine._pending_reward
        else:
            reward = -1  # timeout_penalty TODO: checking if this is what ruins the spatial scenario (have been using the timeout_penalty which is at -0.1, a hundred magnitude difference in learning signal, if it is a more robust solution must be found, can be worth to rewrite this whole module file into something more robust, like i mean...i pass world in here, this was definitely a last, sort of, minute fix for the thesis

        if delay_mode:
            reward = timeout_penalty

        terminated = True

    # === max tier reached ===#
    # TODO: will be changed moving forward...somehow, not sure yet into what, think this whole class needs rework
    elif world.droid.digestion_engine.stats[LogKey.CHAINS_COMPLETED] > 0:
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
