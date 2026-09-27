from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.utils.episode_logging.keys import LogKey


def check_episode_end(
    world: GridWorld,
    steps_left: int,
    delay_mode: bool,
    timeout_penalty: float,
    reward: float,
) -> tuple[bool, bool, float]:
    """
    Decide whether the episode is over, and settle the terminal reward.

    :param world: The core env.
    :param steps_left: Number of steps left in the episode.
    :param delay_mode: Whether or not we're running the delay scenario.
    :param timeout_penalty: Penalty for reaching the step limit unfinished.
    :param reward: The episode reward.
    """

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
    # module needs rework
    if world.droid.digestion_engine.stats[LogKey.CHAINS_BROKEN] > 0 and not delay_mode:
        terminated = True

    # === max steps reached === #
    elif steps_left <= 0:
        if not world._conf.max_tier_scoring:
            reward = world.droid.digestion_engine._pending_reward
        else:
            # Max-tier scoring never accumulates a partial reward, so there is
            # nothing to settle up and the timeout reward is simply the
            # configured penalty. This used to be a hardcoded -1, which made the
            # timeout magnitude untunable and silently coupled it to whatever
            # chain_break_penalty happened to be set to.
            reward = timeout_penalty

        if delay_mode:
            reward = timeout_penalty

        terminated = True

    # === max tier reached ===#
    # TODO: will be changed moving forward...somehow, not sure yet into what, think this whole module needs rework
    elif world.droid.digestion_engine.stats[LogKey.CHAINS_COMPLETED] > 0:
        if not world._conf.curriculum_training and world._conf.max_tier_scoring:
            # Overrides the reward from the consumption to a fixed ceiling
            reward = 10.0

        terminated = True

    # === last orb consumed in delay mode ===#
    elif delay_mode and len(world._active_orbs) == 0:
        terminated = True
        # True division: floor division collapsed any small negative penalty to
        # -1.0 (e.g. -0.01 // 2 == -1.0), amplifying the penalty ~100x.
        # NOTE: why do I use a penalty as reward for this scenario though? The elif branch is messy
        # and I think a clear refactor of this module is in place. Look into a strategy design
        # pattern? Seems that when we reach this, is when we're in delay mode, steps remaining, and
        # no chain is completed, but perhaps the chain is broken? Or does this happen on every step?
        reward = timeout_penalty / 2

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
