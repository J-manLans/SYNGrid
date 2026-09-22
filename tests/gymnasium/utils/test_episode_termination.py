from unittest.mock import MagicMock

from syn_grid.gymnasium.utils.episode_logging.keys import LogKey
from syn_grid.gymnasium.utils.episode_termination import check_episode_end


class TestEpisodeTermination:
    def test_timeout_uses_timeout_penalty_with_max_tier_scoring(self):
        world = MagicMock()
        world._conf.single_chain_mode = True
        world._conf.max_tier_scoring = True
        world._conf.curriculum_training = False
        world.droid.score = 10
        world.droid.digestion_engine.stats = {
            LogKey.CHAINS_BROKEN: 0,
            LogKey.CHAINS_COMPLETED: 0,
        }

        terminated, truncated, reward = check_episode_end(
            world=world,
            steps_left=0,
            delay_mode=False,
            timeout_penalty=-10.0,
            reward=0.0,
        )

        assert terminated
        assert not truncated
        assert reward == -10.0