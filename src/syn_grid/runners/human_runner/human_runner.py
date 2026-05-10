from syn_grid.config.models import WorldConfig
from syn_grid.rendering.pygame_renderer import PygameRenderer
from syn_grid.core.grid_world import GridWorld


class HumanRunner:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, run_conf: WorldConfig, steps_left: int):
        self._renderer = PygameRenderer(run_conf.renderer_conf, 60)
        self._world = GridWorld(
            run_conf.grid_world_conf,
            run_conf.orb_factory_conf,
            run_conf.droid_conf,
            run_conf.negative_orb_conf,
            run_conf.tier_orb_conf,
        )
        self._steps_left = steps_left

    # ================= #
    #        API        #
    # ================= #

    def human_player_loop(self) -> None:
        self._world.reset()
        self._render()
        action = None

        while True:
            if action is not None:
                rew = self._world.perform_droid_action(action)
                self._steps_left -= 1
                terminated, truncated, rew = self._check_episode_end(rew)
                self._render()

                if terminated or truncated:
                    break

            action = self._renderer.get_user_action()

    # ================= #
    #      Helpers      #
    # ================= #

    def _render(self):
        self._renderer.render(
            self._world.droid.position,
            self._world.get_orb_is_active_status(True),
            self._world.get_orb_positions(True),
            self._world.get_orb_meta(True),
            self._get_hud_data(),
        )

    def _get_hud_data(self) -> dict[str, int | float]:
        hud_data: dict[str, int | float] = {}

        hud_data["score"] = self._world.droid.score
        hud_data["moves"] = self._steps_left
        hud_data["current tier chain"] = (
            self._world.droid.digestion_engine.chained_tiers
        )

        return hud_data

    def _check_episode_end(self, reward: float) -> tuple[bool, bool, float]:
        terminated = False
        truncated = False

        if self._world.droid.score <= 0:
            # always terminate when agent is out of score
            terminated = True

        if self._world._conf.single_chain_mode:
            terminated, truncated, reward = self._check_single_chain_end(
                terminated, truncated, reward
            )
        else:
            terminated, truncated, reward = self._check_continuous_mode_end(
                terminated, truncated, reward
            )

        return terminated, truncated, reward

    def _check_single_chain_end(
        self, terminated: bool, truncated: bool, reward: float
    ) -> tuple[bool, bool, float]:
        # === tier chain broken ===#
        if self._world.droid.digestion_engine.tier_chain_broken:
            terminated = True

        # === max steps reached === #
        elif self._steps_left <= 0:
            if not self._world._conf.max_tier_scoring:
                reward = self._world.droid.digestion_engine._pending_reward
            else:
                reward += -0.1

            terminated = True

        # === max tier reached ===#
        elif (
            self._world._conf.termination_on_max_tier
            and self._world.droid.digestion_engine.max_tier_reached
        ):
            if (
                not self._world._conf.curriculum_training
                and self._world._conf.max_tier_scoring
            ):
                reward = 10

            terminated = True

        return terminated, truncated, reward

    def _check_continuous_mode_end(
        self, terminated: bool, truncated: bool, reward: float
    ) -> tuple[bool, bool, float]:
        # === max steps reached === #
        if self._steps_left <= 0:
            if not self._world._conf.max_tier_scoring:
                reward = self._world.droid.digestion_engine._pending_reward
            terminated = True

        return terminated, truncated, reward
