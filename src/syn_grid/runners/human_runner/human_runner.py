from syn_grid.config.models import WorldConfig
from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.utils.episode_termination import check_episode_end
from syn_grid.rendering.pygame_renderer import PygameRenderer

# NOTE: This whole class is to a large extent stitched together, code duplication I think etc. When
# I go full GUI, this needs to be connected to that in a better way because after loading or
# customizing a scenario, the ability to play or train will be available, so either this will start
# or the training...and it feels...I don't know, a little bit apart, like how rewards are
# calculated and so on, there is some mismatch, and it needs to be an exact replica of the agent
# version of the core.


class HumanRunner:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, world_conf: WorldConfig, steps_left: int):
        self._renderer = PygameRenderer(world_conf.renderer_conf, 60)

        self.delay_mode = world_conf.grid_world_conf.delay_mode
        self.chain_break_penalty = world_conf.droid_conf.chain_break_penalty

        self._world = GridWorld(
            world_conf.grid_world_conf,
            world_conf.orb_factory_conf,
            world_conf.droid_conf,
            world_conf.negative_orb_conf,
            world_conf.tier_orb_conf,
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
                terminated, truncated, rew = check_episode_end(
                    self._world,
                    self._steps_left,
                    self.delay_mode,
                    self.chain_break_penalty,
                    rew,
                )
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
            "human",
        )

    def _get_hud_data(self) -> dict[str, int | float]:
        hud_data: dict[str, int | float] = {}

        hud_data["score"] = self._world.droid.score
        hud_data["moves"] = self._steps_left
        hud_data["current tier chain"] = (
            self._world.droid.digestion_engine.chained_tiers
        )

        return hud_data
