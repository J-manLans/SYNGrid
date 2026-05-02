from syn_grid.config.models import WorldConfig, ObsConfig
from syn_grid.core.grid_world import GridWorld
from syn_grid.rendering.pygame_renderer import PygameRenderer
from syn_grid.gymnasium.action_space import DroidAction
from syn_grid.gymnasium.observation_space.observation_handler import (
    ObservationHandler,
)

import gymnasium as gym
from gymnasium import spaces


class SYNGridEnv(gym.Env):
    """
    SYNGrid reinforcement learning environment.

    A discrete grid-world environment for benchmarking single-agent RL.
    """

    # ================= #
    #       Init        #
    # ================= #

    # Metadata required by Gym.
    # "human" for Pygame visualization.
    # render_fps caps the update rate of render(); each call corresponds to one logic step, not the
    # full game framerate. Simply put: render_fps controls the speed of the environment’s logic,
    # while a sub-loop in the renderer would handle smooth animation between steps.
    metadata = {"render_modes": ["human"], "render_fps": 16}

    def __init__(
        self,
        run_conf: WorldConfig,
        obs_conf: ObsConfig,
        render_mode: str | None = None,
    ):
        # Set up bench environment;
        self.render_mode = render_mode
        self.world = GridWorld(
            run_conf.grid_world_conf,
            run_conf.orb_factory_conf,
            run_conf.droid_conf,
            run_conf.negative_orb_conf,
            run_conf.tier_orb_conf,
        )

        if self.render_mode == "human":
            self.renderer = PygameRenderer(
                run_conf.renderer_conf, self.metadata["render_fps"]
            )

        # Set up Gymnasium environment:

        # Gymnasium also requires us to define action_space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(DroidAction))

        # Same goes with observation_space: this provides the agent with a structured view
        # of the world that it uses to decide its actions.
        self._observation_handler = ObservationHandler(
            obs_conf, len(self.world.ALL_ORBS)
        )
        self.observation_space = self._observation_handler.setup_obs_space()

    # ======================== #
    #    Gymnasium contract    #
    # ======================== #

    def reset(self, *, seed=None, options=None):
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)

        # Reset the environment.
        self.world.reset(self.np_random)
        self._observation_handler.reset()

        if self.render_mode == "human":
            self.render()

        self.obs = self._observation_handler.get_observation(self.world)

        # Return observation and info (not used)
        return self.obs, {}

    def step(self, action: int):
        # Perform action and adjust variables affected by it
        reward = self.world.perform_droid_action(DroidAction(action))
        self._observation_handler.steps_left -= 1
        terminated, truncated, reward = self._check_episode_end(reward)

        if self.render_mode == "human":
            self.render()

        self.obs = self._observation_handler.get_observation(self.world)

        info = {
            "reward": reward,
            "steps_left": self._observation_handler.steps_left,
            "score": self.world.droid.score,
            "chain": self.world.droid.digestion_engine.chained_tiers,
        }

        # Return observation, reward, terminated, truncated and info
        return (
            self.obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        self.renderer.render(
            self.world.droid.position,
            self.world.get_orb_is_active_status(True),
            self.world.get_orb_positions(True),
            self.world.get_orb_meta(True),
            self._get_hud_data(),
        )
        self.renderer.get_user_action()

    # ================== #
    #       Helpers      #
    # ================== #

    # === Gymnasium contract === #

    def _get_hud_data(self) -> dict[str, int | float]:
        hud_data: dict[str, int | float] = {}

        hud_data["score"] = self.world.droid.score
        hud_data["moves"] = self._observation_handler.steps_left
        hud_data["current tier chain"] = self.world.droid.digestion_engine.chained_tiers

        return hud_data

    def _check_episode_end(self, reward: float) -> tuple[bool, bool, float]:
        terminated = False
        truncated = False

        if self.world.droid.score <= 0:
            terminated = True

        if (self._observation_handler.steps_left <= 0) and not terminated:
            truncated = True

        return terminated, truncated, reward
