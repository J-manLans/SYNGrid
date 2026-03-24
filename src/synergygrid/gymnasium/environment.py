import gymnasium as gym
from gymnasium import spaces

from synergygrid.core.grid_world import GridWorld
from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.gymnasium.action_space import AgentAction
from synergygrid.gymnasium.observation_space import ObservationHandler
from synergygrid.rendering.pygame_renderer import PygameRenderer


class SYNGridEnv(gym.Env):
    """
    SYNGrid reinforcement learning environment.

    A discrete grid-world environment for benchmarking single-agent RL.
    """

    # Metadata required by Gym.
    # "human" for Pygame visualization.
    # render_fps caps the update rate of render(); each call corresponds to one logic step, not the
    # full game framerate. Simply put: render_fps controls the speed of the environment’s logic,
    # while a sub-loop in the renderer would handle smooth animation between steps.
    metadata = {"render_modes": ["human"], "render_fps": 4}

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        max_active_resources: int = 3,
        grid_rows: int = 5,
        grid_cols: int = 5,
        max_steps: int = 100,
        render_mode: str | None = None,
        human_control: bool = False,
    ):
        # Set up bench environment;

        self._init_vars(max_active_resources, grid_rows, grid_cols, render_mode)
        self._init_world(
            human_control, max_steps, max_active_resources, grid_rows, grid_cols
        )
        if human_control:
            return

        if self.render_mode == "human":
            self._init_renderer(grid_rows, grid_cols, self.metadata["render_fps"])

        # if human_control:
        #     self._max_steps = max_steps
        #     self._human_play_loop()
        #     return

        # Set up Gymnasium environment:

        # Gymnasium also requires us to define action_space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(AgentAction))

        # Same goes with observation_space: this provides the agent with a structured view
        # of the world that it uses to decide its actions.
        self._observation_handler = ObservationHandler(
            self.world, grid_rows, grid_cols, max_steps
        )
        self.observation_space = self._observation_handler.setup_obs_space()

    # ======================== #
    #    Gymnasium contract    #
    # ======================== #

    def reset(self, *, seed=None, options=None):
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)

        # Reset the environment.
        self._observation_handler.reset()
        self.world.reset(self.np_random)

        self._obs = self._observation_handler.get_observation()
        norm_obs = self._observation_handler.normalize_obs(self._obs)

        if self.render_mode == "human":
            self.render()

        # Return observation and info (not used)
        return norm_obs, {}

    def step(self, action: AgentAction):
        # Perform action and adjust variables affected by it
        reward = self.world.perform_agent_action(AgentAction(action))
        self._observation_handler.step_count_down -= 1
        truncated = self._observation_handler.step_count_down <= 0
        terminated = self.world.agent.score <= 0

        self._obs = self._observation_handler.get_observation()
        norm_obs = self._observation_handler.normalize_obs(self._obs)

        if self.render_mode == "human":
            self.render()

        # Return observation, reward, terminated, truncated and info (TODO: not used now, but add it at termination or truncation so result can be persisted in the evaluate_agent() method)
        return (
            norm_obs,
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self) -> None:
        self.renderer.render(
            self.world.agent.position,
            self.world.get_resource_is_active_status(True),
            self.world.get_resource_positions(True),
            self.world.get_resource_meta(True),
            self._get_hud_data(),
        )

    # ================== #
    #       Helpers      #
    # ================== #

    # === Init === #

    def _init_vars(
        self,
        max_active_resources: int,
        grid_rows: int,
        grid_cols: int,
        render_mode: str | None,
    ) -> None:
        self.max_active_resources = max_active_resources
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode

    def _init_world(
        self,
        human_control: bool,
        max_steps: int,
        max_active_resources: int,
        grid_rows: int,
        grid_cols: int,
    ) -> None:
        if human_control:
            self._init_renderer(self.grid_rows, self.grid_rows, 60)
            self.world = GridWorld(
                max_active_resources,
                grid_rows,
                grid_cols,
                renderer=self.renderer,
                steps_left=max_steps,
            )
        else:
            self.world = GridWorld(max_active_resources, grid_rows, grid_cols)

    def _init_renderer(self, grid_rows: int, grid_cols: int, fps: int) -> None:
        self.renderer = PygameRenderer(grid_rows, grid_cols, fps)

    # === Gymnasium contract === #

    def _get_hud_data(self) -> dict[str, int]:
        hud_data: dict[str, int] = {}
        hud_data["score"] = self._obs["agent data"][3]
        hud_data["moves"] = self._obs["agent data"][2]
        hud_data["current tier chain"] = self._obs["agent data"][4]

        return hud_data
