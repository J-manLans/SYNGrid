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

        self._init_vars(
            max_active_resources, grid_rows, grid_cols, render_mode, human_control
        )
        self._init_world(max_active_resources, grid_rows, grid_cols)
        if self.render_mode == "human":
            self._init_renderer(grid_rows, grid_cols)

        if human_control:
            self._max_steps = max_steps
            self._human_play_loop()
            return

        # Set up Gymnasium environment:

        # Gymnasium also requires us to define action_space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(AgentAction))

        # Same goes with observation_space: this provides the agent with a structured view
        # of the world that it uses to decide its actions.
        self._observation_handler = ObservationHandler(
            self._world, grid_rows, grid_cols, max_steps
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
        self._world.reset(self.np_random)

        if self.render_mode == "human":
            self.render()

        obs = self._observation_handler.get_observation()
        norm_obs = self._observation_handler.normalize_obs(obs)

        # Return observation and info (not used)
        return norm_obs, {}

    def step(self, action: AgentAction):
        # Perform action and adjust variables affected by it
        reward = self._world.perform_agent_action(AgentAction(action))
        self._observation_handler.step_count_down -= 1
        truncated = self._observation_handler.step_count_down <= 0
        terminated = self._world._agent.score <= 0

        if self.render_mode == "human":
            self.render()

        obs = self._observation_handler.get_observation()
        norm_obs = self._observation_handler.normalize_obs(obs)

        # Return observation, reward, terminated, truncated and info (TODO: not used now, but add it at termination or truncation so result can be persisted in the evaluate_agent() method)
        return (
            norm_obs,
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self) -> None | str:
        return self._renderer.render(
            self._world._agent.position,
            self._world.get_resource_is_active_status(True),
            self._world.get_resource_positions(True),
            self._world.get_resource_meta(True),
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
        human_control: bool,
    ) -> None:
        self.max_active_resources = max_active_resources
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.human_control = human_control

    def _init_world(
        self, max_active_resources: int, grid_rows: int, grid_cols: int
    ) -> None:
        self._world = GridWorld(
            max_active_resources, grid_rows=grid_rows, grid_cols=grid_cols
        )

    def _init_renderer(self, grid_rows: int, grid_cols: int) -> None:
        self._renderer = PygameRenderer(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=self.metadata["render_fps"],
        )

    # === Gymnasium contract === #

    def _get_hud_data(self) -> dict[str, int]:
        hud_data: dict[str, int] = {}
        hud_data["score"] = self._world._agent.score
        if self.human_control:
            hud_data["moves"] = self._step_count_down
        else:
            hud_data["moves"] = self._observation_handler.step_count_down
        if len(BaseResource._chained_tiers) > 0:
            if BaseResource._chained_tiers[-1] == self._world.max_tier:
                hud_data["current tier chain"] = 0
            else:
                hud_data["current tier chain"] = BaseResource._chained_tiers[-1]
        else:
            hud_data["current tier chain"] = 0

        return hud_data

    # === Human control === #

    def _human_play_loop(self):
        self._step_count_down = 100
        self._renderer._step_fps = 60
        self._world.reset()
        action = str(self.render()).upper()

        while True:
            if not action == "NONE":
                agent_action = AgentAction[action]
                self._world.perform_agent_action(agent_action)
                self._step_count_down -= 1
                truncated = self._step_count_down <= 0
                terminated = self._world._agent.score <= 0

                if terminated or truncated:
                    break

            action = str(self.render()).upper()
