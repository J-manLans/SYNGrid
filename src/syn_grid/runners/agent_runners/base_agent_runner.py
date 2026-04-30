from syn_grid.config.models import AgentConfig, GlobalAgentConf, WorldConfig, ObsConfig
from syn_grid.utils.paths_util import get_project_path
from syn_grid.utils.date_utils import get_date
from syn_grid.gymnasium.env_factory import make, check_my_env

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from gymnasium import Env


class BaseAgentRunner(ABC):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        self._conf = conf.global_agent_conf
        self._train_conf = conf.train_agent_conf
        self._eval_conf = conf.eval_agent_conf
        self._obs_conf = obs_conf
        self._run_conf = run_conf

        # Get current date and time to us as an identifier for unique file naming
        self._date = get_date()

        # Create directories for saving models and logs
        self._model_dir = Path(get_project_path("output", "models"))
        self.log_dir = Path(get_project_path("output", "results", "logs"))

        Path(self._model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def _set_id(self, id: str) -> None:
        self._id = id

    # ================= #
    #  Abstract methods #
    # ================= #

    @abstractmethod
    def _construct_model_id(
        self, agent_conf: GlobalAgentConf, run_conf: WorldConfig
    ) -> str:
        """
        Construct a unique model identifier for saving/loading.

        This method is called directly after `super().__init__()` and should:
            1. Call `super()._get_model_base_id()` to retrieve base identifiers
            2. Combine them with any subclass-specific agent parameters
            3. Set the result via `super()._set_id()`

        Returns:
            A unique string identifier (used as filename/folder name)
        """

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def eval(self) -> None: ...

    # ================= #
    #      Helpers      #
    # ================= #

    def _get_model_base_id(self) -> tuple[str, str]:
        tag = (
            f"TAG_{self._conf.id_tag}__"
            if self._conf.id_tag
            else ""
        )
        perception = self._obs_conf.observation_handler.perception
        tier = f"Tier{self._run_conf.orb_factory_conf.max_tier}"
        reward = f"{self._run_conf.tier_orb_conf.base_reward}rew"
        growth = (
            ""
            if (
                self._run_conf.orb_factory_conf.max_tier == 1
                or self._run_conf.tier_orb_conf.linear_reward_growth
            )
            else f"_{self._run_conf.tier_orb_conf.growth_factor}growth"
        )
        neg = "_Neg" if self._run_conf.orb_factory_conf.types.negative.enabled else ""
        score = f"{self._run_conf.droid_conf.starting_score}score"
        step_offset = f"{self._run_conf.droid_conf.step_penalty}step_offset"
        tier_consumption_penalty = (
            ""
            if self._run_conf.droid_conf.tier_consumption_penalty == 0.0
            else f"_{self._run_conf.droid_conf.tier_consumption_penalty}cons_offset"
        )

        base_tier_id = f"{perception}{neg}__{tier}_{reward}{growth}{tier_consumption_penalty}__{score}{step_offset}__{tag}{self._conf.alg}"
        base_non_tier_id = (
            f"{perception}_NoTier{neg}__{score}_{step_offset}__{tag}{self._conf.alg}"
        )

        return base_tier_id, base_non_tier_id

    def _make_raw_env(self, render_mode: str | None) -> Env:
        env = make(render_mode, self._run_conf, self._obs_conf)

        if self._conf.check_env:
            check_my_env(env)
            sys.exit("Environment is fine")

        return env

    def _get_saved_path(self, dir: Path) -> Path:
        """
        Returns a path that matches the latest model of the timesteps specified in the config file
        """

        if self._conf.agent_steps == "":
            sys.exit("You forgot to specify the models steps")

        file_name = f"{self._conf.agent_steps}_{self._id}*"

        # list all files
        matches = list(dir.glob(file_name))
        if not matches:
            raise FileNotFoundError(f"No model found for path: {file_name}")

        # return the one with the highest value of the timestamp
        return max(matches, key=lambda p: p.stat().st_mtime)

    def _get_model_id(self) -> str:
        return f"{self._id}_{self._date}"
