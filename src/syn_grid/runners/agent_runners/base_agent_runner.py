from abc import ABC, abstractmethod
from pathlib import Path

from gymnasium import Env
from gymnasium.wrappers import RecordVideo

from syn_grid.gymnasium.utils.env_factory import make
from syn_grid.gymnasium.utils.episode_logging.episode_stats_wrapper import (
    EpisodeStatsWrapper,
)
from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.utils.date_utils import get_date
from syn_grid.utils.paths_util import get_project_path


class BaseAgentRunner(ABC):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, agent_bundle: AgentBundle):
        self._agent_conf = agent_bundle.agent_conf.global_agent_conf
        self._train_conf = agent_bundle.agent_conf.train_agent_conf
        self._eval_conf = agent_bundle.agent_conf.eval_agent_conf
        self._obs_conf = agent_bundle.obs_conf
        self._world_conf = agent_bundle.world_conf
        # Get current date and time to us as id for unique file naming
        self._date = get_date()

        self._init_output_directories()
        self._set_models_base_id()

    # ================= #
    #  Abstract methods #
    # ================= #

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def eval(self) -> None: ...

    # ================= #
    #         API       #
    # ================= #

    def get_unique_model_id(self) -> str:
        """Return the model ID, with a timestamp to uniquely identify each run."""
        return f"{self._id}_{self._date}"

    # ================= #
    #      Helpers      #
    # ================= #

    # === Setup === #

    def _init_output_directories(self) -> None:
        """
        Create and store paths for model checkpoints and TensorBoard logs.

        Uses the save_folder config value if provided, otherwise saves directly
        under the default 'models' directory.
        """

        model_dir = get_project_path("output", "models")
        log_dir = get_project_path("output", "results", "logs")

        save_folder = self._agent_conf.save_folder
        if save_folder:
            model_dir /= save_folder
            log_dir /= save_folder

        self._model_dir = model_dir
        self._log_dir = log_dir

        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _set_models_base_id(self) -> None:
        perception = self._obs_conf.observation_handler.perception
        tier = self._world_conf.orb_factory_conf.types.tier.enabled
        negative = self._world_conf.orb_factory_conf.types.negative.enabled

        tag = (
            f"TAG_{self._agent_conf.id_tag}_seed{self._agent_conf.seed}_"
            if self._agent_conf.id_tag
            else ""
        )
        tier_suffix = "" if tier else "_NoTier"
        negative_suffix = "_Neg" if negative else ""

        self._id = (
            f"{perception}{tier_suffix}{negative_suffix}__{tag}{self._agent_conf.alg}"
        )

    # === Env factory === #

    def _make_raw_env(self, render_mode: str | None) -> Env:
        return make(render_mode, self._world_conf, self._obs_conf)

    # === Wrappers === #

    def _logger_wrapper(self, env: Env, sub_dir: str) -> Env:
        """
        Wrap the environment with EpisodeStatsWrapper for logging.
        Tracks episode metrics and saves them to a CSV for training and eval analysis.
        """

        # NOTE: this one I need to look over once I refactor the logger itself

        return EpisodeStatsWrapper(
            env, self._log_dir / sub_dir, self.get_unique_model_id()
        )

    def _rec_video_wrapper(self, env: Env, **trigger) -> RecordVideo:
        video_output = get_project_path("output", "results", "videos")
        return RecordVideo(
            env,
            str(video_output),
            **trigger,
            name_prefix=self.get_unique_model_id(),
        )

    # === Persistence === #

    def _find_latest_saved_path(self, dir: Path) -> Path:
        """
        Find the most recently modified saved file matching the configured agent steps and ID
        """

        if self._agent_conf.agent_steps == "":
            raise ValueError("You forgot to specify the models steps")

        file_name = f"{self._agent_conf.agent_steps}_{self._id}*"

        matches = list(dir.glob(file_name))
        if not matches:
            raise FileNotFoundError(
                f"\nNo model found for path: {file_name}"
                f"\nIn: {self._agent_conf.save_folder if self._agent_conf.save_folder else 'base_dir'}"
            )

        # Multiple matching files may exist, so use the most recently modified one.
        return max(matches, key=lambda p: p.stat().st_mtime)
