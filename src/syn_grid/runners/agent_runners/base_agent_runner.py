from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig
from syn_grid.utils.paths_util import get_project_path
from syn_grid.utils.date_utils import get_date
from syn_grid.gymnasium.utils.env_factory import make, check_my_env
from syn_grid.gymnasium.utils.episode_logging.episode_stats_wrapper import (
    EpisodeStatsWrapper,
)

from pathlib import Path
from abc import ABC, abstractmethod
from gymnasium import Env
from gymnasium.wrappers import RecordVideo


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
        # Get current date and time to us as id for unique file naming
        self._date = get_date()

        self._init_output_directories()
        self._get_model_base_id()

    def _set_id(self, id: str) -> None:
        self._id = id

    # ================= #
    #  Abstract methods #
    # ================= #

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def eval(self) -> None: ...

    # ================= #
    #      Helpers      #
    # ================= #

    # === Setup === #

    def _init_output_directories(self):
        """
        Create and store paths for model checkpoints and TensorBoard logs.

        Uses the save_folder config value if provided, otherwise saves directly
        under the default 'models' directory."""

        base_model_dir = get_project_path("output", "models")
        base_log_dir = get_project_path("output", "results", "logs")

        self._model_dir = (
            base_model_dir / self._conf.save_folder
            if self._conf.save_folder
            else base_model_dir
        )

        self._log_dir = (
            base_log_dir / self._conf.save_folder
            if self._conf.save_folder
            else base_log_dir
        )

        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_base_id(self) -> None:
        tag = (
            f"TAG_{self._conf.id_tag}_seed{self._conf.seed}_"
            if self._conf.id_tag
            else ""
        )
        perception = self._obs_conf.observation_handler.perception
        neg = "_Neg" if self._run_conf.orb_factory_conf.types.negative.enabled else ""

        if self._run_conf.orb_factory_conf.types.tier.enabled:
            self._id = f"{perception}{neg}__" f"{tag}{self._conf.alg}"
        else:
            self._id = f"{perception}_NoTier{neg}__" f"{tag}{self._conf.alg}"

    # === Factory === #

    def _make_raw_env(self, render_mode: str | None) -> Env:
        return make(render_mode, self._run_conf, self._obs_conf)

    # === Wrappers === #

    def _logger_wrapper(self, env: Env, sub_dir: str) -> Env:
        """
        Wrap the environment with EpisodeStatsWrapper for logging.
        Tracks episode metrics and saves them to a CSV for training and eval analysis.
        """

        return EpisodeStatsWrapper(env, self._log_dir / sub_dir, self._get_model_id())

    def _rec_video_wrapper(self, env: Env, **trigger) -> RecordVideo:
        video_output = get_project_path("output", "results", "videos")
        return RecordVideo(
            env,
            str(video_output),
            **trigger,
            name_prefix=self._get_model_id(),
        )

    # === Persistence === #

    def _get_saved_path(self, dir: Path) -> Path:
        """
        Returns a path that matches the latest model of the timesteps specified in the config file
        """

        if self._conf.agent_steps == "":
            raise ValueError("You forgot to specify the models steps")

        file_name = f"{self._conf.agent_steps}_{self._id}*"

        # list all files
        matches = list(dir.glob(file_name))
        if not matches:
            raise FileNotFoundError(
                f"\nNo model found for path: {file_name}"
                f"\nIn: {self._conf.save_folder if self._conf.save_folder else 'base_dir'}"
            )

        # return the one with the highest value of the timestamp
        return max(matches, key=lambda p: p.stat().st_mtime)

    # === Identification === #

    def _get_model_id(self) -> str:
        return f"{self._id}_{self._date}"
