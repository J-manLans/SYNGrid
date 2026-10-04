from abc import ABC, abstractmethod
from pathlib import Path

from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from syn_grid.gymnasium.utils.env_factory import make
from syn_grid.gymnasium.utils.episode_logging.csv_episode_logger import (
    CSVEpisodeLogger,
)
from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.utils.date_utils import get_date
from syn_grid.utils.paths_util import get_project_path


class BaseAgentRunner(ABC):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, runner_bundle: RunnerBundle):
        self._runner_conf = runner_bundle.runner_conf.common_conf
        self._train_conf = runner_bundle.runner_conf.train_conf
        self._eval_conf = runner_bundle.runner_conf.eval_conf
        self._world_conf = runner_bundle.world_conf
        self.scenario = runner_bundle.scenario
        self._save_folder = self.scenario.name
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

        if self._save_folder:
            model_dir /= self._save_folder
            log_dir /= self._save_folder

        self._model_dir = model_dir
        self._log_dir = log_dir

        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _set_models_base_id(self) -> None:
        perception = self.scenario.perception

        # The glob this feeds must not match a checkpoint trained on a different grid — the observation vector is a fixed length at every grid size.
        rows, cols = self.scenario.grid_dimensions
        grid_suffix = f"_{rows}x{cols}"

        self._id = (
            f"{perception}_seed{self._runner_conf.seed}_{grid_suffix}"
            f"__TAG_{self.scenario.tag}_{self._runner_conf.alg}"
        )

    # === Env factory === #

    def _make_raw_env(self, render_mode: str | None) -> Env:
        return make(self.scenario, render_mode)

    # === Wrappers === #

    # --- Logger ---#

    def _wrap_record_episode_statistics(self, env: Env) -> Env:
        """
        Wrap the environment with Gymnasium's episode statistics wrapper. It records episode reward, length, and elapsed time.
        """

        return RecordEpisodeStatistics(env)

    def _wrap_episode_csv_logger(self, env: Env, sub_dir: str, env_idx: int = 0) -> Env:
        """
        Log standard and SYNGrid episode statistics to CSV.

        Expects the environment to provide episode statistics through
        ``RecordEpisodeStatistics``.

        Args:
            env: Environment to wrap.
            sub_dir: Sub-directory (under the run's log dir) to write the CSV into.
            env_idx: Index of this environment among parallel environments, if
                running more than one. Appended to the filename so each parallel
                environment writes to its own file instead of colliding on one.
                Defaults to 0, which is all a single-environment runner needs.
        """

        return CSVEpisodeLogger(
            env, self._log_dir / sub_dir, self.get_unique_model_id(), env_idx
        )

    # --- Video recording ---#

    def _maybe_wrap_video(self, env: Env) -> Env:
        # Training video
        if self._runner_conf.training and self._train_conf.record_video:
            local_interval = max(
                1, self._train_conf.rec_interval // self._train_conf.n_envs
            )

            return self._rec_video_wrapper(
                env,
                step_trigger=lambda t: t % local_interval == 0,
                video_length=self._train_conf.rec_length,
            )
        # Evaluation video
        elif not self._runner_conf.training and self._eval_conf.record_video:
            return self._rec_video_wrapper(
                env,
                episode_trigger=lambda t: t == self._eval_conf.rec_episode,
            )

        return env

    def _rec_video_wrapper(self, env: Env, **trigger) -> RecordVideo:
        video_output = (
            get_project_path("output", "results", "videos") / self.get_unique_model_id()
        )

        return RecordVideo(
            env,
            str(video_output),
            **trigger,
        )

    # === Persistence === #

    def _find_latest_saved_path(self, dir: Path) -> Path:
        """
        Find the most recently modified saved file matching the configured agent steps and ID
        """

        if self._runner_conf.agent_steps == "":
            raise ValueError("You forgot to specify the models steps")

        file_name = f"{self._runner_conf.agent_steps}_{self._id}*"

        matches = list(dir.glob(file_name))
        if not matches:
            raise FileNotFoundError(
                f"\nNo model found for path: {file_name}"
                f"\nIn: {self._save_folder if self._save_folder else 'base_dir'}"
            )

        # Multiple matching files may exist, so use the most recently modified one.
        return max(matches, key=lambda p: p.stat().st_mtime)
