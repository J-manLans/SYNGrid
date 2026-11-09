from abc import ABC, abstractmethod
from pathlib import Path

import pygame
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from syn_grid.config.models.runner_models import RunnerConf
from syn_grid.gymnasium.utils.env_factory import make
from syn_grid.gymnasium.utils.episode_logging.csv_episode_logger import (
    CSVEpisodeLogger,
)
from syn_grid.rendering.pygame_renderer import PygameRenderer
from syn_grid.scenario.scenario import Scenario
from syn_grid.utils.date_utils import get_date
from syn_grid.utils.paths_util import get_project_path, get_syn_grid_path


class BaseAgentRunner(ABC):
    """
    What every agent runner shares, and the base for a runner of your own.

    A runner implements `train` and `eval`. It is made from the scenario and the runner config,
    and registered with a member in `Algorithm` (`config/models/runner_models.py`) and an entry
    in `RUNNER` (`runner_registry.py`).

    It gets, ready to use:

    - output folders for models and logs, named after the scenario.
    - the run's id, from the perception, the seed, the scenario's tag, the config's tag and the
      algorithm, through `get_unique_model_id`.
    - the scenario's env, from `_make_raw_env`, and the wrappers SYNGrid's runs use: episode
      statistics with the CSV episode logger on top if asked (`_wrap_episode_stats`), and video
      recording as the config asks (`_maybe_wrap_video`).
    - the run controls of a run with a human renderer: Space pauses, Escape or closing the window
      quits.
    - finding the latest saved file of the configured steps and id, and a sound when training
      stops.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, scenario: Scenario, runner_conf: RunnerConf):
        self._runner_conf = runner_conf.common_conf
        self._train_conf = runner_conf.train_conf
        self._eval_conf = runner_conf.eval_conf
        self.scenario = scenario
        self._save_folder = self.scenario.scenario_name
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

        Both live in a folder named after the scenario.
        """

        self._model_dir = get_project_path("output", "models") / self._save_folder
        self._log_dir = (
            get_project_path("output", "results", "logs") / self._save_folder
        )

        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _set_models_base_id(self) -> None:
        perception = self.scenario.observation.perception.value

        # The scenario's tag is its own axis; the user's tag, if set, marks a
        # change outside it. The algorithm closes the id because checkpoints
        # are found by prefix, so a tag must never come last.
        tag = self.scenario.scenario_tag
        if self._runner_conf.tag:
            tag = f"{tag}_{self._runner_conf.tag}"

        self._id = (
            f"{perception}_seed{self._runner_conf.seed}"
            f"__TAG_{tag}__{self._runner_conf.alg.value}"
        )

    # === Env factory === #

    def _make_raw_env(
        self, render_mode: str | None, render_fps: int | None = None
    ) -> Env:
        return make(self.scenario, render_mode, render_fps)

    # === Wrappers === #

    def _wrap_episode_stats(
        self, env: Env, csv_sub_dir: str | None = None, env_idx: int = 0
    ) -> Env:
        """
        Record each episode's reward, length and time, and log them to CSV if asked.

        The statistics are Gymnasium's `RecordEpisodeStatistics`, which SB3's training logs read
        too. The CSV logger reads them, so it always goes on top of them, with the scenario's
        metrics beside them.

        Args:
            env: Environment to wrap.
            csv_sub_dir: Sub-directory, under the run's log dir, to write the CSV into. None
                writes no CSV.
            env_idx: Index of this environment among parallel environments. Appended to the CSV
                file's name, so each environment writes its own file.
        """

        env = RecordEpisodeStatistics(env)

        if csv_sub_dir is None:
            return env

        return CSVEpisodeLogger(
            env,
            self._log_dir / csv_sub_dir,
            self.get_unique_model_id(),
            tuple(metric.name for metric in self.scenario.metrics),
            env_idx,
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

    # === Run controls === #

    def _handle_run_controls(self, renderer: PygameRenderer) -> bool:
        """
        Read the window's input between two steps of a run with a human renderer.

        Space pauses the run and shows the pause screen until Space is pressed again. Escape or
        closing the window quits, paused or not. Any other input is ignored.

        Returns:
            Whether the player quit, so the run stops.
        """

        for event in pygame.event.get():
            if self._is_quit_event(event):
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # What was pressed before the pause is dropped, so nothing acts after resuming.
                return self._pause(renderer)

        return False

    def _pause(self, renderer: PygameRenderer) -> bool:
        """
        Show the pause screen and block until Space resumes or the player quits.

        Returns:
            Whether the player quit instead of resuming.
        """

        renderer.draw_pause()

        while True:
            event = pygame.event.wait()

            if self._is_quit_event(event):
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return False

    @staticmethod
    def _is_quit_event(event: pygame.event.Event) -> bool:
        return event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        )

    # === Sound === #

    def _play_training_stopped_sound(self) -> None:
        """
        Play a sound when training stops, so a long run can be left to finish.

        A machine without an audio device skips the sound and says so.
        """

        try:
            pygame.mixer.init()
        except pygame.error as error:
            print(f"Training stopped. No sound played: {error}")
            return

        try:
            sound = pygame.mixer.Sound(
                f"{get_syn_grid_path('assets', 'sounds')}/training_complete.wav"
            )
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
        finally:
            pygame.mixer.quit()

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
                f"\nNo model found for path: {file_name}\nIn: {self._save_folder}"
            )

        # Multiple matching files may exist, so use the most recently modified one.
        return max(matches, key=lambda p: p.stat().st_mtime)
