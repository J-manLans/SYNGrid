from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
)

from syn_grid.config.models.runner_models import RunnerConf
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.sb3.artifact_manager import ArtifactManager
from syn_grid.runners.agent_runners.sb3.policy_resolver import resolve_policy
from syn_grid.runners.agent_runners.sb3.run_controls_callback import (
    RunControlsCallback,
)
from syn_grid.scenario.scenario import Scenario
from syn_grid.utils.paths_util import get_project_path

# ================= #
#       Spec        #
# ================= #


@dataclass(frozen=True)
class SB3Spec:
    """One SB3 algorithm as SYNGrid runs it. Calling it with a scenario and a runner config
    makes its runner.

    Attributes:
        algorithm: the SB3 algorithm class, such as `PPO` or `RecurrentPPO`.
        device: where the model runs. SB3 leaves it out of a saved model, so it has to be stated
            to continue a run or evaluate one.
        hyper_parameters: what this algorithm adds to the shared hyperparameters, such as its
            `policy_kwargs`.
        frame_stack: how many observations are stacked into one, for short-term memory without
            a recurrent policy. None stacks nothing.
    """

    algorithm: type[BaseAlgorithm]
    device: str
    hyper_parameters: Mapping[str, Any] = field(default_factory=dict)
    frame_stack: int | None = None

    def __call__(self, scenario: Scenario, runner_conf: RunnerConf) -> "SB3Runner":
        return SB3Runner(scenario, runner_conf, self)


# ================= #
#      Runner       #
# ================= #


class SB3Runner(BaseAgentRunner):
    """
    Trains and evaluates any SB3 algorithm, as its `SB3Spec` describes it.

    Every algorithm shares the flow: a vectorized env, normalized and stacked if the spec asks,
    a model made or loaded, and an eval loop that carries a recurrent policy's state between
    steps. A policy that is not recurrent is handed the same state and ignores it.
    """

    # ================= #
    #       Init        #
    # ================= #

    _TRAIN: Final[str] = "train"
    _EVAL: Final[str] = "eval"
    _SHARED_HYPER_PARAMETERS: Final[dict[str, Any]] = {
        "ent_coef": 0.025,
        "n_steps": 128,
        "batch_size": 128,
        "n_epochs": 4,
    }

    def __init__(self, scenario: Scenario, runner_conf: RunnerConf, spec: SB3Spec):
        super().__init__(scenario, runner_conf)
        self._spec = spec
        print(f"Initializing {spec.algorithm.__name__}...")

        policy = resolve_policy(self.scenario.observation.perception, spec.algorithm)
        hyper_parameters = {
            "policy": policy,
            **self._SHARED_HYPER_PARAMETERS,
            "device": spec.device,
            **spec.hyper_parameters,
        }

        self._artifact_manager = ArtifactManager(
            algorithm=spec.algorithm,
            hyper_parameters=hyper_parameters,
            model_dir=self._model_dir,
            vec_norm_stats_dir=self._init_vec_norm_stats_dir(),
            find_latest_saved_path=self._find_latest_saved_path,
        )

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._build_env(self._train_conf.render_mode, self._TRAIN)
        model = self._resolve_training_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        env = self._build_env(self._eval_conf.render_mode, self._EVAL)
        model = self._artifact_manager.load_model(env, self._runner_conf.seed)

        self._eval_model(env, model)

    # ================= #
    #      Helpers      #
    # ================= #

    # === Env === #

    def _build_env(self, render_mode: str | None, sub_dir: str) -> VecEnv:
        """The vectorized env a run trains or evaluates on: normalized, then stacked if the
        spec asks for it."""

        env: VecEnv = self._resolve_normalized_env(
            self._make_dummy_vec_env(render_mode, sub_dir)
        )

        if self._spec.frame_stack is not None:
            env = VecFrameStack(env, self._spec.frame_stack)

        return env

    # === Properties === #

    @property
    def _is_fresh_training_run(self) -> bool:
        return self._runner_conf.training and not self._train_conf.continue_training

    # === Init === #

    def _init_vec_norm_stats_dir(self):
        """
        Directory for saving environment normalization statistics.
        Required for consistent eval, also for resuming training.
        """

        vec_norm_stats_dir = (
            get_project_path("output", "vec_norm_stats") / self._save_folder
        )
        vec_norm_stats_dir.mkdir(parents=True, exist_ok=True)
        return vec_norm_stats_dir

    def _make_dummy_vec_env(self, render_mode: str | None, sub_dir: str) -> DummyVecEnv:
        # If we are training, create as many envs that the config specifies.
        # If we're evaluating a trained agent — just create one env since no batching is needed.
        n_envs = self._train_conf.n_envs if self._runner_conf.training else 1

        return DummyVecEnv(
            [
                lambda i=i: self._make_env(render_mode, sub_dir, env_idx=i)
                for i in range(n_envs)
            ]
        )

    def _make_env(self, render_mode: str | None, sub_dir: str, env_idx: int) -> Env:
        """
        Create the environment with logging and optional video recording.

        Video recording is applied only to the first environment to avoid
        duplicate video output when running multiple environments in parallel.
        """

        env = self._make_raw_env(render_mode)

        csv_output = (
            self._train_conf.csv_output
            if self._runner_conf.training
            else self._eval_conf.csv_output
        )
        # Training always needs RecordEpisodeStatistics since tensorboard depends on it to log rew
        # and length, eval only uses it if we write to csv.
        record_episode_stats = self._runner_conf.training or csv_output

        if record_episode_stats:
            env = self._wrap_episode_stats(
                env, csv_sub_dir=sub_dir if csv_output else None, env_idx=env_idx
            )

        if env_idx == 0:
            env = self._maybe_wrap_video(env)

        return env

    def _resolve_normalized_env(self, env: DummyVecEnv) -> VecNormalize:
        """
        Create or load the normalization wrapper for the environment.

        Normalizes rewards and relevant environment statistics during training
        and restores the saved normalization state when evaluating a trained model.
        """

        if self._is_fresh_training_run:
            return self._artifact_manager.create_normalize_wrapper(env)
        else:
            return self._artifact_manager.load_normalize_wrapper(
                env, training=self._runner_conf.training
            )

    # === Model === #

    def _resolve_training_model(self, env: Env | VecEnv) -> BaseAlgorithm:
        if self._is_fresh_training_run:
            # If enabled, start TensorBoard with: tensorboard --logdir results/logs/<env_name>
            tensorboard_log = (
                str(self._log_dir / self._TRAIN)
                if self._train_conf.tensorboard_output
                else None
            )

            return self._artifact_manager.create_model(
                env, tensorboard_log=tensorboard_log, seed=self._runner_conf.seed
            )
        else:
            return self._artifact_manager.load_model(env, self._runner_conf.seed)

    # === Train === #

    def _train_model(self, model: BaseAlgorithm, env: VecEnv):
        """Train the model for the configured number of iterations and save progress."""

        debugging_callback = (
            RunControlsCallback(self._handle_run_controls)
            if self._train_conf.render_mode == "human"
            else None
        )

        try:
            for _ in range(self._train_conf.iterations):
                model.learn(
                    total_timesteps=self._train_conf.timesteps,
                    callback=debugging_callback,
                    tb_log_name=self.get_unique_model_id(),
                    reset_num_timesteps=False,
                )

                self._maybe_save_model(model, env)

                # Quitting from the window ends learn() early, and the remaining iterations too.
                if debugging_callback is not None and debugging_callback.stopped:
                    print("Training stopped from the window")
                    break
        except KeyboardInterrupt:
            print("Training interrupted")
            # If you interrupt training manually, the model gets saved on that step count
            self._maybe_save_model(model, env)
        finally:
            env.close()
            self._play_training_stopped_sound()

    def _maybe_save_model(self, model: BaseAlgorithm, env: VecEnv) -> None:
        if self._train_conf.model_output:
            self._artifact_manager.save_model(model, env, self.get_unique_model_id())

    # === Eval === #

    def _eval_model(self, env: VecEnv, model: BaseAlgorithm):
        obs = env.reset()
        try:
            num_episodes = self._eval_conf.num_eval_episodes
            renderer = (
                env.get_attr("renderer", indices=0)[0]
                if self._eval_conf.render_mode == "human"
                else None
            )

            for episode in range(1, num_episodes + 1):
                if renderer is not None:
                    renderer.set_title(f"SYNGrid · episode {episode}/{num_episodes}")

                # A recurrent policy's state, and which envs start an episode, carried between steps
                state = None
                episode_start = np.ones((env.num_envs,), dtype=bool)

                while True:
                    if renderer is not None:
                        player_quit = self._handle_run_controls(renderer)
                        if player_quit:
                            return

                    action, state = model.predict(
                        obs,  # type: ignore[arg-type]
                        state=state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    obs, _, dones, _ = env.step(action)
                    episode_start = dones

                    if dones[0]:
                        break
        finally:
            env.close()
