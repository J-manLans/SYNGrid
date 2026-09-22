from typing import Any, Final, Generic, TypeVar

from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.sb3.artifact_manager import ArtifactManager
from syn_grid.runners.agent_runners.sb3.execution_strategy import (
    ExecutionStrategy,
)
from syn_grid.runners.agent_runners.sb3.utils.plateau_callback import PlateauCallback
from syn_grid.utils.paths_util import get_project_path

T = TypeVar("T", bound=BaseAlgorithm)


class BaseSB3Runner(BaseAgentRunner, Generic[T]):
    """
    Shared orchestration for every SB3-backed agent runner.

    train() and eval() are template methods: the flow lives here once.
    Concrete agents (StatelessPPO, FrameStackPPO, LstmPPO, ...) configure
    this class with their algorithm, hyperparameters, and an execution
    strategy for eval-time action selection, then only override `_build_env`
    if they need extra env wrapping on top of the shared normalized-vec-env
    pipeline (see FrameStackPPO). No agent should ever reach into another
    agent's internals -- everything an agent needs, it supplies here.
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

    def __init__(
        self,
        agent_bundle: AgentBundle,
        hyper_parameters: dict[str, Any],
        algorithm: type[T],
        execution_strategy: ExecutionStrategy,
    ):
        super().__init__(agent_bundle)
        self._execution_strategy = execution_strategy

        self._artifact_manager: ArtifactManager[T] = ArtifactManager(
            algorithm=algorithm,
            hyper_parameters=hyper_parameters,
            model_dir=self._model_dir,
            vec_norm_stats_dir=self._init_vec_norm_stats_dir(),
            find_latest_saved_path=super()._find_latest_saved_path,
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
        model = self._artifact_manager.load_model(env)

        self._eval_model(env, model)

    # ================= #
    #       Hooks       #
    # ================= #

    def _build_env(self, render_mode: str | None, sub_dir: str) -> VecEnv:
        """
        Build the vectorized env this agent trains/evaluates on.

        Default pipeline: a DummyVecEnv wrapped with VecNormalize, shared by
        every current agent. Override and call `super()._build_env(...)`
        first to layer additional wrappers on top (see FrameStackPPO).
        """

        env = self._make_dummy_vec_env(render_mode, sub_dir)
        return self._resolve_normalized_env(env)

    # ================= #
    #      Helpers      #
    # ================= #

    # === Properties === #

    @property
    def _is_fresh_training_run(self) -> bool:
        return self._agent_conf.training and not self._train_conf.continue_training

    # === Init === #

    def _init_vec_norm_stats_dir(self):
        """
        Directory for saving environment normalization statistics.
        Required for consistent eval, also for resuming training.
        """

        base = get_project_path("output", "vec_norm_stats")
        vec_norm_stats_dir = (
            base / self._agent_conf.save_folder
            if self._agent_conf.save_folder
            else base
        )
        vec_norm_stats_dir.mkdir(parents=True, exist_ok=True)
        return vec_norm_stats_dir

    # === Env === #

    def _make_dummy_vec_env(self, render_mode: str | None, sub_dir: str) -> DummyVecEnv:
        # If we are training, create as many envs that the config specifies.
        # If we're evaluating a trained agent — just create one env since no batching is needed.
        n_envs = self._train_conf.n_envs if self._agent_conf.training else 1

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
            if self._agent_conf.training
            else self._eval_conf.csv_output
        )
        # Training always needs RecordEpisodeStatistics since tensorboard depends on it to log rew
        # and length, eval only uses it if we write to csv.
        record_episode_stats = self._agent_conf.training or csv_output

        if record_episode_stats:
            env = self._wrap_record_episode_statistics(env)

        if csv_output:
            env = self._wrap_episode_csv_logger(env, sub_dir, env_idx)

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
                env, training=self._agent_conf.training
            )

    # === Model === #

    def _resolve_training_model(self, env: Env | VecEnv) -> T:
        if self._is_fresh_training_run:
            # If enabled, start TensorBoard with: tensorboard --logdir results/logs/<env_name>
            tensorboard_log = (
                str(self._log_dir / self._TRAIN)
                if self._train_conf.tensorboard_output
                else None
            )

            return self._artifact_manager.create_model(
                env, tensorboard_log=tensorboard_log, seed=self._agent_conf.seed
            )
        else:
            return self._artifact_manager.load_model(env)

    # === Train === #

    def _train_model(self, model: T, env: VecEnv):
        """Train the model for the configured number of iterations and save progress."""

        try:
            for i in range(1, self._train_conf.iterations + 1):
                model.learn(
                    total_timesteps=self._train_conf.timesteps,
                    tb_log_name=self.get_unique_model_id(),
                    reset_num_timesteps=False,
                    callback=(
                        PlateauCallback(
                            self._agent_conf.terminate_threshold,
                            self._agent_conf.plateau_threshold,
                        )
                        if self._agent_conf.plateau_detection
                        else None
                    ),
                )

                self._maybe_save_model(model, env)
        except KeyboardInterrupt:
            print("Training interrupted")
            # If you interrupt training manually, the model gets saved on that step count
            self._maybe_save_model(model, env)
        finally:
            env.close()

    def _maybe_save_model(self, model: T, env: VecEnv) -> None:
        if self._train_conf.model_output:
            self._artifact_manager.save_model(model, env, self.get_unique_model_id())

    # === Eval === #

    def _eval_model(self, env: VecEnv, model: T):
        obs = env.reset()
        try:
            for _ in range(self._eval_conf.num_eval_episodes):
                self._execution_strategy.reset(env.num_envs)
                while True:
                    action = self._execution_strategy.predict(model, obs)
                    obs, _, dones, _ = env.step(action)
                    self._execution_strategy.on_step(dones)

                    if dones[0]:
                        break
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()
