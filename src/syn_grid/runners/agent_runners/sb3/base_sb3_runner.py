from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig
from syn_grid.utils.paths_util import get_project_path
from syn_grid.runners.agent_runners.sb3.utils.plateau_callback import PlateauCallback

import os
from typing import Type, TypeVar, Any, Generic
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecEnv,
    unwrap_vec_normalize,
)
from gymnasium import Env

T = TypeVar("T", bound=BaseAlgorithm)


class BaseSB3Runner(BaseAgentRunner, Generic[T]):
    # ================= #
    #       Init        #
    # ================= #

    _POLICY_MAP = {"vector": "Mlp", "composite": "MultiInput", "grid": "Cnn"}
    _TRAIN = "train"
    _EVAL = "eval"

    def __init__(
        self,
        conf: AgentConfig,
        obs_conf: ObsConfig,
        run_conf: WorldConfig,
        hyper_parameters: dict[str, Any],
        algorithm: Type[T],
    ):
        super().__init__(conf, obs_conf, run_conf)
        self._HYPER_PARAMETERS = hyper_parameters
        self._ALGORITHM = algorithm
        self._writer = None

        self._init_normalization_stats_dir()

    @classmethod
    def _get_policy_from_perception(
        cls, perception_str: str, use_lstm: bool = False
    ) -> str:
        """Extract SB3 policy string from perception configuration."""

        policy = ""

        for perception_key, base_policy in cls._POLICY_MAP.items():
            if perception_key in perception_str:
                suffix = "LstmPolicy" if use_lstm else "Policy"
                policy = base_policy + suffix
                break

        return policy

    # ================= #
    #      Helpers      #
    # ================= #

    def _init_normalization_stats_dir(self):
        """
        Create directory for saving environment normalization statistics.
        Required for consistent eval, also for resuming training.
        """

        base = get_project_path("output", "models_vec_norms")
        self._vec_norm_stats_dir = (
            base / self._conf.save_folder if self._conf.save_folder else base
        )

        self._vec_norm_stats_dir.mkdir(parents=True, exist_ok=True)

    # === Env === #

    def _make_wrapped_dummy_vec_env(
        self, render_mode: str | None, sub_dir: str
    ) -> DummyVecEnv:
        n_envs = self._train_conf.n_envs if self._conf.training else 1
        return DummyVecEnv(
            [
                lambda i=i: self._make_env(render_mode, sub_dir, env_idx=i)
                for i in range(n_envs)
            ]
        )

    def _make_env(self, render_mode: str | None, sub_dir: str, env_idx: int) -> Env:
        # During training, only env 0 gets a render mode so that it can be
        # used for rendering/video recording; the remaining environments do not
        # need to render. During evaluation, the render mode is passed to the
        # single environment.
        #
        # NOTE: DummyVecEnv requires all environments to have the same render_mode.
        # Therefore, using a render mode for env 0 and None for the others causes
        # a render_mode mismatch when training with multiple environments. So need to
        # rethink this one.
        env = self._make_raw_env(
            render_mode if (not self._conf.training or env_idx == 0) else None
        )

        # if logging is enabled
        # fmt: off
        if env_idx == 0:
            if (
                (self._conf.training and self._train_conf.csv_output)
                or (not self._conf.training and self._eval_conf.csv_output)
            ):
                env = self._logger_wrapper(env, sub_dir)

            # if video recording for training is on record at a specific timestep interval
            if self._conf.training and self._train_conf.render_mode == "rgb_array":
                env = self._rec_video_wrapper(
                    env,
                    step_trigger=lambda t: t % self._train_conf.rec_interval == 0,
                    video_length=self._train_conf.rec_length,
                )
        # fmt: on

        # if video recording for evaluation is on record selected episode
        if not self._conf.training and self._eval_conf.render_mode == "rgb_array":
            env = self._rec_video_wrapper(
                env, episode_trigger=lambda t: t == self._eval_conf.rec_episode
            )

        return env

    def _get_normalized_env(self, env: DummyVecEnv) -> VecNormalize:
        if self._conf.training and not self._train_conf.continue_training:
            return self._apply_normalize_wrapper(env)
        else:
            return self._load_normalize_wrapper(env)

    # --- Wrappers --- #

    # If we're training from scratch
    def _apply_normalize_wrapper(self, env: DummyVecEnv) -> VecNormalize:
        return VecNormalize(env, norm_obs=True, norm_reward=False)

    # If we are loading a checkpoint for evaluation or continual training
    def _load_normalize_wrapper(self, env: DummyVecEnv) -> VecNormalize:
        evn_load_path = self._get_saved_path(self._vec_norm_stats_dir)
        vec_env = VecNormalize.load(str(evn_load_path), env)
        vec_env.training = False
        return vec_env

    # === Model === #

    def _get_model(self, env: Env | VecEnv, sub_dir: str) -> T:
        if self._conf.training and not self._train_conf.continue_training:
            return self._create_model(env, sub_dir)
        else:
            return self._load_model(env)

    def _create_model(self, env: Env | VecEnv, sub_dir: str) -> T:
        return self._ALGORITHM(
            env=env,
            verbose=1,
            tensorboard_log=(
                str(self._log_dir / sub_dir)
                if self._train_conf.tensorboard_output
                else None
            ),
            seed=self._conf.seed,
            **self._HYPER_PARAMETERS,
        )

    def _load_model(self, env: Env | VecEnv) -> T:
        model_path = self._get_saved_path(self._model_dir)
        return self._ALGORITHM.load(path=model_path, env=env, **self._HYPER_PARAMETERS)

    # === Train === #

    def _train_model(self, model: T, env: VecEnv):
        try:
            # This loop will keep training based on timesteps and iterations.
            # After the timesteps are completed, the model is saved and training
            # continues for the next iteration. When training is done, start another
            # cmd prompt and launch Tensorboard:
            # tensorboard --logdir results/logs/<env_name>
            # Once Tensorboard is loaded, it will print a URL. Follow the URL to see
            # the status of the training.

            for i in range(1, self._train_conf.iterations + 1):
                # Train the model
                model.learn(
                    total_timesteps=self._train_conf.timesteps,
                    tb_log_name=self._get_model_id(),
                    reset_num_timesteps=False,
                    callback=(
                        PlateauCallback(
                            self._conf.terminate_threshold, self._conf.plateau_threshold
                        )
                        if self._conf.plateau_detection
                        else None
                    ),
                )

                self._save_model(model, env)
        except KeyboardInterrupt:
            print("Training interrupted")
            self._save_model(model, env)
        finally:
            env.close()

    def _save_model(self, model: T, env):
        if self._train_conf.model_output:
            # Save the model
            checkpoint = f"{model.num_timesteps}_{self._get_model_id()}.zip"
            model.save(self._model_dir / checkpoint)
            print(f"\nModel saved with {model.num_timesteps} time steps")

            vec_normalize = unwrap_vec_normalize(env)
            if vec_normalize is not None:
                evn_save_path = f"{self._vec_norm_stats_dir}/{model.num_timesteps}_{self._get_model_id()}.pkl"
                vec_normalize.save(evn_save_path)
                print(f"VecNormalize stats saved at {model.num_timesteps} timesteps")

    # === Eval === #

    def _eval_model(self, env: VecEnv, model: T):
        obs = env.reset()
        try:
            for i in range(self._eval_conf.num_eval_episodes):
                while True:
                    action, states = model.predict(
                        obs, deterministic=True  # type: ignore[arg-type]
                    )
                    obs, rewards, dones, info = env.step(action)

                    if dones[0]:
                        break
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()
