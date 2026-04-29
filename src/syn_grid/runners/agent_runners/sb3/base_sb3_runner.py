from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.config.models import AgentConfig, GlobalAgentConf, WorldConfig, ObsConfig
from syn_grid.utils.paths_util import get_project_path

import os
from typing import Type, TypeVar, Any, Generic
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from gymnasium import Env

T = TypeVar("T", bound=BaseAlgorithm)


class BaseSB3Runner(BaseAgentRunner, Generic[T]):
    # ================= #
    #       Init        #
    # ================= #

    _POLICY_MAP = {"vector": "Mlp", "composite": "MultiInput", "spatial": "Cnn"}

    def __init__(
        self,
        conf: AgentConfig,
        obs_conf: ObsConfig,
        run_conf: WorldConfig,
        hyper_parameters: dict[str, Any],
        algorithm: Type[T],
        lstm_hidden_size: int | None = None,
    ):
        super().__init__(conf, obs_conf, run_conf)
        id = self._construct_model_id(
            conf.global_agent_conf, run_conf, lstm_hidden_size
        )
        super()._set_id(id)
        self._HYPER_PARAMETERS = hyper_parameters
        self._ALGORITHM = algorithm

        # Create directory for saving learned statistics for normalization
        self._vec_norm_stats_dir = Path(
            get_project_path("output", "results", "saved_vec_norms")
        )
        Path(self._vec_norm_stats_dir).mkdir(parents=True, exist_ok=True)

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

        return policy

    # ================= #
    #      Helpers      #
    # ================= #

    def _construct_model_id(
        self,
        agent_conf: GlobalAgentConf,
        run_conf: WorldConfig,
        lstm_hidden_size: int | None = None,
    ) -> str:
        base_tier_id, base_non_tier_id = super()._get_model_base_id()

        # --- RecurrentPPO with tier orbs --- #
        if agent_conf.alg == "RPPO" and run_conf.orb_factory_conf.types.tier.enabled:
            return f"{base_tier_id}{lstm_hidden_size}"
        # --- RecurrentPPO without tier orbs --- #
        elif (
            agent_conf.alg == "RPPO"
            and not run_conf.orb_factory_conf.types.tier.enabled
        ):
            return f"{base_non_tier_id}{lstm_hidden_size}"
        # --- With tier orbs --- #
        elif (
            not agent_conf.alg == "RPPO"
            and run_conf.orb_factory_conf.types.tier.enabled
        ):
            return f"{base_tier_id}"
        # --- Without tier orbs --- #
        elif (
            not agent_conf.alg == "RPPO"
            and not run_conf.orb_factory_conf.types.tier.enabled
        ):
            return f"{base_non_tier_id}"
        else:
            raise ValueError(
                f"Unhandled case:\n"
                f"alg={agent_conf.alg}\n"
                f"tier_enabled={run_conf.orb_factory_conf.types.tier.enabled}"
            )

    # === Env === #

    def _make_env(self, render_mode: str | None) -> Env:
        env = self._make_raw_env(render_mode)

        if self._conf.training and self._train_conf.enable_output:
            env = self._wrap_in_monitor(env)

        return env

    def _get_vec_env(self, env: DummyVecEnv) -> VecNormalize:
        if self._conf.training and not self._train_conf.continue_training:
            return self._apply_normalize_wrapper(env)
        else:
            return self._load_normalize_wrapper(env)

    # --- Wrappers --- #

    def _wrap_in_monitor(self, env: Env) -> Env:
        """
        Wrap the environment with a Monitor for logging.
        The created csv is needed for plotting our own graphs with matplotlib later.
        """

        return Monitor(env=env, filename=str(self.log_dir / self._get_model_id()))

    def _make_wrapped_dummy_vec_env(self, render_mode: str | None) -> DummyVecEnv:
        return DummyVecEnv([lambda: self._make_env(render_mode)])

    def _load_normalize_wrapper(self, env: DummyVecEnv) -> VecNormalize:
        evn_load_path = self._get_saved_path(self._vec_norm_stats_dir)
        vec_env = VecNormalize.load(str(evn_load_path), env)
        vec_env.training = False
        return vec_env

    def _apply_normalize_wrapper(self, env: DummyVecEnv) -> VecNormalize:
        return VecNormalize(env, norm_obs=True, norm_reward=False)

    # === Model === #

    def _get_model(self, env: Env | VecEnv) -> T:
        if self._conf.training and not self._train_conf.continue_training:
            return self._create_model(env)
        else:
            return self._load_model(env)

    def _load_model(self, env: Env | VecEnv) -> T:
        model_path = self._get_saved_path(self._model_dir)
        return self._ALGORITHM.load(
            model_path, env=env, device=self._HYPER_PARAMETERS["device"]
        )

    def _create_model(self, env: Env | VecEnv) -> T:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        return self._ALGORITHM(
            env=env,
            verbose=1,
            tensorboard_log=(
                str(self.log_dir) if self._train_conf.enable_output else None
            ),
            **self._HYPER_PARAMETERS,
        )

    # === Train === #

    def _train_model(self, model: T, env: Env | VecEnv):
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
                )

                if self._train_conf.enable_output:
                    # Save the model
                    checkpoint = f"{model.num_timesteps}_{self._get_model_id()}.zip"
                    model.save(self._model_dir / checkpoint)
                    print(f"\nModel saved with {model.num_timesteps} time steps")

                    if isinstance(env, VecNormalize):
                        evn_save_path = f"{self._vec_norm_stats_dir}/{model.num_timesteps}_{self._get_model_id()}.pkl"
                        env.save(evn_save_path)
        finally:
            env.close()
