from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig


import os
from typing import Type, Any
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env


class BaseSB3Runner(BaseAgentRunner):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        conf: AgentConfig,
        obs_conf: ObsConfig,
        run_conf: WorldConfig,
        hyper_parameters: dict[str, Any],
        algorithm: Type[BaseAlgorithm],
    ):
        super().__init__(conf, obs_conf, run_conf)
        self._HYPER_PARAMETERS = hyper_parameters
        self._ALGORITHM = algorithm

    # ================= #
    #      Helpers      #
    # ================= #

    def _wrap_in_monitor(self, env: Env) -> Env:
        """
        Wrap the environment with a Monitor for logging.
        The created csv is needed for plotting our own graphs with matplotlib later.
        """

        log = str(self.log_dir / self._get_log_identifier())

        return Monitor(env=env, filename=str(self.log_dir / self._get_log_identifier()))

    def _load_model(self, env: Env) -> BaseAlgorithm:
        model_path = self._get_model_path()
        return self._ALGORITHM.load(
            model_path, env=env, device=self._HYPER_PARAMETERS["device"]
        )

    def _create_model(self, env: Env) -> BaseAlgorithm:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        return self._ALGORITHM(
            env=env,
            verbose=1,
            tensorboard_log=(
                str(self.log_dir) if self.train_conf.enable_output else None
            ),
            **self._HYPER_PARAMETERS,
        )

    def _train_model(self, model: BaseAlgorithm, env: Env):
        try:
            # This loop will keep training based on timesteps and iterations.
            # After the timesteps are completed, the model is saved and training
            # continues for the next iteration. When training is done, start another
            # cmd prompt and launch Tensorboard:
            # tensorboard --logdir results/logs/<env_name>
            # Once Tensorboard is loaded, it will print a URL. Follow the URL to see
            # the status of the training.
            for i in range(1, self.train_conf.iterations + 1):
                # Train the model
                model.learn(
                    total_timesteps=self.train_conf.timesteps,
                    tb_log_name=self._get_log_identifier(),
                    reset_num_timesteps=False,
                )

                if self.train_conf.enable_output:
                    # Save the model
                    save_identifier = (
                        f"{model.num_timesteps}_{self._get_log_identifier()}.zip"
                    )
                    model.save(Path(self.model_dir) / save_identifier)
                    print(f"\nModel saved with {model.num_timesteps} time steps")
        finally:
            env.close()
