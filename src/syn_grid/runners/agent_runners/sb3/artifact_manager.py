"""Persistence for SB3 models and their VecNormalize statistics.

Owns everything that touches disk for an SB3 runner: creating and loading
models, and creating and loading the running normalization stats that must
travel alongside a checkpoint for eval or continued training to be valid.

Whether to create or load is an orchestration decision (it depends on
whether we're training from scratch, continuing training, or evaluating),
so that branching stays in BaseSB3Runner. This class only knows how to do
each half once asked.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, unwrap_vec_normalize

T = TypeVar("T", bound=BaseAlgorithm)


class ArtifactManager(Generic[T]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        algorithm: type[T],
        hyper_parameters: dict[str, Any],
        model_dir: Path,
        vec_norm_stats_dir: Path,
        find_latest_saved_path: Callable[[Path], Path],
    ):
        self._algorithm = algorithm
        self._hyper_parameters = hyper_parameters
        self._model_dir = model_dir
        self._vec_norm_stats_dir = vec_norm_stats_dir
        self._find_latest_saved_path = find_latest_saved_path

    # ================= #
    #        API        #
    # ================= #

    # === Normalization === #

    def create_normalize_wrapper(self, env: VecEnv) -> VecNormalize:
        """Wrap a fresh env for training from scratch."""
        return VecNormalize(env, norm_obs=True, norm_reward=False)

    def load_normalize_wrapper(self, env: VecEnv) -> VecNormalize:
        """Load saved normalization stats onto an env, for eval or resumed training."""
        stats_path = str(self._find_latest_saved_path(self._vec_norm_stats_dir))
        vec_env = VecNormalize.load(stats_path, env)
        vec_env.training = False
        return vec_env

    # === Model === #

    def create_model(
        self, env: Env | VecEnv, tensorboard_log: str | None, seed: int
    ) -> T:
        return self._algorithm(
            env=env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=seed,
            **self._hyper_parameters,
        )

    def load_model(self, env: Env | VecEnv) -> T:
        model_path = self._find_latest_saved_path(self._model_dir)
        return self._algorithm.load(path=model_path, env=env, **self._hyper_parameters)

    def save_model(self, model: T, env, unique_model_id: str) -> Path:
        checkpoint = f"{model.num_timesteps}_{unique_model_id}.zip"
        model_path = self._model_dir / checkpoint
        model.save(model_path)
        print(f"\nModel saved with {model.num_timesteps} time steps")

        vec_normalize = unwrap_vec_normalize(env)
        if vec_normalize is not None:
            stats_path = (
                self._vec_norm_stats_dir
                / f"{model.num_timesteps}_{unique_model_id}.pkl"
            )
            vec_normalize.save(str(stats_path))
            print(f"VecNormalize stats saved at {model.num_timesteps} timesteps")

        return model_path
