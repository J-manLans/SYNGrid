
from typing import Literal

from pydantic import BaseModel, model_validator

# ======================= #
#  Nested Configurations   #
# ======================= #

class CommonConf(BaseModel, frozen=True, extra="forbid", strict=True):
    alg: str
    agent_steps: str
    seed: int
    training: bool
    check_env: bool


class TrainConf(BaseModel, frozen=True, extra="forbid", strict=True):
    continue_training: bool
    csv_output: bool
    tensorboard_output: bool
    model_output: bool
    n_envs: int
    timesteps: int
    iterations: int
    render_mode: Literal["human", "rgb_array"] | None
    record_video: bool = False
    rec_interval: int
    rec_length: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.n_envs <= 0:
            raise ValueError(
                f"envs:{self.n_envs}. Can't train if there isn't an environment to train on."
            )
        if self.render_mode == "human" and self.n_envs > 1:
            raise ValueError(
                "render_mode 'human' requires n_envs=1 (live rendering doesn't "
                "support parallel environments)"
            )
        if self.record_video and self.render_mode != "rgb_array":
            raise ValueError("record_video requires render_mode='rgb_array'")
        return self


class EvalConf(BaseModel, frozen=True, extra="forbid", strict=True):
    num_eval_episodes: int
    render_mode: Literal["human", "rgb_array"] | None
    record_video: bool = False
    rec_episode: int
    csv_output: bool

    @model_validator(mode="after")
    def validate_config(self):
        if self.record_video and self.render_mode != "rgb_array":
            raise ValueError("record_video requires render_mode='rgb_array'")
        return self

# ============================= #
#    Top-Level Configurations   #
# ============================= #

class RunnerConf(BaseModel, frozen=True, extra="forbid", strict=True):
    common_conf: CommonConf
    train_conf: TrainConf
    eval_conf: EvalConf