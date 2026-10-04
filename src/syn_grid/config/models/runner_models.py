
from pydantic import BaseModel, model_validator

# ======================= #
#  Nested Configurations   #
# ======================= #

class CommonConf(BaseModel, frozen=False):
    alg: str
    agent_steps: str
    seed: int
    human_control: bool
    training: bool


class TrainConf(BaseModel, frozen=False):
    continue_training: bool
    csv_output: bool
    tensorboard_output: bool
    model_output: bool
    n_envs: int
    timesteps: int
    iterations: int
    render_mode: str | None
    record_video: bool = False
    rec_interval: int
    rec_length: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.render_mode not in ["human", "rgb_array", None]:
            raise ValueError("The value of render mode is not allowed")
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


class EvalConf(BaseModel, frozen=False):
    num_eval_episodes: int
    render_mode: str | None
    record_video: bool = False
    rec_episode: int
    csv_output: bool

    @model_validator(mode="after")
    def validate_config(self):
        if self.render_mode not in ["human", "rgb_array", None]:
            raise ValueError("The value of render mode is not allowed")
        if self.record_video and self.render_mode != "rgb_array":
            raise ValueError("record_video requires render_mode='rgb_array'")
        return self

# ============================= #
#    Top-Level Configurations   #
# ============================= #

class RunnerConf(BaseModel, frozen=False):
    common_conf: CommonConf
    train_conf: TrainConf
    eval_conf: EvalConf