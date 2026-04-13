from pydantic import BaseModel, model_validator

# ======================= #
#   Experiment Settings   #
# ======================= #


class SnapshotConf(BaseModel, frozen=True):
    enabled: bool
    id: str


# ----------------------- #
#   World Configuration   #
# ----------------------- #


class GridWorldConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_active_orbs: int


class RendererConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int


class DroidConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    starting_score: int


# === OrbFactory START === #


class OrbConf(BaseModel, frozen=True):
    enabled: bool
    weight: int


class TypesConf(BaseModel, frozen=True):
    negative: OrbConf
    tier: OrbConf

    class Config:
        extra = "allow"


class OrbFactoryConf(BaseModel, frozen=True):
    max_active_orbs: int
    max_tier: int
    types: TypesConf


# === OrbFactory END === #


class NegativeConf(BaseModel, frozen=True):
    reward: float
    cool_down: int


class TierConf(BaseModel, frozen=True):
    linear_reward_growth: bool
    step_wise_scoring: bool
    growth_factor: float
    base_reward: float
    cool_down: int


# ----------------------- #
#    Obs Configuration    #
# ----------------------- #


class ObservationConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_tier: int
    max_curriculum_tier: int
    max_steps: int
    modality: str
    difficulty: str


    @model_validator(mode="after")
    def clamp_curriculum(self):
        if self.max_curriculum_tier < self.max_tier:
            raise ValueError(
                f"max_curriculum_tier ({self.max_curriculum_tier}) "
                f"must be equal to or exceed max_tier ({self.max_tier})"
            )
        return self


# ----------------------- #
#   Agent Configuration   #
# ----------------------- #


class GlobalAgentConf(BaseModel, frozen=False):
    algorithm_index: int
    agent_steps: str
    identifier: str
    human_control: bool
    training: bool


class TrainAgentConf(BaseModel, frozen=False):
    continue_training: bool
    timesteps: int
    iterations: int


class EvalAgentConf(BaseModel, frozen=False):
    trained_model: bool


# ======================= #
#   Domain Config Blocks  #
# ======================= #


class WorldConfig(BaseModel, frozen=True):
    grid_world_conf: GridWorldConf
    orb_factory_conf: OrbFactoryConf
    renderer_conf: RendererConf
    droid_conf: DroidConf
    negative_orb_conf: NegativeConf
    tier_orb_conf: TierConf


class ObsConfig(BaseModel, frozen=True):
    observation_handler: ObservationConf


class AgentConfig(BaseModel, frozen=False):
    global_agent_conf: GlobalAgentConf
    train_agent_conf: TrainAgentConf
    eval_agent_conf: EvalAgentConf


###########################
#    Top Configurations   #
###########################


class ExperimentConfig(BaseModel, frozen=True):
    snapshot: SnapshotConf


class FullConf(BaseModel):
    world: WorldConfig
    obs: ObsConfig
    agent: AgentConfig
