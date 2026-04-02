from pydantic import BaseModel

###########################
#   Experiment Settings   #
###########################


class SnapshotConf(BaseModel, frozen=True):
    enabled: bool
    id: str


###########################
#    Run Configuration    #
###########################


class GridWorldConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_active_resources: int
    max_tier: int


class RendererConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int


class DroidConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    starting_score: int


class NegativeConf(BaseModel, frozen=True):
    reward: float
    cool_down: int


class TierConf(BaseModel, frozen=True):
    linear_reward_growth: bool
    step_wise_scoring: bool
    growth_factor: float
    base_reward: float
    cool_down: int


###########################
#    Obs Configuration    #
###########################


class ObservationConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_steps: int


###########################
#   Agent Configuration   #
###########################


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


###########################
#    Top Configurations   #
###########################


class ExperimentConfig(BaseModel, frozen=True):
    snapshot: SnapshotConf


class RunConfig(BaseModel, frozen=True):
    grid_world_conf: GridWorldConf
    renderer_conf: RendererConf
    droid_conf: DroidConf
    negative_resource_conf: NegativeConf
    tier_resource_conf: TierConf


class ObsConfig(BaseModel, frozen=True):
    observation_handler: ObservationConf


class AgentConfig(BaseModel, frozen=False):
    global_agent_conf: GlobalAgentConf
    train_agent_conf: TrainAgentConf
    eval_agent_conf: EvalAgentConf


class FullConf(BaseModel):
    run: RunConfig
    obs: ObsConfig
    agent: AgentConfig
