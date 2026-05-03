from pydantic import BaseModel, model_validator

# ======================= #
#   Experiment Settings   #
# ======================= #


class SnapshotConf(BaseModel, frozen=True):
    enabled: bool


# ----------------------- #
#   World Configuration   #
# ----------------------- #


class GridWorldConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_active_orbs: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError("grid_cols and grid_rows should be larger than 0")
        if self.max_active_orbs <= 0:
            raise ValueError("max_active_orbs should be larger than 0")
        return self


# === Renderer START === #


class AssetsConf(BaseModel, frozen=True):
    droid_img: str
    positive_orb_img: str
    negative_orb_img: str
    floor_img: str
    hud_img: str


class RendererConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    img_assets: AssetsConf


# === Renderer END === #


class DroidConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    starting_score: float
    step_penalty: float
    tier_consumption_penalty: float

    @model_validator(mode="after")
    def validate_config(self):
        if self.tier_consumption_penalty > 0:
            raise ValueError("tier_consumption_penalty must be 0 or negative")
        return self


# === OrbFactory START === #


class OrbConf(BaseModel, frozen=True):
    enabled: bool
    weight: int


class TypesConf(BaseModel, frozen=True):
    negative: OrbConf
    tier: OrbConf


class OrbFactoryConf(BaseModel, frozen=True):
    grid_rows: int
    grid_cols: int
    max_active_orbs: int
    max_tier: int
    de_spawn_tiers: bool
    types: TypesConf

    @model_validator(mode="after")
    def validate_config(self):
        if self.max_tier <= 0:
            raise ValueError("max_tier should be larger than 0")
        return self


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


class ObservationHandlerConf(BaseModel, frozen=True):
    perception: str
    max_steps: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.perception not in [
            "vector_markovian",
            "vector_fully_pomdp",
            "composite_markovian",
            "composite_fully_pomdp",
            "composite_grid_markovian",
            "grid_pixel",
        ]:
            raise ValueError("The value of difficulty is not allowed")
        return self


class PerceptionConf(BaseModel, frozen=True):
    max_score: int
    max_steps: int
    max_tier: int
    grid_rows: int
    grid_cols: int
    max_active_orbs: int


# ----------------------- #
#   Agent Configuration   #
# ----------------------- #


class GlobalAgentConf(BaseModel, frozen=False):
    alg: str
    agent_steps: str
    id_tag: str | None
    save_folder: str | None
    seed: int
    human_control: bool
    training: bool
    check_env: bool


class TrainAgentConf(BaseModel, frozen=False):
    continue_training: bool
    monitor_output: bool
    tensorboard_output: bool
    model_output: bool
    render_mode: str | None
    timesteps: int
    iterations: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.render_mode not in ["human", None]:
            raise ValueError("The value of render mode is not allowed")
        return self


class EvalAgentConf(BaseModel, frozen=False):
    num_eval_episodes: int
    render_mode: str | None

    @model_validator(mode="after")
    def validate_config(self):
        if self.render_mode not in ["human", None]:
            raise ValueError("The value of render mode is not allowed")
        return self


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
    observation_handler: ObservationHandlerConf
    perception: PerceptionConf


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
