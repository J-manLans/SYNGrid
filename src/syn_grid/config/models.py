from enum import Enum

from pydantic import BaseModel, model_validator


class ScoringMode(Enum):
    """How a tier chain's reward is paid out.

    One value rather than three mutually exclusive booleans. The booleans had
    to be validated against each other on every load, and a fourth copy of the
    question lived in the world config where the digestion engine could not see
    it and episode termination read the wrong one.
    """

    STEP_WISE = "step_wise"
    THRESHOLD = "threshold"
    MAX_TIER = "max_tier"


# ======================= #
#   Experiment Settings   #
# ======================= #


class SnapshotConf(BaseModel, frozen=True):
    enabled: bool


# ----------------------- #
#   World Configuration   #
# ----------------------- #


class GridConf(BaseModel, frozen=True):
    """World geometry and orb-field tunables.

    Deliberately holds no field that identifies a scenario. Which of these
    values are meaningful, and what they mean, is the scenario's business -- a
    tier chain refuses ``de_spawn_tiers`` and a chain that cannot terminate on
    completion is not a goal -- and those rules live with the scenario rather
    than here, where they used to have to be written as one conditional over a
    flag that named none of them.

    ``delay_mode`` used to be here. It is not any more: delay is what makes
    ``goal_tier_chain_delay`` a different scenario from
    ``goal_tier_chain_spatial``, so a flag restating that could only ever
    contradict the scenario name. ``delay``, the length of the cooldown, stays,
    because how long is a tunable and whether there is one is not.

    ``termination_on_max_tier`` was here too and was worse: validated, then read
    by nothing. A completed chain ended the episode because the scoring mode
    said so, and the flag could only disagree with that in silence.

    The validator is gone with them. It existed to reject combinations of
    scenario flags, and the scenario now rejects them against itself.
    """

    grid_rows: int
    grid_cols: int
    delay: int
    de_spawn_tiers: bool
    max_tier: int
    max_active_orbs: int


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
    boundary_penalty: float
    chain_break_penalty: float
    tier_consumption_penalty: float
    reward_multiplier: float
    timeout_penalty: float

    @model_validator(mode="after")
    def validate_config(self):
        penalties = {
            "step_penalty": self.step_penalty,
            "boundary_penalty": self.boundary_penalty,
            "chain_break_penalty": self.chain_break_penalty,
            "tier_consumption_penalty": self.tier_consumption_penalty,
            "timeout_penalty": self.timeout_penalty,
        }
        positive_penalty = [name for name, value in penalties.items() if value > 0]
        if positive_penalty:
            raise ValueError(f"{', '.join(positive_penalty)} must be 0 or negative")
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
    scoring: ScoringMode
    growth_factor: float
    base_reward: float
    cool_down: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.growth_factor <= 0:
            raise ValueError(f"{self.growth_factor} must be a positive value.")
        return self


# ----------------------- #
#    Obs Configuration    #
# ----------------------- #


class ObservationHandlerConf(BaseModel, frozen=True):
    perception: str
    max_steps: int

    @model_validator(mode="after")
    def validate_config(self):
        if self.perception not in [
            "vector_markovian_easy",
            "vector_markovian",
            "vector_fog_of_war",
            "composite_markovian",
            "composite_fully_pomdp",
            "composite_grid_markovian",
            "grid_pixel",
        ]:
            raise ValueError("The value of difficulty is not allowed")
        return self


# === PerceptionConf START === #


class EnabledOrbsConf(BaseModel, frozen=True):
    neg_enabled: bool
    tier_enabled: bool


class PerceptionConf(BaseModel, frozen=True):
    """How an observation is encoded.

    The world-derived counts are not here. How many orb slots an observation
    holds and how far its tier channel reaches are properties of the world,
    which the scenario already describes; this block used to carry a second
    copy of each that had to be kept in step by hand through YAML anchors, and
    an observation space was only fully knowable by reading four files at once.
    """

    max_score: int
    max_steps: int
    grid_rows: int
    grid_cols: int
    include_timer: bool
    enabled_orbs: EnabledOrbsConf
    tiers: int


# === PerceptionConf END === #


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


class TrainAgentConf(BaseModel, frozen=False):
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


class EvalAgentConf(BaseModel, frozen=False):
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


# ======================= #
#   Domain Config Blocks  #
# ======================= #


class WorldConfig(BaseModel, frozen=True):
    grid_conf: GridConf
    orb_factory_conf: OrbFactoryConf
    renderer_conf: RendererConf
    droid_conf: DroidConf
    negative_orb_conf: NegativeConf
    tier_orb_conf: TierConf


class ObsConfig(BaseModel, frozen=True):
    observation_handler_conf: ObservationHandlerConf
    perception_conf: PerceptionConf


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
    """A complete experiment.

    ``scenario`` selects which scenario the world implements. Everything below
    it is a tunable: a parameter the selected scenario reads, not a combination
    that identifies it. That distinction is the whole point. A config used to
    define its own scenario through a set of booleans which six separate places
    then had to interpret consistently, and any two of them could disagree
    about what the same file meant.
    """

    scenario: str
    world: WorldConfig
    obs: ObsConfig
    agent: AgentConfig

    @model_validator(mode="after")
    def validate_scenario(self):
        # Imported here rather than at module scope: the registry builds
        # scenarios out of these very models, so a top-level import would be
        # circular. Resolving during validation means a config naming a
        # scenario that does not exist, or one whose parameters its scenario
        # rejects, fails at load rather than at environment setup.
        from syn_grid.scenario.registry import build_scenario

        try:
            build_scenario(self.scenario, self.world, self.obs)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        return self

    @model_validator(mode="after")
    def validate_grid_dimensions(self):
        grid_dimensions = {
            "grid_world_conf": (
                self.world.grid_conf.grid_rows,
                self.world.grid_conf.grid_cols,
            ),
            "orb_factory_conf": (
                self.world.orb_factory_conf.grid_rows,
                self.world.orb_factory_conf.grid_cols,
            ),
            "renderer_conf": (
                self.world.renderer_conf.grid_rows,
                self.world.renderer_conf.grid_cols,
            ),
            "droid_conf": (
                self.world.droid_conf.grid_rows,
                self.world.droid_conf.grid_cols,
            ),
            "perception": (
                self.obs.perception_conf.grid_rows,
                self.obs.perception_conf.grid_cols,
            ),
        }

        # Check grid size bigger than 0
        invalid = {
            name: dims
            for name, dims in grid_dimensions.items()
            if dims[0] <= 0 or dims[1] <= 0
        }
        if invalid:
            raise ValueError(f"grid_rows/grid_cols must be larger than 0: {invalid}")

        # Check that all grid_row/col are equal across configurations
        reference = grid_dimensions["grid_world_conf"]
        mismatched = {
            name: dims for name, dims in grid_dimensions.items() if dims != reference
        }

        if mismatched:
            raise ValueError(
                f"grid_rows/grid_cols mismatch: grid_world_conf has {reference}, "
                f"but these blocks differ: {mismatched}"
            )
        return self
