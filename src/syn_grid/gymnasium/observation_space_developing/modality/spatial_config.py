from pydantic import BaseModel, ConfigDict


class SpatialDifficultyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str

    # How much of each feature is visible
    visibility: dict[str, float]

    # Global masking value for hidden info
    mask_value: float = -1.0


EASY = SpatialDifficultyConfig(
    name="easy", visibility={"orbs": 1.0, "tiers": 1.0, "effects": 1.0}, mask_value=-1.0
)

DEFAULT = SpatialDifficultyConfig(
    name="default", visibility={"orbs": 0.6, "tiers": 0.5, "effects": 0.2}
)

HARD = SpatialDifficultyConfig(
    name="hard", visibility={"orbs": 0.2, "tiers": 0.1, "effects": 0.0}
)
