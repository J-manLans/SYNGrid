from syn_grid.gymnasium.observation_space.modality.spatial_modality import (
    SpatialModality,
)
from syn_grid.gymnasium.observation_space.difficulty.easy import (
    EasyDifficulty,
)
from syn_grid.gymnasium.observation_space.difficulty.medium import (
    MediumDifficulty,
)
from syn_grid.gymnasium.observation_space.difficulty.hard import (
    HardDifficulty,
)

MODALITIES = {"spatial": SpatialModality, "vector": None}

DIFFICULTIES = {
    "easy": EasyDifficulty,
    "medium": MediumDifficulty,
    "hard": HardDifficulty,
}
