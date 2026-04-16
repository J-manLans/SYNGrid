from syn_grid.gymnasium.observation_space.modality.vector_modality import VectorModality
from syn_grid.gymnasium.observation_space.modality.composite_modality import (
    CompositeModality,
)
from syn_grid.gymnasium.observation_space.modality.spatial_modality import (
    SpatialModality,
)
from syn_grid.gymnasium.observation_space.difficulty.easy import EasyDifficulty
from syn_grid.gymnasium.observation_space.difficulty.medium import MediumDifficulty
from syn_grid.gymnasium.observation_space.difficulty.hard import HardDifficulty

MODALITIES = {
    "vector": VectorModality,
    "composite": CompositeModality,
    "spatial": SpatialModality,
}

DIFFICULTIES = {
    "easy": EasyDifficulty,
    "medium": MediumDifficulty,
    "hard": HardDifficulty,
}
