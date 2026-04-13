from syn_grid.gymnasium.observation_space_developing.modality.spatial_modality import SpatialModality
from syn_grid.gymnasium.observation_space_developing.difficulty.easy import EasyDifficulty
from syn_grid.gymnasium.observation_space_developing.difficulty.medium import MediumDifficulty
from syn_grid.gymnasium.observation_space_developing.difficulty.hard import HardDifficulty

MODALITIES = {
    "spatial": SpatialModality,
    "vector": None
}

DIFFICULTIES = {
    "easy": EasyDifficulty,
    "medium": MediumDifficulty,
    "hard": HardDifficulty,
}
