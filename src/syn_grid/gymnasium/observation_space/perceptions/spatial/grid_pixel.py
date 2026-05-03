from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)


class GridPixel(BasePerception):
    """
    Planned future implementation.

    A pixel-based observation represented as a Box(H, W, 3) RGB image,
    where the grid is rendered at a fixed pixel resolution. H and W refer
    to rendered image dimensions, not grid cell counts — each grid cell
    is represented as a block of pixels. Intended for CNN-based agents
    that infer all state information purely from visual input, with no
    additional spaces or globals.
    """

    NotImplementedError()
