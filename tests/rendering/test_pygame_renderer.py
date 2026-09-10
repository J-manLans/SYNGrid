from unittest.mock import patch

import numpy as np
import pytest

from syn_grid.core.orbs.orb_meta import OrbCategory, OrbMeta, SynergyType
from syn_grid.rendering.pygame_renderer import PygameRenderer
from tests.utils.config_helpers import get_test_config


class TestPygameRendererModeAwareness:
    """
    PygameRenderer should only ever touch the display/window subsystem in "human" mode. "rgb_array"
    must stay a pure off-screen Surface so N parallel envs never spawn N OS windows, and so
    rgb_array construction works on a headless machine with no video driver at all.

    pygame.display is patched throughout so these tests are independent of whatever global pygame
    state earlier tests in the same session left behind, and don't depend on there being a real
    display available.
    """

    # ================= #
    #     Fixtures      #
    # ================= #

    @pytest.fixture
    def renderer_conf(self):
        return get_test_config().world.renderer_conf

    @pytest.fixture
    def hud_data(self) -> dict[str, int | float]:
        return {"score": 0.0, "moves": 10, "current tier chain": -1}

    @pytest.fixture
    def single_tier_orb(self) -> list[OrbMeta]:
        return [OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, tier=1)]

    # ================= #
    #       Tests       #
    # ================= #

    def test_rgb_array_never_touches_display_subsystem(self, renderer_conf):
        with (
            patch("pygame.display.init") as mock_display_init,
            patch("pygame.display.set_mode") as mock_set_mode,
        ):
            renderer = PygameRenderer(renderer_conf, "rgb_array", fps=8)

        mock_display_init.assert_not_called()
        mock_set_mode.assert_not_called()
        assert isinstance(renderer._window_surface, __import__("pygame").Surface)

    def test_human_mode_initializes_a_real_display(self, renderer_conf):
        with (
            patch("pygame.display.init") as mock_display_init,
            patch("pygame.display.set_mode") as mock_set_mode,
            patch("pygame.display.set_caption"),
        ):
            PygameRenderer(renderer_conf, "human", fps=8)

        mock_display_init.assert_called_once()
        mock_set_mode.assert_called_once()

    def test_rgb_array_render_returns_pixel_array(
        self, renderer_conf, hud_data, single_tier_orb
    ):
        renderer = PygameRenderer(renderer_conf, "rgb_array", fps=8)

        frame = renderer.render(
            droid_pos=[0, 0],
            is_active_statuses=[True],
            orb_positions=[[0, 0]],
            orb_meta=single_tier_orb,
            hud_data=hud_data,
        )

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        # render() transposes to (height, width, 3); window_size is (width, height)
        assert frame.shape[:2] == (renderer.window_size[1], renderer.window_size[0])

    def test_human_mode_render_returns_none(
        self, renderer_conf, hud_data, single_tier_orb
    ):
        with (
            patch("pygame.display.init"),
            patch("pygame.display.set_mode") as mock_set_mode,
            patch("pygame.display.set_caption"),
            patch("pygame.display.update"),
        ):
            mock_set_mode.return_value = __import__("pygame").Surface((10, 10))
            renderer = PygameRenderer(renderer_conf, "human", fps=8)

            result = renderer.render(
                droid_pos=[0, 0],
                is_active_statuses=[True],
                orb_positions=[[0, 0]],
                orb_meta=single_tier_orb,
                hud_data=hud_data,
            )

        assert result is None

    def test_rgb_array_construction_is_headless_safe(self, renderer_conf):
        """
        No pygame.init()/pygame.display.init() dependency at all for
        rgb_array — only pygame.font.init(), which needs no video driver.
        This is what makes it safe to construct on a machine with no
        display (e.g. a remote training box without a virtual display).
        """

        with patch("pygame.init") as mock_pygame_init:
            PygameRenderer(renderer_conf, "rgb_array", fps=8)

        mock_pygame_init.assert_not_called()
