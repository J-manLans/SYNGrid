from typing import Final

import numpy as np
import pygame

from syn_grid.core.orbs.orb_meta import (
    DirectKind,
    OrbMeta,
    SynergyKind,
)
from syn_grid.scenario.blocks.hud import HudBar, HudElement
from syn_grid.utils.paths_util import get_syn_grid_path


class PygameRenderer:
    _GRAPHIC_RESOURCES: Final[dict[str, str]] = {
        "droid_img": "assets/sprites/droid.png",
        "positive_orb_img": "assets/sprites/positive_orb.png",
        "negative_orb_img": "assets/sprites/negative_orb.png",
        "floor_img": "assets/tiles/floor.png",
        "hud_img": "assets/sprites/hud.png",
    }
    _FONT: Final[str] = "assets/fonts/Minecraft.ttf"

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        grid_dimensions: tuple[int, int],
        hud: tuple[HudElement, ...],
        render_mode: str | None,
        fps: int,
    ) -> None:
        """
        Initializes the Pygame renderer.

        Sets:
        - Size of grid world
        - The full value of each HUD bar, by label
        - One logic step fps, not game.
        - Game colors
        - Game font
        - Graphic elements
        """

        self._rows: Final[int] = grid_dimensions[0]
        self._cols: Final[int] = grid_dimensions[1]
        self._render_mode: Final[str | None] = render_mode
        self._bars_max: Final[dict[str, float]] = {
            hud_element.label: hud_element.maximum
            for hud_element in hud
            if isinstance(hud_element, HudBar)
        }

        # Game font, one per size, loaded the first time a size is used
        pygame.font.init()
        self._fonts: dict[int, pygame.font.Font] = {}

        # Initialize assets
        self._init_colors()
        self._init_text_sizes()
        self._init_vars()
        self._load_graphics()

        # Define game window size (width, height)
        self.window_size = (
            self._window_width,
            (self._cell_height * self._rows)
            + (self._grid_offset * 3)
            + self._hud_height,
        )

        # human: a real OS window. rgb_array: an off-screen pixel buffer, no
        # display/window manager involved, and safe to construct on a headless
        # machine with no video driver.
        if self._render_mode == "human":
            pygame.display.init()  # initialize the display module
            self._clock = pygame.time.Clock()  # Game clock
            self._step_fps = fps
            self._window_surface = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("SYNGrid")
        else:
            self._window_surface = pygame.Surface(self.window_size)

    # ================= #
    #        API        #
    # ================= #

    def render(
        self,
        droid_pos: list[int],
        is_active_statuses: list[bool],
        orb_positions: list[list[int]],
        orb_meta: list[OrbMeta],
        hud_data: dict[str, int | float],
    ) -> np.ndarray | None:
        """
        Draws the game to the surface. Updates the display and limits fps in
        human mode; returns a pixel array in rgb_array mode.
        """

        # Draw the background first of all
        self._window_surface.fill(self._dark_grey_clr)

        col = (droid_pos[1] * self._cell_width) + self._grid_offset
        row = (droid_pos[0] * self._cell_height) + self._grid_offset

        # Draw the graphics with pygame. blit() draws things in order, so we need to stack elements
        # in the order we want them to be drawn
        self._draw_floor_and_orbs(orb_positions, orb_meta, is_active_statuses)
        self._draw_droid((col, row))
        self._draw_hud(hud_data)

        # If we have actual visualization, update the display, else if we're running in rgb_array
        # mode, compress the visual information into an array and return that one.
        if self._render_mode == "human":
            self._update()
            return None
        elif self._render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self._window_surface), axes=(1, 0, 2)
            )

    def draw_summary(self, summary: dict[str, float]) -> None:
        """
        Draws an episode summary over the last frame, one line per entry, in human mode.

        The frame is dimmed, the keys that play again and quit are written centred at the top, and
        below them a "Stats" header and each entry as `label: value`, in the order given.
        """

        self._dim_frame()

        instructions = [
            self._render_text("Space: play again", self._large_text, self._yellow_clr),
            self._render_text("Escape: quit", self._large_text, self._yellow_clr),
        ]
        top = self._grid_offset * 2
        self._draw_lines(instructions, top=top, centred=True)

        stats = [self._render_text("Stats", self._medium_text, self._blue_clr)]
        # :g shows up to six significant digits, so float noise is cut and whole numbers stay whole
        stats += [
            self._render_text(f"{label}: {value:g}", self._small_text, self._purple_clr)
            for label, value in summary.items()
        ]
        # One instruction line's height of space between the instructions and the stats
        instructions_height = sum(line.get_height() for line in instructions)
        gap = instructions[0].get_height()
        self._draw_lines(stats, top=top + instructions_height + gap, centred=False)

        pygame.display.update()

    def draw_pause(self) -> None:
        """
        Draws the pause screen over the last frame, in human mode.

        The frame is dimmed, and a "Paused" header and the keys that resume and quit are written
        centred over the grid.
        """

        self._dim_frame()

        lines = [
            self._render_text("Paused", self._large_text, self._yellow_clr),
            self._render_text("Space: resume", self._medium_text, self._purple_clr),
            self._render_text("Escape: quit", self._medium_text, self._purple_clr),
        ]
        grid_height = self._cell_height * self._rows
        text_height = sum(line.get_height() for line in lines)
        top = self._grid_offset + (grid_height - text_height) // 2
        self._draw_lines(lines, top=top, centred=True)

        pygame.display.update()

    def set_title(self, title: str) -> None:
        """Sets the window's title in human mode."""

        if self._render_mode == "human":
            pygame.display.set_caption(title)

    def close(self) -> None:
        """Closes the window and shuts pygame down."""

        pygame.quit()

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init ===#

    def _init_colors(self) -> None:
        """Sets the game's colors. Yellow and blue are from droid.png, green from hud.png."""

        self._dark_grey_clr = (26, 26, 26)
        self._grey_clr = (75, 75, 75)
        self._purple_clr = (165, 102, 192)
        self._yellow_clr = (242, 255, 0)
        self._green_clr = (38, 202, 40)
        self._blue_clr = (0, 194, 251)

    def _init_text_sizes(self) -> None:
        """Sets the game's text sizes, in points."""

        self._small_text = 20
        self._medium_text = 25
        self._large_text = 30

    def _init_vars(self):
        # Sizes of each cell in the grid
        self._cell_width = 64
        self._cell_height = 64

        # Used for created tier surfaces in draw_orbs()
        self._tier_text_cache: dict[int, pygame.Surface] = {}

        # Different measurements for coherence
        self._grid_offset = self._cell_width // 4
        self._window_width = (self._cell_width * self._cols) + self._grid_offset * 2
        self._hud_height = self._cell_height * 4
        self._hud_width = self._cell_width * 5
        self._padding = 10

    def _load_graphics(self) -> None:
        """Load the application's static graphics."""

        self.graphics = {
            name: pygame.image.load(get_syn_grid_path(path))
            for name, path in self._GRAPHIC_RESOURCES.items()
        }

    # === API ===#

    def _draw_floor_and_orbs(self, orb_positions, orb_meta, is_active_statuses):
        """Draw floor tiles and orbs"""

        for r in range(self._rows):
            for c in range(self._cols):
                pos = (
                    (c * self._cell_width) + self._grid_offset,
                    (r * self._cell_height) + self._grid_offset,
                )
                self._window_surface.blit(self.graphics["floor_img"], pos)

                for i in range(len(is_active_statuses)):
                    if is_active_statuses[i]:
                        res_r, res_c = orb_positions[i]
                        if (r, c) == (res_r, res_c):
                            self._draw_orb(orb_meta[i], pos)

    def _draw_orb(self, orb_meta: OrbMeta, pos: tuple[int, int]):
        """Draw orb at pixel position `pos` (top-left)"""

        if orb_meta.KIND == DirectKind.NEGATIVE:
            self._window_surface.blit(self.graphics["negative_orb_img"], pos)
        else:
            if orb_meta.KIND == SynergyKind.TIER and orb_meta.TIER is not None:
                self._draw_tier_orb(orb_meta.TIER, pos)

    def _draw_tier_orb(self, tier: int, pos: tuple[int, int]):
        base_img = self.graphics["positive_orb_img"]
        # create (or fetch cached) combined surface with number
        tier_surf = self._make_tier_surface(tier, base_img)
        self._window_surface.blit(tier_surf, pos)

    def _make_tier_surface(
        self, tier: int, base_img: "pygame.Surface"
    ) -> "pygame.Surface":
        """
        Create a copy of base_img with the tier number drawn centered on it.
        Cache by (tier, base_size).
        """

        # Get cached image if it exists
        if tier in self._tier_text_cache:
            return self._tier_text_cache[tier]

        # render main text
        text_surf = self._render_text(str(tier), self._small_text, self._dark_grey_clr)

        # create target surface (copy of base)
        surf = base_img.copy()
        # get centered rect
        text_rect = text_surf.get_rect(
            center=(surf.get_width() // 2, (surf.get_height() // 2) + 2)
        )

        # blit the main text
        surf.blit(text_surf, text_rect)

        # cache and return
        self._tier_text_cache[tier] = surf
        return surf

    def _draw_droid(self, pos: tuple[int, int]):
        """Draw droid at a specific pixel position"""

        self._window_surface.blit(self.graphics["droid_img"], pos)

    def _draw_hud(self, hud_data: dict[str, int | float]):
        """Draw HUD / score with background rectangle for multiple data"""

        # --- Hud element --- #
        hud_img = self.graphics["hud_img"]
        hud_rect = hud_img.get_rect(
            topleft=(
                self._grid_offset + (self._cols - 5) * self._cell_width // 2,
                (self._cell_height * self._rows) + self._grid_offset * 2,
            )
        )
        self._window_surface.blit(hud_img, hud_rect)

        # --- Energy and moves bar --- #
        self._draw_energy_bar(hud_data["energy"], hud_rect)
        self._draw_steps_bar(hud_data["steps left"], hud_rect)

        # --- Current tier chain --- #
        # TODO(phase 8b): drawn by kind instead of picked by label, since it is scenario specific
        chained_tiers = hud_data["current tier chain"]
        if chained_tiers > -1:
            self._draw_hud_stat(
                chained_tiers, hud_rect.x + 33, hud_rect.y + (hud_rect.height - 68)
            )

    def _draw_energy_bar(self, current_energy: float, hud_rect: pygame.Rect):
        """
        Draw the energy bar inside hud_rect.
        Bar fills relative to the energy bar's maximum.
        """

        bar_height = 10
        bar_width = hud_rect.width - 116
        # --- Position bar at top of HUD --- #
        bar_x = hud_rect.x + 55
        bar_y = hud_rect.y + 23

        # --- Calculate fill proportion --- #
        fill_ratio = current_energy / self._bars_max["energy"]
        fill_width = int(fill_ratio * bar_width)
        fill_width = max(0, fill_width)  # clamp to [0, fill_width]

        # --- Draw filled portion --- #
        pygame.draw.rect(
            self._window_surface,
            self._yellow_clr,
            (bar_x, bar_y, fill_width, bar_height),
        )

        # --- Draw border --- #
        status_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self._window_surface, self._grey_clr, status_rect, 2)

        self._draw_hud_stat(round(current_energy, 2), hud_rect.x + 120, hud_rect.y + 45)

    def _draw_steps_bar(self, remaining_steps: float, hud_rect: pygame.Rect):
        """
        Draw the steps-left bar inside hud_rect.
        Bar fills relative to the steps-left bar's maximum.
        """

        ratio = remaining_steps / self._bars_max["steps left"]

        bar_height = hud_rect.height - 73
        bar_width = 10
        # --- Position bar at right side of HUD --- #
        bar_x = hud_rect.x + (hud_rect.width - 47)
        bar_y = hud_rect.y + (hud_rect.height - bar_height) - 40

        current_height = int(bar_height * ratio)

        # --- Draw filled portion --- #
        pygame.draw.rect(
            self._window_surface,
            self._green_clr,
            (bar_x, bar_y + (bar_height - current_height), bar_width, current_height),
        )

        # --- Draw border --- #
        status_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self._window_surface, self._grey_clr, status_rect, 2)

    def _draw_hud_stat(self, stat: float, x, y):
        """
        Draw a numeric stat centered at position (x,y) in the HUD.

        Parameters:
            stat: Numeric value to display (formatted to 2 decimals)
            x: Left coordinate of the stat's bounding box
            y: Top coordinate of the stat's bounding box
        """

        tier_surf = self._render_text(str(stat), self._large_text, self._purple_clr)

        # Place it in the hud
        tier_rect = pygame.Rect(x, y, 64, 52)

        # Center it inside a rect
        rect = tier_surf.get_rect(center=tier_rect.center)
        self._window_surface.blit(tier_surf, rect)

    def _render_text(
        self, text: str, size: int, color: tuple[int, int, int]
    ) -> pygame.Surface:
        """Renders text in the game font at `size`, loading that size the first time it is used."""

        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(get_syn_grid_path(self._FONT), size)

        return self._fonts[size].render(text, True, color)

    def _dim_frame(self) -> None:
        """Dims what is drawn, so a screen can be written over the last frame."""

        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        overlay.fill((*self._dark_grey_clr, 220))
        self._window_surface.blit(overlay, (0, 0))

    def _draw_lines(self, lines: list[pygame.Surface], top: int, centred: bool) -> None:
        """
        Draws rendered text lines stacked downwards from `top`.

        Each line is centred in the window, or left-aligned at the grid's margin.
        """

        y = top
        for line in lines:
            if centred:
                rect = line.get_rect(midtop=(self.window_size[0] // 2, y))
            else:
                rect = line.get_rect(topleft=(self._grid_offset * 2, y))
            self._window_surface.blit(line, rect)
            y += line.get_height()

    def _update(self):
        """Refreshes the display and limits FPS"""

        pygame.display.update()
        self._clock.tick(self._step_fps)
