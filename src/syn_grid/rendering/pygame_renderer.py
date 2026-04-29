from syn_grid.config.models import RendererConf, AssetsConf
from syn_grid.core.orbs.orb_meta import (
    OrbMeta,
    OrbCategory,
    DirectType,
    SynergyType,
)
from syn_grid.utils.paths_util import get_package_path
from syn_grid.gymnasium.action_space import DroidAction

import pygame
import json
import sys


class PygameRenderer:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, renderer_conf: RendererConf, fps: int) -> None:
        """
        Initializes the Pygame renderer.

        Sets:
        - Size of grid world
        - One logic step fps, not game.
        - Game colors
        - Game font
        - Graphic elements
        """

        self._conf = renderer_conf

        pygame.init()  # Initialize pygame
        pygame.display.init()  # initialize the display module
        self.clock = pygame.time.Clock()  # Game clock
        self._step_fps = fps

        # Default font
        self.tier_font = pygame.font.Font(
            get_package_path("assets", "fonts", "Minecraft.ttf"), 20
        )
        self.hud_font = pygame.font.Font(
            get_package_path("assets", "fonts", "Minecraft.ttf"), 30
        )

        self._init_colors()
        self._init_vars()
        self._load_graphics()

        # Define game window size (width, height)
        self.window_size = (
            self._window_width,
            (self._cell_height * self._conf.grid_rows)
            + (self._grid_offset * 3)
            + self._hud_height,
        )

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("SYNGrid")

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
    ) -> None | str:
        """
        Draws the game window and all its content, updates and limits the fps.
        """

        self.window_surface.fill(self._background_clr)

        col = droid_pos[1] * self._cell_width
        row = droid_pos[0] * self._cell_height

        self.window_surface.fill(self._background_clr)

        # Draw the graphics with pygame. blit() draws things in order, so we need to stack elements
        # in the order we want them to be drawn
        self._draw_floor_and_orbs(orb_positions, orb_meta, is_active_statuses)
        self._draw_droid((col + self._grid_offset, row + self._grid_offset))
        self._draw_hud(hud_data)

        self._update()

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init ===#

    def _init_colors(self) -> None:
        """Sets all the colors the game uses"""

        self._background_clr = (26, 26, 26)
        self._hud_text_clr = (165, 102, 192)

    def _init_vars(self):
        # Sizes of each cell in the grid
        self._cell_width = 64
        self._cell_height = 64

        # Used for created tier surfaces in draw_orbs()
        self._tier_text_cache: dict[int, pygame.Surface] = {}

        self._grid_offset = self._cell_width // 4
        self._window_width = (
            self._cell_width * self._conf.grid_cols
        ) + self._grid_offset * 2
        self._hud_height = self._cell_height * 4
        self._hud_width = self._cell_width * 5
        self._padding = 10

    def _load_graphics(self) -> None:
        """Load graphics via the config file"""

        self.graphics = {}
        for field_name, relative_path in self._conf.img_assets.model_dump().items():
            full_path = get_package_path(relative_path)
            self.graphics[field_name] = pygame.image.load(full_path)

    # === API ===#

    def _draw_floor_and_orbs(self, orb_positions, orb_meta, is_active_statuses):
        """Draw floor tiles and orbs"""

        for r in range(self._conf.grid_rows):
            for c in range(self._conf.grid_cols):
                pos = (
                    (c * self._cell_width) + self._grid_offset,
                    (r * self._cell_height) + self._grid_offset,
                )
                self.window_surface.blit(self.graphics['floor_img'], pos)

                for i in range(len(is_active_statuses)):
                    if is_active_statuses[i]:
                        res_r, res_c = orb_positions[i]
                        if (r, c) == (res_r, res_c):
                            self._draw_orb(orb_meta[i], pos)

    def _draw_orb(self, orb_meta: OrbMeta, pos: tuple[int, int]):
        """Draw orb at pixel position `pos` (top-left)"""

        if orb_meta.CATEGORY == OrbCategory.DIRECT:
            if orb_meta.TYPE == DirectType.NEGATIVE:
                self.window_surface.blit(self.graphics['negative_orb_img'], pos)
        else:
            if orb_meta.TYPE == SynergyType.TIER and orb_meta.TIER is not None:
                self._draw_tier_orb(orb_meta.TIER, pos)

    def _draw_tier_orb(self, tier: int, pos: tuple[int, int]):
        base_img = self.graphics['positive_orb_img']
        # create (or fetch cached) combined surface with number
        tier_surf = self._make_tier_surface(tier, base_img)
        self.window_surface.blit(tier_surf, pos)

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
        text_surf = self.tier_font.render(str(tier), True, self._background_clr)

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

        self.window_surface.blit(self.graphics['droid_img'], pos)

    def _draw_hud(self, hud_data: dict[str, int | float]):
        """Draw HUD / score with background rectangle for multiple data"""

        # --- Hud element --- #
        hud_img = self.graphics['hud_img']
        hud_rect = hud_img.get_rect(
            topleft=(
                self._grid_offset,
                (self._cell_height * self._conf.grid_rows) + self._grid_offset * 2,
            )
        )
        self.window_surface.blit(hud_img, hud_rect)

        # --- Energy and moves bar --- #
        self._draw_life_bar(hud_data["score"], hud_rect)
        self._draw_moves_bar(hud_data["moves"], hud_rect)

        # --- Current tier chain --- #

        chained_tiers = hud_data["current tier chain"]
        if chained_tiers > -1:
            self._draw_hud_stat(
                chained_tiers, hud_rect.x + 33, hud_rect.y + (hud_rect.height - 68)
            )

    def _draw_life_bar(self, current_score: int | float, hud_rect: pygame.Rect):
        """
        Draw a dynamic life bar inside hud_rect.
        Bar fills relative to highest score reached so far (max_seen_score).
        """

        # --- Update max_seen_score dynamically --- #
        if not hasattr(self, "_max_seen_score"):
            self._max_seen_score = max(current_score, 1)  # init, avoid div by zero
        self._max_seen_score = max(self._max_seen_score, current_score)

        bar_height = 10
        bar_width = hud_rect.width - 116
        # --- Position bar at top of HUD --- #
        bar_x = hud_rect.x + 55
        bar_y = hud_rect.y + 23

        # --- Calculate fill proportion --- #
        fill_ratio = current_score / self._max_seen_score
        fill_width = int(fill_ratio * bar_width)
        fill_width = max(0, fill_width)  # clamp to [0, fill_width]

        # --- Draw filled portion --- #
        pygame.draw.rect(
            self.window_surface, (255, 255, 65), (bar_x, bar_y, fill_width, bar_height)
        )

        # --- Draw border --- #
        status_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.window_surface, (75, 75, 75), status_rect, 2)

        self._draw_hud_stat(current_score, hud_rect.x + 120, hud_rect.y + 45)

    def _draw_moves_bar(self, remaining_moves: int | float, hud_rect: pygame.Rect):
        """
        Draw a dynamic life bar inside hud_rect.
        Bar fills relative to highest score reached so far (max_seen_score).
        """

        # --- Catch initial episode moves --- #
        if not hasattr(self, "_initial_moves"):
            self._initial_moves = remaining_moves
        ratio = remaining_moves / self._initial_moves

        bar_height = hud_rect.height - 73
        bar_width = 10
        # --- Position bar at right side of HUD --- #
        bar_x = hud_rect.x + (hud_rect.width - 47)
        bar_y = hud_rect.y + (hud_rect.height - bar_height) - 40

        current_height = int(bar_height * ratio)

        # --- Draw filled portion --- #
        pygame.draw.rect(
            self.window_surface,
            (58, 216, 48),
            (bar_x, bar_y + (bar_height - current_height), bar_width, current_height),
        )

        # --- Draw border --- #
        status_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.window_surface, (75, 75, 75), status_rect, 2)

    def _draw_hud_stat(self, stat: int | float, x, y):
        '''
        Draw a numeric stat centered at position (x,y) in the HUD.

        Parameters:
            stat: Numeric value to display (formatted to 2 decimals)
            x: Left coordinate of the stat's bounding box
            y: Top coordinate of the stat's bounding box
        '''

        tier_surf = self.hud_font.render(f"{stat:.2f}", True, self._hud_text_clr)

        # Place it in the hud
        tier_rect = pygame.Rect(x, y, 64, 52)

        # Center it inside a rect
        rect = tier_surf.get_rect(center=tier_rect.center)
        self.window_surface.blit(tier_surf, rect)

    def _update(self):
        """Refreshes the display and limits FPS"""

        pygame.display.update()
        self.clock.tick(self._step_fps)

    def get_user_action(self) -> DroidAction | None:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_LEFT:
                    action = DroidAction(0)
                if event.key == pygame.K_DOWN:
                    action = DroidAction(1)
                if event.key == pygame.K_RIGHT:
                    action = DroidAction(2)
                if event.key == pygame.K_UP:
                    action = DroidAction(3)

        return action
