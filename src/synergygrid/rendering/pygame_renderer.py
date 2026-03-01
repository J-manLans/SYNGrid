import pygame
import json
from os import path
import sys
import numpy as np
from synergygrid.core import ResourceMeta, ResourceCategory, DirectType, SynergyType


class PygameRenderer:
    RENDER_FPS = 60

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows: int = 5, grid_cols: int = 5, fps: int = 2) -> None:
        """
        Initializes the Pygame renderer.

        Sets:
        - Size of grid world
        - One logic step fps, not game.
        - Game colors
        - Game font
        - Graphic elements
        """

        pygame.init()  # Initialize pygame
        pygame.display.init()  # initialize the display module
        self.clock = pygame.time.Clock()  # Game clock

        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._step_fps = fps

        # Default font
        self.font = pygame.font.Font("assets/fonts/Minecraft.ttf", 20)
        self.font_height = self.font.get_height()

        self._init_colors()
        self._init_vars()
        self._load_graphics()
        self._set_frames_per_step()

        # Define game window size (width, height)
        self.window_size = (
            self._window_width,
            (self._cell_height * self._grid_rows)
            + (self._grid_offset * 3)
            + self._hud_height,
        )

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size)

    # ================= #
    #        API        #
    # ================= #

    def render(
        self,
        agent_pos: list[int],
        is_active_statuses: list[bool],
        resource_positions: list[list[np.int64]],
        resource_meta: list[ResourceMeta],
        hud_data: dict[str, int],
    ) -> None | str:
        """
        Draws the game window and all its content, updates and limits the fps.

        Also catches user events (clicking the X top right or hitting ESC) for closing the game.
        """

        # TODO: decide if we go all in on animation and either keep this method or fix the
        # animation render method so it looks slicker, Keep both as of now

        self.window_surface.fill(self._background_clr)

        col = agent_pos[1] * self._cell_width
        row = agent_pos[0] * self._cell_height

        self.window_surface.fill(self._background_clr)

        # Draw the graphics with pygame. blit() draws things in order, so we need to stack elements
        # in the order we want them to be drawn
        self._draw_floor_and_resources(
            resource_positions, resource_meta, is_active_statuses
        )
        self._draw_agent((col + self._grid_offset, row + self._grid_offset))
        self._draw_hud(hud_data)

        self._update(
            self._step_fps
        )  # TODO: also remove self._step_fps when approach is chosen
        return self._process_quit_events()

    def render_with_animation(
        self,
        agent_pos: list[int],
        is_active_statuses: list[bool],
        resource_positions: list[list[np.int64]],
        resource_meta: list[ResourceMeta],
        hud_data: dict[str, int],
    ) -> None:
        """
        Blocking render: animate agent over a fixed number of frames while everything
        else stays visually static.
        """

        # Track last cell and score
        if not hasattr(self, "_last_agent_cell"):
            self._last_agent_cell = list(agent_pos)
            self._last_agent_score = hud_data["score"]
        last = self._last_agent_cell

        # Animate agent if moving
        if agent_pos != last:
            for frame in range(self.frames_per_step + 1):
                # Sets the x and y coordinates for where the agent should be drawn, updating only
                # the axis the it is currently moving along
                progress = frame / self.frames_per_step
                distance_to_target = 1 - progress
                col = (distance_to_target * last[1]) + (progress * agent_pos[1])
                row = (distance_to_target * last[0]) + (progress * agent_pos[0])
                coordinates = (
                    int(col * self._cell_width),
                    int(row * self._cell_height),
                )

                self.window_surface.fill(self._background_clr)

                # Draw the graphics with pygame. blit() draws things in order, so we need to stack
                # elements in the order we want them to be drawn
                self._draw_floor_and_resources(
                    resource_positions,
                    resource_meta,
                    is_active_statuses,
                )
                self._draw_agent(coordinates)

                if not frame == self.frames_per_step:
                    # NOTE: if we render like this this should be the last score, so it doesn't
                    # change before the whole animation is over, just do like this now to not get
                    # any warnings
                    self._draw_hud(hud_data)
                else:
                    self._draw_hud(hud_data)

                self._update(self.RENDER_FPS)

            # snap logical cell
            self._last_agent_score = hud_data["score"]
            self._last_agent_cell = [agent_pos[0], agent_pos[1]]
        else:
            # no movement, single frame
            self.render(
                agent_pos,
                is_active_statuses,
                resource_positions,
                resource_meta,
                hud_data,
            )

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init ===#

    def _init_colors(self) -> None:
        """Sets all the colors the game uses"""

        self._background_clr = (26, 26, 26)
        self._hud_text_clr = (0, 210, 210)

    def _init_vars(self):
        # Sizes of each cell in the grid
        self._cell_width = 64
        self._cell_height = 64

        self._grid_offset = self._cell_width // 4
        self._window_width = (
            self._cell_width * self._grid_cols
        ) + self._grid_offset * 2
        self._hud_height = self._cell_height * 3
        self._hud_width = self._cell_width * 4
        self._padding = 10

    def _load_graphics(self) -> None:
        """Load graphics via JSON file"""

        # Used for created tier surfaces
        self._tier_text_cache: dict[int, pygame.Surface] = {}

        ROOT_DIR = path.abspath(path.join(path.dirname(__file__), "..", "..", ".."))
        json_file = path.join(ROOT_DIR, "assets/paths.json")

        with open(json_file, "r") as f:
            graphics_paths = json.load(f)

        self.graphics = {}
        for attr, rel_path in graphics_paths.items():
            full_path = path.join(ROOT_DIR, rel_path)
            self.graphics[attr] = pygame.image.load(full_path)

    def _set_frames_per_step(self):
        for f in range(self._step_fps, 0, -1):
            if self.RENDER_FPS % f == 0:
                self.frames_per_step = self.RENDER_FPS // f
                break

    # === API ===#

    def _draw_floor_and_resources(
        self, resource_positions, resource_meta, is_active_statuses
    ):
        """Draw floor tiles and resources"""

        for r in range(self._grid_rows):
            for c in range(self._grid_cols):
                pos = (
                    (c * self._cell_width) + self._grid_offset,
                    (r * self._cell_height) + self._grid_offset,
                )
                self.window_surface.blit(self.graphics["floor_img"], pos)

                for i in range(len(is_active_statuses)):
                    if is_active_statuses[i]:
                        res_r, res_c = resource_positions[i]
                        if (r, c) == (res_r, res_c):
                            self._draw_resource(resource_meta[i], pos)

    def _draw_resource(self, resource_meta: ResourceMeta, pos: tuple[int, int]):
        """Draw resource at pixel position `pos` (top-left)"""

        if resource_meta.category == ResourceCategory.DIRECT:
            if resource_meta.type == DirectType.POSITIVE:
                self._draw_tier_resource(resource_meta, pos)
            elif resource_meta.type == DirectType.NEGATIVE:
                self.window_surface.blit(self.graphics["negative_resource"], pos)
        else:
            if resource_meta.type == SynergyType.TIER:
                self._draw_tier_resource(resource_meta, pos)

    def _draw_tier_resource(self, resource_meta: ResourceMeta, pos: tuple[int, int]):
        tier = resource_meta.tier
        base_img = self.graphics["positive_resource"]
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
        text_surf = self.font.render(str(tier), True, self._background_clr)

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

    def _draw_agent(self, pos: tuple[int, int]):
        """Draw agent at a specific pixel position"""

        self.window_surface.blit(self.graphics["agent_img"], pos)

    def _draw_hud(self, hud_data: dict[str, int]):
        """Draw HUD / score with background rectangle for multiple data"""

        # --- HUD rectangle --- #
        hud_rect = pygame.Rect(
            self._window_width // 7.5,
            (self._cell_height * self._grid_rows) + self._grid_offset * 2,
            self._hud_width,
            self._hud_height,
        )

        # draw background rectangle
        pygame.draw.rect(self.window_surface, (75, 75, 75), hud_rect, 2, 10)

        self._draw_life_bar(hud_data["score"], hud_rect)

        # --- Texts --- #
        hud_texts = [
            f"Moves: {hud_data['moves']}",
            f"Current Tier Chain: {hud_data['current tier chain']}",
            f"Score: {hud_data['score']}",
        ]

        # Render all text surfaces first
        text_surfaces = [
            self.font.render(line, True, self._hud_text_clr) for line in hud_texts
        ]

        # Calculate total height for vertical centering
        line_spacing = self._padding
        total_height = sum(
            surf.get_height() for surf in text_surfaces
        ) + line_spacing * (len(text_surfaces) - 1)

        # Start y position so block is vertically centered
        y = hud_rect.y + (hud_rect.height - total_height) // 2

        # Blit each text, centered horizontally
        for surf in text_surfaces:
            rect = surf.get_rect()
            rect.centerx = hud_rect.centerx  # center horizontally
            rect.y = y
            self.window_surface.blit(surf, rect)
            y += surf.get_height() + line_spacing

    def _draw_life_bar(self, current_score, hud_rect):
        """
        Draw a dynamic life bar inside hud_rect.
        Bar fills relative to highest score reached so far (max_seen_score).
        """

        # --- Update max_seen_score dynamically ---
        if not hasattr(self, "_max_seen_score"):
            self._max_seen_score = max(current_score, 1)  # init, avoid div by zero
        self._max_seen_score = max(self._max_seen_score, current_score)

        bar_padding = 5  # padding inside HUD rect
        bar_height = 10  # height of the life bar
        bar_width = hud_rect.width - (4 * bar_padding)
        # --- Position bar at bottom of HUD ---
        bar_x = hud_rect.x + bar_padding * 2
        bar_y = hud_rect.y + hud_rect.height - bar_height - (bar_padding * 2)
        status_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)

        # --- Calculate fill proportion ---
        fill_ratio = current_score / self._max_seen_score
        fill_width = int(fill_ratio * bar_width)
        fill_width = max(0, fill_width)  # clamp to [0, fill_width]

        # --- Draw filled portion ---
        pygame.draw.rect(
            self.window_surface, (255, 255, 65), (bar_x, bar_y, fill_width, bar_height)
        )
        # --- Optional: draw border ---
        pygame.draw.rect(self.window_surface, (75, 75, 75), status_rect, 2)

    def _update(self, render_fps: int):
        """Refreshes the display and limits FPS"""

        pygame.display.update()
        self.clock.tick(render_fps)

    def _process_quit_events(self) -> None | str:
        """Handle quitting / ESC"""

        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    pass
                if event.key == pygame.K_LEFT:
                    action = "left"
                if event.key == pygame.K_DOWN:
                    action = "down"
                if event.key == pygame.K_RIGHT:
                    action = "right"
                if event.key == pygame.K_UP:
                    action = "up"

        return action
