# Phase 8a: The renderer rework

**Produces:** a renderer split by reason to change, and a fog overlay that shows what the agent sees.

**Files:** `rendering/pygame_renderer.py`, `rendering/world_view.py`, `rendering/hud_view.py`, `rendering/assets/`,
`pyproject.toml`.

1. Move the renderer's images and font into `rendering/assets/` with `git mv`: `sprites/`, `tiles/` and `fonts/`.
   `sounds/` stays in `assets/`, since the runner uses it. Add the folder to the package data. A pure move, in its
   own commit.
2. Split `pygame_renderer.py` into a coordinator, `world_view.py` and `hud_view.py`, as pure moves, with no base
   classes. The coordinator keeps the window, the summary and pause screens, the dimming and text helpers, the title
   and `close()`. The world view takes the floor, the orbs and the droid. The HUD view takes the HUD.
3. The world view takes the view radius from phase 7 and dims every cell farther from the droid, with the same
   distance the perception uses. No radius, no overlay.

**Done when:** a human run looks as before apart from the fog, and the observation and the reward are unchanged.
