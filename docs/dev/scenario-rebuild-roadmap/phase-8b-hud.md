# Phase 8b: The modular HUD

**Produces:** a HUD stacked from drawn pieces and drawn by element kind, so a scenario can list any number of
elements in the same style. It replaces the single `hud.png` and the matching by label.

**Files:** `rendering/hud_view.py`, the HUD theme file and the HUD pieces in `rendering/assets/`.

**Settle first** (see `open-questions.md`, phase 8b): what the renderer is handed about the HUD; where a bar sits;
what the window does when the HUD and the grid differ in height.

1. The pieces, drawn by the owner: a top cap, a row piece that tiles with matching connector edges, and a bottom cap,
   all of one width.
2. The theme file describes each piece: its picture and the anchors inside it, measured from the piece's own corner.
3. The HUD view stacks the cap, one row per element in `scenario.hud`, and the bottom cap.
4. Each element is drawn by its kind, so no element is picked by label. The energy number under the energy bar goes,
   and the score is drawn from its own element. The theme file replaces the hand-tuned numbers in the draw helpers.

**Done when:** a scenario that lists one more element gets one more row with no change to the art, the code or the
theme file, and the observation and the reward are unchanged.
