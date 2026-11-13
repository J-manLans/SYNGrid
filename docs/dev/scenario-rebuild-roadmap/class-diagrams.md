# Before phase 7: The class diagrams

**Produces:** Mermaid class diagrams of the rebuilt strain, grouped as the owner thinks of the code, to redraw in
draw.io.

The script draws the whole strain and one diagram per group, with a legend of the arrows. What is left:

1. The owner reads the diagrams and says what is missing or wrong for the redraw in draw.io.
2. Two links the script cannot see, to draw or to leave out: the runners make their environment through
   `env_factory.make`, a function, so the runners diagram has no environment; and the CSV logger wraps a `gym.Env`,
   so it has no arrows.

**Done when:** the diagrams show what the owner wants to redraw in draw.io.
