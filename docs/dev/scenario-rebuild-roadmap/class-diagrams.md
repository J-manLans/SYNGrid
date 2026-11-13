# Before phase 7: The class diagrams

**Produces:** Mermaid class diagrams of the rebuilt strain, grouped as the owner thinks of the code, to redraw in
draw.io.

The script draws every class of the rebuilt strain into one Mermaid diagram, with protocol arrows. The diagrams are
a starting point the owner redraws in draw.io, which imports Mermaid. To fine-tune:

1. Private attributes: pyreverse hides them by default, and with them their "has a" arrows, such as
   `ArtifactManager --* SB3Runner`. `-f ALL` with `-k` (class names only) shows them: 62 relations against 57.
2. Diagrams grouped around how the owner thinks of the code, beside the whole one. The whole one is for spotting
   connections, not for reading: at around ninety classes it is too dense. Each diagram is a name and the folders or
   classes it covers, kept as a short list in the script. A narrow diagram loses its arrows to the rest, so it shows
   its neighbours outside it as empty boxes.
   1. **Everything.**
   2. **Orbs:** the base orb, its meta, the direct and tier orbs, and the timer.
   3. **Droid and digestion:** the droid, the engine, the digester protocol and the digesters, with the orbs outside.
      The droid has its digesters, which digest orbs: how it works when it runs.
   4. **Grid world:** the world owning the droid and the orbs, without their insides.
   5. **Scenario:** the scenario, the registry, the recipes, the populations, termination and the builders. How a
      world is put together: the recipe that makes the digesters and hands them to the droid is a dashed «create»
      arrow here, while diagram 3 keeps the droid owning them.
   6. **Observation:** the handler, the base perception, the perceptions and the observation rules they read.
   7. **Environment:** the Gymnasium environment, with the scenario, the world and the observation handler outside.
   8. **Runners:** the base runner, the SB3 runners, the artifact manager, the policy resolver, the human runner and
      the episode logger, with the environment outside.

   `rendering`, `plot` and `utils` show only in the whole one. The renderer gets a diagram of its own after its
   rework (8a), and the extractor joins the runners diagram once it is written (7b). Plotting reads the CSV the
   logger writes and calls nothing of it, a link a class diagram cannot show; whether the two get a diagram is for
   their rebuild (phase 9).
3. Arrows for what is handed in. pyreverse draws an attribute's arrow only where it sees the class built, as in
   `self.x = SomeClass()`. What a scenario injects arrives as a parameter, so links such as the droid's to its
   digesters are missing. The script adds them from the source, as it adds the protocol arrows: for each
   constructor, the parameters stored on `self`, with their type hints, lists and dicts unwrapped to what they hold.
   The arrow goes to the hinted type, usually a protocol, which is what the class knows. It covers only parameters
   with a hint and stored on `self` as they are.

**Done when:** the diagrams show what the owner wants to redraw in draw.io.
