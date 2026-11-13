# Observation

- **Observation:** what the agent is given each step.
- **Perception:** the class that turns the world into an observation, and the config value that names it, such as
  `vector_fog_of_war`.
- **Visibility:** how much of the world a perception shows: all of it, or a window around the droid. It is a way of
  talking about perceptions and not a setting.
- **Encoding:** the form the observation takes: a flat vector, a composite of named parts, a grid of cells, or
  pixels. It is also not a setting.
- **Markovian:** a perception that shows the whole world, so one observation is enough to act on.
- **Fog of war:** a perception that shows only the orbs within one cell of the droid.
- **POMDP:** a task where one observation is not enough, so the agent needs memory. A perception with a limited view
  makes one.
- **Slot:** the place for one orb in an observation.
- **Global feature:** one value in the observation that is not about an orb, such as energy or steps left.
- **Orb feature:** one value in an orb's slot beyond its active flag and position, such as its identity. The scenario
  lists them, as it lists global features.
- **View radius:** how far from the droid the agent sees orbs. None is the full view; 1 and up is a limited view.
- **Format:** what a perception class is once the view radius is the scenario's: vector, composite, grid or pixel.
- **HUD:** what the renderer shows beside the grid. A scenario lists its elements.
- **Metric:** one named number a scenario reports at the end of an episode.
