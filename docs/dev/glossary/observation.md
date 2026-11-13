# Observation

- **Observation:** what the agent is given each step.
- **Perception:** the class that turns the world into an observation, and the config value that names it, such as
  `vector_fog_of_war`. See `two-senses.md`.
- **Visibility:** how much of the world the agent is shown: all of it under a floodlight, or the orbs within a
  lantern. The scenario sets it, as the view radius. See `two-senses.md`.
- **Encoding:** the form the observation takes: a flat vector, a composite of named parts, a grid of cells, or
  pixels. A run picks it, as the format. See `two-senses.md`.
- **Markovian:** a task in which one observation is enough to act on. It is a property of a scenario and its view
  together, not of the view: delay under a floodlight is not Markovian. Also in the perception name
  `vector_markovian`, the vector under a floodlight.
- **POMDP:** a task where one observation is not enough, so the agent needs memory. A lantern makes one, and so does
  anything else the observation hides, such as time.
- **Fog of war:** an older word for the lantern, kept in the perception name `vector_fog_of_war`, a lantern of
  radius 1.
- **Slot:** the place for one orb in an observation.
- **Global feature:** one value in the observation that is not about an orb, such as energy or steps left.
- **Orb feature:** one value in an orb's slot beyond its active flag and position, such as its identity. The scenario
  lists them, as it lists global features.
- **View radius:** how far from the droid the agent sees orbs. None is the floodlight; 1 and up is a lantern.
- **Floodlight:** the full view: no view radius, so the agent sees every orb on the field.
- **Lantern:** the limited view: a view radius of 1 and up, so the agent sees only the orbs within it. What the droid
  leaves goes dark again.
- **Format:** what a perception class is once the view radius is the scenario's: vector, composite, grid or pixel.
- **HUD:** what the renderer shows beside the grid. A scenario lists its elements.
- **Metric:** one named number a scenario reports at the end of an episode.
