# A guide entity that walks the chain (2026-11-09)

**Returns when:** a scenario with a second body on the grid is planned: a guide, a rival or a companion.

**Idea:** A second body on the grid, a ghost, that shows the agent the tier order and does nothing else. This
scenario withholds the identity, so the agent cannot see the tiers and the ghost is the only place the order can come
from. Elsewhere tier order is visible, by "An orb's identity is the radix scalar, and tier order is meant to be seen".

The rules as talked through:

- The orbs land in new positions every episode, and their tiers are hidden from the observation. Everything else is
  visible, the ghost included, on every step.
- The ghost walks the shortest path from orb to orb in tier order. It only passes over them and never eats one, so
  it touches no orb, no digester and no score.
- It walks every tier but one, stops on the last tier it walks, and leaves one orb untouched. Which tier is left alone
  is the ghost's one dial: tier 1 or tier 4.
- With tier 1 left alone, the ghost walks 2, 3 and 4. The agent has to find the orb nobody visited and eat it first,
  then follow the ghost's route. An agent that only copies the ghost eats a tier 2 first, which breaks the chain and
  ends the episode, so imitation earns nothing.
- With tier 4 left alone, the ghost walks 1, 2 and 3. Copying gets the agent through three tiers, and the test is
  that it must then finish the chain on the orb the ghost never touched. It is the easier setting.
- The ghost's behaviour is locked, so it is the same in every episode and the dial is the only thing that changes
  what the scenario asks. It always takes the shortest path with a fixed tie rule, never eats, and does the same
  thing when it stops. Whether it then stays or leaves is decided once and not tunable.
- The ghost's trail can be switched on as an option. With it off, the agent has to remember what the ghost did, which
  suits a recurrent policy. With it on, the history is in the observation and a stateless policy can read it. One
  scenario, two questions: can the agent learn the order, and can it remember it. Results must say which setting
  they came from.

Leaving tier 1 untouched is the harder setting: the first eat is the test. Leaving tier 4 untouched is easier,
because following the ghost already gets the agent through three tiers.

**Where it came from:** The owner, by voice on 2026-11-09, after asking what a rival or ghost droid could be besides
a score fight (the obvious design: contact decided by who has the higher score). The idea moved from enemy to
teacher once the tiers were hidden.

**Why it might matter:**

- It asks for something no scenario does today: learning a hidden order by watching it, not by trial and error with
  the end reward as the only teacher. The untouched first orb is what stops it being plain imitation.
- The trail is a vector, since the observation is one. A cell the ghost never visited reads 0. A visited cell reads a
  value that falls with age, down to a floor above 0, so the oldest cell is still readable when the ghost leaves.
  How fast it falls is a dial. Listing only the visited cells, as position and age, keeps the size fixed on a large
  grid. Order numbers in the cells would do the same with less shading.
- Companions, rivals and guides would share the same machinery: a second entity with its own movement rule. The owner
  would file this under an entity family, but one scenario does not show the shape yet. A second or third entity
  scenario is the moment to build it.

**Open points:**

- The ghost needs a fixed rule when a shortest path has several routes, such as rows before columns, or the walk is
  not reproducible and the replay probe cannot hold it.
- Where the ghost starts: the centre, where the droid starts, or somewhere else.
- Whether the ghost stays on the grid after it stops or leaves. It is decided once, and a leaving ghost adds a memory
  test on top.
- How long the ghost walks, and how many steps the agent has after it stops. Locked with the rest of its behaviour.
- Whether the ghost is a real entity in the world or a scripted route the scenario advances, with the observation
  reading a pointer on it. For the model they are the same: a few numbers in the observation, and drawing happens
  only for eval and human viewing, so training speed does not depend on it. The owner leans towards a real ghost,
  because the renderer then draws it like anything else and the human view needs no special case. The scripted route
  is lighter, since the ghost touches nothing, but it would be drawn by a separate path.
- The identity the agent sees for an orb numbers tier orbs in tier order, so it would give the order away. The ghost
  scenario withholds the identity for itself alone, which leaves the identity as it is everywhere
  else.
- What the ghost needs from the engine: a place in the world's step, a place in the observation, and a way to
  leave. A watcher may fit the first, see "Field effects through a watcher" in
  `effect-orbs-magic-from-engine.md`.
- The word "trace" in the glossary is the replay probe's recording. The ghost's trail needs another name so the two
  do not mix.
- With four orbs a blind guess hits tier 1 one time in four. It is a weak shortcut, since a wrong pick ends the
  episode, but a small agent may still settle for it. Training one early and watching what it does would show.
- What the agent learns is a rule, the unvisited orb first and the rest in the ghost's order. Whether a larger chain
  makes that harder or easier is open.
