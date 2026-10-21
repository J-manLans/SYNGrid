# Ideas About the Codebase

Things noticed while rebuilding that aren't problems to solve now, just
worth revisiting later. Low bar for entry: half-formed is fine.

---

## Entry template

**Idea:**
**Where it came from:**
**Why it might matter:**

---

## A locked scenario keeps only its axis in the YAML (2026-10-10)

**Idea:** Once a scenario is declared done, its values move out of the YAML and are fixed in its builder, next to the
scoring mode the builder already chooses. The YAML and the config models keep only what sits on the scenario's
difficulty axis (grid size for spatial, chain length and grid for scaling, the delay for delay) and the perception.
Penalties, rewards, `max_steps` and `max_energy` are no longer fields of that scenario's config, so a YAML cannot set
them. The continuous sandbox scenario stays fully tunable, for every parameter that matters and
can't break something.

Varying a locked scenario means changing its axis value, not registering a new scenario per variation, so there is
no `spatial_6x6` beside `spatial_5x5`.

**Where it came from:** Reshaping the config models on 2026-10-10. Dense got defaults on its own classes so its YAML
could leave values out. Those dense classes were removed on 2026-10-11, since every scenario now sets its own
penalties.

**Why it might matter:** The scenario name is a run's identity. While every value is in the YAML, two runs with the
same name can describe different worlds, which is the reason scoring mode is already kept out of config. A short YAML
also shows at a glance what is meant to be varied.

Decided by the owner on 2026-10-11: the fixed values go in the builder, not in the config models and not in the
YAML. A fixed value is then no longer a field, so it cannot be overridden and a forgotten YAML line cannot load a
value nobody chose. The config models describe only what can be tuned.

Turned down: fixed values as defaults on the scenario's config models. A default can still be overridden, which
brings back two runs with the same name describing different worlds.

Not yet. A scenario is locked after it has been rebuilt and has reproduced its legacy results, because until then
the values aren't settled and comparing with legacy is easier with every value in view. Until then every scenario
states all of its values in its YAML.

**Settled on 2026-10-12:** how the config models are shaped for a locked scenario. The models stay as the full
description of a world, a locked scenario gets a small model holding only what a run may choose, and its builder makes
the full config from its fixed values. This is the third of the three sketches that came up on 2026-10-11. The
decision, the two sketches that were turned down and what is still open are in `design-decisions.md` under "A locked
scenario loads a small model, and its builder makes the full config".

The optional negative orb is still to settle. Its `reward` and `cool_down` are tuning values, so a locked YAML would
only say whether the orb is present.

---

## A GUI on top of the registry (2026-10-12)

**Idea:** The owner's picture of the GUI the project is heading for.

- The first step is picking a scenario: one of the registered ones, or the continuous sandbox.
- For a locked scenario only its axis can be edited. Its other values are shown read-only.
- The values are on one side and a preview of the scenario on the other. The preview follows the form, so a larger
  grid for spatial shows as a larger grid.
- In the sandbox every value that is meant to be tunable can be edited, and orbs are added from a list of every
  existing kind.
- Version 1 stops there. A user who finds a good setup in the sandbox turns it into a scenario by hand, in the code.
- Version 2 has a "create scenario" step: choose continuous or goal, define the goal, and save the result as a
  scenario.

**Where it came from:** Talking through whether the locked-scenario models would also serve a GUI, on 2026-10-12.
Nothing in the repo is written for a GUI; the only mention is a TODO in `config/config_manager.py`.

**Why it might matter:** It is a second consumer of the config models, and it asks a few things of them.

What already fits:

- `SCENARIOS` is the list of scenarios, and an entry's config class is what a user may set for that scenario. A form
  can be made from it without a special case for locked scenarios or the sandbox.
- `ConfigManager.load_config` ends in `model_class(**raw)`, so a dict from a form takes the same path as a dict from
  a YAML.
- A `Scenario` holds nothing that changes and `build_world()` returns a new world on each call, so a preview can
  build a world from the form on every edit.
- Each orb kind is a `*_orb_conf` block on a world model. A sandbox world model that declares every kind as optional
  is the list of existing orbs, and adding an orb is switching a block on. It allows one block per kind.

What it asks for:

- The full config a locked builder makes has to be reachable without building a `Scenario`, for the read-only panel.
  The snapshot needs the same.
- An error has to name a field the user can see. See the open points in the design decision.
- A range on a locked axis would serve the form, the validation and the preview at once.
- The small models have no defaults, so a form needs starting values from somewhere. The YAML could keep that role.
- The explanations of the fields are in class docstrings. A form would want them per field.
- The renderer's settings have no place in the new models yet, and a preview needs them.

Version 2 is the only part that goes against the current design. A scenario is a builder in code, a member of
`ScenarioName` and an entry in the registry, and a scenario made in a GUI would have to be data that a general builder
composes from. Nothing is shaped for that now. The blocks (termination as a protocol with a file per strategy,
`OrbBundle`) are what such a composer would choose from.

Turning a sandbox setup into a goal scenario is not a copy of its config: the sandbox is continuous and has its own
world model, so the goal and the termination are decided by hand.

The owner wants to learn pygame_gui and would use it for this GUI. It is built for interactive widgets and themes
them through a JSON file, which is what a form of fields, dropdowns and buttons needs. The game HUD stays in plain
pygame, see "The renderer takes plain values, and the HUD is the scenario's list" in `design-decisions.md`.

---

## A bundle that carries everything its orb kind brings (2026-10-12)

**Idea:** An `OrbBundle` holds a population and a way to make a digester. A kind brings more than that to a scenario:
the metrics that read its digester, a tag suffix, HUD elements and observation features. The bundle could carry some
of these, and a family's helper would collect them from its bundles as it now sums their counts.

**Where it came from:** Adding the negative orb to the tier-chain family. Its `if` block in `_tier_chain_scenario`
adds the bundle, extends the tag and adds the `negatives_eaten` metric: three lines that together mean "add this
brick".

**Why it might matter:**

- The contributions are of three sorts. The population and the digester must exist together: an orb with no digester
  makes the engine raise. The count and the metrics follow from the kind being present, with nothing to decide. What
  the HUD shows and what the agent observes are the scenario's choice, and a kind should not make it.
- Only the first sort is in the bundle, and the count is taken from the population. That is enough while one family
  uses the negative orb, because its `if` block is one readable place.
- The moment to act is a second family using the same kind. Each would then repeat the lines, and they could drift.
  The first candidate is a `metrics` field on the bundle.
- A metric's reader is the same in every family: take the kind's digester from the engine and read a counter. So the
  readers would move out of a family's builders and live with the kind. Followed to the end, that is one module per
  kind, which groups by kind where `scenario/blocks/populations/` and `scenario/blocks/recipes/` group by role.
- The HUD's form and the observation's features are open, in phases 9 and 7. Putting either in the bundle decides
  them early.

---

## A perception is two things: what the droid sees and how it is encoded (2026-10-12)

**Idea:** Split the perception into two settings. One is the visibility: the whole world, or a window around the
droid. The other is the encoding: a flat vector, a composite of named parts, a grid of cells, or pixels. A scenario
fixes the visibility when its test depends on it, and the encoding stays a choice.

**Where it came from:** Deciding which perceptions a locked scenario offers, on 2026-10-12. The fog of war in
`vector_fog_of_war.py` is one filter, which keeps the orbs within one cell of the droid, and nothing ties it to the
vector encoding. The class names already read as a pair: `vector`, `composite`, `composite_grid` and `grid_pixel` on
one side, and `markovian`, `fog_of_war` and `fully_pomdp` on the other.

**Why it might matter:**

- "Spatial offers only the perceptions with a limited view" then needs no list per scenario. With one enum, every new
  perception has to be added by hand to each scenario that allows it.
- The two halves may belong to different sides. The visibility follows from the task, which is the scenario's. The
  encoding follows from the policy and its feature extractor, which is the runner's. This bears on the shape of
  `obs_conf`, which is open.
- The composite perceptions return `Dict` spaces that SB3 flattens into one vector, so they give the model nothing a
  vector does not until a custom extractor is written. Their own notes say so.
- A limited pixel observation can be made in two ways. A crop of the window is the same size on every grid and says
  nothing about where the droid is. The full grid with everything outside the window blacked out keeps the position.
  `vector_fog_of_war` gives the droid its position beside the orbs it sees, so the second is the closer match.
- `composite_grid_markovian` and a pixel observation are not the same thing. The first is a rows x cols x channels
  array where a cell holds facts (an active flag, the orb kind, the tier). The second is a picture the model has to
  read. Both suit a CNN; the first is small and exact, the second large and slower.

---

## A held chain that keeps the droid alive (2026-10-16)

**Idea:** In threshold scoring a held chain is worth something not yet paid. A droid that runs out of energy while
holding one could have the chain cashed into energy and go on, so the chain doubles as a reserve.

If it is ever built, it happens in the world and not at termination: after the droid acts and eats, a droid at 0
asks its digestion to cash out, and termination then reads energy as usual. Seen as a revive at termination it needs
a check for steps left (`terminated and steps_left > 0`), which reads poorly and lets a payout undo an ending. Seen as
a cash-out before termination looks, no clock check is needed: if the clock ran out too, the episode ends on the clock
and nothing is owed.

**Where it came from:** Deciding on 2026-10-16 that the ending's payout goes to the score and not to energy, so a
droid that ran out of energy stays at 0 on its last frame. See "Energy and score are two numbers" in
`design-decisions.md`.

**Why it might matter:** Probably as a reason not to build it. The droid can already cash out without the rule, by
eating a tier orb out of order while it still has moves left: a threshold break pays `chain_break_penalty` plus what
the chain had gathered. Learning when to do that is the long-horizon trade-off the benchmark wants to test, and an
automatic cash-out at 0 would hand the agent the answer. That deliberate cash-out only matters where a break does not
end the episode, in a family where chains repeat. In a goal tier chain a break ends it, and running out of energy pays
the chain without the penalty, so breaking on purpose never helps there.

If it is built anyway, it is a new path for a reward into the droid and changes what threshold teaches, so it would be
a scenario of its own. It fits a continuous family better than a goal one: a cash-out would have to end the chain
without counting as a break or a completion, and a goal episode would then go on with a chain that cannot finish.

**Where it earns its place (2026-10-16):** a highly stochastic goal scenario, with a continuous threshold tier chain
running beside the goal and the cash-out at 0 turned on. Orb rewards are uncertain, so the droid has to take chances,
and one payout can take it to 0 on a single step. A deliberate break cannot save it then, because the droid cannot
know beforehand which orb is the bad one. The only defence is to already carry an undigested chain, so the cash-out is
no longer an answer handed to the agent but what makes insurance a strategy. The agent learns how much chain is worth
carrying against the step costs of building it, and that a large held chain makes risky orbs affordable.

The goal ends the episode, and a break or a completion of the side chain ends nothing, which is a termination of the
scenario's own. Open when it is built:

- Whether a cash-out pays the whole chain, with anything above `max_energy` lost or left undigested, or only what
  keeps the droid alive. Paying only what is needed makes the chain a battery and changes what carrying it is worth.
- What a cash-out counts as. It is not a break or a completion, so it likely needs a counter of its own.
- What "stochastic" means: a reward drawn per orb, or kinds whose values are hidden. That decides whether the agent
  can learn to read the risk at all.
