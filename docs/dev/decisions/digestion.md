# Decisions: Digestion

How an eaten orb becomes a reward: the engine, the digesters, the scoring modes and the recipes that make them.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## Digestion lives in the droid, with a folder of digesters (2026-10-12)

**Decision:** Digestion is the droid's own organ, in `core/droid/digestion/`:

- `digester.py`, the `OrbDigester` protocol: `KIND`, `reset`, `digest` returning the reward as a float, and
  `undigested`. `KIND` is the routing key, an `OrbKind` from `core/orbs/orb_meta.py`.
- `engine.py`, a router with no scoring rules. It hands an eaten orb to the digester for the orb's kind, and lets a
  reader fetch a digester by class with `get`, which matches with `isinstance`, so asking for `TierDigester` finds
  whichever scoring mode is registered.
- `digesters/`, one digester per orb kind, a folder for a kind with several files.

The engine raises on every sign of wrong wiring: a `ValueError` when two digesters own a kind, a `KeyError` when an
eaten orb's kind has no digester or `get` finds no digester of the class.

The droid creates the engine from the digesters it is handed. The scenario holds a way to make each digester, one
per `OrbBundle` (`make_digester`), and never a digester, since a digester holds episode state and one `Scenario` is
shared by every env in the process. `build_world` makes them, so every world gets its own.

**Why:** The engine stays a plain mechanism any family can use. The scenario decides which digesters exist and how
each is configured, but does not own their code. The droid never imports a concrete digester or learns its scenario.

**Rejected:**

- `core/digestion/`, or digesters in the scenario package: digesting is something the droid does.
- `build_world` or the scenario creating the engine: nothing outside the droid needs to.
- One scenario method that returns every digester: adding an optional orb kind would then mean editing it as well as
  adding the bundle.
- `get` returning `None`, an unrouted orb paying 0, or a second digester replacing the first: each can only come from
  a scenario wired wrong, and each would go unnoticed. A metric reading 0 because its digester is missing looks like
  one reading 0 because nothing was eaten.

## Digesters do not talk to each other (2026-10-12)

**Decision:** A digester scores its own kind and knows nothing of the others. `digest` returns a float. There are no
events, no result object, no tally in the engine and no hook by which a digester hears of another kind's orb.

A tier chain is broken only by a tier orb eaten out of order. A negative orb costs its own reward and leaves the chain
alone.

A reader that needs what a digester knows asks the droid, `world.droid.get_digester(TierDigester)`, which forwards to
the engine's `get`, and reads its attributes. The engine stays private to the droid. A class that more than one
registered digester belongs to, such as `DirectDigester` with both direct kinds registered, raises instead of picking
one: the reader asks for the exact class.

**Why:** Each digester can be read and tried alone, and each kind has one meaning: a tier orb is about order, a
negative orb costs what its reward says. Adding a negative orb to a scenario changes one thing, so its effect on a
learning curve can be read.

**Rejected:**

- Events returned from `digest` and counted by the engine (the reference): nothing reacted to an event, every reader
  wanted a count since reset, which a digester already keeps.
- A `notice` hook on digesters: every pair becomes a possible conversation, the result depends on which is told
  first, and a step's reward can no longer be explained from one place.
- Any non-tier orb breaking a chain (legacy): a direct orb changes the score and nothing else.
- A public `digestion_engine` attribute on the droid: every reader would walk three levels into it.

**Revisit if:** an orb has to change what another kind pays (see "How effect orbs get their magic from the engine"
in `open-questions.md`), something has to react to what happened and not only to what it paid, or a scenario wants an
existing kind to break chains, in which case the scenario's entry hands the tier digester the set of breaking kinds.

## The tier digester is a template with one subclass per scoring mode (2026-10-15)

**Decision:** `TierDigester` is a base class that owns the chain. For each tier orb it decides what happened, asks the
subclass what that pays, counts it and moves the chain:

- **Progress:** the next tier, not the last. A tier 1 with no chain running starts one.
- **Completion:** the next tier, and the last. The chain is over.
- **Break:** any other tier. It leaves no chain, whatever its tier; a new chain starts with the next tier 1.

A mode writes `_score_progress`, `_score_break` and `_score_completion`, each handed the orb and called before the
chain moves, and `undigested`, what the chain has gathered and not paid. The base holds what readers take:
`chain_tier`, `chain_worth`, and the counters `chain_progresses`, `chains_broken` and `chains_completed`, as plain
public attributes. Each tier orb counts once, under the one thing it did, so the counters add up to the tier orbs
eaten. The base takes `max_tier`; each mode takes what it pays with. The scenario's entry picks the subclass.

**Why:** Adding a mode is adding a class, not editing one that works. A scenario loads only its mode, and each mode
can be tried alone.

**Rejected:**

- One digester with a `ScoringMode` enum and a branch per mode (the reference): the scenario's entry already decides the
  mode, so the enum states it twice, and it is a conditional in the wrong component.
- The orb carrying its own scoring (the old engine): orbs are the same in every mode.
- `max_tier` read off the orb (legacy): `TierOrb` does not store it.
- The last orb's reward held in the base (legacy's `_pterminal_reward`): no mode reads it.

## The scoring modes pay the same orbs at different times (2026-10-12)

**Decision:** A tier orb is worth the same in every mode; a mode decides only when it is paid. A break always costs
`chain_break_penalty`.

| | Progress | Break | Completion | Owed at the end |
|---|---|---|---|---|
| Step-wise | the orb's reward | the penalty | the orb's reward | 0 |
| Threshold | 0 | the penalty, plus what the chain had gathered | what the chain had gathered, plus the orb's reward | what the chain has gathered |
| Max-tier | 0 | the penalty | what the chain had gathered, plus the orb's reward | 0 |

A chain completed without a break pays the same total in all three, and step-wise and threshold pay the same total for
any sequence once threshold has paid what it holds.

The break is one rule everywhere: a tier orb that is not the next one, and it leaves no chain. The orb that breaks is
never paid or held, a tier 1 included, so a new chain starts with the next tier 1 and every completed chain pays its
full worth. Nothing is paid for a tier 1 eaten out of order, so tier 1 orbs cannot be farmed where orbs respawn.

**Why:** The modes form one family, so comparing two isolates when the reward arrives. The agent gets one rule: eat
the tier one above the chain.

**Rejected:**

- Max-tier paying a fixed prize with tier orbs worth nothing (the reference; legacy's fixed 10): sparse against dense
  at one chain length is what the scaling scenarios compare, and with the same worth only the timing differs. The
  cost is accepted: the prize grows with the chain unless the base reward is set per scenario.
- Max-tier paying only the last orb's reward (legacy): the lower tiers would never be paid.
- A completion bonus in step-wise (legacy): a completed chain would be worth nearly double max-tier's.
- A separate `tier_consumption_penalty` (legacy): a second number for the same idea.
- A tier 1 never costing a penalty (legacy step-wise): it can be farmed.
- A threshold break paying only the last tier (legacy), or no penalty: a special case, and a free cash-out.
- A tier 1 that breaks a chain starting a new one (legacy, and this decision until 2026-11-08): the metrics could not
  tell what a break left behind, and it took three special cases: the restart in the base, step-wise paying the
  tier 1's reward on top of the penalty, and threshold paying a repeated tier 1 the penalty alone so it could not be
  farmed. In threshold the restarted tier 1 was also settled at the end, one base reward more than the break paid.

**Revisit if:** long chains train worse because the prize grows (try the discount and exploration first, then the base
reward, then reward normalisation).

## A new scoring mode is a new scenario (2026-10-11)

**Decision:** Scoring mode is not config. A different mode is a new registered scenario, an entry of its own in the
family's catalogue. The continuous sandbox is the exception.

**Why:** The scenario name is the run's identity: run ids, result folders and comparisons hang on it. A mode in config
would let two runs of one name pay differently.

**Rejected:** the mode in the YAML with a lookup from its name to a class.

**Revisit if:** one experiment has to sweep modes. The lookup would sit at the edge, beside `SCENARIOS`, and not touch
the digesters.

## A tier bundle makes its digester from a frozen recipe (2026-10-14)

**Decision:** A tier bundle's `make_digester` is a `TierDigesterRecipe`, a frozen dataclass of `scoring_mode` (the tier
digester class), `max_tier` and `chain_break_penalty`. Calling it returns a new digester. A scenario's entry states its
mode by naming the class as `scoring`, which reaches `_tier_bundle`. A digester whose constructor takes no values, as
`NegDigester`, is its own `make_digester`; there is no recipe base class.

`scenario/blocks/recipes/` holds every recipe: a frozen object a scenario holds that makes something with episode
state when a world is built. The other is `WorldRecipe`, a scenario's `build_world`, shared by every family. It holds
the world config, the bundles and `max_active_orbs`, reads only what the base `WorldConf` has, and makes the
digesters, a `SynergyDroid` (with the grid, `max_energy`, `step_penalty` and `boundary_penalty` as plain values) and
the `GridWorld`. The full world config is reachable as `scenario.build_world.world_conf`.

Because both compare by value, two scenarios built from one config compare equal. That lets `==` answer "is this the
scenario I think it is?" when a scenario is locked, when a saved agent is evaluated and when a GUI saves one. It
describes the scenario, not its behaviour; that is for a replay with one seed.

A digester sets `self.KIND: Final[OrbKind]` in `__init__`. A `Final` with a value in the class body is a class
variable to Pyright and does not satisfy the protocol's instance variable.

**Why:** It compares by value, it is the one place a mode is chosen, and the choice is a class, not a name to look up.

**Rejected:**

- A `functools.partial`: two partials never compare equal.
- A module-level function per mode: it has nowhere to get `max_tier` and the penalty.
- The recipe in the family's `builders.py`, or in `core/`: it holds nothing of the family, and core is handed digesters.
- Recipes handed down a layer at a time (the world given a droid maker, the droid given digester recipes): nothing is
  gained, and it puts makers in core's constructors.
- Inheriting from the digester protocol: Pyright refuses to assign a `Final` the parent declares, and a subclass that
  forgets a method inherits an empty one.
