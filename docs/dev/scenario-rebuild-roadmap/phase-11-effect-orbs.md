# Phase 11: Effect orbs

**Produces:** what effect orbs need from the engine, and a first set of effect orbs, each tried in the sandbox.

An effect orb changes something other than itself, such as what another kind's orb pays. It comes after the sandbox
(phase 10), since what an effect orb is is found by trying them together, and the sandbox is where orbs are tried
together.

**Settle first** (see `open-questions.md`, phase 11): how effect orbs get their magic from the engine, whether an
effect orb kind despawns like the others, and which effect orbs make the first set.

1. **What the engine gives them:** a tick, a stage before routing, a way to change what another kind's orb pays, and a
   way to reach the world, as the engine question settles them. A digester with none of them stays as it is.
2. **The first set:** each effect orb a kind, a digester, a population and a bundle, as "A new orb kind" in
   `code-map.md` lists, and each with a metric that shows what it did.
3. **Each tried in the sandbox:** played, then trained on by a small agent.

**Done when:** each effect orb of the first set has been played in the sandbox, and a small agent shows learning on
a sandbox setup that holds it.
