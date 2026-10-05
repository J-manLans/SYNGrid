# v1.0.0 TODO

Ordered by dependency, not severity — each item unblocks the ones below it.
Grouped into phases by urgency, numbered continuously across all of them.

Unrelated in-flight work: see `rppo-regression.md`.
Scenario architecture: see `scenario-refactor.md`.

## Make the numbers mean something

1. ~~Seed the load path.~~ **Not a defect — closed, premise was wrong.**
   `ArtifactManager.load_model` does not pass a seed, but it does not need to.
   SB3 persists `seed` in the checkpoint, `load()` restores it via
   `__dict__.update(data)`, and `OnPolicyAlgorithm._setup_model` then calls
   `set_random_seed(self.seed)`, which re-seeds torch, numpy, the action space
   and the VecEnv. Verified on 2.9.0 and against the declared 2.7.1 floor: two
   independent unseeded loads of one checkpoint produce identical trajectories
   under a fixed action tape. Passing `seed=` to `load()` is *not* a no-op — it
   overrides the checkpoint's seed and changes the rollout, which would decouple
   eval from the seed the checkpoint was trained with. One real residual, much
   smaller than the original claim: the seed comes from the checkpoint, so
   `agent_conf.seed` is silently ignored for `eval` and `continue_training`.
   Worth a warning, not a behaviour change.

2. Fix the reward engine. Both defects closed.
   - `OrbFactory._normalize_counts` sorted the count list before distributing the
     remainder, destroying the index-to-orb-type mapping. Weights inverted
     whenever the first enabled type outweighed the second. **Fixed in
     `3c166eb`**; the code moved to `WeightedPopulation` in the scenario
     refactor and is covered by `tests/scenario/rules/test_population.py`,
     including the weight ordering that used to break it. The 4
     `xfail(strict=True)` cases that documented it went with the fix.
   - `DigestionEngine._max_reward_bonus` is never cleared on a chain break, so a
     later completion pays out a bonus accumulated on an abandoned chain.
     **Ruled off the spatial path** — it lives on `_step_wise_scoring` and the
     spatial config uses max-tier scoring. Still open on the step-wise path.

3. ~~Pin every reward-shaping ambiguity to one documented value.~~ **Timeout
   penalty: done.** It is now `DroidConf.timeout_penalty` (default `-1.0`),
   replacing a hardcoded `-1` inside `episode_termination.py`. The old hardcode
   shadowed a parameter, and that parameter was itself being fed
   `chain_break_penalty` — so the timeout and chain-break rewards were pinned at
   a 1:1 ratio and the interesting ratios were unreachable by config. They are
   independent fields now; the spatial scenario runs at
   `chain_break_penalty: -0.01` / `timeout_penalty: -1.0` (100:1). A related
   floor-division bug on the delay-mode path (`timeout_penalty // 2`, which
   collapsed any small negative penalty to `-1.0`) is fixed too. Termination now
   lives in `scenario/rules/termination.py` as a rule object rather than a free
   function reading `world._conf`. **Remaining open:** the value is now tunable
   but not yet *established* — the sweep in `rppo-regression.md` has to be redone
   one variable at a time, because every run recorded before this change is a
   confounded comparison.

4. ~~Decide `truncated` semantics.~~ **Done: timeouts terminate.** Running out of
   steps is a punishment and sets `terminated`, so no value function is
   bootstrapped at the horizon. Pinned by a test so it is not later "fixed" by
   accident. The open cost of that choice — no bootstrapping at the horizon — is
   now a documented trade-off rather than an oversight. Continuous mode is a
   separate, still-open question and deliberately untouched.

## Make it run

5. `ConfigDict(extra="forbid")` on the config models. Unknown keys are currently
   swallowed by pydantic's default, so a typo in any shipped YAML trains on a
   different config than the file describes. The scenario refactor removed the
   worst instance — a typo'd scenario flag used to change which world you got,
   and there is nothing left to typo — but a typo'd *tunable* is still silent.

6. Add `configs.yaml` / `test_configs.yaml` / `scenarios/*.yaml` to
   `[tool.setuptools.package-data]`. Only `assets/**/*` is declared, so a
   non-editable install cannot find its default config and dies at startup. The
   scenario refactor added a directory of configs, so this got worse.

7. Add a `--config` flag so a run can point at a config outside the installed
   package. With 17 scenario configs in the package this matters more than it did.

## Make it honest

8. Every remaining NOTE/TODO → resolved, or pinned with a tracked issue. A
   documented frozen quirk is fine; an undocumented one is a defect, because a
   reader cannot tell which behaviour produced the published numbers. Two
   preserved oddities are now written down and pinned by tests rather than left
   to be rediscovered — the spatial scenario's over-wide observation, and
   `curriculum_training` being inert in continuous mode. See
   `scenario-refactor.md`.

9. CI step that loads every shipped config against `FullConf`. This catches dead
   config keys, invalid perception names and typos permanently. `FullConf` now
   also resolves the selected scenario during validation, so such a step would
   catch an unknown scenario name and an illegal parameter set too. The check
   itself is one line per file over `src/syn_grid/config/scenarios/`.

10. CHANGELOG marking 1.0.0 as the frozen reference for the thesis figures.
    Note that the scenario refactor is a breaking config change: the archived
    `reproduction_package/` YAMLs no longer load, which is accepted (the thesis
    is graded and the package is slated for removal).

11. `reproduction_package/` — slated for removal before publication. Its four
    inbound references are listed in git history; the scenario refactor has
    already broken its configs, which resolves the concern in practice. Two
    `scripts/` mentions (`replay_probe.py`'s worked example,
    `replay_sweep.sh`'s comment) still point at a directory that no longer
    produces runnable configs.

12. Consider folding the scenario name into `_set_models_base_id`. Run identity
    is still derived from perception, grid size and enabled orb types. The
    scenario is now a first-class concept and belongs in the run id — but this
    orphans every existing checkpoint path, so it needs a migration, not a
    drive-by change. Deliberately left out of the refactor.

## Adoption, if there is room

13. Widen the Python ceiling. `requires-python = ">=3.10,<3.11"` blocks 3.11+
    for no reason the code depends on.

14. Content-hash run ids, so hand-written `id_tag`s stop being load-bearing.
    Related to item 12: a run's identity is currently assembled from a
    perception name, a grid size and which orb types are enabled, which is
    exactly the kind of reconstruction the scenario concept exists to stop.

---

## Claude

Ja. Jag skulle koka ner Claudes config-fynd till det här, separerat från de senare `environment.py`/`GridWorld`-problemen:

### Config-modellerna

* **`max_active_orbs` ligger på fel nivå.** Det är inte universellt; tier-chain använder i praktiken `max_tier` som poolstorlek. Det bör därför inte vara ett krav i en generell orb-poolmodell.
* **`TierOrbPoolConf` deklarerar `negative` igen**, trots att det redan finns i basmodellen. Dubblett → ta bort.
* **`EnabledOrbsConf` är död kod** och kan tas bort.
* **`tiers` vs `max_tier`:** båda är avsiktliga — `tiers` är observations/kurrikulum-relaterat och kan vara större än `max_tier` — men det saknas en validator om relationen ska garanteras, t.ex. `tiers >= max_tier`.
* **`max_steps` finns på två ställen**, `ObservationHandlerConf` och `PerceptionConf`. Kontrollera om båda verkligen behövs eller om det är en kvarleva.
* **Renderer-konfigurationen har tappat sitt hem.** `RendererConf` låg i den gamla borttagna modellen och renderaren behöver nu en ny plats för den konfigurationen.
* **`WeightedPopulation` använder fortfarande gamla `OrbFactoryConf`**, alltså kvarleva från den gamla configstrukturen.
* **`check_env.py` använder fortfarande `FullConf`**, ytterligare en gammal modellreferens.

### Registry / typningen

* **`cast(TierScenarioConf, ...)` finns fortfarande i builders.** Det beror på att `SCENARIO_MODELS` och `SCENARIO_BUILDERS` är två separata register. Ett gemensamt register som kopplar `scenario name → rätt configmodell + builder` skulle kunna ge korrekt typning utan cast.
* `neg_orb(...)` längst ner är gammal/död kod.
* `_require_scoring` refererar fortfarande till gamla `tier_orb_conf.scoring`.
* Tre builders är fortfarande placeholders (`...`).

### Den större arkitekturfrågan

Det viktigaste fyndet är egentligen inte ett enskilt configfel:

> **Config är fortfarande på väg att läcka in i runtime.**

`GridWorld`, `Environment` osv. letar efter gamla `world_conf`, `droid_conf`, `obs_conf` etc. medan den nya riktningen är att config används för att **bygga ett färdigt `Scenario`**, och runtime sedan arbetar med scenario-/regelobjekten.

Det är också därför frågan vi nyss diskuterade om `Scenario` som runtime-root är ganska central. Om vi går den vägen blir mycket av Claudes "hur får Environment tag på confen?" ett icke-problem: **den ska inte ha confen.**

Och jag skulle nog **inte fixa alla punkterna mekaniskt ännu**. Några av dem kommer sannolikt försvinna när du gör Scenario/World-beslutet och städar den nya configmodellen.

---

## Chatty
Absolut. Jag skulle sammanfatta riktningen så här:

**Scenario-first runtime**

* `app.py` bygger `Scenario` från config.
* `Scenario` blir runtime-root för den konkreta simulationen.
* Scenario-specifika komponenter skapas/komponeras där: `Droid`, `GridWorld`, population, spawning, digestion, observation, termination osv.
* `Env` tar bara emot `Scenario` och fungerar främst som Gymnasium-adapter.
* Undvik att `Env` tar emot `Scenario` bara för att sedan skicka det vidare till `GridWorld`.
* Scenario-specifik mekanik kapslas bakom `Scenario`, så calling-kod behöver inte casta för att komma åt tier-specifika detaljer.
* Behåll `DigestionEngine` tills vidare; om den visar sig bara delegera till scenariot kan den tas bort.

Den centrala principen:

> **Scenario definierar och komponerar den konkreta simulationen. Env exponerar den som en Gymnasium-miljö.**
