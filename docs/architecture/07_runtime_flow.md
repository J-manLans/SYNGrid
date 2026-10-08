# 07 — Runtime flow

Startup, for `python3 -m syn_grid` with the current YAMLs
(`goal_tier_chain_spatial`, PPO, training):

```text
app.main()
  ├─ load_experiment_configs()      3 YAMLs -> GlobalConf, TierScenarioConf, RunnerConf
  ├─ build_scenario()               -> Scenario                         (once)
  ├─ build_runner()                 -> StatelessPPO
  └─ dispatch() -> runner.train()
        ├─ _build_env()             16 × SYNGridEnv, each calls scenario.build_world()
        ├─ create_model()           SB3 PPO on VecNormalize(DummyVecEnv)
        └─ model.learn()            SB3 loop -> SYNGridEnv.step
```

One environment step:

```text
SYNGridEnv.step(action)
  ├─▶ GridWorld.perform_droid_action
  │     ├─▶ SynergyDroid.perform_action            move, step/boundary penalty
  │     ├─▶ SynergyDroid.consume_orb               if the droid is on an orb
  │     │      └─▶ DigestionEngine.digest
  │     │             └─▶ TierOrbDigester.digest   reward + chain events
  │     ├─▶ SpawningRules.on_orb_consumed
  │     └─▶ SpawningRules.after_step
  │   ◀── step penalty + orb reward
  ├─  steps_left -= 1
  ├─▶ TierChainTermination.evaluate(world, steps_left, reward)
  │   ◀── EpisodeOutcome(terminated, truncated=False, reward)
  ├─▶ ObservationHandler.get_observation
  │      └─▶ VectorFogOfWar.get_observation
  └── return obs, reward, terminated, truncated, info
```

Config is read once, the scenario is built once, and every environment builds
its own world from that one scenario. From then on SB3 drives the environments;
the repo's own code runs inside `SYNGridEnv.reset` and `SYNGridEnv.step`.

Within a step the world resolves movement and consumption first and returns a
raw reward. Termination then decides whether the episode ends and may replace
that reward (the timeout penalty, for example). The observation is encoded
last, from the world as it stands after the step.

With the current config an episode ends when the chain is broken, when the
chain is completed, when the 100 steps run out, or when the droid's score
reaches zero. `truncated` is always `False`.
