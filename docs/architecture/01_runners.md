# 01 — Runners

```text
runners/
├── agent_runners/
│   ├── sb3/
│   │   ├── artifact_manager.py
│   │   ├── base_sb3_runner.py
│   │   ├── execution_strategy.py
│   │   ├── frame_stack_ppo.py
│   │   ├── lstm_ppo.py
│   │   ├── policy_resolver.py
│   │   └── stateless_ppo.py
│   ├── utils/
│   │   └── extractors.py
│   ├── agent_registry.py
│   ├── base_agent_runner.py
│   └── runner_bundle.py
└── human_runner/
    └── human_runner.py
```

```text
BaseAgentRunner (ABC)
├── BaseSB3Runner[T: BaseAlgorithm]
│   ├── StatelessPPO   (T = PPO)
│   ├── FrameStackPPO  (T = PPO)
│   └── LstmPPO        (T = RecurrentPPO)
└── HumanRunner
```

A runner takes a `RunnerBundle` (the built `Scenario` plus `RunnerConf`) and
drives environments with it. `agent_registry.build_runner` picks the class:
`HumanRunner` when `human_control` is set, otherwise the entry for
`common_conf.alg` in the `RUNNER` table (`PPO`, `FSPPO`, `RPPO`). `app.dispatch`
then calls `train()` or `eval()` on it.

`BaseAgentRunner` owns what every run needs regardless of algorithm: the run
id, the output directories, and the per-environment wrappers (episode
statistics, CSV logging, video). `BaseSB3Runner` holds the whole train and eval
flow once. It builds a `DummyVecEnv` of `SYNGridEnv`s wrapped in
`VecNormalize`, and delegates model and normalisation-stat persistence to
`ArtifactManager`. The three concrete runners only supply an algorithm,
hyperparameters, a policy string from `policy_resolver`, and an
`ExecutionStrategy` for eval-time action selection; `FrameStackPPO` also adds
frame stacking to the environment.

`HumanRunner` skips all of that. It builds one bare `SYNGridEnv` in `human`
render mode and loops on keyboard input until the episode ends.
