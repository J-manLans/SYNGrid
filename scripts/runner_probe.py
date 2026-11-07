#!/usr/bin/env python3
"""
Fingerprint each algorithm's runner: what a short seeded training leaves, and what the runner does
with it.

For every algorithm in `RUNNER`, two lines are printed:

- The pieces: the runner is built as a run builds it, its env trains 512 steps on two envs with
  CSV logging on, and its eval loop runs three episodes on the same env.
- The round trip: `train()` runs as a run does, for 512 steps, and saves the model and its
  normalization stats. A second runner in eval mode finds them by their steps, loads them and
  runs `eval()` for three episodes.

Each line has `weights`, a digest of the policy's parameters (after training, or as loaded), and
`eval_actions`, how many eval steps there were with a digest of every action. The pieces also
print `csv`: how many CSV files and rows training wrote, and a digest of the rows without the
wall-clock column `t`.

Not reached: the run controls, which wait for a key press in a window, and video recording.

Run it before and after a change to the runners, or to anything they train on, and compare the
lines: the replay probe never reaches the runners. Each line repeats exactly on one machine and
one set of package versions. After a change of Python, torch or numpy the weights and actions can
move without a bug; read how far before calling it behavioural.

The scenario is built from its run YAML, as a run builds it. The probe works in its own folder,
so what the runners write under `output/` goes there and not into the project. Imports sit
inside functions, so PYTHONPATH can point this at another checkout.

Usage:
    runner_probe.py [--scenario NAME] [--work-dir DIR]

Example:
    PYTHONPATH=src python scripts/runner_probe.py
"""

import argparse
import csv
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

TRAIN_STEPS = 512


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="goal_single_tier_chain_delay")
    ap.add_argument(
        "--work-dir",
        help="the folder the runners write in; a temporary one if not given",
    )
    args = ap.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import syn_grid
    from syn_grid.config.config_manager import ConfigManager
    from syn_grid.config.models.global_models import ScenarioName
    from syn_grid.config.models.runner_models import Algorithm, RunnerConf
    from syn_grid.gymnasium.utils.env_factory import register_env
    from syn_grid.runners.agent_runners.runner_registry import build_runner
    from syn_grid.scenario.registry import SCENARIOS, build_scenario

    register_env()
    manager = ConfigManager()
    name = ScenarioName(args.scenario)
    conf = SCENARIOS[name].full_conf(
        manager.load_raw(manager.scenario_config_path(name))
    )
    scenario = build_scenario(name, conf)
    base = manager.load_config("runner_config.yaml", RunnerConf)
    # The runners build every output path from the working directory
    work_dir = Path(args.work_dir or tempfile.mkdtemp(prefix="runner_probe_")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    print(f"syn_grid from {syn_grid.__file__}")
    print(f"scenario {name.value}, tag {scenario.scenario_tag}, working in {work_dir}")

    for alg in Algorithm:
        runner_conf = base.model_copy(
            update={
                "common_conf": base.common_conf.model_copy(
                    update={"alg": alg, "training": True, "tag": "probe"}
                ),
                "train_conf": base.train_conf.model_copy(
                    update={
                        "n_envs": 2,
                        "render_mode": None,
                        "csv_output": True,
                        "tensorboard_output": False,
                        "model_output": False,
                    }
                ),
                "eval_conf": base.eval_conf.model_copy(
                    update={
                        "num_eval_episodes": 3,
                        "render_mode": None,
                        "csv_output": False,
                    }
                ),
            }
        )
        runner = build_runner(scenario, runner_conf)
        runner._log_dir = work_dir / "pieces" / alg.value

        env = runner._build_env(None, "train")
        model = runner._resolve_training_model(env)
        model.learn(TRAIN_STEPS)

        weights = hashlib.sha256()
        for key, value in sorted(model.policy.state_dict().items()):
            weights.update(key.encode())
            weights.update(value.detach().cpu().numpy().tobytes())

        actions: list[np.ndarray] = []
        step = env.step

        def recording_step(action, step=step, actions=actions):
            actions.append(np.asarray(action).copy())
            return step(action)

        env.step = recording_step
        runner._eval_model(env, model)

        print(
            f"{alg.value:6s} policy={type(model.policy).__name__:28s} "
            f"obs={env.observation_space.shape} weights={weights.hexdigest()[:16]} "
            f"eval_actions={len(actions)}:{digest(np.concatenate(actions).tobytes())}"
        )

        files = []
        for path in sorted((work_dir / "pieces" / alg.value).rglob("*.csv")):
            with path.open() as f:
                rows = [
                    {k: v for k, v in row.items() if k != "t"}
                    for row in csv.DictReader(f)
                ]
            files.append((path.name.split("_")[-1], rows))

        print(
            f"       csv files={len(files)} rows={sum(len(rows) for _, rows in files)} "
            f"digest={digest(repr(files).encode())}"
        )

        _round_trip(alg, runner_conf, scenario, build_runner)

    return 0


def _round_trip(alg, runner_conf, scenario, build_runner) -> None:
    """Train and save through `train()`, then load and evaluate through `eval()`."""

    train_conf = runner_conf.model_copy(
        update={
            "common_conf": runner_conf.common_conf.model_copy(
                update={"tag": "probe_round_trip"}
            ),
            "train_conf": runner_conf.train_conf.model_copy(
                update={
                    "csv_output": False,
                    "model_output": True,
                    "timesteps": TRAIN_STEPS,
                    "iterations": 1,
                }
            ),
        }
    )
    build_runner(scenario, train_conf).train()

    eval_conf = train_conf.model_copy(
        update={
            "common_conf": train_conf.common_conf.model_copy(
                update={"training": False, "agent_steps": str(TRAIN_STEPS)}
            )
        }
    )
    runner = build_runner(scenario, eval_conf)

    # Watch what eval() loads and does, around the runner's own methods
    loaded = []
    load_model = runner._artifact_manager.load_model

    def recording_load(*args, **kwargs):
        loaded.append(load_model(*args, **kwargs))
        return loaded[-1]

    runner._artifact_manager.load_model = recording_load

    actions: list[np.ndarray] = []
    build_env = runner._build_env

    def recording_build_env(*args, **kwargs):
        env = build_env(*args, **kwargs)
        step = env.step

        def recording_step(action, step=step):
            actions.append(np.asarray(action).copy())
            return step(action)

        env.step = recording_step
        return env

    runner._build_env = recording_build_env
    runner.eval()

    weights = hashlib.sha256()
    for key, value in sorted(loaded[0].policy.state_dict().items()):
        weights.update(key.encode())
        weights.update(value.detach().cpu().numpy().tobytes())

    print(
        f"       round trip: saved and loaded at {loaded[0].num_timesteps} steps "
        f"weights={weights.hexdigest()[:16]} "
        f"eval_actions={len(actions)}:{digest(np.concatenate(actions).tobytes())}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
