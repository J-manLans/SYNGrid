#!/usr/bin/env python3
"""
Compare two replay_probe recordings and report where they first diverge.

Takes the two <label>.npz files that replay_probe.py wrote, and for each of
obs, rewards, terminated, truncated and ep_lens either confirms they are
identical or names the first index where they differ. Given that index, the
reward on both sides and the observation slots that moved usually identify the
mechanism without reading any source diff.

Usage:
    replay_diff.py <a.npz> <b.npz>

    Both paths are required and are the .npz files, not the .json summaries.
    Only numpy is needed; this never imports syn_grid, so it runs against any
    pair of recordings regardless of which commits produced them.

Examples:
    # did this refactor change anything?
    python scripts/replay_diff.py output/replay_diff/pre-sc.npz \
                                 output/replay_diff/post-sc.npz

    # narrow a sweep down to the commit that caused it
    python scripts/replay_diff.py output/replay_diff/2048379-sc.npz \
                                 output/replay_diff/5e93079-sc.npz

Reading the output:
    "identical over N entries"   that array is unchanged
    "first differs at N"         divergence point; A= and B= are the two values
    "SHAPE DIFFERS"              the observation space itself changed, which
                                  invalidates any reward comparison
    "reward sum A=... B=..."     total drift across the whole run

For a behaviour-preserving refactor every line should read "identical". Any
line that does not is something the refactor moved.

Note the comparison is exact, with no epsilon. A refactor that reassociates
floating-point arithmetic will show differences of order 1e-16; read the reported
magnitudes before treating them as behavioural.
"""

from __future__ import annotations

import sys

import numpy as np


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    a_name, b_name = sys.argv[1], sys.argv[2]
    a, b = load(a_name), load(b_name)

    print(f"A = {a_name}")
    print(f"B = {b_name}\n")

    for key in ("rewards", "terminated", "truncated", "ep_lens"):
        if key not in a or key not in b:
            continue
        x, y = a[key], b[key]
        n = min(x.size, y.size)
        if x.shape != y.shape:
            print(f"  {key:<11} SHAPE DIFFERS: A{x.shape} vs B{y.shape}")
        if n and not np.array_equal(x[:n], y[:n]):
            idx = int(np.argmax(x[:n] != y[:n]))
            print(f"  {key:<11} first differs at {idx}: A={x[idx]} B={y[idx]}")
        else:
            print(f"  {key:<11} identical over {n} entries")

    if a["obs"].shape != b["obs"].shape:
        print(f"  obs         SHAPE DIFFERS: A{a['obs'].shape} vs B{b['obs'].shape}")
        return 0

    n = min(a["obs"].shape[0], b["obs"].shape[0])
    diff = np.abs(a["obs"][:n] - b["obs"][:n])
    per_step = diff.max(axis=1)
    if n and per_step.max() > 0:
        idx = int(np.argmax(per_step > 0))
        slots = np.where(diff[idx] > 1e-6)[0]
        print(
            f"\n  obs         first differs at step {idx} "
            f"(max |delta| = {per_step[idx]:.4f})"
        )
        print(f"              slots {slots.tolist()[:12]}")
        print(f"              A: {a['obs'][idx][slots].tolist()[:12]}")
        print(f"              B: {b['obs'][idx][slots].tolist()[:12]}")
        print(f"  obs         total differing steps: {int((per_step > 0).sum())}/{n}")
    else:
        print(f"  obs         identical over {n} steps")

    if a["rewards"].size and b["rewards"].size:
        print(
            f"\n  reward sum  A={a['rewards'].sum():.3f}  B={b['rewards'].sum():.3f}  "
            f"delta={b['rewards'].sum() - a['rewards'].sum():+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
