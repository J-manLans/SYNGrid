#!/usr/bin/env python3
"""
Compare two probe.py recordings and report where they first diverge.

For rewards, terminated, truncated, ep_lens and obs, it says "identical" or names the first index
that differs, with both values; for obs, also the slots that moved. The comparison is exact, so read
the size of a difference before calling it behavioural: reordered float arithmetic shows as ~1e-16.
Needs only numpy.

Usage:
    diff.py A.npz B.npz
"""

import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    a, b = (dict(np.load(path)) for path in sys.argv[1:])
    print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n")

    for key in ("rewards", "terminated", "truncated", "ep_lens"):
        x, y = a[key], b[key]
        n = min(x.size, y.size)
        if x.shape != y.shape:
            print(f"  {key:<11} SHAPE DIFFERS: A{x.shape} vs B{y.shape}")
        if n and not np.array_equal(x[:n], y[:n]):
            i = int(np.argmax(x[:n] != y[:n]))
            print(f"  {key:<11} first differs at {i}: A={x[i]} B={y[i]}")
        else:
            print(f"  {key:<11} identical over {n} entries")

    if a["obs"].shape[1:] != b["obs"].shape[1:]:
        print(f"  obs         SHAPE DIFFERS: A{a['obs'].shape} vs B{b['obs'].shape}")
        return 0

    n = min(len(a["obs"]), len(b["obs"]))
    delta = np.abs(a["obs"][:n] - b["obs"][:n])
    per_step = delta.max(axis=1)
    if n and per_step.max() > 0:
        i = int(np.argmax(per_step > 0))
        slots = np.flatnonzero(delta[i])[:12]
        print(
            f"\n  obs         first differs at step {i} (max |delta| = {per_step[i]:.4g})"
        )
        print(f"              slots {slots.tolist()}")
        print(f"              A: {a['obs'][i][slots].tolist()}")
        print(f"              B: {b['obs'][i][slots].tolist()}")
        print(f"  obs         differing steps: {int((per_step > 0).sum())}/{n}")
    else:
        print(f"  obs         identical over {n} steps")

    ra, rb = a["rewards"].sum(), b["rewards"].sum()
    print(f"\n  reward sum  A={ra:.3f}  B={rb:.3f}  delta={rb - ra:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
