"""Quick sanity check: env builds, reset/step return correct shapes."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scripts._robots import get_env_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["franka", "ainex", "ainex_mujoco"], default="ainex")
    args = ap.parse_args()

    env_cls = get_env_class(args.robot)["env"]
    env = env_cls(render_mode=None, max_steps=20)
    obs, info = env.reset(seed=0)
    print("obs shape:", obs.shape, "dtype:", obs.dtype)
    assert obs.shape == (10,)

    total = 0.0
    t = 0
    for t in range(20):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total += r
        if term or trunc:
            print(f"  ended at step {t+1}: term={term} trunc={trunc}")
            break
    print(f"random-policy return over {t+1} steps: {total:.3f}")
    print(f"final distance: {info['distance']:.3f}m")
    env.close()
    print("OK")


if __name__ == "__main__":
    main()
