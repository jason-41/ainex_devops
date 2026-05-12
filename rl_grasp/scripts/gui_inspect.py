"""
Open a PyBullet GUI to visually inspect the env (no training, no policy).

Usage:
    python scripts/gui_inspect.py --robot ainex --mode zero
    python scripts/gui_inspect.py --robot ainex --mode random
    python scripts/gui_inspect.py --robot franka --mode zero

modes:
    zero    - send action=0 every step (robot stays still, you can rotate the camera)
    random  - random actions (lets you see the env's full motion range)
"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scripts._robots import get_env_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["franka", "ainex"], default="ainex")
    ap.add_argument("--mode", choices=["zero", "random"], default="zero")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env_cls = get_env_class(args.robot)["env"]
    env = env_cls(render_mode="human", max_steps=10_000)
    obs, _ = env.reset(seed=args.seed)
    action_zero = np.zeros(env.action_space.shape, dtype=np.float32)
    print(f"[gui_inspect] robot={args.robot}  mode={args.mode}  initial_dist={float(np.linalg.norm(obs[6:9])):.3f}m")
    print("Drag mouse to rotate view, scroll to zoom, Ctrl+C in this terminal to quit.")

    try:
        while True:
            action = env.action_space.sample() if args.mode == "random" else action_zero
            obs, r, term, trunc, info = env.step(action)
            time.sleep(1 / 60)
            if term or trunc:
                print(f"  episode reset: dist={info['distance']:.3f}m, term={term}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        env.close()


if __name__ == "__main__":
    main()
