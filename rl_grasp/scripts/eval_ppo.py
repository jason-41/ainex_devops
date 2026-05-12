"""Watch a trained policy interactively.

  --robot franka         → PyBullet GUI
  --robot ainex          → PyBullet GUI
  --robot ainex_mujoco   → MuJoCo viewer (no ROS, no Gazebo)
"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from scripts._robots import get_env_class, per_robot_dirs


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["franka", "ainex", "ainex_mujoco"], default="ainex")
    ap.add_argument("--ckpt", default="ppo_grasp_final",
                    help="checkpoint name without .zip, e.g. ppo_grasp_300000_steps")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    cfg = get_env_class(args.robot)
    env_cls, max_steps = cfg["env"], cfg["max_steps"]
    _, ckpt_dir = per_robot_dirs(ROOT, args.robot)

    ckpt_path = os.path.join(ckpt_dir, f"{args.ckpt}.zip")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")

    is_mujoco = args.robot == "ainex_mujoco"

    if is_mujoco:
        # MuJoCo env in headless DIRECT mode for physics; an external mujoco.viewer
        # attaches to env.model/env.data and renders interactively.
        env = env_cls(render_mode=None, max_steps=max_steps)
        model = PPO.load(ckpt_path, env=env)
        run_mujoco_viewer(env, model, args.episodes)
    else:
        env = env_cls(render_mode="human", max_steps=max_steps)
        model = PPO.load(ckpt_path, env=env)
        run_pybullet_eval(env, model, args.episodes)


def run_pybullet_eval(env, model, episodes):
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        ep_ret, dist, term = 0.0, None, False
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            dist = info["distance"]
            time.sleep(1 / 60)
            if term or trunc:
                break
        if term:
            successes += 1
        print(f"ep {ep}: return={ep_ret:7.2f}  final_dist={dist:.3f}m  success={term}")
    print(f"\nsuccess rate: {successes}/{episodes}")
    env.close()


def run_mujoco_viewer(env, model, episodes):
    """Use mujoco.viewer.launch_passive — interactive window, our control loop ticks physics."""
    import mujoco.viewer

    successes = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # decent default camera looking at the workspace from front-right
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.0, -0.15, 0.45]

        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            ep_ret, dist, term = 0.0, None, False
            for _ in range(env.max_steps):
                if not viewer.is_running():
                    print("viewer closed — bye")
                    env.close()
                    return
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_ret += r
                dist = info["distance"]
                viewer.sync()
                time.sleep(1 / 60)
                if term or trunc:
                    break
            if term:
                successes += 1
            print(f"ep {ep}: return={ep_ret:7.2f}  final_dist={dist:.3f}m  success={term}")
    print(f"\nsuccess rate: {successes}/{episodes}")
    env.close()


if __name__ == "__main__":
    main()
