"""Train a PPO policy on either Franka or ainex grasp env."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from scripts._robots import get_env_class, per_robot_dirs


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_env(env_cls, max_steps, rank):
    def _init():
        return env_cls(render_mode=None, max_steps=max_steps, seed=rank)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["franka", "ainex", "ainex_mujoco"], default="ainex")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--vec", choices=["subproc", "dummy"], default="subproc",
                    help="subproc = real parallel processes (faster), dummy = serial in one process")
    args = ap.parse_args()

    cfg = get_env_class(args.robot)
    env_cls, max_steps = cfg["env"], cfg["max_steps"]
    log_dir, ckpt_dir = per_robot_dirs(ROOT, args.robot)

    VecCls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    env = VecCls([make_env(env_cls, max_steps, i) for i in range(args.n_envs)])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir,
        verbose=1,
        device="auto",
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_grasp",
    )
    print(f"[train] robot={args.robot}  steps={args.total_steps}  ckpt={ckpt_dir}  log={log_dir}")
    model.learn(total_timesteps=args.total_steps, callback=ckpt_cb, progress_bar=True)
    model.save(os.path.join(ckpt_dir, "ppo_grasp_final"))
    print("saved:", os.path.join(ckpt_dir, "ppo_grasp_final.zip"))


if __name__ == "__main__":
    main()
