"""Robot registry: --robot flag → env class + per-robot dirs."""
from envs.grasp_env_franka import GraspEnv as FrankaEnv
from envs.grasp_env_ainex import GraspEnv as AinexEnv
from envs.grasp_env_mujoco import GraspEnv as AinexMuJoCoEnv


REGISTRY = {
    "franka":        {"env": FrankaEnv,       "max_steps": 300},
    "ainex":         {"env": AinexEnv,        "max_steps": 200},
    "ainex_mujoco":  {"env": AinexMuJoCoEnv,  "max_steps": 200},
}


def get_env_class(robot):
    if robot not in REGISTRY:
        raise ValueError(f"unknown robot {robot!r}, choose from {list(REGISTRY)}")
    return REGISTRY[robot]


def per_robot_dirs(root, robot):
    import os
    log_dir = os.path.join(root, "logs", robot)
    ckpt_dir = os.path.join(root, "checkpoints", robot)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return log_dir, ckpt_dir
