"""
Replay a MuJoCo-trained PPO policy in Gazebo Harmonic.

(Filename kept as `replay_pybullet_policy.py` for git history; the policy
itself was trained in MuJoCo. PyBullet still runs as an in-process IK
sidecar — same FK/IK math used during training.)

Pipeline each control step:
    Gazebo joint state ─► (FK in PyBullet copy) ─► current EE
    Gazebo pose info   ─►                           cube xyz
                                          │
                                          ▼ obs (10D)
                                       PPO policy
                                          │ action (4D)
                                          ▼
    target EE = ee + action[:3]*delta ─► (IK in PyBullet copy) ─► joint targets
                                                                       │
                                                                       ▼
                                                            27 cmd_pos publishers

Usage:
    # 1) Have Gazebo running in another terminal:
    #    ros2 launch rl_grasp_gazebo/launch/spawn.launch.py
    # 2) From rl_venv:
    source ~/rl_venv/bin/activate
    python rl_grasp_gazebo/scripts/replay_pybullet_policy.py
    # add --dry-run to print obs but NOT send commands (debug)
"""
import argparse
import os
import sys
import threading
import time

# Import venv packages FIRST so we get the right numpy/torch
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO

# Now make gz-transport (system-installed only) findable.
# Append (not insert-0) so venv numpy takes priority over /usr's numpy 1.26.
sys.path.append("/usr/lib/python3/dist-packages")
from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_v_pb2 import Pose_V

assert np.__version__.startswith("2."), \
    f"expected numpy 2.x from venv, got {np.__version__} from {np.__file__}"


# -------- paths / topics --------
REPO_ROOT = "/home/jason/Desktop/TUM/backup_practice/ainex_devops"
AINEX_URDF = os.path.join(REPO_ROOT, "src/ainex/ainex_description/urdf/ainex.urdf")
AINEX_PKG_PARENT = os.path.join(REPO_ROOT, "src/ainex")
DEFAULT_CKPT = os.path.join(
    REPO_ROOT, "rl_grasp/checkpoints/ainex_mujoco/ppo_grasp_final.zip"
)

WORLD = "grasp_world"
TOPIC_JOINT_STATE = f"/world/{WORLD}/model/ainex/joint_state"
TOPIC_POSE_INFO = f"/world/{WORLD}/dynamic_pose/info"
TOPIC_CMD_FMT = "/ainex/{joint}/cmd_pos"

# -------- joint config (mirrors grasp_env_ainex.py) --------
ARM_JOINT_NAMES = ["r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw"]
GRIPPER_JOINT_NAME = "r_gripper"
LOCKED_JOINT_NAMES = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
    "l_sho_pitch", "l_sho_roll", "l_el_pitch", "l_el_yaw", "l_gripper",
    "head_pan", "head_tilt",
]
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME] + LOCKED_JOINT_NAMES

CUBE_MODEL_NAME = "cube"
STEP_DELTA = 0.02
GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 0.7
# 10 Hz, not 30 — gives Gazebo PID time to actually track each IK target
# instead of overshooting. Training was in PyBullet's stiff-constraint motor.
CONTROL_HZ = 10


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.joint_q = {}                  # joint_name -> position (rad)
        self.cube_pos = None               # np.array(3,) or None
        self.last_joint_msg_t = 0.0
        self.last_pose_msg_t = 0.0


def joint_state_cb(state: SharedState):
    def _cb(msg: Model):
        with state.lock:
            for j in msg.joint:
                # gz Model.joint[].axis1.position is the rad position for revolute joints
                state.joint_q[j.name] = float(j.axis1.position)
            state.last_joint_msg_t = time.time()
    return _cb


def pose_cb(state: SharedState):
    def _cb(msg: Pose_V):
        with state.lock:
            for pose in msg.pose:
                if pose.name == CUBE_MODEL_NAME:
                    state.cube_pos = np.array(
                        [pose.position.x, pose.position.y, pose.position.z],
                        dtype=np.float32,
                    )
            state.last_pose_msg_t = time.time()
    return _cb


def build_pybullet_helper():
    """Headless PyBullet client used only for FK / IK on the same URDF."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(AINEX_PKG_PARENT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(AINEX_URDF, [0, 0, 0.30], useFixedBase=True)

    name_to_idx = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name_to_idx[info[1].decode()] = i

    movable = [
        i for i in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED
    ]
    ee_link = name_to_idx["r_gripper"]  # joint idx == link idx in PyBullet
    arm_idx = [name_to_idx[n] for n in ARM_JOINT_NAMES]
    return robot, name_to_idx, movable, ee_link, arm_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--dry-run", action="store_true",
                    help="print obs and predicted action, but do NOT publish commands")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt}")

    print(f"[replay] loading policy from {args.ckpt}")
    model = PPO.load(args.ckpt, device="cpu")  # MlpPolicy, CPU is plenty

    print(f"[replay] booting headless PyBullet for FK/IK ...")
    robot, name_to_idx, movable, ee_link, arm_idx = build_pybullet_helper()

    state = SharedState()
    node = Node()
    print(f"[replay] subscribing to {TOPIC_JOINT_STATE}")
    if not node.subscribe(Model, TOPIC_JOINT_STATE, joint_state_cb(state)):
        sys.exit("failed to subscribe joint_state")
    print(f"[replay] subscribing to {TOPIC_POSE_INFO}")
    if not node.subscribe(Pose_V, TOPIC_POSE_INFO, pose_cb(state)):
        sys.exit("failed to subscribe pose")

    pubs = {}
    if not args.dry_run:
        # only the joints that have a controller in URDF (right arm + gripper)
        active = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
        for jname in active:
            pub = node.advertise(TOPIC_CMD_FMT.format(joint=jname), Double)
            if not pub.valid():
                sys.exit(f"failed to advertise cmd topic for {jname}")
            pubs[jname] = pub
        print(f"[replay] advertising {len(pubs)} cmd_pos topics")

    # wait for first state messages
    print("[replay] waiting for first joint_state + cube pose ...")
    deadline = time.time() + 10
    while time.time() < deadline:
        with state.lock:
            ok = bool(state.joint_q) and state.cube_pos is not None
        if ok:
            break
        time.sleep(0.1)
    else:
        sys.exit("timed out waiting for Gazebo state — is gz running?")
    print("[replay] state stream alive, entering control loop")

    period = 1.0 / CONTROL_HZ
    next_t = time.time()
    step = 0
    while True:
        next_t += period

        # read shared state snapshot
        with state.lock:
            joints = dict(state.joint_q)
            cube = None if state.cube_pos is None else state.cube_pos.copy()
        if cube is None:
            time.sleep(period); continue

        # mirror gz joint angles into PyBullet
        for jname, q in joints.items():
            if jname in name_to_idx:
                p.resetJointState(robot, name_to_idx[jname], q)
        # FK: EE in current pose
        ee = np.array(p.getLinkState(robot, ee_link)[0], dtype=np.float32)

        # observation matches grasp_env_ainex._obs(): ee(3) + cube(3) + diff(3) + gripper(1)
        gripper_q = float(joints.get(GRIPPER_JOINT_NAME, 0.0))
        obs = np.concatenate([ee, cube, cube - ee, [gripper_q]]).astype(np.float32)

        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        target_ee = ee + action[:3] * STEP_DELTA
        sol = p.calculateInverseKinematics(
            robot, ee_link, target_ee.tolist(),
            maxNumIterations=100, residualThreshold=1e-4,
        )
        # IK returns one value per movable joint, in order
        sol_by_joint_idx = {movable[k]: sol[k] for k in range(len(sol))}

        if step % 30 == 0:
            d = float(np.linalg.norm(cube - ee))
            print(f"[replay] t={step/CONTROL_HZ:5.1f}s  ee={ee.round(3)}  "
                  f"cube={cube.round(3)}  dist={d:.3f}m  action={action.round(2)}")
        step += 1

        if args.dry_run:
            time.sleep(max(0.0, next_t - time.time())); continue

        # publish targets only to joints that have a Gazebo controller:
        # - arm joints from IK
        for jname, jidx in zip(ARM_JOINT_NAMES, arm_idx):
            msg = Double(); msg.data = float(sol_by_joint_idx[jidx])
            pubs[jname].publish(msg)
        # - gripper from action[3]
        g = GRIPPER_CLOSE if action[3] > 0 else GRIPPER_OPEN
        msg = Double(); msg.data = g
        pubs[GRIPPER_JOINT_NAME].publish(msg)
        # locked joints have no controller in URDF — they hold at 0 via static friction

        # pace
        time.sleep(max(0.0, next_t - time.time()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
