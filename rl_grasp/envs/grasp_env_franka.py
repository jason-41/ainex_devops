"""
Grasp env in PyBullet with Franka Panda (7-DoF arm + parallel gripper).

Action: [dx, dy, dz, gripper] in [-1, 1]
        - dx,dy,dz: end-effector displacement (gripper points down, IK solved)
        - gripper:  >0 close, <=0 open

Reward = -dist(ee, cube)
       + 1.0   if both fingers in contact with cube
       + 5.0   if grasping AND cube lifted above table
Termination: cube lifted above SUCCESS_Z while still grasping (success).
"""
import math
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class GraspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Panda joint layout in pybullet_data's franka_panda/panda.urdf
    ARM_JOINTS = list(range(7))         # 7 revolute arm joints
    FINGER_JOINTS = [9, 10]             # prismatic fingers, range [0, 0.04]
    EE_LINK = 11                        # panda_grasptarget (between fingers)
    HOME_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    FINGER_OPEN = 0.04
    FINGER_CLOSE = 0.0

    TABLE_TOP_Z = 0.625
    CUBE_HALF = 0.025                   # cube_small.urdf side ~ 5cm
    CUBE_INIT_Z = TABLE_TOP_Z + CUBE_HALF + 0.001
    LIFT_Z = TABLE_TOP_Z + 0.08         # cube clearly off the table
    SUCCESS_Z = TABLE_TOP_Z + 0.15

    STEP_DELTA = 0.03                   # 3 cm per env step
    SIM_SUBSTEPS = 20

    def __init__(self, render_mode=None, max_steps=300, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # obs: ee_xyz(3) + cube_xyz(3) + (cube - ee)(3) + gripper_width(1) = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._connect()

    def _connect(self):
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        p.loadURDF("table/table.urdf", [0.5, 0, 0], physicsClientId=self.cid)

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf", [0, 0, self.TABLE_TOP_Z],
            useFixedBase=True, physicsClientId=self.cid,
        )
        for j, q in zip(self.ARM_JOINTS, self.HOME_Q):
            p.resetJointState(self.robot, j, q, physicsClientId=self.cid)
        for j in self.FINGER_JOINTS:
            p.resetJointState(self.robot, j, self.FINGER_OPEN, physicsClientId=self.cid)

        cube_x = 0.5 + float(self._rng.uniform(-0.05, 0.05))
        cube_y = float(self._rng.uniform(-0.10, 0.10))
        self.cube = p.loadURDF(
            "cube_small.urdf", [cube_x, cube_y, self.CUBE_INIT_Z],
            physicsClientId=self.cid,
        )

        self.steps = 0
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.cid)
        return self._obs(), {}

    def _ee_pos(self):
        return np.array(p.getLinkState(self.robot, self.EE_LINK, physicsClientId=self.cid)[0])

    def _cube_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.cube, physicsClientId=self.cid)[0])

    def _gripper_width(self):
        s1 = p.getJointState(self.robot, self.FINGER_JOINTS[0], physicsClientId=self.cid)[0]
        s2 = p.getJointState(self.robot, self.FINGER_JOINTS[1], physicsClientId=self.cid)[0]
        return s1 + s2

    def _is_grasping(self):
        cl = p.getContactPoints(self.robot, self.cube, self.FINGER_JOINTS[0], -1, physicsClientId=self.cid)
        cr = p.getContactPoints(self.robot, self.cube, self.FINGER_JOINTS[1], -1, physicsClientId=self.cid)
        return bool(cl) and bool(cr)

    def _obs(self):
        ee = self._ee_pos()
        cube = self._cube_pos()
        return np.concatenate([ee, cube, cube - ee, [self._gripper_width()]]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_ee = self._ee_pos() + action[:3] * self.STEP_DELTA
        target_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])  # gripper points down
        joints = p.calculateInverseKinematics(
            self.robot, self.EE_LINK, target_ee.tolist(), target_orn,
            maxNumIterations=50, residualThreshold=1e-4,
            physicsClientId=self.cid,
        )
        for j, q in zip(self.ARM_JOINTS, joints[:7]):
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL, targetPosition=q,
                force=200, physicsClientId=self.cid,
            )

        finger_target = self.FINGER_CLOSE if action[3] > 0 else self.FINGER_OPEN
        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL, targetPosition=finger_target,
                force=20, physicsClientId=self.cid,
            )

        for _ in range(self.SIM_SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)

        obs = self._obs()
        ee, cube = obs[:3], obs[3:6]
        dist = float(np.linalg.norm(cube - ee))
        grasping = self._is_grasping()

        reward = -dist
        if grasping:
            reward += 1.0
        if grasping and cube[2] > self.LIFT_Z:
            reward += 5.0

        self.steps += 1
        terminated = bool(grasping and cube[2] > self.SUCCESS_Z)
        truncated = self.steps >= self.max_steps
        info = {
            "distance": dist,
            "cube_z": float(cube[2]),
            "gripper_width": float(obs[9]),
            "grasping": grasping,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
