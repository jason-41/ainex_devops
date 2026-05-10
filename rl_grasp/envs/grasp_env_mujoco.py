"""
MuJoCo-based reach env for ainex right arm. Mirrors grasp_env_ainex.py
(PyBullet) but runs physics in MuJoCo. IK is computed via a sidecar
PyBullet client because PyBullet's damped-LS IK is well-tested and
deterministic; MuJoCo's role is the physics + contact dynamics that
training learns against.

Action: [dx, dy, dz, gripper] in [-1,1]
Obs:    [ee_xyz, cube_xyz, cube-ee, gripper_q]  (10D)
"""
import os
import gymnasium as gym
import numpy as np
import mujoco
import pybullet as p
import pybullet_data
from gymnasium import spaces


REPO_ROOT = "/home/jason/Desktop/TUM/backup_practice/ainex_devops"
SCENE_XML = os.path.join(REPO_ROOT, "rl_grasp/urdf/grasp_scene.xml")
AINEX_URDF = os.path.join(REPO_ROOT, "rl_grasp/urdf/ainex_mujoco.urdf")  # for sidecar IK


class GraspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    ARM_JOINTS = ["r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw"]
    GRIPPER_JOINT = "r_gripper"
    EE_BODY = "r_gripper_link"
    GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 0.7
    STEP_DELTA = 0.02
    # ainex's IK error floor at the 20-cm reach sphere edge is ~20-30 mm,
    # so the touch threshold must sit above that.
    SUCCESS_DIST = 0.04

    # base now at z=0.24 (feet on ground); cube/platform z dropped by 0.06 to match
    CUBE_BASE_XYZ = (0.13, -0.17, 0.27)
    CUBE_RANDOM = (0.02, 0.02, 0.015)
    # Reward shaping weights (added in step()):
    JOINT_REG_WEIGHT = 0.02     # discourage extreme joint angles → avoids singular straight-arm
    ACTION_REG_WEIGHT = 0.01    # discourage large action norms → smoother trajectories

    PHYSICS_SUBSTEPS = 10  # mj_step calls per env step (timestep 0.002 → 20ms env step)

    def __init__(self, render_mode=None, max_steps=200, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)

        # MuJoCo physics
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self._cache_mj_ids()

        # PyBullet sidecar — only used for IK (calculateInverseKinematics)
        self._pb = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._pb)
        self._pb_robot = p.loadURDF(
            AINEX_URDF, [0, 0, 0], useFixedBase=True, physicsClientId=self._pb,
        )
        self._pb_arm_idx, self._pb_ee_link, self._pb_movable = self._cache_pb_ids()

        self._renderer = None  # lazy

    def _cache_mj_ids(self):
        m = self.model
        self._mj_ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.EE_BODY)
        self._mj_cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        # qpos addresses for arm joints (used to set targets)
        self._mj_arm_qpos = [
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.ARM_JOINTS
        ]
        self._mj_grip_qpos = m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.GRIPPER_JOINT)
        ]
        # cube freejoint qpos (7 elements: x,y,z, qw,qx,qy,qz)
        self._mj_cube_qpos = m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        ]
        # joints we DON'T control (legs, head, left arm) — must be force-held at 0
        # every step or gravity will slowly drift them, causing visible tremor.
        active = set(self.ARM_JOINTS) | {self.GRIPPER_JOINT, "cube_free"}
        self._mj_lock_qpos = []
        self._mj_lock_qvel = []
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name not in active:
                self._mj_lock_qpos.append(m.jnt_qposadr[j])
                self._mj_lock_qvel.append(m.jnt_dofadr[j])

    def _cache_pb_ids(self):
        name_to_idx = {}
        for i in range(p.getNumJoints(self._pb_robot, physicsClientId=self._pb)):
            info = p.getJointInfo(self._pb_robot, i, physicsClientId=self._pb)
            name_to_idx[info[1].decode()] = i
        arm_idx = [name_to_idx[n] for n in self.ARM_JOINTS]
        ee_link = name_to_idx[self.GRIPPER_JOINT]
        movable = [
            i for i in range(p.getNumJoints(self._pb_robot, physicsClientId=self._pb))
            if p.getJointInfo(self._pb_robot, i, physicsClientId=self._pb)[2] != p.JOINT_FIXED
        ]
        return arm_idx, ee_link, movable

    def _ee_pos(self):
        return np.array(self.data.xpos[self._mj_ee_bid], dtype=np.float32)

    def _cube_pos(self):
        return np.array(self.data.xpos[self._mj_cube_bid], dtype=np.float32)

    def _gripper_q(self):
        return float(self.data.qpos[self._mj_grip_qpos])

    def _obs(self):
        ee, cube = self._ee_pos(), self._cube_pos()
        return np.concatenate([ee, cube, cube - ee, [self._gripper_q()]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        # randomize cube position by writing into the freejoint qpos
        cx, cy, cz = self.CUBE_BASE_XYZ
        cube_x = cx + float(self._rng.uniform(-self.CUBE_RANDOM[0], self.CUBE_RANDOM[0]))
        cube_y = cy + float(self._rng.uniform(-self.CUBE_RANDOM[1], self.CUBE_RANDOM[1]))
        cube_z = cz + float(self._rng.uniform(-self.CUBE_RANDOM[2], self.CUBE_RANDOM[2]))
        addr = self._mj_cube_qpos
        self.data.qpos[addr:addr+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[addr+3:addr+7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion

        # let things settle
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Use PyBullet sidecar for IK
        # 1) sync PyBullet joint state to MuJoCo's current arm joints
        for jname, qadr, jidx in zip(self.ARM_JOINTS, self._mj_arm_qpos, self._pb_arm_idx):
            p.resetJointState(self._pb_robot, jidx, float(self.data.qpos[qadr]), physicsClientId=self._pb)

        # 2) target EE = current EE + scaled action
        target_ee = self._ee_pos() + action[:3] * self.STEP_DELTA

        # 3) IK
        sol = p.calculateInverseKinematics(
            self._pb_robot, self._pb_ee_link, target_ee.tolist(),
            maxNumIterations=100, residualThreshold=1e-4,
            physicsClientId=self._pb,
        )
        sol_by_jidx = {self._pb_movable[k]: sol[k] for k in range(len(sol))}

        # 4) write joint targets directly to MuJoCo qpos (constraint-style stiff motor)
        for jname, qadr, jidx in zip(self.ARM_JOINTS, self._mj_arm_qpos, self._pb_arm_idx):
            self.data.qpos[qadr] = sol_by_jidx[jidx]
        # gripper
        gtarget = self.GRIPPER_CLOSE if action[3] > 0 else self.GRIPPER_OPEN
        self.data.qpos[self._mj_grip_qpos] = gtarget

        # 5) advance physics so cube/contact dynamics resolve
        for _ in range(self.PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        # 6) hard-clamp the 19 unactuated joints (legs, head, left arm) back to 0.
        # The model has no actuators on them, so gravity slowly drifts them; we
        # don't want that drift in the obs nor visually. Cube/arm dynamics are
        # unaffected because the chain is fixed-base and these joints are isolated.
        for a in self._mj_lock_qpos:
            self.data.qpos[a] = 0.0
        for a in self._mj_lock_qvel:
            self.data.qvel[a] = 0.0

        obs = self._obs()
        ee, cube = obs[:3], obs[3:6]
        dist = float(np.linalg.norm(cube - ee))

        reward = -dist
        if dist < 0.05: reward += 0.5
        if dist < self.SUCCESS_DIST: reward += 5.0

        # discourage straight-out / singular arm: penalise joints far from home (q=0)
        arm_qs = np.array([self.data.qpos[a] for a in self._mj_arm_qpos])
        reward -= self.JOINT_REG_WEIGHT * float(np.sum(arm_qs ** 2))
        # encourage smooth trajectories
        reward -= self.ACTION_REG_WEIGHT * float(np.linalg.norm(action[:3]))

        self.steps += 1
        terminated = bool(dist < self.SUCCESS_DIST)
        truncated = self.steps >= self.max_steps
        info = {"distance": dist}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if hasattr(self, "_pb") and self._pb is not None:
            try:
                p.disconnect(self._pb)
            except Exception:
                pass
            self._pb = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
