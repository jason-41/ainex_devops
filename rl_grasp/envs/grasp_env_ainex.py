"""
Reach env using the actual ainex humanoid URDF in PyBullet.

The robot only has ~2 effective positioning DoF in the right arm
(sho_pitch and el_pitch are twist joints that barely move the EE),
so this env trains a REACH task: minimize distance(EE, cube).
A grasp+lift task is unrealistic with the 1-DoF revolute gripper.

Locked joints (held with motor at q=0): all except right arm + right gripper.
Active joints: r_sho_pitch(17), r_sho_roll(18), r_el_pitch(19), r_el_yaw(20), r_gripper(21)
EE link: r_gripper_link (index 21)
"""
import os
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


REPO_ROOT = "/home/jason/Desktop/TUM/backup_practice/ainex_devops"
AINEX_URDF = os.path.join(REPO_ROOT, "src/ainex/ainex_description/urdf/ainex.urdf")
AINEX_PKG_PARENT = os.path.join(REPO_ROOT, "src/ainex")  # for `package://ainex_description/...`


class _SilenceStderr:
    """Suppress C-level stdout+stderr (Bullet 'No inertial data...' warnings) in a with-block."""
    def __enter__(self):
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *args):
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)
        os.close(self._saved_out)
        os.close(self._saved_err)
        os.close(self._null)


class GraspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # right-arm joint indices (probed from URDF):
    R_ARM_JOINTS = [17, 18, 19, 20]   # sho_pitch, sho_roll, el_pitch, el_yaw
    R_GRIPPER = 21
    EE_LINK = 21                       # r_gripper_link

    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSE = 0.7

    BASE_Z = 0.24                      # base lowered so feet rest on ground (was 0.30)
    R_SHOULDER_XYZ = (0.0, -0.064, 0.328)  # right shoulder world pos at base_z=0.24

    # workspace where the right hand can actually reach
    # ainex right-arm IK error is ~20-30 mm at the edge of the 20 cm reach
    # sphere, so SUCCESS_DIST must be larger than that or the policy can
    # never collect the +5 touch bonus and won't learn.
    CUBE_BASE_XYZ = (0.13, -0.17, 0.27)
    CUBE_RANDOM = (0.02, 0.02, 0.015)  # tighter jitter so we stay inside reach
    PLATFORM_HALF = (0.08, 0.08, 0.04) # small box "table" supporting the cube
    SUCCESS_DIST = 0.04                # 4 cm — above the IK floor for ainex

    # Reward shaping
    JOINT_REG_WEIGHT = 0.02            # discourage extreme arm angles (anti-singularity)
    ACTION_REG_WEIGHT = 0.01           # discourage large action norms (smoothness)

    STEP_DELTA = 0.02                  # 2 cm per env step (smaller workspace than Franka)
    SIM_SUBSTEPS = 10

    def __init__(self, render_mode=None, max_steps=200, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # obs: ee_xyz(3) + cube_xyz(3) + (cube - ee)(3) + gripper(1) = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._world_built = False
        self._connect()

    def _connect(self):
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.cid = p.connect(mode)
        p.setAdditionalSearchPath(AINEX_PKG_PARENT, physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)

    def _build_world_once(self):
        """Heavy one-time setup: load plane, robot URDF, platform, cube. Reused across resets."""
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        with _SilenceStderr():
            self.robot = p.loadURDF(
                AINEX_URDF, [0, 0, self.BASE_Z], useFixedBase=True, physicsClientId=self.cid,
            )

        plat_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=list(self.PLATFORM_HALF), physicsClientId=self.cid
        )
        plat_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=list(self.PLATFORM_HALF),
            rgbaColor=[0.55, 0.35, 0.20, 1.0], physicsClientId=self.cid,
        )
        self.platform = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=plat_col, baseVisualShapeIndex=plat_vis,
            basePosition=[0, 0, -1],  # parked; reset() will move it
            physicsClientId=self.cid,
        )

        cube_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], physicsClientId=self.cid
        )
        cube_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015],
            rgbaColor=[0.10, 0.70, 0.10, 1.0], physicsClientId=self.cid,
        )
        self.cube = p.createMultiBody(
            baseMass=0.05, baseCollisionShapeIndex=cube_col, baseVisualShapeIndex=cube_vis,
            basePosition=[0, 0, -1], physicsClientId=self.cid,
        )

        self._movable = [
            j for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid))
            if p.getJointInfo(self.robot, j, physicsClientId=self.cid)[2] != p.JOINT_FIXED
        ]
        # cached buffers for batched motor control: only arm + gripper slots change per step
        n = len(self._movable)
        self._target_buf = [0.0] * n          # locked joints stay at 0 forever
        self._force_buf = [200.0] * n         # default lock force
        self._arm_slots = [k for k, j in enumerate(self._movable) if j in self.R_ARM_JOINTS]
        self._gripper_slot = self._movable.index(self.R_GRIPPER)
        for k in self._arm_slots:
            self._force_buf[k] = 50.0
        self._force_buf[self._gripper_slot] = 10.0
        self._world_built = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self._world_built:
            self._build_world_once()

        # zero all joints (fast, no URDF reload)
        for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            p.resetJointState(self.robot, j, 0.0, physicsClientId=self.cid)

        # randomize cube + matching platform position
        cx, cy, cz = self.CUBE_BASE_XYZ
        cube_x = cx + float(self._rng.uniform(-self.CUBE_RANDOM[0], self.CUBE_RANDOM[0]))
        cube_y = cy + float(self._rng.uniform(-self.CUBE_RANDOM[1], self.CUBE_RANDOM[1]))
        cube_z = cz + float(self._rng.uniform(-self.CUBE_RANDOM[2], self.CUBE_RANDOM[2]))

        p.resetBasePositionAndOrientation(
            self.platform,
            [cube_x, cube_y, cube_z - self.PLATFORM_HALF[2] - 0.015],
            [0, 0, 0, 1], physicsClientId=self.cid,
        )
        p.resetBasePositionAndOrientation(
            self.cube, [cube_x, cube_y, cube_z], [0, 0, 0, 1], physicsClientId=self.cid,
        )
        p.resetBaseVelocity(
            self.cube, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid,
        )

        self.steps = 0
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.cid)
        return self._obs(), {}

    def _ee_pos(self):
        return np.array(p.getLinkState(self.robot, self.EE_LINK, physicsClientId=self.cid)[0])

    def _cube_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.cube, physicsClientId=self.cid)[0])

    def _gripper_q(self):
        return float(p.getJointState(self.robot, self.R_GRIPPER, physicsClientId=self.cid)[0])

    def _obs(self):
        ee, cube = self._ee_pos(), self._cube_pos()
        return np.concatenate([ee, cube, cube - ee, [self._gripper_q()]]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_ee = self._ee_pos() + action[:3] * self.STEP_DELTA
        sol = p.calculateInverseKinematics(
            self.robot, self.EE_LINK, target_ee.tolist(),
            maxNumIterations=100, residualThreshold=1e-4,
            physicsClientId=self.cid,
        )
        # update only the arm + gripper slots; locked joints keep their baked target=0
        for k in self._arm_slots:
            self._target_buf[k] = sol[k]
        gtarget = self.GRIPPER_CLOSE if action[3] > 0 else self.GRIPPER_OPEN
        self._target_buf[self._gripper_slot] = gtarget

        # one batched motor command for all 22 movable joints (vs 22 individual calls)
        p.setJointMotorControlArray(
            self.robot, jointIndices=self._movable,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self._target_buf, forces=self._force_buf,
            physicsClientId=self.cid,
        )

        for _ in range(self.SIM_SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)

        obs = self._obs()
        ee, cube = obs[:3], obs[3:6]
        dist = float(np.linalg.norm(cube - ee))

        # shaped reach reward
        reward = -dist
        if dist < 0.05:
            reward += 0.5     # close-by bonus to fight noise floor
        if dist < self.SUCCESS_DIST:
            reward += 5.0    # touch bonus

        # discourage straight/singular arm: penalise joints far from home (q=0)
        arm_qs = np.array([p.getJointState(self.robot, j, physicsClientId=self.cid)[0]
                           for j in self.R_ARM_JOINTS])
        reward -= self.JOINT_REG_WEIGHT * float(np.sum(arm_qs ** 2))
        # encourage smooth trajectories
        reward -= self.ACTION_REG_WEIGHT * float(np.linalg.norm(action[:3]))

        self.steps += 1
        terminated = bool(dist < self.SUCCESS_DIST)
        truncated = self.steps >= self.max_steps
        info = {"distance": dist, "ee": tuple(ee), "cube": tuple(cube)}
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
