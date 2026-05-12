"""
Gym env that drives Gazebo Harmonic via gz-transport (no rclpy, no ros2_control).

STATUS: STUB. The connect/step plumbing is sketched but not wired end-to-end.
        To finish:
          1. Pick a Python gz-transport library (gz-transport13 has Python bindings;
             alternative: ros_gz_bridge + rclpy in groupE_venv).
          2. Implement _read_joint_state() and _send_joint_targets() against
             the topics published by gazebo_harmonic.xacro:
                state:  /world/grasp_world/model/ainex/joint_state
                cmd:    /ainex/<joint>/cmd_pos     (one topic per joint)
          3. Implement deterministic resets — Gazebo doesn't have a clean
             "reset" service; usually you teleport bodies back via gz service
                /world/grasp_world/set_pose
          4. Implement step pacing — pause/unpause sim, or use /world/control
             with multi_step to run a fixed number of physics steps.

The EE position has to be derived from joint angles via an FK call. Either
import pinocchio (project already uses it) or call PyBullet just for FK on
the same URDF. Cube position comes from the bridged Pose topic.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


# joint topics that ainex_gz.urdf.xacro creates
ARM_JOINTS = ["r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw"]
GRIPPER_JOINT = "r_gripper"
LOCKED_JOINTS = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
    "l_sho_pitch", "l_sho_roll", "l_el_pitch", "l_el_yaw", "l_gripper",
    "head_pan", "head_tilt",
]


class GraspEnvGz(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=200, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        # TODO: import gz.transport13 and create a Node here
        # self._gz_node = gz.transport13.Node()
        # self._joint_state_sub = self._gz_node.subscribe(...)
        # self._cmd_pubs = {j: self._gz_node.advertise(f"/ainex/{j}/cmd_pos", DoubleMsg) for j in ALL_JOINTS}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # TODO: send 0 commands to all joints; teleport cube to random pose
        #       via /world/grasp_world/set_pose service; advance N physics
        #       steps so things settle.
        raise NotImplementedError("see module docstring TODO list")

    def step(self, action):
        # TODO: build dict of joint targets (arm from IK; locked joints = 0;
        #       gripper from action[3]); publish to /ainex/<j>/cmd_pos topics;
        #       advance physics (sim_step service or sleep + read clock); read
        #       new joint_state + cube pose; assemble obs / reward / done.
        raise NotImplementedError

    def close(self):
        pass
