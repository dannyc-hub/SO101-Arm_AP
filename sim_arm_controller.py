import time
import numpy as np
import os
import lerobot

import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation as R
from PIL import Image
from ikpy.chain import Chain

from lerobot.robots.so_follower.so_follower import SOFollower

class SimSO101ArmController:
    """Drop-in replacement for SO101ArmController for data generation.
    Mirrors the real class's interface so ActivePerceptionController
    works unmodified against either."""

    CAMERA_DOWN_TARGET = [0.0, 0.0, 1.0] # inverted from real controller
    CAMERA_AXIS_MODE = "Y"

    # Overhead camera: fixed position above the workspace, looking straight down.
    # Adjust position to roughly center over your arm's actual reachable area.
    OVERHEAD_CAM_POSITION = [0.0, -0.12, 0.5]
    OVERHEAD_CAM_TARGET = [0.0, -0.12, 0.0]  # looking straight down at that XY

    def __init__(self, urdf_path, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)

        # Build a name -> PyBullet joint index map, rather than assuming position.
        # URDF joint names here are "1".."6" (not descriptive), matching the
        # kinematic order: 1=shoulder_pan, 2=shoulder_lift, 3=elbow_flex,
        # 4=wrist_flex, 5=wrist_roll, 6=gripper/jaw.
        name_to_index = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode()
            name_to_index[joint_name] = i

        print("[SimSO101ArmController] Discovered joint mapping:", name_to_index)

        # Explicit, verified mapping — order matches motor_keys /
        # active_joints_rad order used by solve_ik (joint "1" through "5").
        self.joint_indices = [
            name_to_index["1"],  # shoulder_pan
            name_to_index["2"],  # shoulder_lift
            name_to_index["3"],  # elbow_flex
            name_to_index["4"],  # wrist_flex
            name_to_index["5"],  # wrist_roll
        ]
        self.gripper_joint_index = name_to_index["6"]  # jaw/camera-mount joint

        self.ee_link_index = name_to_index["5"]  # moving_jaw link — matches real controller's FK target

        self.chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=["base"],
            active_links_mask=[False, True, True, True, True, True, False],
        )

        # Instantly place joints at a safe, in-bounds starting configuration
        safe_start_rad = [0.0, 0.0, 0.0, 0.0, 0.0]
        for idx, angle in zip(self.joint_indices, safe_start_rad):
            p.resetJointState(self.robot_id, idx, angle)

    def _solve_ik(self, target_position, maintain_camera_down=True):
        current_angles_rad = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]

        # Clip to each active joint's declared bounds (with a tiny safety margin)
        EPS = 1e-4
        clipped_angles_rad = []
        for joint_angle, link in zip(current_angles_rad, self.chain.links[1:6]):
            lower, upper = link.bounds
            if lower is not None and upper is not None:
                joint_angle = float(np.clip(joint_angle, lower + EPS, upper - EPS))
            clipped_angles_rad.append(joint_angle)

        initial_position = [0.0] + clipped_angles_rad + [0.0]

        if maintain_camera_down:
            ik_solution = self.chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=self.CAMERA_DOWN_TARGET,
                orientation_mode=self.CAMERA_AXIS_MODE,
                initial_position=initial_position,
            )
        else:
            ik_solution = self.chain.inverse_kinematics(
                target_position=target_position,
                initial_position=initial_position,
            )

        active_joints_rad = ik_solution[1:6]
        return np.degrees(active_joints_rad).tolist()

    def move_to_joint_angles(self, angles_deg, gripper_pos=0.0, duration=0.2):
        angles_rad = np.radians(angles_deg[:5])
        for idx, angle in zip(self.joint_indices, angles_rad):
            p.setJointMotorControl2(
                self.robot_id, idx, p.POSITION_CONTROL, targetPosition=angle
            )
        for _ in range(int(duration * 240)):  # PyBullet default 240Hz sim step
            p.stepSimulation()
            
    def get_overhead_camera_frame(self):
        """Renders a synthetic frame from a fixed overhead viewpoint,
        matching the real controller's separate overhead camera."""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.OVERHEAD_CAM_POSITION,
            cameraTargetPosition=self.OVERHEAD_CAM_TARGET,
            cameraUpVector=[0, 1, 0],  # "up" in the image, since looking straight down along Z
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=102, aspect=1.77, nearVal=0.05, farVal=1.5)

        width, height = 1280, 720

        _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
        return Image.fromarray(rgb_array)

    def move_to_cartesian_pose(self, x, y, z, gripper_pos=0.0, duration=0.2, maintain_camera_down=True):
        """Sim equivalent of the real controller's IK-based Cartesian move.
        NOTE: this does NOT yet run actual inverse kinematics — see below."""
        target_angles_deg = self._solve_ik([x, y, z], maintain_camera_down=maintain_camera_down)
        self.move_to_joint_angles(target_angles_deg, gripper_pos=gripper_pos, duration=duration)

    def move_to_cartesian_pose_with_wrist(
        self, x, y, z, wrist_roll_deg=None, wrist_flex_deg=None, gripper_pos=0.0, duration=0.2
    ):
        target_angles_deg = self._solve_ik([x, y, z], maintain_camera_down=True)

        if wrist_flex_deg is not None:
            target_angles_deg[3] = float(wrist_flex_deg)
        if wrist_roll_deg is not None:
            target_angles_deg[4] = float(wrist_roll_deg)

        self.move_to_joint_angles(target_angles_deg, gripper_pos=gripper_pos, duration=duration)

    def get_current_cartesian_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        pos = state[0]
        return [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)]

    def get_current_wrist_angles_deg(self):
        flex = np.degrees(p.getJointState(self.robot_id, 3)[0])
        roll = np.degrees(p.getJointState(self.robot_id, 4)[0])
        return flex, roll

    def get_wrist_camera_frame(self):
        """Renders a synthetic frame from the end-effector's pose."""
        state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        pos, orn = state[0], state[1]
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # Extract local link axes (Columns: 0 = Local X, 1 = Local Y, 2 = Local Z)
        local_x = rot_matrix[:, 0]
        local_y = rot_matrix[:, 1]
        local_z = rot_matrix[:, 2]

        forward_vec = -local_z 
        up_vec = local_y  

        CAMERA_MOUNT_OFFSET = 0.07
        cam_eye = np.array(pos) + forward_vec * CAMERA_MOUNT_OFFSET
        cam_target = cam_eye + forward_vec * 0.1

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=up_vec.tolist(),
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=102, aspect=1.77, nearVal=0.01, farVal=1.0)

        width, height = 1280, 720
        _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
        return Image.fromarray(rgb_array)

    def disconnect(self):
        p.disconnect(self.client)
