import time
import numpy as np
import cv2
import os
import lerobot

from scipy.spatial.transform import Rotation as R
from PIL import Image
from ikpy.chain import Chain
from search import ActiveSearchPattern

from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

class SO101ArmController:

    # Camera boresight (viewing direction) target: straight down in world frame.
    CAMERA_DOWN_TARGET = [0.0, 0.0, 1.0]
    CAMERA_AXIS_MODE = "Y"

    def __init__(self, urdf_path="so101.urdf", wrist_cam_idx=2, overhead_cam_idx=0, port="/dev/ttyACM0"):
        # 1. Initialize Kinematic Chain from URDF for Inverse Kinematics
        self.chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=["base"],
            active_links_mask=[False, True, True, True, True, True, False]  # 5 active motor links
        )

        # 2. Initialize LeRobot hardware connection using degrees mode
        config = SOFollowerRobotConfig(port=port, use_degrees=True, id="follower_arm")
        self.robot = SOFollower(config=config)
        self.robot.connect()

        # 3. Enable motor torque explicitly so the arm holds position
        if hasattr(self.robot, "bus") and self.robot.bus is not None:
            self.robot.bus.enable_torque()

        # 4. Motor key mapping matching LeRobot SOFollower action space
        self.motor_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
        ]

        # 5. Initialize camera streams
        self.wrist_cap = cv2.VideoCapture(wrist_cam_idx)
        self.overhead_cap = cv2.VideoCapture(overhead_cam_idx)

        # 6. Initialize search pattern engine with default IK Cartesian pose [X, Y, Z]
        self.search_pattern = ActiveSearchPattern(
            arm_controller=self, center_pose=[0.25, 0.0, 0.15]
        )
        
    def solve_ik(
        self,
        target_position: list[float],
        target_orientation: list[float] = None,
        orientation_mode: str = None,
        maintain_camera_down: bool = True,
    ) -> list[float]:
        """Calculates joint angles for a Cartesian position, constrained by
        default to keep the wrist camera facing straight down.

        :param maintain_camera_down: if True and no explicit target_orientation
            is given, enforces CAMERA_DOWN_TARGET orientation. Set False for
            unconstrained position-only IK.
        """
        observation = self.robot.get_observation()
        current_angles_deg = list(observation.values())[:-1]
        current_angles_rad = np.radians(current_angles_deg)
        initial_position = [0.0] + list(current_angles_rad) + [0.0]

        if target_orientation is None and maintain_camera_down:
            target_orientation = self.CAMERA_DOWN_TARGET
            orientation_mode = orientation_mode or self.CAMERA_AXIS_MODE

        if target_orientation is not None:
            try:
                ik_solution = self.chain.inverse_kinematics(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    orientation_mode=orientation_mode or "all",
                    initial_position=initial_position,
                )
            except Exception as e:
                # Orientation-constrained IK can fail to converge near
                # workspace limits. Fall back to position-only rather than
                # crashing the perception loop.
                print(f"[SO101ArmController] Orientation IK failed ({e}); falling back to position-only IK.")
                ik_solution = self.chain.inverse_kinematics(
                    target_position=target_position,
                    initial_position=initial_position,
                )
        else:
            ik_solution = self.chain.inverse_kinematics(
                target_position=target_position,
                initial_position=initial_position,
            )

        active_joints_rad = ik_solution[1:6]
        return np.degrees(active_joints_rad).tolist()
       
    def move_to_cartesian_pose(
        self,
        x: float,
        y: float,
        z: float,
        gripper_pos: float = 0.0,
        duration: float = 0.2,
        maintain_camera_down: bool = True,
    ):
        """Commands arm to move to a 3D Cartesian space coordinate (X, Y, Z in meters),
        keeping the wrist camera facing down by default.

        :param x: Target X position (meters forward/backward).
        :param y: Target Y position (meters left/right).
        :param z: Target Z position (meters height).
        :param gripper_pos: Target gripper position.
        :param duration: Pause time to allow physical movement execution.
        :param maintain_camera_down: set False for unconstrained position-only IK.
        """
        target_angles_deg = self.solve_ik(
            target_position=[x, y, z],
            maintain_camera_down=maintain_camera_down,
        )
        self.move_to_joint_angles(target_angles_deg, gripper_pos=gripper_pos, duration=duration)

    def move_to_cartesian_pose_with_wrist(
        self,
        x: float,
        y: float,
        z: float,
        wrist_roll_deg: float = None,
        wrist_flex_deg: float = None,
        gripper_pos: float = 0.0,
        duration: float = 0.2,
    ):
        """Like move_to_cartesian_pose, but lets you override wrist_flex/wrist_roll
        explicitly after IK — useful for sampling camera viewing angles at a fixed
        Cartesian position.

        Overriding wrist_roll_deg is safe — it rotates the image about the
        camera's own viewing axis without breaking the down-facing constraint.
        Overriding wrist_flex_deg WILL break the down-facing guarantee, since
        flex is what IK solves for to keep the camera pointed down — only pass
        it if you deliberately want a tilted view (as in your optimization sweep).
        """
        target_angles_deg = self.solve_ik(target_position=[x, y, z])  # maintain_camera_down=True by default

        if wrist_flex_deg is not None:
            target_angles_deg[3] = float(wrist_flex_deg)
        if wrist_roll_deg is not None:
            target_angles_deg[4] = float(wrist_roll_deg)

        self.move_to_joint_angles(target_angles_deg, gripper_pos=gripper_pos, duration=duration)
    
    def get_current_cartesian_pose(self) -> list[float]:
        """Reads physical motor positions and uses FK to calculate the current 3D Cartesian point (X, Y, Z)."""
        # 1. Read actual motor angles from Feetech servos
        observation = self.robot.get_observation()
        
        current_angles_deg = list(observation.values())
        
        # Exclude claw angle
        current_angles_deg = current_angles_deg[:-1]
        
        # 2. Convert degrees to radians
        current_angles_rad = np.radians(current_angles_deg)
        
        # 3. Construct full joint vector matching ikpy chain size
        full_joint_vector = [0.0] + list(current_angles_rad) + [0.0]
        
        # 4. Compute Forward Kinematics matrix
        transformation_matrix = self.chain.forward_kinematics(full_joint_vector)
        
        # Extract X, Y, Z from translation column of 4x4 transform
        x, y, z = transformation_matrix[:3, 3]
        return [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
    
    def get_current_wrist_roll_rad(self) -> float:
        """Reads current wrist_roll joint angle for camera-frame correction."""
        observation = self.robot.get_observation()
        wrist_roll_deg = observation["wrist_roll.pos"]
        return np.radians(wrist_roll_deg)
        
    def get_camera_facing_vector(self) -> np.ndarray:
        """Debug/calibration helper: returns the world-frame direction the
        camera's local CAMERA_AXIS_MODE axis currently points, via forward
        kinematics. Use this to calibrate CAMERA_AXIS_MODE / CAMERA_DOWN_TARGET —
        a correctly down-facing camera should print roughly [0, 0, -1].
        """
        observation = self.robot.get_observation()
        current_angles_deg = list(observation.values())[:-1]
        current_angles_rad = np.radians(current_angles_deg)
        full_joint_vector = [0.0] + list(current_angles_rad) + [0.0]

        transform = self.chain.forward_kinematics(full_joint_vector)
        axis_index = {"X": 0, "Y": 1, "Z": 2}[self.CAMERA_AXIS_MODE]
        facing_vector = transform[:3, axis_index]
        return facing_vector
    
    def move_to_joint_angles(
        self,
        angles_deg: list[float],
        gripper_pos: float = 0.0,
        duration: float = 0.2,
        max_step_deg: float = 3.0,   # max degrees moved per interpolation step
        min_steps: int = 3,          # always take at least this many steps
    ):
        """Smoothed joint controller: interpolates from current pose to target
        pose over `duration` seconds using an ease-in/ease-out (smoothstep) profile,
        so large pose changes don't snap the arm."""
        if len(angles_deg) < 5:
            raise ValueError(f"Expected 5 joint angles, got {len(angles_deg)}")

        # Read current joint state (explicit key lookup, not positional order)
        observation = self.robot.get_observation()
        current_arm_angles = [float(observation[k]) for k in self.motor_keys]
        current_gripper = float(observation.get("gripper.pos", gripper_pos))

        target_arm_angles = [float(a) for a in angles_deg[:5]]
        target_gripper = float(gripper_pos)

        # Scale step count to the largest joint delta so big transitions get
        # proportionally more waypoints (and therefore stay smooth) while tiny
        # IBVS corrections don't waste time on unnecessary steps.
        max_delta = max(abs(t - c) for t, c in zip(target_arm_angles, current_arm_angles))
        steps = max(min_steps, int(np.ceil(max_delta / max_step_deg)))
        step_duration = duration / steps

        for i in range(1, steps + 1):
            t = i / steps
            alpha = t * t * (3 - 2 * t)  # cubic smoothstep: zero velocity at both ends

            interp_angles = [
                c + (tgt - c) * alpha
                for c, tgt in zip(current_arm_angles, target_arm_angles)
            ]
            interp_gripper = current_gripper + (target_gripper - current_gripper) * alpha

            target_dict = {
                key: float(angle) for key, angle in zip(self.motor_keys, interp_angles)
            }
            target_dict["gripper.pos"] = float(interp_gripper)

            self.robot.send_action(target_dict)
            time.sleep(step_duration)

    def get_current_wrist_angles_deg(self) -> tuple[float, float]:
        """Returns (wrist_flex_deg, wrist_roll_deg) from current observation."""
        observation = self.robot.get_observation()
        return float(observation["wrist_flex.pos"]), float(observation["wrist_roll.pos"])
    
    def get_wrist_camera_frame(self) -> Image.Image:
        ret, frame = self.wrist_cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get_overhead_camera_frame(self) -> Image.Image:
        ret, frame = self.overhead_cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def disconnect(self):
        if self.wrist_cap.isOpened():
            self.wrist_cap.release()
        if self.overhead_cap.isOpened():
            self.overhead_cap.release()
        self.robot.disconnect()


if __name__ == "__main__":
    lerobot_dir = os.path.dirname(lerobot.__file__)
    urdf_path = os.path.join(lerobot_dir,"configs", "so101_cleaned.urdf")
    
    arm = SO101ArmController(urdf_path=urdf_path, port="/dev/ttyACM0")

    try:
        rest_joint_angles = [-3.0, -110.0, 96.0, 62.0, -2.0, 0.5]
        home_joint_angles = [0.0, 0.0, 0.0, 90.0, 0.0]

        print("Moving to baseline joint pose")
        arm.move_to_joint_angles(home_joint_angles, duration=2.0)
        
        for axis in ["X", "Y", "Z"]:
            arm.CAMERA_AXIS_MODE = axis
            print(f"Local {axis} axis points (world frame): {np.round(arm.get_camera_facing_vector(), 3)}")

        print("Returning to home pose...")
        arm.move_to_joint_angles(home_joint_angles, duration=2.0)
        
        print("Returning to rest pose...")
        arm.move_to_joint_angles(rest_joint_angles, duration=2.0)

    finally:
        arm.disconnect()
