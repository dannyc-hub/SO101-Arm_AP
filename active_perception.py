import time
import numpy as np
from PIL import Image

from arm_controller import SO101ArmController
from detection import detect_target


class ActivePerceptionController:
    def __init__(
        self,
        arm: SO101ArmController,
        target_text: str,
        confidence_threshold: float = 0.80,
        target_bbox_ratio: float = 0.30,
        max_iterations: int = 10,
    ):
        """Active perception optimizer using Image-Based Visual Servoing and Cartesian offset sampling.

        :param arm: Connected SO101ArmController instance.
        :param target_text: Text prompt for OWLv2 detection (e.g., "screwdriver").
        :param confidence_threshold: Stopping threshold for OWLv2 score.
        :param target_bbox_ratio: Desired target bounding box area relative to camera frame area.
        :param max_iterations: Maximum alignment iterations per search phase.
        """
        self.arm = arm
        self.target_text = target_text
        self.confidence_threshold = confidence_threshold
        self.target_bbox_ratio = target_bbox_ratio
        self.max_iterations = max_iterations

    def calculate_quality_metric(self, score: float, box: list[float], img_size: tuple[int, int]) -> float:
        """Calculates combined certainty score incorporating detection confidence and optimal visual scale."""
        if score == 0.0 or box is None:
            return 0.0

        img_w, img_h = img_size
        img_area = img_w * img_h

        # Compute bounding box fill ratio relative to image frame
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        area_ratio = (box_w * box_h) / img_area

        # Penalty score for camera being too far or too close
        size_penalty = 1.0 - min(abs(area_ratio - self.target_bbox_ratio) / self.target_bbox_ratio, 1.0)

        # Weighted quality metric
        quality = (0.7 * score) + (0.3 * size_penalty)
        return float(quality)

    def align_wrist_camera_2d(self, kp_xy: float = 0.01, kp_z: float = 0.01) -> tuple[float, list[float], int]:
        """Centering loop (Image-Based Visual Servoing): Moves arm in X, Y, Z to center object and normalize scale."""
        print(f"[ActivePerception] Aligning eye-in-hand camera over target '{self.target_text}'...")

        best_score = 0.0
        best_box = None
        actuation_count = 0

        for i in range(self.max_iterations):
            frame = self.arm.get_wrist_camera_frame()
            if frame is None:
                print("[ActivePerception] Error: Failed to capture frame from wrist camera.")
                break

            img_w, img_h = frame.size
            found, score, box = detect_target(frame, self.target_text, confidence_threshold=0.2)

            if not found:
                print(f"[ActivePerception] Target lost during alignment step {i+1}.")
                break

            best_score = score
            best_box = box

            box_cx = (box[0] + box[2]) / 2.0
            box_cy = (box[1] + box[3]) / 2.0
            error_x = box_cx - (img_w / 2.0)
            error_y = box_cy - (img_h / 2.0)

            quality = self.calculate_quality_metric(score, box, frame.size)
            if abs(error_x) < 20 and abs(error_y) < 20 and score >= self.confidence_threshold:
                print(f"[ActivePerception] Target centered and high quality reached: {quality:.3f}")
                break

            current_x, current_y, current_z = self.arm.get_current_cartesian_pose()

            MAX_STEP_METERS = 0.012

            dy = np.clip(-error_x * kp_xy, -MAX_STEP_METERS, MAX_STEP_METERS)
            dx = np.clip(-error_y * kp_xy, -MAX_STEP_METERS, MAX_STEP_METERS)

            box_area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / (img_w * img_h)
            dz = np.clip((box_area_ratio - self.target_bbox_ratio) * kp_z, -MAX_STEP_METERS, MAX_STEP_METERS)

            new_x = float(np.clip(current_x + dx, -0.05, 0.10))
            new_y = float(np.clip(current_y + dy, -0.22, -0.02))
            new_z = float(np.clip(current_z + dz, 0.16, 0.35))

            print("center coordinates", new_x, new_y, new_z, f"| actuation #{actuation_count + 1}")

            self.arm.move_to_cartesian_pose(new_x, new_y, new_z, duration=1.0)
            actuation_count += 1

        print(f"[ActivePerception] Alignment stage used {actuation_count} actuation(s).")
        return best_score, best_box, actuation_count
     
    def optimize_viewpoint_positions(
        self,
        base_x: float,
        base_y: float,
        base_z: float,
        step_m: float = 0.03,
        wrist_roll_offsets_deg: list[float] = (-45.0, -20.0, 0.0, 20.0, 45.0),
    ) -> tuple[list[float], float, float, int]:
        """Cartesian offset search + wrist angle sweep: samples adjacent X/Y/Z
        positions AND wrist_roll angles to find the best combined viewpoint."""
        print("[ActivePerception] Target low confidence. Starting Cartesian + wrist spatial optimization scan...")

        base_wrist_flex, base_wrist_roll = self.arm.get_current_wrist_angles_deg()

        best_pose = [base_x, base_y, base_z]
        best_wrist_roll = base_wrist_roll
        best_quality = 0.0
        actuation_count = 0

        candidate_offsets = [
            (0.0, 0.0, 0.0),
            (step_m, 0.0, 0.0),
            (-step_m, 0.0, 0.0),
            (0.0, step_m, 0.0),
            (0.0, -step_m, 0.0),
            (0.0, 0.0, step_m),
            (0.0, 0.0, -step_m),
        ]

        for dx, dy, dz in candidate_offsets:
            target_x = float(np.clip(base_x + dx, -0.05, 0.10))
            target_y = float(np.clip(base_y + dy, -0.22, -0.02))
            target_z = float(np.clip(base_z + dz, 0.16, 0.35))

            for roll_offset in wrist_roll_offsets_deg:
                candidate_roll = base_wrist_roll + roll_offset

                self.arm.move_to_cartesian_pose_with_wrist(
                    target_x, target_y, target_z,
                    wrist_roll_deg=candidate_roll,
                    duration=0.25,
                )
                actuation_count += 1
                time.sleep(0.1)

                frame = self.arm.get_wrist_camera_frame()
                if frame is None:
                    continue

                found, score, box = detect_target(frame, self.target_text, confidence_threshold=0.1)
                if not found:
                    continue

                quality = self.calculate_quality_metric(score, box, frame.size)
                print(
                    f" -> Position: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}] "
                    f"| Wrist roll: {candidate_roll:.1f} deg | Score: {score:.3f} | Quality: {quality:.3f} "
                    f"| actuation #{actuation_count}"
                )

                if quality > best_quality:
                    best_quality = quality
                    best_pose = [target_x, target_y, target_z]
                    best_wrist_roll = candidate_roll

                if best_quality >= self.confidence_threshold:
                    print(f"[ActivePerception] Optimum certainty threshold met ({best_quality:.3f}).")
                    break

            if best_quality >= self.confidence_threshold:
                break

        # Final move to optimal Cartesian + wrist configuration also counts as an actuation
        self.arm.move_to_cartesian_pose_with_wrist(*best_pose, wrist_roll_deg=best_wrist_roll, duration=0.3)
        actuation_count += 1

        print(f"[ActivePerception] Optimization stage used {actuation_count} actuation(s).")
        return best_pose, best_wrist_roll, best_quality, actuation_count

    def run(self) -> tuple[list[float], float]:
        """Executes full active perception loop using standard solve_ik."""
        print("[ActivePerception] Capturing overhead initialization frame...")
        overhead_frame = self.arm.get_overhead_camera_frame()
        if overhead_frame is not None:
            found, score, box = detect_target(overhead_frame, self.target_text, confidence_threshold=0.25)
            if found:
                print(f"[ActivePerception] Overhead detection found target (score: {score:.3f}).")

        print("Centering arm to object")
        score, box, align_actuations = self.align_wrist_camera_2d()

        curr_x, curr_y, curr_z = self.arm.get_current_cartesian_pose()
        wrist_frame = self.arm.get_wrist_camera_frame()

        current_quality = 0.0
        if wrist_frame is not None and box is not None:
            current_quality = self.calculate_quality_metric(score, box, wrist_frame.size)

        optimize_actuations = 0
        if current_quality < self.confidence_threshold:
            print("Finding optimal viewpoint...")
            best_pose, best_wrist_roll, final_quality, optimize_actuations = self.optimize_viewpoint_positions(
                curr_x, curr_y, curr_z
            )
        else:
            best_pose = [curr_x, curr_y, curr_z]
            best_wrist_roll = self.arm.get_current_wrist_angles_deg()[1]
            final_quality = current_quality

        total_actuations = align_actuations + optimize_actuations
        # Stored for external inspection without changing this method's return signature.
        self.last_run_actuation_counts = {
            "align": align_actuations,
            "optimize": optimize_actuations,
            "total": total_actuations,
        }

        print(
            f"[ActivePerception] Execution complete. Final Pose: {best_pose} with Quality: {final_quality:.3f} "
            f"| Actuations — align: {align_actuations}, optimize: {optimize_actuations}, total: {total_actuations}"
        )
        return best_pose, final_quality

if __name__ == "__main__":
    import os
    import lerobot

    lerobot_dir = os.path.dirname(lerobot.__file__)
    urdf_path = os.path.join(lerobot_dir, "configs", "so101_cleaned.urdf")

    arm_controller = SO101ArmController(urdf_path=urdf_path, port="/dev/ttyACM0")

    rest_joint_angles = [-3.0, -110.0, 96.0, 62.0, -2.0, 0.5]
    home_joint_angles = [-12.0, -52.0, 40.0,74.0, 0.0]

    try:
        print("Moving to baseline joint pose...")
        arm_controller.move_to_joint_angles(home_joint_angles, duration=2.0)
        
        print("Executing active perception...")
        perception = ActivePerceptionController(
            arm=arm_controller,
            target_text="green screwdriver",
            confidence_threshold=0.75
        )
        optimal_pose, certainty = perception.run()
        print(perception.last_run_actuation_counts)
        
        print("Returning to home pose...")
        arm_controller.move_to_joint_angles(home_joint_angles, duration=2.0)
        
        print("Returning to rest pose...")
        arm_controller.move_to_joint_angles(rest_joint_angles, duration=2.0)

    finally:
        arm_controller.disconnect()
