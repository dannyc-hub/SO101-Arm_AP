import os
import time
import json
import csv
from datetime import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from arm_controller import SO101ArmController
from detection import detect_target


# Define candidate grid matching candidate_offsets x wrist_roll_offsets_deg (7 * 5 = 35)
CANDIDATE_GRID = [
    (dx, dy, dz, roll)
    for dx, dy, dz in [
        (0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (-0.03, 0.0, 0.0),
        (0.0, 0.03, 0.0), (0.0, -0.03, 0.0), (0.0, 0.0, 0.03), (0.0, 0.0, -0.03)
    ]
    for roll in [-45.0, -20.0, 0.0, 20.0, 45.0]
]

class NBVModel(nn.Module):
    def __init__(self, num_classes: int = 35, aux_dim: int = 5):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=None)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(576 + aux_dim, 128), # 576 + 5 = 581
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, aux_feature: torch.Tensor = None) -> torch.Tensor:
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Shape: (batch_size, 576)

        if aux_feature is not None:
            x = torch.cat([x, aux_feature], dim=1)  # Shape: (batch_size, 581)

        x = self.head(x)
        return x


# Image transformations matching model training parameters
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_trained_nbv_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Instantiates model and loads trained PyTorch state dictionary."""
    model = NBVModel(num_classes=len(CANDIDATE_GRID), aux_dim=5)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Model Loader] Successfully loaded weights from '{checkpoint_path}'")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model


class ActivePerceptionController:
    def __init__(
        self,
        arm: SO101ArmController,
        target_text: str,
        confidence_threshold: float = 0.80,
        target_bbox_ratio: float = 0.30,
        max_iterations: int = 10,
        log_dir: str = "nbv_training_data",
    ):
        """Active perception optimizer using Image-Based Visual Servoing, Cartesian offset sampling, and ML models.

        :param arm: Connected SO101ArmController instance.
        :param target_text: Text prompt for OWLv2 detection (e.g., "screwdriver").
        :param confidence_threshold: Stopping threshold for OWLv2 score.
        :param target_bbox_ratio: Desired target bounding box area relative to camera frame area.
        :param max_iterations: Maximum alignment iterations per search phase.
        :param log_dir: Output directory for image and jsonl training data logs.
        """
        self.arm = arm
        self.target_text = target_text
        self.confidence_threshold = confidence_threshold
        self.target_bbox_ratio = target_bbox_ratio
        self.max_iterations = max_iterations

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sample_idx = 0
        self.last_run_actuation_counts = {"align": 0, "optimize": 0, "total": 0}

    def _log_sample(
        self,
        frame: Image.Image,
        base_pose: list[float],
        offset: tuple[float, float, float],
        wrist_roll_offset: float,
        score: float,
        quality: float,
        found: bool,
        base_frame_name: str = None,
    ):
        """Saves one (state, action, outcome) tuple for NBV model training."""
        sample_name = f"{self.episode_id}_{self.sample_idx:04d}"
        frame.save(os.path.join(self.log_dir, f"{sample_name}.jpg"))

        record = {
            "image_file": f"{sample_name}.jpg",
            "base_pose": base_pose,
            "offset": offset,
            "wrist_roll_offset_deg": wrist_roll_offset,
            "score": score,
            "quality": quality,
            "found": found,
            "base_frame_name": base_frame_name,
            "target_text": self.target_text,
        }
        with open(os.path.join(self.log_dir, "labels.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        self.sample_idx += 1

    def _build_aux_features(self, frame: Image.Image, box: list[float], score: float) -> list[float]:
        """Generates auxiliary features normalized identically to training."""
        if box is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        img_w, img_h = frame.size
        box_cx = float(((box[0] + box[2]) / 2.0) / img_w)
        box_cy = float(((box[1] + box[3]) / 2.0) / img_h)
        box_w = float((box[2] - box[0]) / img_w)
        box_h = float((box[3] - box[1]) / img_h)
        return [box_cx, box_cy, box_w, box_h, float(score)]

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

        base_frame = self.arm.get_wrist_camera_frame()
        base_found, base_score, base_box = detect_target(base_frame, self.target_text, confidence_threshold=0.1)
        base_frame_name = f"{self.episode_id}_{self.sample_idx:04d}_base.jpg"
        if base_frame is not None:
            base_frame.save(os.path.join(self.log_dir, base_frame_name))
        self.sample_idx += 1

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
                quality = self.calculate_quality_metric(score, box, frame.size) if found else 0.0

                self._log_sample(
                    frame,
                    [base_x, base_y, base_z],
                    (dx, dy, dz),
                    roll_offset,
                    score,
                    quality,
                    found,
                    base_frame_name=base_frame_name,
                )

                if not found:
                    continue

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

        # Final move to optimal Cartesian + wrist configuration
        self.arm.move_to_cartesian_pose_with_wrist(*best_pose, wrist_roll_deg=best_wrist_roll, duration=0.3)
        actuation_count += 1

        print(f"[ActivePerception] Optimization stage used {actuation_count} actuation(s).")
        return best_pose, best_wrist_roll, best_quality, actuation_count

    def optimize_viewpoint_positions_learned(
        self,
        base_x: float,
        base_y: float,
        base_z: float,
        model: nn.Module,
        top_k: int = 3,
        device: str = "cpu",
    ) -> tuple[list[float], float, float, int]:
        """Evaluates trained model candidates (top-k) to quickly find the optimal viewpoint."""
        frame = self.arm.get_wrist_camera_frame()
        found, score, box = detect_target(frame, self.target_text, confidence_threshold=0.1)
        aux = self._build_aux_features(frame, box, score)

        image_t = transform(frame).unsqueeze(0).to(device)
        aux_t = torch.tensor(aux, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_t, aux_t)
            top_indices = torch.topk(logits, top_k).indices[0].tolist()

        actuation_count = 0
        best_pose, best_wrist_roll, best_quality = [base_x, base_y, base_z], 0.0, 0.0

        for idx in top_indices:
            dx, dy, dz, roll = CANDIDATE_GRID[idx]
            target_x = float(np.clip(base_x + dx, -0.05, 0.10))
            target_y = float(np.clip(base_y + dy, -0.22, -0.02))
            target_z = float(np.clip(base_z + dz, 0.16, 0.35))

            self.arm.move_to_cartesian_pose_with_wrist(target_x, target_y, target_z, wrist_roll_deg=roll, duration=0.25)
            actuation_count += 1

            frame = self.arm.get_wrist_camera_frame()
            found, score, box = detect_target(frame, self.target_text, confidence_threshold=0.1)
            quality = self.calculate_quality_metric(score, box, frame.size) if found else 0.0

            if quality > best_quality:
                best_quality = quality
                best_pose = [target_x, target_y, target_z]
                best_wrist_roll = roll

            if best_quality >= self.confidence_threshold:
                break

        return best_pose, best_wrist_roll, best_quality, actuation_count

    def run(
        self,
        model: nn.Module = None,
        use_learned: bool = True,
        device: str = "cpu",
    ) -> tuple[list[float], float]:
        """Executes full active perception loop using standard or learned optimization search."""
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
            if use_learned and model is not None:
                best_pose, best_wrist_roll, final_quality, optimize_actuations = (
                    self.optimize_viewpoint_positions_learned(
                        curr_x, curr_y, curr_z, model=model, device=device
                    )
                )
            else:
                best_pose, best_wrist_roll, final_quality, optimize_actuations = (
                    self.optimize_viewpoint_positions(curr_x, curr_y, curr_z)
                )
        else:
            best_pose = [curr_x, curr_y, curr_z]
            best_wrist_roll = self.arm.get_current_wrist_angles_deg()[1]
            final_quality = current_quality

        total_actuations = align_actuations + optimize_actuations
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


def save_results_to_csv_and_json(
    episode_records: list[dict],
    csv_filepath: str = "active_perception_results.csv",
    json_filepath: str = "active_perception_results.json",
):
    """Saves structured experiment metadata and metrics to CSV and JSON files by appending."""
    if not episode_records:
        print("[Data Export] No episode records available to export.")
        return

    headers = [
        "episode",
        "target_text",
        "target_x",
        "target_y",
        "target_z",
        "num_clutter_objects",
        "learned_align_actuations",
        "learned_opt_actuations",
        "learned_total_actuations",
        "learned_quality",
        "heuristic_align_actuations",
        "heuristic_opt_actuations",
        "heuristic_total_actuations",
        "heuristic_quality",
    ]

    file_exists = os.path.exists(csv_filepath)
    try:
        with open(csv_filepath, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerows(episode_records)
        print(f"[Data Export] Successfully appended metrics to CSV spreadsheet: '{csv_filepath}'")
    except Exception as e:
        print(f"[Data Export] Failed to save CSV file: {e}")

    existing_json = []
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, "r") as json_file:
                existing_json = json.load(json_file)
        except Exception:
            existing_json = []

    existing_json.extend(episode_records)

    try:
        with open(json_filepath, mode="w") as json_file:
            json.dump(existing_json, json_file, indent=4)
        print(f"[Data Export] Successfully exported metrics to JSON: '{json_filepath}'")
    except Exception as e:
        print(f"[Data Export] Failed to save JSON file: {e}")


if __name__ == "__main__":
    import lerobot

    MODEL_PATH = "nbv_best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nbv_model = None
    if os.path.exists(MODEL_PATH):
        nbv_model = load_trained_nbv_model(MODEL_PATH, device=device)
    else:
        print(f"[Warning] '{MODEL_PATH}' not found. Execution will default to heuristic search mode.")

    lerobot_dir = os.path.dirname(lerobot.__file__)
    urdf_path = os.path.join(lerobot_dir, "configs", "so101_cleaned.urdf")

    arm_controller = SO101ArmController(urdf_path=urdf_path, port="/dev/ttyACM0")

    rest_joint_angles = [-3.0, -110.0, 96.0, 62.0, -2.0, 0.5]
    home_joint_angles = [-12.0, -52.0, 40.0, 74.0, 0.0]

    try:
        print("Moving to baseline joint pose...")
        arm_controller.move_to_joint_angles(home_joint_angles, duration=2.0)

        target_text = "tape roll"
        curr_x, curr_y, curr_z = arm_controller.get_current_cartesian_pose()

        record = {
            "episode": 1,
            "target_text": target_text,
            "target_x": round(curr_x, 4),
            "target_y": round(curr_y, 4),
            "target_z": round(curr_z, 4),
            "num_clutter_objects": 0,
            "learned_align_actuations": None,
            "learned_opt_actuations": None,
            "learned_total_actuations": None,
            "learned_quality": None,
            "heuristic_align_actuations": None,
            "heuristic_opt_actuations": None,
            "heuristic_total_actuations": None,
            "heuristic_quality": None,
        }

        csv_filepath = "active_perception_results.csv"
        episode_num = 1
        if os.path.exists(csv_filepath):
            try:
                with open(csv_filepath, mode="r", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        episode_num = int(rows[-1].get("episode", 0)) + 1
            except Exception:
                pass
        record["episode"] = episode_num

        # 1. Learned Model Loop (if model present)
        if nbv_model is not None:
            print("\n--- Executing Active Perception with Learned Model ---")
            perception_learned = ActivePerceptionController(
                arm=arm_controller,
                target_text=target_text,
                confidence_threshold=0.75,
                log_dir="real_arm_test_data",
            )
            _, l_qual = perception_learned.run(model=nbv_model, use_learned=True, device=device)
            counts_l = perception_learned.last_run_actuation_counts

            record["learned_align_actuations"] = counts_l["align"]
            record["learned_opt_actuations"] = counts_l["optimize"]
            record["learned_total_actuations"] = counts_l["total"]
            record["learned_quality"] = round(l_qual, 4)

        # Reset pose between runs
        print("\nReturning to home pose before candidate sweep...")
        arm_controller.move_to_joint_angles(home_joint_angles, duration=2.0)
        time.sleep(1.0)

        # 2. Heuristic Full Candidate Sweep Loop
        print("\n--- Executing Active Perception with Full Candidate Sweep (Heuristic) ---")
        perception_heuristic = ActivePerceptionController(
            arm=arm_controller,
            target_text=target_text,
            confidence_threshold=0.75,
            log_dir="real_arm_test_data",
        )
        _, h_qual = perception_heuristic.run(use_learned=False)
        counts_h = perception_heuristic.last_run_actuation_counts

        record["heuristic_align_actuations"] = counts_h["align"]
        record["heuristic_opt_actuations"] = counts_h["optimize"]
        record["heuristic_total_actuations"] = counts_h["total"]
        record["heuristic_quality"] = round(h_qual, 4)

        # Save results by appending
        save_results_to_csv_and_json([record], csv_filepath=csv_filepath, json_filepath="active_perception_results.json")

        print("\nReturning to home pose...")
        arm_controller.move_to_joint_angles(home_joint_angles, duration=2.0)

        print("Returning to rest pose...")
        arm_controller.move_to_joint_angles(rest_joint_angles, duration=2.0)

    finally:
        arm_controller.disconnect()