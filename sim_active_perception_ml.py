import time
import numpy as np
from PIL import Image

import json
import os
import csv
from datetime import datetime

import pybullet as p
import pybullet_data

from arm_controller import SO101ArmController
from sim_arm_controller import SimSO101ArmController
from detection import detect_target

from torchvision import transforms
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

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
    def __init__(self, num_classes=35, aux_dim=5):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(576 + aux_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, aux):
        feats = self.pool(self.backbone(image)).flatten(1)
        return self.head(torch.cat([feats, aux], dim=1))


# Match image transformations used during model training
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
    model.eval()  # Put model into evaluation mode
    return model

# Object spawn options using consistent string identifiers
SHAPE_OPTIONS = ["box", "cylinder", "sphere", "rectangular_prism"]
COLOR_OPTIONS = {
    "red":    [0.9, 0.1, 0.1, 1],
    "blue":   [0.2, 0.2, 0.9, 1],
    "yellow": [0.9, 0.9, 0.1, 1],
    "green":  [0.1, 0.8, 0.1, 1],
    "purple": [0.6, 0.2, 0.8, 1],
    "orange": [0.9, 0.5, 0.1, 1],
    "grey":   [0.5, 0.5, 0.5, 1],
}

def _create_primitive_body(shape, color_rgba, position, orientation, size, mass=0.02):
    """Shared helper — builds one PyBullet primitive body of the given
    shape/color/size at a pose. Uses mass=0.02 so physics resolves overlaps,
    plus high damping/friction to prevent spheres from rolling away."""
    if shape == "box":
        half_extents = [size, size, size]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color_rgba)
    elif shape == "rectangular_prism":
        half_extents = [
            np.random.uniform(0.01, 0.04),
            np.random.uniform(0.005, 0.02),
            np.random.uniform(0.01, 0.03)
        ]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color_rgba)
    elif shape == "cylinder":
        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=size, height=size * 2)
        vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=size, length=size * 2, rgbaColor=color_rgba)
    else:  # "sphere"
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
        vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color_rgba)

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=position,
        baseOrientation=orientation,
    )

    # Prevent spheres from rolling away while allowing physics to resolve overlaps
    p.changeDynamics(
        body_id, -1,
        linearDamping=2.0,
        angularDamping=2.0,
        lateralFriction=1.5
    )

    return body_id

def spawn_random_target(
    shape=None,
    color_name=None,
    workspace_x=(0.0, 0.11),
    workspace_y=(-0.21, 0.0),
    z=0.02,
):
    """Spawns the actual detection target."""
    if shape is None:
        shape = np.random.choice(SHAPE_OPTIONS)
    if color_name is None:
        color_name = np.random.choice(list(COLOR_OPTIONS.keys()))

    x = np.random.uniform(*workspace_x)
    y = np.random.uniform(*workspace_y)
    yaw = np.random.uniform(0, 2 * np.pi)
    size = np.random.uniform(0.008, 0.04)
    orientation = p.getQuaternionFromEuler([0, 0, yaw])

    obj_id = _create_primitive_body(shape, COLOR_OPTIONS[color_name], [x, y, z], orientation, size)

    return obj_id, (x, y, z), shape, color_name

def spawn_clutter_objects(
    target_pos,
    target_shape,
    target_color_name,
    num_objects: int = None,
    min_count: int = 2,
    max_count: int = 6,
    min_dist_from_target: float = 0.035,
    min_dist_between_clutter: float = 0.025,
    workspace_x=(-0.04, 0.09),
    workspace_y=(-0.21, -0.03),
    z=0.02,
):
    """Spawns distractor objects around the target."""
    if num_objects is None:
        num_objects = np.random.randint(min_count, max_count + 1)

    available_colors = list(COLOR_OPTIONS.keys())
    target_xy = np.array(target_pos[:2])

    clutter_ids = []
    attempts = 0
    max_attempts = num_objects * 15

    while len(clutter_ids) < num_objects and attempts < max_attempts:
        attempts += 1

        x = np.random.uniform(*workspace_x)
        y = np.random.uniform(*workspace_y)

        if np.linalg.norm(np.array([x, y]) - target_xy) < min_dist_from_target:
            continue

        too_close_to_existing = False
        for existing_id in clutter_ids:
            existing_pos, _ = p.getBasePositionAndOrientation(existing_id)
            if np.linalg.norm(np.array([x, y]) - np.array(existing_pos[:2])) < min_dist_between_clutter:
                too_close_to_existing = True
                break
        if too_close_to_existing:
            continue

        shape = np.random.choice(SHAPE_OPTIONS)
        color_name = np.random.choice(available_colors)
        if shape == target_shape and color_name == target_color_name:
            other_colors = [c for c in available_colors if c != target_color_name]
            color_name = np.random.choice(other_colors)

        size = np.random.uniform(0.008, 0.04)
        yaw = np.random.uniform(0, 2 * np.pi)
        orientation = p.getQuaternionFromEuler([0, 0, yaw])

        obj_id = _create_primitive_body(shape, COLOR_OPTIONS[color_name], [x, y, z], orientation, size)
        clutter_ids.append(obj_id)

    return clutter_ids

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

    def _log_sample(self, frame, base_pose, offset, wrist_roll_offset, score, quality, found, base_frame_name=None):
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
    
    def calculate_quality_metric(self, score: float, box: list[float], img_size: tuple[int, int]) -> float:
        """Calculates combined certainty score incorporating detection confidence and optimal visual scale."""
        if score == 0.0 or box is None:
            return 0.0

        img_w, img_h = img_size
        img_area = img_w * img_h

        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        area_ratio = (box_w * box_h) / img_area

        size_penalty = 1.0 - min(abs(area_ratio - self.target_bbox_ratio) / self.target_bbox_ratio, 1.0)
        quality = (0.7 * score) + (0.3 * size_penalty)
        return float(quality)

    def align_wrist_camera_2d(self, kp_xy: float = 0.01, kp_z: float = 0.01) -> tuple[float, list[float], int]:
        """Centering loop (Image-Based Visual Servoing)."""
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
        """Full heuristic grid search across offsets and wrist roll angles."""
        print("[ActivePerception] Target low confidence. Starting Cartesian + wrist spatial optimization scan...")

        base_wrist_flex, base_wrist_roll = self.arm.get_current_wrist_angles_deg()

        base_frame = self.arm.get_wrist_camera_frame()
        base_found, base_score, base_box = detect_target(base_frame, self.target_text, confidence_threshold=0.1)
        base_frame_name = f"{self.episode_id}_{self.sample_idx:04d}_base.jpg"
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
                
                self._log_sample(frame, [base_x, base_y, base_z], (dx, dy, dz), roll_offset, score, quality, found, base_frame_name=base_frame_name)
                
                if not found:
                    continue
                
                if quality > best_quality:
                    best_quality = quality
                    best_pose = [target_x, target_y, target_z]
                    best_wrist_roll = candidate_roll

                if best_quality >= self.confidence_threshold:
                    break

            if best_quality >= self.confidence_threshold:
                break

        self.arm.move_to_cartesian_pose_with_wrist(*best_pose, wrist_roll_deg=best_wrist_roll, duration=0.3)
        actuation_count += 1

        print(f"[ActivePerception] Optimization stage used {actuation_count} actuation(s).")
        return best_pose, best_wrist_roll, best_quality, actuation_count

    def _build_aux_features(self, frame, box, score):
        """Generates auxiliary features normalized identically to training."""
        if box is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        img_w, img_h = frame.size
        box_cx = float(((box[0] + box[2]) / 2.0) / img_w)
        box_cy = float(((box[1] + box[3]) / 2.0) / img_h)
        box_w = float((box[2] - box[0]) / img_w)
        box_h = float((box[3] - box[1]) / img_h)
        return [box_cx, box_cy, box_w, box_h, float(score)]


    def optimize_viewpoint_positions_learned(self, base_x, base_y, base_z, model, top_k=3, device="cpu"):
        """Evaluates model top-k candidates before executing full sweep if necessary."""
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

    def run(self, model: nn.Module = None, use_learned: bool = True, device: str = "cpu") -> tuple[list[float], float]:
        """Executes full active perception loop."""
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
                best_pose, best_wrist_roll, final_quality, optimize_actuations = self.optimize_viewpoint_positions_learned(
                    curr_x, curr_y, curr_z, model=model, device=device
                )
            else:
                best_pose, best_wrist_roll, final_quality, optimize_actuations = self.optimize_viewpoint_positions(
                    curr_x, curr_y, curr_z
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


def save_results_to_csv_and_json(episode_records, csv_filepath="sim_active_perception_results.csv", json_filepath="sim_active_perception_results.json"):
    """Saves structured experiment metadata and metrics to CSV and JSON files."""
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
        "heuristic_quality"
    ]

    try:
        with open(csv_filepath, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(episode_records)
        print(f"[Data Export] Successfully exported metrics to CSV spreadsheet: '{csv_filepath}'")
    except Exception as e:
        print(f"[Data Export] Failed to save CSV file: {e}")

    try:
        with open(json_filepath, mode="w") as json_file:
            json.dump(episode_records, json_file, indent=4)
        print(f"[Data Export] Successfully exported metrics to JSON: '{json_filepath}'")
    except Exception as e:
        print(f"[Data Export] Failed to save JSON file: {e}")


if __name__ == "__main__":
    import lerobot

    NUM_EPISODES = 100
    ENABLE_CLUTTER = True
    MODEL_PATH = "/home/danny/thesis_scripts/nbv_best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device', device)

    nbv_model = None
    if os.path.exists(MODEL_PATH):
        nbv_model = load_trained_nbv_model(MODEL_PATH, device=device)
    else:
        print(f"[Warning] '{MODEL_PATH}' not found. Test will run in heuristic mode only.")

    lerobot_dir = os.path.dirname(lerobot.__file__)
    urdf_path = os.path.join(lerobot_dir, "configs", "so101_cleaned.urdf")

    sim_arm = SimSO101ArmController(urdf_path=urdf_path, gui=False)
    home_joint_angles = [-12.0, -52.0, 40.0, 74.0, 0.0]

    SHAPE_NAMES = {
        "box": "cube",
        "cylinder": "cylinder",
        "sphere": "sphere",
        "rectangular_prism": "rectangular prism",
    }

    episode_data_records = []

    try:
        results = {"heuristic": [], "learned": []}
        for episode in range(NUM_EPISODES):
            try:
                obj_id, obj_pos, obj_shape, obj_color_name = spawn_random_target()
                target_text = f"{obj_color_name} {SHAPE_NAMES[obj_shape]}"
                print(f"\n--- Episode {episode + 1}/{NUM_EPISODES}: Target '{target_text}' at {obj_pos} ---")

                clutter_ids = []
                if ENABLE_CLUTTER:
                    clutter_ids = spawn_clutter_objects(
                        target_pos=obj_pos,
                        target_shape=obj_shape,
                        target_color_name=obj_color_name,
                    )
                    print(f"Spawned {len(clutter_ids)} clutter object(s)")

                record = {
                    "episode": episode + 1,
                    "target_text": target_text,
                    "target_x": round(obj_pos[0], 4),
                    "target_y": round(obj_pos[1], 4),
                    "target_z": round(obj_pos[2], 4),
                    "num_clutter_objects": len(clutter_ids),
                    "learned_align_actuations": None,
                    "learned_opt_actuations": None,
                    "learned_total_actuations": None,
                    "learned_quality": None,
                    "heuristic_align_actuations": None,
                    "heuristic_opt_actuations": None,
                    "heuristic_total_actuations": None,
                    "heuristic_quality": None,
                }

                # Test 1: Learned Model Execution (if model present)
                if nbv_model is not None:
                    sim_arm.move_to_joint_angles(home_joint_angles, duration=1.0)
                    perception_learned = ActivePerceptionController(
                        arm=sim_arm,
                        target_text=target_text,
                        confidence_threshold=0.75,
                        log_dir="sim_test_data",
                    )
                    _, l_qual = perception_learned.run(model=nbv_model, use_learned=True, device=device)
                    
                    counts_l = perception_learned.last_run_actuation_counts
                    results["learned"].append(counts_l["optimize"])

                    record["learned_align_actuations"] = counts_l["align"]
                    record["learned_opt_actuations"] = counts_l["optimize"]
                    record["learned_total_actuations"] = counts_l["total"]
                    record["learned_quality"] = round(l_qual, 4)

                # Test 2: Heuristic Baseline Execution
                sim_arm.move_to_joint_angles(home_joint_angles, duration=1.0)
                perception_heuristic = ActivePerceptionController(
                    arm=sim_arm,
                    target_text=target_text,
                    confidence_threshold=0.75,
                    log_dir="sim_test_data",
                )
                _, h_qual = perception_heuristic.run(use_learned=False)
                
                counts_h = perception_heuristic.last_run_actuation_counts
                results["heuristic"].append(counts_h["optimize"])

                record["heuristic_align_actuations"] = counts_h["align"]
                record["heuristic_opt_actuations"] = counts_h["optimize"]
                record["heuristic_total_actuations"] = counts_h["total"]
                record["heuristic_quality"] = round(h_qual, 4)

                episode_data_records.append(record)
            
            except Exception as e:
                print(f"[Episode {episode + 1}] FAILED: {e}")
            
            finally:
                p.removeBody(obj_id)
                for c_id in clutter_ids:
                    p.removeBody(c_id)

        print("\n=== Final Test Summary ===")
        if results["learned"]:
            print(f"Mean actuations (Learned Model): {np.mean(results['learned']):.2f}")
        print(f"Mean actuations (Heuristic Grid): {np.mean(results['heuristic']):.2f}")

        save_results_to_csv_and_json(episode_data_records)

    finally:
        sim_arm.disconnect()