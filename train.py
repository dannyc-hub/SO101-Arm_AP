import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


CANDIDATE_GRID = [
    (dx, dy, dz, roll)
    for dx, dy, dz in [(0,0,0),(0.03,0,0),(-0.03,0,0),(0,0.03,0),(0,-0.03,0),(0,0,0.03),(0,0,-0.03)]
    for roll in [-45.0, -20.0, 0.0, 20.0, 45.0]
]


def offset_to_class_index(offset, roll):
    key = tuple(offset) + (roll,)
    for i, c in enumerate(CANDIDATE_GRID):
        if np.allclose(c, key, atol=1e-6):
            return i
    raise ValueError(f"Offset {key} not found in CANDIDATE_GRID")


def load_and_group_samples(log_dir="sim_training_data"):
    groups = defaultdict(list)
    with open(f"{log_dir}/labels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            key = (r["image_file"].rsplit("_", 1)[0], tuple(r["base_pose"]))
            groups[key].append(r)

    training_examples = []
    for key, candidates in groups.items():
        if len(candidates) < 2:
            continue

        score_vector = np.zeros(len(CANDIDATE_GRID), dtype=np.float32)
        has_positive = False
        
        for cand in candidates:
            try:
                idx = offset_to_class_index(cand["offset"], cand["wrist_roll_offset_deg"])
                score_vector[idx] = cand["quality"]
                if cand["quality"] > 0:
                    has_positive = True
            except ValueError:
                continue

        if not has_positive:
            continue

        sample = candidates[0]
        training_examples.append({
            "base_image": sample["base_frame_name"],
            "box_cx_norm": sample["base_box_cx_norm"],
            "box_cy_norm": sample["base_box_cy_norm"],
            "box_w_norm": sample["base_box_w_norm"],
            "box_h_norm": sample["base_box_h_norm"],
            "score": sample["base_score"],
            "target_scores": score_vector,
        })
    return training_examples


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NBVRegressionDataset(Dataset):
    def __init__(self, examples, log_dir, transform=None):
        self.examples = examples
        self.log_dir = log_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = Image.open(f"{self.log_dir}/{ex['base_image']}").convert("RGB")
        image = self.transform(image)

        aux = torch.tensor([
            ex["box_cx_norm"], ex["box_cy_norm"],
            ex["box_w_norm"], ex["box_h_norm"], ex["score"],
        ], dtype=torch.float32)

        target_scores = torch.tensor(ex["target_scores"], dtype=torch.float32)
        return image, aux, target_scores


class NBVScoringModel(nn.Module):
    def __init__(self, num_candidates=35, aux_dim=5):
        super().__init__()
        backbone = mobilenet_v3_small(weights="DEFAULT")
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(576 + aux_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_candidates),
        )

    def forward(self, image, aux):
        feats = self.pool(self.backbone(image)).flatten(1)
        return self.head(torch.cat([feats, aux], dim=1))


def combined_criterion(pred_scores, target_scores, alpha=1.0, beta=1.0, temperature=1.0):
    """
    Combines MSE Loss (for absolute magnitude matching) with 
    KL Divergence Loss (as a listwise ranking loss on softmax distributions).
    """
    loss_mse = F.mse_loss(pred_scores, target_scores)
    
    pred_log_prob = F.log_softmax(pred_scores / temperature, dim=-1)
    target_prob = F.softmax(target_scores / temperature, dim=-1)
    loss_rank = F.kl_div(pred_log_prob, target_prob, reduction='batchmean')
    
    return alpha * loss_mse + beta * loss_rank


if __name__ == "__main__":
    examples = load_and_group_samples()
    print(f"Loaded {len(examples)} training examples")
    if len(examples) == 0:
        raise RuntimeError("No training examples found.")

    dataset = NBVRegressionDataset(examples, log_dir="sim_training_data")

    train_size = int(0.85 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NBVScoringModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    ALPHA = 1.0  
    BETA = 0.5   
    TEMPERATURE = 1.0

    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    for epoch in range(50):
        model.train()
        total_loss = 0
        for images, aux, target_scores in train_loader:
            images, aux, target_scores = images.to(device), aux.to(device), target_scores.to(device)
            optimizer.zero_grad()
            pred_scores = model(images, aux)
            
            loss = combined_criterion(pred_scores, target_scores, alpha=ALPHA, beta=BETA, temperature=TEMPERATURE)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct, correct_top3, total = 0, 0, 0
        with torch.no_grad():
            for images, aux, target_scores in val_loader:
                images, aux, target_scores = images.to(device), aux.to(device), target_scores.to(device)
                pred_scores = model(images, aux)
                
                val_loss += combined_criterion(pred_scores, target_scores, alpha=ALPHA, beta=BETA, temperature=TEMPERATURE).item()

                true_best = target_scores.argmax(dim=1)
                pred_best = pred_scores.argmax(dim=1)
                top3 = pred_scores.topk(3, dim=1).indices

                correct += (pred_best == true_best).sum().item()
                correct_top3 += sum(true_best[i] in top3[i] for i in range(len(true_best)))
                total += len(true_best)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        top1 = correct / total
        top3 = correct_top3 / total

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.3f} val_loss={avg_val_loss:.3f} "
              f"top1_acc={top1:.3f} top3_acc={top3:.3f} lr={current_lr:.1e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "nbv_best_model.pt")
            print(f"  -> Saved new best model (val_loss: {best_val_loss:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.3f}")
                break

    print("Training finished. Best model saved to nbv_best_model.pt")