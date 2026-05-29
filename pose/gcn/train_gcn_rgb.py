# =========================================================
# train_gcn.py
# ---------------------------------------------------------
# Purpose:
#   Train a GCN-based residual refiner for 14-joint SLP pose
#
# Input:
#   C:\Users\Konyang\SLP_project\gcn_dataset\train.npz
#   C:\Users\Konyang\SLP_project\gcn_dataset\val.npz
#   C:\Users\Konyang\SLP_project\gcn_dataset\test.npz
#
# Output:
#   C:\Users\Konyang\SLP_project\gcn_ckpt\
#       best_gcn.pt
#       last_gcn.pt
#       train_log.csv
#       test_metrics.json
#
# Requirements:
#   pip install torch numpy pandas matplotlib
# =========================================================

import os
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# 1. PATH SETTINGS
# ---------------------------------------------------------
DATA_ROOT = r"C:\Users\Konyang\SLP_project\gcn_dataset"
TRAIN_NPZ = os.path.join(DATA_ROOT, "train.npz")
VAL_NPZ   = os.path.join(DATA_ROOT, "val.npz")
TEST_NPZ  = os.path.join(DATA_ROOT, "test.npz")

OUT_ROOT = r"C:\Users\Konyang\SLP_project\gcn_ckpt"
BEST_PATH = os.path.join(OUT_ROOT, "best_gcn.pt")
LAST_PATH = os.path.join(OUT_ROOT, "last_gcn.pt")
LOG_CSV_PATH = os.path.join(OUT_ROOT, "train_log.csv")
TEST_JSON_PATH = os.path.join(OUT_ROOT, "test_metrics.json")


# ---------------------------------------------------------
# 2. TRAIN SETTINGS
# ---------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12

NUM_WORKERS = 0   # Windows에서는 0 추천
PIN_MEMORY = True if DEVICE == "cuda" else False

# residual 학습 안정화용
GRAD_CLIP_NORM = 1.0

# loss weights
L1_LOSS_WEIGHT = 1.0
BONE_LOSS_WEIGHT = 0.2

# confidence weighting
USE_CONF_WEIGHT = True
MIN_CONF_WEIGHT = 0.20
MAX_CONF_WEIGHT = 1.00

# node dropout / noise augmentation
USE_INPUT_AUG = True
INPUT_NOISE_STD = 0.01
CONF_DROPOUT_PROB = 0.10


# ---------------------------------------------------------
# 3. JOINT / GRAPH SETTINGS
# ---------------------------------------------------------
SLP14_NAMES = [
    "right_ankle", "right_knee", "right_hip",
    "left_hip", "left_knee", "left_ankle",
    "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist",
    "thorax", "head"
]

# 14-joint skeleton edges
EDGES = [
    (0, 1), (1, 2),           # right leg
    (5, 4), (4, 3),           # left leg
    (2, 3),                   # pelvis bridge
    (6, 7), (7, 8),           # right arm
    (11, 10), (10, 9),        # left arm
    (8, 12), (9, 12),         # shoulders -> thorax
    (2, 12), (3, 12),         # hips -> thorax
    (12, 13)                  # thorax -> head
]


# ---------------------------------------------------------
# 4. REPRODUCIBILITY
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ---------------------------------------------------------
# 5. GRAPH UTILS
# ---------------------------------------------------------
def build_adjacency(num_nodes: int, edges):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # self-loop
    for i in range(num_nodes):
        A[i, i] = 1.0

    # D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(D, 1e-6, None)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


A_NORM = build_adjacency(14, EDGES)


# ---------------------------------------------------------
# 6. DATASET
# ---------------------------------------------------------
class PoseGCNDataset(Dataset):
    def __init__(self, npz_path: str, augment: bool = False):
        data = np.load(npz_path, allow_pickle=True)

        self.X = data["X"].astype(np.float32)   # (N,14,3) [x,y,conf]
        self.Y = data["Y"].astype(np.float32)   # (N,14,2) GT xy
        self.augment = augment

        assert self.X.ndim == 3 and self.X.shape[1:] == (14, 3), f"Bad X shape: {self.X.shape}"
        assert self.Y.ndim == 3 and self.Y.shape[1:] == (14, 2), f"Bad Y shape: {self.Y.shape}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()   # (14,3)
        y = self.Y[idx].copy()   # (14,2)

        if self.augment and USE_INPUT_AUG:
            # 좌표 noise
            x[:, :2] += np.random.normal(0, INPUT_NOISE_STD, size=(14, 2)).astype(np.float32)

            # confidence dropout
            drop_mask = (np.random.rand(14) < CONF_DROPOUT_PROB).astype(np.float32)
            x[:, 2] = x[:, 2] * (1.0 - drop_mask)

            # clamp normalized coords softly
            x[:, :2] = np.clip(x[:, :2], -0.5, 1.5)

        return {
            "x": torch.from_numpy(x),  # (14,3)
            "y": torch.from_numpy(y),  # (14,2)
        }


# ---------------------------------------------------------
# 7. MODEL
# ---------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A):
        # x: (B, J, C)
        x = torch.matmul(A, x)   # (B, J, C) using broadcast
        x = self.linear(x)
        return x


class ResidualGCN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=2, num_layers=4, dropout=0.1):
        super().__init__()
        assert num_layers >= 2

        self.input_proj = GraphConv(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        self.output_proj = GraphConv(hidden_dim, out_dim)

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        # x: (B,14,3)
        h = self.input_proj(x, A)
        h = self.norms[0](h)
        h = self.act(h)
        h = self.dropout(h)

        for i, block in enumerate(self.blocks):
            res = h
            h = block(h, A)
            h = self.norms[i + 1](h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + res

        out = self.output_proj(h, A)  # residual delta: (B,14,2)
        return out


# ---------------------------------------------------------
# 8. LOSS
# ---------------------------------------------------------
def build_conf_weight(conf: torch.Tensor):
    """
    conf: (B,14)
    confidence 높을수록 weight를 너무 크게 주면
    낮은 confidence 관절 보정이 약해질 수 있어서
    적당히 clamp
    """
    w = conf.clamp(0.0, 1.0)
    w = MIN_CONF_WEIGHT + (MAX_CONF_WEIGHT - MIN_CONF_WEIGHT) * w
    return w  # (B,14)

def l1_joint_loss(pred_xy, gt_xy, conf=None):
    """
    pred_xy, gt_xy: (B,14,2)
    conf: (B,14)
    """
    diff = torch.abs(pred_xy - gt_xy).sum(dim=-1)  # (B,14)

    if USE_CONF_WEIGHT and conf is not None:
        w = build_conf_weight(conf)
        loss = (diff * w).mean()
    else:
        loss = diff.mean()

    return loss

def bone_length_loss(pred_xy, gt_xy):
    """
    bone length consistency
    """
    loss = 0.0
    count = 0
    for i, j in EDGES:
        pred_len = torch.norm(pred_xy[:, i] - pred_xy[:, j], dim=-1)
        gt_len = torch.norm(gt_xy[:, i] - gt_xy[:, j], dim=-1)
        loss = loss + torch.abs(pred_len - gt_len).mean()
        count += 1
    return loss / max(count, 1)

def compute_total_loss(pred_xy, gt_xy, conf):
    l1 = l1_joint_loss(pred_xy, gt_xy, conf)
    bone = bone_length_loss(pred_xy, gt_xy)
    total = L1_LOSS_WEIGHT * l1 + BONE_LOSS_WEIGHT * bone
    return total, l1.detach(), bone.detach()


# ---------------------------------------------------------
# 9. METRICS
# ---------------------------------------------------------
def mpjpe(pred_xy, gt_xy):
    """
    Mean Per Joint Position Error in normalized bbox coordinates
    pred_xy, gt_xy: (B,14,2)
    """
    err = torch.norm(pred_xy - gt_xy, dim=-1)  # (B,14)
    return err.mean()

def pck(pred_xy, gt_xy, thr=0.05):
    """
    normalized bbox frame 기준 PCK
    """
    err = torch.norm(pred_xy - gt_xy, dim=-1)  # (B,14)
    correct = (err < thr).float()
    return correct.mean()


# ---------------------------------------------------------
# 10. TRAIN / EVAL
# ---------------------------------------------------------
@dataclass
class EpochResult:
    loss: float
    l1: float
    bone: float
    mpjpe: float
    pck05: float

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_l1 = 0.0
    total_bone = 0.0
    total_mpjpe = 0.0
    total_pck05 = 0.0
    n_batches = 0

    A = A_NORM.to(DEVICE)

    for batch in loader:
        x = batch["x"].to(DEVICE)  # (B,14,3)
        y = batch["y"].to(DEVICE)  # (B,14,2)

        pred_input_xy = x[:, :, :2]
        conf = x[:, :, 2]

        residual = model(x, A)              # (B,14,2)
        pred_xy = pred_input_xy + residual  # residual learning

        loss, l1, bone = compute_total_loss(pred_xy, y, conf)
        cur_mpjpe = mpjpe(pred_xy, y)
        cur_pck05 = pck(pred_xy, y, thr=0.05)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += float(loss.item())
        total_l1 += float(l1.item())
        total_bone += float(bone.item())
        total_mpjpe += float(cur_mpjpe.item())
        total_pck05 += float(cur_pck05.item())
        n_batches += 1

    if n_batches == 0:
        return EpochResult(0, 0, 0, 0, 0)

    return EpochResult(
        loss=total_loss / n_batches,
        l1=total_l1 / n_batches,
        bone=total_bone / n_batches,
        mpjpe=total_mpjpe / n_batches,
        pck05=total_pck05 / n_batches,
    )


# ---------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("=" * 60)
    print("train_gcn.py")
    print("=" * 60)
    print("DEVICE:", DEVICE)
    print("TRAIN_NPZ:", TRAIN_NPZ)
    print("VAL_NPZ  :", VAL_NPZ)
    print("TEST_NPZ :", TEST_NPZ)

    train_ds = PoseGCNDataset(TRAIN_NPZ, augment=True)
    val_ds   = PoseGCNDataset(VAL_NPZ, augment=False)
    test_ds  = PoseGCNDataset(TEST_NPZ, augment=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples  : {len(val_ds)}")
    print(f"Test samples : {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    model = ResidualGCN(
        in_dim=3,
        hidden_dim=64,
        out_dim=2,
        num_layers=4,
        dropout=0.10
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4
    )

    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_res = run_epoch(model, train_loader, optimizer=optimizer)
        val_res = run_epoch(model, val_loader, optimizer=None)

        scheduler.step(val_res.loss)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_res.loss,
            "train_l1": train_res.l1,
            "train_bone": train_res.bone,
            "train_mpjpe": train_res.mpjpe,
            "train_pck05": train_res.pck05,
            "val_loss": val_res.loss,
            "val_l1": val_res.l1,
            "val_bone": val_res.bone,
            "val_mpjpe": val_res.mpjpe,
            "val_pck05": val_res.pck05,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"LR={current_lr:.6f} | "
            f"Train Loss={train_res.loss:.6f}, MPJPE={train_res.mpjpe:.6f}, PCK@0.05={train_res.pck05:.4f} | "
            f"Val Loss={val_res.loss:.6f}, MPJPE={val_res.mpjpe:.6f}, PCK@0.05={val_res.pck05:.4f}"
        )

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_res.loss,
                "config": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "dropout": 0.10,
                    "in_dim": 3,
                    "out_dim": 2
                }
            },
            LAST_PATH
        )

        # save best
        if val_res.loss < best_val_loss:
            best_val_loss = val_res.loss
            best_epoch = epoch
            early_stop_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_res.loss,
                    "config": {
                        "hidden_dim": 64,
                        "num_layers": 4,
                        "dropout": 0.10,
                        "in_dim": 3,
                        "out_dim": 2
                    }
                },
                BEST_PATH
            )
            print(f"  -> [BEST] saved to {BEST_PATH}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print(f"[EARLY STOP] no improvement for {PATIENCE} epochs.")
            break

    # save log
    log_df = pd.DataFrame(history)
    log_df.to_csv(LOG_CSV_PATH, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Training finished")
    print("Best epoch   :", best_epoch)
    print("Best val loss:", best_val_loss)
    print("Best model   :", BEST_PATH)
    print("Last model   :", LAST_PATH)
    print("Log CSV      :", LOG_CSV_PATH)
    print("=" * 60)

    # -----------------------------------------------------
    # TEST EVALUATION
    # -----------------------------------------------------
    print("[INFO] Loading best model for test evaluation...")
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    test_res = run_epoch(model, test_loader, optimizer=None)

    test_metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_res.loss,
        "test_l1": test_res.l1,
        "test_bone": test_res.bone,
        "test_mpjpe": test_res.mpjpe,
        "test_pck05": test_res.pck05,
        "device": DEVICE,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }

    with open(TEST_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print("[TEST]")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print("Saved:", TEST_JSON_PATH)


if __name__ == "__main__":
    main()