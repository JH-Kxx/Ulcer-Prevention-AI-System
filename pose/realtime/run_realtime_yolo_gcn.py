# =========================================================
# run_realtime_yolo_gcn.py
# ---------------------------------------------------------
# Real-time pipeline:
#   Camera -> YOLO17 -> map 17->14 -> build stable bbox
#          -> GCN refine -> confidence-aware EMA smoothing
#          -> render 14-joint skeleton (body-part colorized)
#
# Requirements:
#   pip install ultralytics opencv-python numpy torch
# =========================================================

import os
import time
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. PATH / MODEL SETTINGS
# ---------------------------------------------------------
YOLO_MODEL_NAME = "yolo26s-pose.pt"
GCN_CKPT_PATH = r"C:\Users\Konyang\SLP_project\gcn_ckpt_rgb\best_gcn.pt"

TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

CAM_INDEX = 0
IMG_SIZE = 640
CONF_THRES = 0.05
IOU_THRES = 0.45
MAX_DET = 5


# ---------------------------------------------------------
# 2. REALTIME FILTER / BBOX SETTINGS
# ---------------------------------------------------------
# predicted joints 기반 안정 bbox
PRED_BBOX_PAD_X = 60.0
PRED_BBOX_PAD_Y = 80.0
MIN_BBOX_W = 160.0
MIN_BBOX_H = 260.0
MAX_ASPECT_RATIO = 5.0

# GCN 입력에 최소한 이 정도 joint는 있어야 함
MIN_PRED14_CONF_POSITIVE = 4

# EMA smoothing
USE_SMOOTHING = True
BASE_ALPHA = 0.65
MIN_ALPHA = 0.20
MAX_ALPHA = 0.85

# confidence gating
MIN_RENDER_CONF = 0.05

# skeleton line / point size
LINE_THICKNESS = 1
POINT_RADIUS = 2


# ---------------------------------------------------------
# 3. JOINT SETTINGS
# ---------------------------------------------------------
SLP14_NAMES = [
    "right_ankle", "right_knee", "right_hip",
    "left_hip", "left_knee", "left_ankle",
    "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist",
    "thorax", "head"
]

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
COCO_IDX = {name: i for i, name in enumerate(COCO17_NAMES)}

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 12), (12, 9), (9, 10), (10, 11),
    (2, 12), (3, 12), (12, 13)
]


# ---------------------------------------------------------
# 3-1. BODY PART COLOR SETTINGS (BGR)
# ---------------------------------------------------------
COLOR_HEAD = (255, 0, 255)      # purple
COLOR_TORSO = (0, 255, 255)     # yellow
COLOR_ARM = (0, 165, 255)       # orange
COLOR_LEG = (255, 255, 0)       # cyan
COLOR_STABLE_BBOX = (255, 255, 255)  # white

JOINT_PART = {
    0: "leg",    # right_ankle
    1: "leg",    # right_knee
    2: "leg",    # right_hip
    3: "leg",    # left_hip
    4: "leg",    # left_knee
    5: "leg",    # left_ankle
    6: "arm",    # right_wrist
    7: "arm",    # right_elbow
    8: "arm",    # right_shoulder
    9: "arm",    # left_shoulder
    10: "arm",   # left_elbow
    11: "arm",   # left_wrist
    12: "torso", # thorax
    13: "head"   # head
}


def part_color(part_name: str):
    if part_name == "head":
        return COLOR_HEAD
    elif part_name == "torso":
        return COLOR_TORSO
    elif part_name == "arm":
        return COLOR_ARM
    elif part_name == "leg":
        return COLOR_LEG
    return (0, 255, 0)


# ---------------------------------------------------------
# 4. GCN MODEL
# ---------------------------------------------------------
def build_adjacency(num_nodes: int, edges):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    for i in range(num_nodes):
        A[i, i] = 1.0

    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(D, 1e-6, None)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


A_NORM = build_adjacency(14, EDGES)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A):
        x = torch.matmul(A, x)
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

        out = self.output_proj(h, A)
        return out


# ---------------------------------------------------------
# 5. UTILS
# ---------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def normalize_by_bbox(kpts_xy: np.ndarray, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(float(x2 - x1), 1e-6)
    bh = max(float(y2 - y1), 1e-6)

    out = kpts_xy.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - x1) / bw
    out[:, 1] = (out[:, 1] - y1) / bh
    return out


def denormalize_by_bbox(kpts_xy_norm: np.ndarray, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(float(x2 - x1), 1e-6)
    bh = max(float(y2 - y1), 1e-6)

    out = kpts_xy_norm.copy().astype(np.float32)
    out[:, 0] = out[:, 0] * bw + x1
    out[:, 1] = out[:, 1] * bh + y1
    return out


def map_coco17_to_slp14(pred_xy: np.ndarray, pred_conf: np.ndarray):
    def get_xyc(name):
        idx = COCO_IDX[name]
        return pred_xy[idx, 0], pred_xy[idx, 1], pred_conf[idx]

    out = np.zeros((14, 3), dtype=np.float32)

    direct = {
        0: "right_ankle",
        1: "right_knee",
        2: "right_hip",
        3: "left_hip",
        4: "left_knee",
        5: "left_ankle",
        6: "right_wrist",
        7: "right_elbow",
        8: "right_shoulder",
        9: "left_shoulder",
        10: "left_elbow",
        11: "left_wrist",
    }

    for slp_idx, coco_name in direct.items():
        x, y, c = get_xyc(coco_name)
        out[slp_idx] = [x, y, c]

    # thorax
    ls = out[9]
    rs = out[8]
    if ls[2] > 0 and rs[2] > 0:
        thorax_x = (ls[0] + rs[0]) / 2.0
        thorax_y = (ls[1] + rs[1]) / 2.0
        thorax_c = (ls[2] + rs[2]) / 2.0
    elif ls[2] > 0:
        thorax_x, thorax_y, thorax_c = ls
    elif rs[2] > 0:
        thorax_x, thorax_y, thorax_c = rs
    else:
        thorax_x = thorax_y = thorax_c = 0.0
    out[12] = [thorax_x, thorax_y, thorax_c]

    # head
    face_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    face_pts = []
    face_confs = []
    for nm in face_names:
        x, y, c = get_xyc(nm)
        if c > 0:
            face_pts.append([x, y])
            face_confs.append(c)

    if len(face_pts) > 0:
        face_pts = np.array(face_pts, dtype=np.float32)
        face_confs = np.array(face_confs, dtype=np.float32)
        head_xy = face_pts.mean(axis=0)
        head_c = float(face_confs.mean())
        out[13] = [head_xy[0], head_xy[1], head_c]
    else:
        out[13] = [0.0, 0.0, 0.0]

    return out


def select_best_person(result):
    if result.boxes is None or result.keypoints is None:
        return None, "boxes_or_keypoints_none"

    boxes = result.boxes
    kpts = result.keypoints

    if boxes is None or len(boxes) == 0:
        return None, "no_boxes"
    if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
        return None, "no_keypoints_xy"

    boxes_conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
    best_idx = 0 if boxes_conf is None or len(boxes_conf) == 0 else int(np.argmax(boxes_conf))

    xyxy = boxes.xyxy[best_idx].detach().cpu().numpy().astype(np.float32)
    box_conf = float(boxes.conf[best_idx].item()) if boxes.conf is not None else 0.0

    pred_xy = kpts.xy[best_idx].detach().cpu().numpy().astype(np.float32)
    if pred_xy.shape[0] != 17:
        return None, f"unexpected_kpt_count_{pred_xy.shape[0]}"

    if kpts.conf is not None:
        pred_conf = kpts.conf[best_idx].detach().cpu().numpy().astype(np.float32)
    else:
        pred_conf = np.ones((pred_xy.shape[0],), dtype=np.float32)

    return {
        "yolo_bbox_xyxy": xyxy,
        "yolo_bbox_conf": box_conf,
        "pred_xy_17": pred_xy,
        "pred_conf_17": pred_conf
    }, "ok"


def build_pred_fullbody_bbox(pred_14_xyc: np.ndarray, img_w: int, img_h: int):
    """
    실시간용 안정 bbox:
    YOLO bbox 대신 predicted 14-joint 전체를 바탕으로 확장
    """
    valid = pred_14_xyc[:, 2] > 0
    pts = pred_14_xyc[valid, :2]

    if len(pts) == 0:
        cx, cy = img_w / 2.0, img_h / 2.0
        bw, bh = MIN_BBOX_W, MIN_BBOX_H
    else:
        x1 = float(np.min(pts[:, 0]) - PRED_BBOX_PAD_X)
        y1 = float(np.min(pts[:, 1]) - PRED_BBOX_PAD_Y)
        x2 = float(np.max(pts[:, 0]) + PRED_BBOX_PAD_X)
        y2 = float(np.max(pts[:, 1]) + PRED_BBOX_PAD_Y)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(x2 - x1, MIN_BBOX_W)
        bh = max(y2 - y1, MIN_BBOX_H)

    if bw / max(bh, 1e-6) > MAX_ASPECT_RATIO:
        bh = bw / MAX_ASPECT_RATIO
    if bh / max(bw, 1e-6) > MAX_ASPECT_RATIO:
        bw = bh / MAX_ASPECT_RATIO

    x1 = cx - bw / 2.0
    x2 = cx + bw / 2.0
    y1 = cy - bh / 2.0
    y2 = cy + bh / 2.0

    x1 = clamp(x1, 0.0, img_w - 1.0)
    y1 = clamp(y1, 0.0, img_h - 1.0)
    x2 = clamp(x2, 0.0, img_w - 1.0)
    y2 = clamp(y2, 0.0, img_h - 1.0)

    if x2 <= x1:
        x2 = min(img_w - 1.0, x1 + 2.0)
    if y2 <= y1:
        y2 = min(img_h - 1.0, y1 + 2.0)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def draw_skeleton_14_colored(img, kpts14_xyc, radius=2, thickness=1):
    img_out = img.copy()

    for a, b in EDGES:
        if kpts14_xyc[a, 2] > MIN_RENDER_CONF and kpts14_xyc[b, 2] > MIN_RENDER_CONF:
            xa, ya = int(kpts14_xyc[a, 0]), int(kpts14_xyc[a, 1])
            xb, yb = int(kpts14_xyc[b, 0]), int(kpts14_xyc[b, 1])

            part_a = JOINT_PART[a]
            part_b = JOINT_PART[b]

            if part_a == part_b:
                edge_color = part_color(part_a)
            else:
                edge_color = COLOR_TORSO

            cv2.line(img_out, (xa, ya), (xb, yb), edge_color, thickness)

    for i in range(14):
        x, y, c = kpts14_xyc[i]
        if c > MIN_RENDER_CONF:
            joint_color = part_color(JOINT_PART[i])
            cv2.circle(img_out, (int(x), int(y)), radius, joint_color, -1)

    return img_out


def confidence_aware_ema(prev_xy, curr_xy, conf, base_alpha=0.65):
    """
    prev_xy, curr_xy: (14,2)
    conf: (14,)
    """
    if prev_xy is None:
        return curr_xy.copy()

    conf = np.clip(conf, 0.0, 1.0)
    alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * conf
    alpha = np.clip(alpha * (base_alpha / MAX_ALPHA), MIN_ALPHA, MAX_ALPHA)
    alpha = alpha[:, None]

    smoothed = alpha * curr_xy + (1.0 - alpha) * prev_xy
    return smoothed.astype(np.float32)


def put_text_with_outline(
    img,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    text_color=(255, 255, 255),
    outline_color=(0, 0, 0),
    thickness=2,
    outline_thickness=4
):
    cv2.putText(img, text, org, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------
def main():
    print("[INFO] Loading YOLO:", YOLO_MODEL_NAME)
    print("[INFO] YOLO device:", YOLO_DEVICE)
    yolo_model = YOLO(YOLO_MODEL_NAME)

    print("[INFO] Loading GCN checkpoint:", GCN_CKPT_PATH)
    print("[INFO] Torch device:", TORCH_DEVICE)
    ckpt = torch.load(GCN_CKPT_PATH, map_location=TORCH_DEVICE)

    cfg = ckpt.get("config", {})
    gcn_model = ResidualGCN(
        in_dim=cfg.get("in_dim", 3),
        hidden_dim=cfg.get("hidden_dim", 64),
        out_dim=cfg.get("out_dim", 2),
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.10),
    ).to(TORCH_DEVICE)

    gcn_model.load_state_dict(ckpt["model_state_dict"])
    gcn_model.eval()

    A = A_NORM.to(TORCH_DEVICE)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없음: index={CAM_INDEX}")

    prev_smoothed_xy = None
    prev_conf = None

    fps_time = time.time()
    fps_counter = 0
    fps_value = 0.0

    print("[INFO] q 누르면 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 프레임 읽기 실패")
            break

        img_h, img_w = frame.shape[:2]
        canvas = frame.copy()

        # YOLO inference
        try:
            results = yolo_model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device=YOLO_DEVICE,
                max_det=MAX_DET,
                verbose=False
            )
        except Exception as e:
            put_text_with_outline(
                canvas,
                f"YOLO ERROR: {str(e)}",
                (20, 40),
                font_scale=0.7,
                text_color=(255, 255, 255),
                outline_color=(0, 0, 255),
                thickness=2,
                outline_thickness=4
            )
            cv2.imshow("YOLO + GCN Realtime", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        status_text = "NO PERSON"

        if len(results) > 0:
            best, status = select_best_person(results[0])

            if best is not None:
                pred_xy_17 = best["pred_xy_17"]
                pred_conf_17 = best["pred_conf_17"]

                pred_14_xyc = map_coco17_to_slp14(pred_xy_17, pred_conf_17)
                num_pred_pos = int((pred_14_xyc[:, 2] > 0).sum())

                if num_pred_pos >= MIN_PRED14_CONF_POSITIVE:
                    stable_bbox_xyxy = build_pred_fullbody_bbox(pred_14_xyc, img_w, img_h)

                    pred_xy_norm = normalize_by_bbox(pred_14_xyc[:, :2], stable_bbox_xyxy)
                    pred_input = np.concatenate(
                        [pred_xy_norm, pred_14_xyc[:, 2:3]],
                        axis=1
                    ).astype(np.float32)

                    with torch.no_grad():
                        x_t = torch.from_numpy(pred_input).unsqueeze(0).to(TORCH_DEVICE)
                        residual = gcn_model(x_t, A)
                        refined_norm = x_t[:, :, :2] + residual
                        refined_norm = refined_norm.squeeze(0).detach().cpu().numpy()

                    refined_xy_img = denormalize_by_bbox(refined_norm, stable_bbox_xyxy)
                    refined_conf = pred_14_xyc[:, 2].copy()

                    if USE_SMOOTHING:
                        smoothed_xy = confidence_aware_ema(
                            prev_smoothed_xy,
                            refined_xy_img,
                            refined_conf,
                            base_alpha=BASE_ALPHA
                        )
                    else:
                        smoothed_xy = refined_xy_img.copy()

                    prev_smoothed_xy = smoothed_xy.copy()
                    prev_conf = refined_conf.copy()

                    # stable bbox only
                    sx1, sy1, sx2, sy2 = stable_bbox_xyxy.astype(int)
                    cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), COLOR_STABLE_BBOX, 1)

                    # GCN + smoothed only
                    gcn_vis = np.concatenate(
                        [smoothed_xy.astype(np.float32), refined_conf[:, None].astype(np.float32)],
                        axis=1
                    )

                    canvas = draw_skeleton_14_colored(
                        canvas,
                        gcn_vis,
                        radius=POINT_RADIUS,
                        thickness=LINE_THICKNESS
                    )

                    status_text = f"PERSON | pred14={num_pred_pos}"
                else:
                    status_text = f"TOO FEW JOINTS ({num_pred_pos})"
                    prev_smoothed_xy = None
                    prev_conf = None
            else:
                status_text = f"NO VALID PERSON ({status})"
                prev_smoothed_xy = None
                prev_conf = None
        else:
            prev_smoothed_xy = None
            prev_conf = None

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_value = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()

        # text overlay
        put_text_with_outline(
            canvas,
            f"FPS: {fps_value:.1f}",
            (20, 30),
            font_scale=0.8,
            text_color=(255, 255, 255),
            outline_color=(0, 0, 0),
            thickness=2,
            outline_thickness=4
        )

        put_text_with_outline(
            canvas,
            status_text,
            (20, 65),
            font_scale=0.7,
            text_color=(255, 255, 255),
            outline_color=(0, 0, 0),
            thickness=2,
            outline_thickness=4
        )

        cv2.imshow("YOLO + GCN Realtime", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()