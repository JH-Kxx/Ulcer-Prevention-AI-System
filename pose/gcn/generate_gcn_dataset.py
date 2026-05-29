# =========================================================
# make_gcn_dataset.py
# ---------------------------------------------------------
# Purpose:
# 1) Load split_output images / GT JSON
# 2) Run YOLO pose inference (17-keypoint COCO format)
# 3) Map YOLO 17 -> SLP 14
# 4) Build a stable full-body bbox from GT 14 joints
# 5) Normalize YOLO14 input and GT14 target by GT-based bbox
# 6) Save GCN training dataset as NPZ
#
# Output:
#   C:\Users\Konyang\SLP_project\gcn_dataset\
#       train.npz
#       val.npz
#       test.npz
#       train_meta.csv
#       val_meta.csv
#       test_meta.csv
#       qc_preview\...
#
# Requirements:
#   pip install ultralytics opencv-python pillow numpy pandas tqdm
# =========================================================

import os
import json
from glob import glob
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. PATH SETTINGS
# ---------------------------------------------------------
SRC_ROOT = r"C:\Users\Konyang\SLP_project\split_output"
IMG_ROOT = os.path.join(SRC_ROOT, "images")
GT_ROOT = os.path.join(SRC_ROOT, "gt_json")

OUT_ROOT = r"C:\Users\Konyang\SLP_project\gcn_dataset"
QC_ROOT = os.path.join(OUT_ROOT, "qc_preview")

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------
# 2. YOLO MODEL SETTINGS
# ---------------------------------------------------------
# Ultralytics 모델 이름 또는 로컬 .pt 경로
YOLO_MODEL_NAME = "yolo26s-pose.pt"

IMG_SIZE = 640
CONF_THRES = 0.05
IOU_THRES = 0.45
DEVICE = 0          # GPU: 0, CPU: "cpu"
MAX_DET = 5

QC_PER_SPLIT = 50
DEBUG_MAX_PRINT_PER_SPLIT = 20


# ---------------------------------------------------------
# 3. BBOX / FILTER SETTINGS
# ---------------------------------------------------------
# GT 기반 full-body bbox 생성 시 padding
GT_BBOX_PAD_X = 40.0
GT_BBOX_PAD_Y = 40.0

# 최소 bbox 크기 보장
MIN_BBOX_W = 80.0
MIN_BBOX_H = 120.0

# 이미지 경계로 clamp
CLAMP_TO_IMAGE = True

# YOLO가 전혀 못 맞힌 샘플도 일부는 학습에 넣을 수 있지만,
# 너무 anchor가 없으면 GCN이 배울 게 없음
MIN_PRED14_CONF_POSITIVE = 4

# GT bbox aspect ratio 보호
MAX_ASPECT_RATIO = 5.0


# ---------------------------------------------------------
# 4. SLP 14-JOINT ORDER
# ---------------------------------------------------------
SLP14_NAMES = [
    "right_ankle", "right_knee", "right_hip",
    "left_hip", "left_knee", "left_ankle",
    "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist",
    "thorax", "head"
]


# ---------------------------------------------------------
# 5. COCO 17-JOINT ORDER (Ultralytics YOLO Pose)
# ---------------------------------------------------------
COCO17_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle"      # 16
]

COCO_IDX = {name: i for i, name in enumerate(COCO17_NAMES)}


# ---------------------------------------------------------
# 6. UTILS
# ---------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_image_paths(folder: str):
    paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        paths.extend(glob(os.path.join(folder, ext)))
    return sorted(paths)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def get_gt_kpts14_from_json(json_path: str):
    data = load_json(json_path)
    kpts = np.array(data["keypoints_xyv"], dtype=np.float32)  # (14,3)
    if kpts.shape != (14, 3):
        raise ValueError(f"GT shape error: {json_path} -> {kpts.shape}")
    return kpts[:, :2].copy()  # v는 안 씀

def sanitize_xy(kpts_xy: np.ndarray, img_w: int, img_h: int):
    out = kpts_xy.copy().astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], 0, img_w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, img_h - 1)
    return out

def build_gt_fullbody_bbox(gt_xy_14: np.ndarray, img_w: int, img_h: int):
    """
    GT 14-joint 전체로 full-body bbox 생성
    cover 상황에서도 전신 기준 좌표계를 유지하기 위한 핵심 함수
    """
    pts = gt_xy_14.astype(np.float32)

    x1 = float(np.min(pts[:, 0]) - GT_BBOX_PAD_X)
    y1 = float(np.min(pts[:, 1]) - GT_BBOX_PAD_Y)
    x2 = float(np.max(pts[:, 0]) + GT_BBOX_PAD_X)
    y2 = float(np.max(pts[:, 1]) + GT_BBOX_PAD_Y)

    # 최소 크기 보장
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(x2 - x1, MIN_BBOX_W)
    bh = max(y2 - y1, MIN_BBOX_H)

    # aspect ratio 보호
    if bw / max(bh, 1e-6) > MAX_ASPECT_RATIO:
        bh = bw / MAX_ASPECT_RATIO
    if bh / max(bw, 1e-6) > MAX_ASPECT_RATIO:
        bw = bh / MAX_ASPECT_RATIO

    x1 = cx - bw / 2.0
    x2 = cx + bw / 2.0
    y1 = cy - bh / 2.0
    y2 = cy + bh / 2.0

    if CLAMP_TO_IMAGE:
        x1 = clamp(x1, 0.0, img_w - 1.0)
        y1 = clamp(y1, 0.0, img_h - 1.0)
        x2 = clamp(x2, 0.0, img_w - 1.0)
        y2 = clamp(y2, 0.0, img_h - 1.0)

    # 최종 degenerate 방지
    if x2 <= x1:
        x2 = min(img_w - 1.0, x1 + 2.0)
    if y2 <= y1:
        y2 = min(img_h - 1.0, y1 + 2.0)

    return np.array([x1, y1, x2, y2], dtype=np.float32)

def normalize_by_bbox(kpts_xy: np.ndarray, bbox_xyxy):
    """
    bbox local frame 정규화
    """
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(float(x2 - x1), 1e-6)
    bh = max(float(y2 - y1), 1e-6)

    out = kpts_xy.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - x1) / bw
    out[:, 1] = (out[:, 1] - y1) / bh
    return out

def draw_skeleton_14(img, kpts14_xyc, color=(0, 255, 0), radius=4, thickness=2):
    edges = [
        (0,1), (1,2), (2,3), (3,4), (4,5),
        (6,7), (7,8), (8,12), (12,9), (9,10), (10,11),
        (2,12), (3,12), (12,13)
    ]

    img_out = img.copy()

    for a, b in edges:
        if kpts14_xyc[a, 2] > 0 and kpts14_xyc[b, 2] > 0:
            xa, ya = int(kpts14_xyc[a, 0]), int(kpts14_xyc[a, 1])
            xb, yb = int(kpts14_xyc[b, 0]), int(kpts14_xyc[b, 1])
            cv2.line(img_out, (xa, ya), (xb, yb), color, thickness)

    for i in range(14):
        x, y, c = kpts14_xyc[i]
        if c > 0:
            cv2.circle(img_out, (int(x), int(y)), radius, color, -1)

    return img_out

def draw_skeleton_14_gt(img, gt_xy_14, color=(255, 0, 0), radius=3, thickness=1):
    gt_vis = np.concatenate(
        [gt_xy_14.astype(np.float32), np.ones((14, 1), dtype=np.float32)],
        axis=1
    )
    return draw_skeleton_14(img, gt_vis, color=color, radius=radius, thickness=thickness)

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

def print_debug_result(img_path, result):
    print(f"[DEBUG] image: {img_path}")
    print(f"        boxes is None: {result.boxes is None}")
    print(f"        keypoints is None: {result.keypoints is None}")

    if result.boxes is not None:
        print(f"        num_boxes: {len(result.boxes)}")
        if result.boxes.conf is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.detach().cpu().numpy()
            print(f"        box_conf[:5]: {confs[:5]}")

    if result.keypoints is not None and result.keypoints.xy is not None:
        print(f"        kpt_xy_shape: {tuple(result.keypoints.xy.shape)}")


# ---------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------
def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(QC_ROOT)
    for split in SPLITS:
        ensure_dir(os.path.join(QC_ROOT, split))

    print("[INFO] Loading YOLO model:", YOLO_MODEL_NAME)
    print("[INFO] device:", DEVICE)
    model = YOLO(YOLO_MODEL_NAME)

    total_fail_stats = defaultdict(int)

    for split in SPLITS:
        print(f"\n==============================")
        print(f"[INFO] Processing split: {split}")
        print(f"==============================")

        img_dir = os.path.join(IMG_ROOT, split)
        gt_dir = os.path.join(GT_ROOT, split)

        img_paths = collect_image_paths(img_dir)
        gt_paths = sorted(glob(os.path.join(gt_dir, "*.json")))

        img_map = {Path(p).stem: p for p in img_paths}
        gt_map = {Path(p).stem: p for p in gt_paths}
        keys = sorted(set(img_map.keys()) & set(gt_map.keys()))

        if len(keys) == 0:
            print(f"[WARNING] No matched files in split={split}")
            continue

        X_list = []
        Y_list = []
        GT_BBOX_list = []
        YOLO_BBOX_list = []
        IMG_WH_list = []
        meta_rows = []

        qc_saved = 0
        debug_printed = 0
        qc_step = max(1, len(keys) // max(QC_PER_SPLIT, 1))
        split_fail_stats = defaultdict(int)

        for idx, key in enumerate(tqdm(keys, desc=f"{split}")):
            img_path = img_map[key]
            gt_path = gt_map[key]

            img = cv2.imread(img_path)
            if img is None:
                split_fail_stats["imread_fail"] += 1
                continue

            h, w = img.shape[:2]

            try:
                gt_xy_14 = get_gt_kpts14_from_json(gt_path)
                gt_xy_14 = sanitize_xy(gt_xy_14, w, h)
            except Exception as e:
                print(f"[WARNING] GT load failed: {gt_path} -> {e}")
                split_fail_stats["gt_load_fail"] += 1
                continue

            gt_bbox_xyxy = build_gt_fullbody_bbox(gt_xy_14, w, h)

            try:
                results = model.predict(
                    source=img,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRES,
                    iou=IOU_THRES,
                    device=DEVICE,
                    max_det=MAX_DET,
                    verbose=False
                )
            except Exception as e:
                print(f"[WARNING] YOLO inference failed: {img_path} -> {e}")
                split_fail_stats["yolo_infer_fail"] += 1
                continue

            if len(results) == 0:
                split_fail_stats["empty_results"] += 1
                continue

            r0 = results[0]
            best, status = select_best_person(r0)

            if best is None:
                split_fail_stats[status] += 1
                if debug_printed < DEBUG_MAX_PRINT_PER_SPLIT:
                    print_debug_result(img_path, r0)
                    print(f"        fail_reason: {status}")
                    debug_printed += 1
                continue

            yolo_bbox_xyxy = best["yolo_bbox_xyxy"]
            pred_xy_17 = best["pred_xy_17"]
            pred_conf_17 = best["pred_conf_17"]

            pred_14_xyc = map_coco17_to_slp14(pred_xy_17, pred_conf_17)

            # 너무 anchor가 적으면 제외
            num_pred_pos = int((pred_14_xyc[:, 2] > 0).sum())
            if num_pred_pos < MIN_PRED14_CONF_POSITIVE:
                split_fail_stats["too_few_pred14_positive"] += 1
                continue

            # 핵심 변경: YOLO bbox가 아니라 GT 기반 full-body bbox로 정규화
            pred_xy_norm = normalize_by_bbox(pred_14_xyc[:, :2], gt_bbox_xyxy)
            gt_xy_norm = normalize_by_bbox(gt_xy_14, gt_bbox_xyxy)

            pred_input = np.concatenate(
                [pred_xy_norm, pred_14_xyc[:, 2:3]],
                axis=1
            ).astype(np.float32)  # (14,3)

            gt_target = gt_xy_norm.astype(np.float32)  # (14,2)

            X_list.append(pred_input)
            Y_list.append(gt_target)
            GT_BBOX_list.append(gt_bbox_xyxy.astype(np.float32))
            YOLO_BBOX_list.append(yolo_bbox_xyxy.astype(np.float32))
            IMG_WH_list.append(np.array([w, h], dtype=np.float32))

            meta_rows.append({
                "id": key,
                "split": split,
                "image_path": img_path,
                "gt_path": gt_path,
                "img_w": w,
                "img_h": h,
                "gt_bbox_x1": float(gt_bbox_xyxy[0]),
                "gt_bbox_y1": float(gt_bbox_xyxy[1]),
                "gt_bbox_x2": float(gt_bbox_xyxy[2]),
                "gt_bbox_y2": float(gt_bbox_xyxy[3]),
                "yolo_bbox_x1": float(yolo_bbox_xyxy[0]),
                "yolo_bbox_y1": float(yolo_bbox_xyxy[1]),
                "yolo_bbox_x2": float(yolo_bbox_xyxy[2]),
                "yolo_bbox_y2": float(yolo_bbox_xyxy[3]),
                "yolo_bbox_conf": float(best["yolo_bbox_conf"]),
                "num_pred14_conf_gt0": num_pred_pos
            })

            if qc_saved < QC_PER_SPLIT and (idx % qc_step == 0):
                qc_img = img.copy()

                # GT bbox = red
                gx1, gy1, gx2, gy2 = gt_bbox_xyxy.astype(int)
                cv2.rectangle(qc_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)

                # YOLO bbox = yellow
                yx1, yy1, yx2, yy2 = yolo_bbox_xyxy.astype(int)
                cv2.rectangle(qc_img, (yx1, yy1), (yx2, yy2), (0, 255, 255), 2)

                # pred = green, gt = blue
                qc_img = draw_skeleton_14(qc_img, pred_14_xyc, color=(0, 255, 0), radius=4, thickness=2)
                qc_img = draw_skeleton_14_gt(qc_img, gt_xy_14, color=(255, 0, 0), radius=3, thickness=1)

                cv2.putText(
                    qc_img,
                    f"{key} | pred(green) gt(blue) gtbbox(red) yolobbox(yellow)",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                qc_path = os.path.join(QC_ROOT, split, f"{key}_qc.jpg")
                cv2.imwrite(qc_path, qc_img)
                qc_saved += 1

        if len(X_list) == 0:
            print(f"[WARNING] No usable samples for split={split}")
            print(f"[FAIL STATS] {dict(split_fail_stats)}")
            for k, v in split_fail_stats.items():
                total_fail_stats[k] += v
            continue

        X = np.stack(X_list, axis=0)                  # (N,14,3)
        Y = np.stack(Y_list, axis=0)                  # (N,14,2)
        GT_BBOX = np.stack(GT_BBOX_list, axis=0)      # (N,4)
        YOLO_BBOX = np.stack(YOLO_BBOX_list, axis=0)  # (N,4)
        IMG_WH = np.stack(IMG_WH_list, axis=0)        # (N,2)

        npz_path = os.path.join(OUT_ROOT, f"{split}.npz")
        meta_path = os.path.join(OUT_ROOT, f"{split}_meta.csv")

        np.savez_compressed(
            npz_path,
            X=X,
            Y=Y,
            gt_bbox_xyxy=GT_BBOX,
            yolo_bbox_xyxy=YOLO_BBOX,
            img_wh=IMG_WH,
            joint_names=np.array(SLP14_NAMES, dtype=object)
        )

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

        print(f"[DONE] {split}")
        print(f"  samples : {len(X)}")
        print(f"  X shape : {X.shape}")
        print(f"  Y shape : {Y.shape}")
        print(f"  npz     : {npz_path}")
        print(f"  meta    : {meta_path}")
        print(f"  fail_stats: {dict(split_fail_stats)}")

        for k, v in split_fail_stats.items():
            total_fail_stats[k] += v

    print("\n==============================")
    print("[ALL DONE] make_gcn_dataset.py finished")
    print("==============================")
    print("output root:", OUT_ROOT)
    print("qc root    :", QC_ROOT)
    print("total fail stats:", dict(total_fail_stats))


if __name__ == "__main__":
    main()