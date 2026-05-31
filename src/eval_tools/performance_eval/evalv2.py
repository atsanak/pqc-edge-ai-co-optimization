"""
eval_yolo_paper.py

Adds "paper-style" detection reporting:

- Detection metrics at a chosen IoU (default 0.50): Precision / Recall / F1
- COCO-style mAP@[0.50:0.95]
- mAP@0.50 and mAP@0.75
- Per-class AP at IoU=0.50 and IoU=0.75 (and per-class GT / Pred counts)

Your existing matching/AP logic is kept (VOC-style AP with precision envelope).
This is "COCO-style" in the sense of averaging AP over IoU thresholds, but it
is not the official pycocotools implementation.

Usage:
  python3 eval_yolo_paper.py
"""

import os
import glob
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------
#  Utilities: loading YOLO files and geometry
# ---------------------------------------------------------

def load_yolo_file(path, with_score=False):
    """
    Expected format per line:
        class_id x_center y_center width height [score]

    Returns:
        if with_score=False:
            list of (class_id, x_center, y_center, w, h)
        if with_score=True:
            list of (class_id, x_center, y_center, w, h, score)
    """
    if not os.path.exists(path):
        return []

    boxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])

            if with_score:
                score = float(parts[5]) if len(parts) >= 6 else 1.0
                boxes.append((cls, x, y, w, h, score))
            else:
                boxes.append((cls, x, y, w, h))
    return boxes


def yolo_to_xyxy(box):
    cls, x, y, w, h = box
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return (cls, x1, y1, x2, y2)


def yolo_to_xyxy_scored(box):
    cls, x, y, w, h, s = box
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return (cls, x1, y1, x2, y2, s)


def iou(box1, box2):
    """
    IoU between two boxes in (cls, x1, y1, x2, y2) format (cls ignored).
    """
    _, x1_1, y1_1, x2_1, y2_1 = box1
    _, x1_2, y1_2, x2_2, y2_2 = box2

    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


# ---------------------------------------------------------
#  AP / mAP
# ---------------------------------------------------------

def compute_ap(rec, prec):
    """
    VOC-style AP with precision envelope.
    rec, prec are 1D numpy arrays.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_map(gt_dir, pred_dir, iou_thresh=0.5, class_agnostic=False):
    """
    Returns:
      per_class_ap: dict {class_id: AP}
      mAP: float (mean of per-class AP over classes that exist in GT)
      gt_count: dict {class_id: number_of_gt_boxes}
      pred_count: dict {class_id: number_of_pred_boxes}
    
    If class_agnostic=True, all boxes are treated as the same class (useful when
    GT and predictions use different class ID schemes).
    """
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

    # Build GT by image and count GTs per class
    gt_by_image = {}  # base -> list of (cls, x1, y1, x2, y2)
    gt_count = {}
    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        gty = load_yolo_file(gt_path, with_score=False)
        gtx = [yolo_to_xyxy(b) for b in gty]
        # If class_agnostic, remap all classes to 0
        if class_agnostic:
            gtx = [(0, g[1], g[2], g[3], g[4]) for g in gtx]
        gt_by_image[base] = gtx
        for g in gtx:
            cls = g[0]
            gt_count[cls] = gt_count.get(cls, 0) + 1

    # Collect predictions across dataset (with scores)
    preds_by_class = {}  # cls -> list of (score, base, pred_box_xyxy)
    pred_count = {}
    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base)
        pry = load_yolo_file(pred_path, with_score=True)
        prx = [yolo_to_xyxy_scored(b) for b in pry]

        for p in prx:
            cls, x1, y1, x2, y2, s = p
            # If class_agnostic, remap all classes to 0
            if class_agnostic:
                cls = 0
            preds_by_class.setdefault(cls, []).append((s, base, (cls, x1, y1, x2, y2)))
            pred_count[cls] = pred_count.get(cls, 0) + 1

    per_class_ap = {}
    classes_in_gt = sorted(gt_count.keys())

    for cls in classes_in_gt:
        npos = gt_count[cls]

        # GTs per image for this class + matched flags
        gt_cls_boxes = {}
        matched = {}
        for base, gts in gt_by_image.items():
            filtered = [g for g in gts if g[0] == cls]
            gt_cls_boxes[base] = filtered
            matched[base] = [False] * len(filtered)

        preds = preds_by_class.get(cls, [])
        preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (score, base, pbox) in enumerate(preds):
            gts = gt_cls_boxes.get(base, [])
            if len(gts) == 0:
                fp[i] = 1.0
                continue

            best_iou = 0.0
            best_j = -1
            for j, gtbox in enumerate(gts):
                if matched[base][j]:
                    continue
                iou_val = iou(pbox, gtbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            if best_j >= 0 and best_iou >= iou_thresh:
                tp[i] = 1.0
                matched[base][best_j] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (npos + 1e-12)
        prec = tp_cum / (tp_cum + fp_cum + 1e-12)

        per_class_ap[cls] = compute_ap(rec, prec)

    mAP = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return per_class_ap, mAP, gt_count, pred_count


def evaluate_map_sweep(gt_dir, pred_dir, iou_thresholds, class_agnostic=False):
    """
    Runs evaluate_map for multiple IoU thresholds.

    Returns:
      sweep: dict {iou: {"per_class_ap":..., "mAP":...}}
      gt_count, pred_count
    """
    sweep = {}
    gt_count_final = None
    pred_count_final = None

    for t in iou_thresholds:
        per_class_ap, mAP, gt_count, pred_count = evaluate_map(gt_dir, pred_dir, iou_thresh=float(t), class_agnostic=class_agnostic)
        t_key = round(float(t), 2)
        sweep[t_key] = {"per_class_ap": per_class_ap, "mAP": mAP}
        gt_count_final = gt_count
        pred_count_final = pred_count

    return sweep, gt_count_final, pred_count_final


# ---------------------------------------------------------
#  Detection-level metrics (single IoU)
# ---------------------------------------------------------

def evaluate_detection(gt_dir, pred_dir, iou_thresh=0.5, class_agnostic=False):
    """
    Evaluate detection metrics (TP, FP, FN, precision, recall, F1, accuracy).
    
    If class_agnostic=True, class IDs are ignored during matching (useful when
    GT and predictions use different class ID schemes).
    """
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base)

        gt_boxes = [yolo_to_xyxy(b) for b in load_yolo_file(gt_path, with_score=False)]
        pred_boxes = [yolo_to_xyxy(b) for b in load_yolo_file(pred_path, with_score=False)]

        gt_matched = [False] * len(gt_boxes)

        for p in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for i, g in enumerate(gt_boxes):
                if gt_matched[i]:
                    continue
                # Skip class check if class_agnostic
                if not class_agnostic and g[0] != p[0]:
                    continue
                iou_val = iou(p, g)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = i

            if best_gt_idx >= 0 and best_iou >= iou_thresh:
                gt_matched[best_gt_idx] = True
                total_tp += 1
            else:
                total_fp += 1

        for matched in gt_matched:
            if not matched:
                total_fn += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom = total_tp + total_fp + total_fn
    accuracy = total_tp / denom if denom > 0 else 0.0

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------
#  Image-level presence (optional)
# ---------------------------------------------------------

def evaluate_image_presence(gt_dir, pred_dir):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

    y_true = []
    y_pred = []

    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base)

        gt_boxes = load_yolo_file(gt_path, with_score=False)
        pred_boxes = load_yolo_file(pred_path, with_score=False)

        y_true.append(1 if len(gt_boxes) > 0 else 0)
        y_pred.append(1 if len(pred_boxes) > 0 else 0)

    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


def image_level_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1v, "confusion_matrix": cm}


# ---------------------------------------------------------
#  Pretty printing for papers
# ---------------------------------------------------------

def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def print_paper_report(gt_dir, pred_dir, show_image_level=False, class_names=None):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    num_images = len(gt_files)

    # A) Detection metrics at IoU=0.50 (common to report alongside mAP)
    det_iou = 0.50
    det = evaluate_detection(gt_dir, pred_dir, iou_thresh=det_iou)

    # B) COCO-style sweep
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    sweep, gt_count, pred_count = evaluate_map_sweep(gt_dir, pred_dir, iou_thresholds)

    map50 = sweep[0.50]["mAP"]
    map75 = sweep[0.75]["mAP"]
    map5095 = float(np.mean([sweep[t]["mAP"] for t in sweep.keys()]))

    for t in sweep.keys():
        print(f"mAP@{t:.2f}: {fmt(sweep[t]['mAP'])}")

    # Per-class AP at 0.50 and 0.75
    ap50 = sweep[0.50]["per_class_ap"]
    ap75 = sweep[0.75]["per_class_ap"]
    classes = sorted(gt_count.keys())

    # ----------------------------
    # Print paper-style block
    # ----------------------------
    print("\n==================== PAPER-STYLE RESULTS ====================")
    print(f"Images: {num_images}")
    print("------------------------------------------------------------")
    print(f"Precision@{det_iou:.2f}: {fmt(det['precision'])}   Recall@{det_iou:.2f}: {fmt(det['recall'])}   F1@{det_iou:.2f}: {fmt(det['f1'])}")
    print(f"Accuracy@{det_iou:.2f}: {fmt(det['accuracy'])}")
    print(f"TP: {det['TP']}   FP: {det['FP']}   FN: {det['FN']}")
    print("------------------------------------------------------------")
    print(f"mAP@[0.50:0.95]: {fmt(map5095)}   mAP@0.50: {fmt(map50)}   mAP@0.75: {fmt(map75)}")
    print("------------------------------------------------------------")
    print("Per-class AP:")
    print(f"{'class':<8}{'GT':>6}{'Pred':>7}{'AP@0.50':>10}{'AP@0.75':>10}")
    for c in classes:
        name = str(c)
        if class_names and c in class_names:
            name = f"{c}({class_names[c]})"

        gt_n = gt_count.get(c, 0)
        pr_n = pred_count.get(c, 0)

        print(
            f"{name:<8}"
            f"{gt_n:>6d}"
            f"{pr_n:>7d}"
            f"{ap50.get(c, 0.0):>10.3f}"
            f"{ap75.get(c, 0.0):>10.3f}"
        )
    print("============================================================\n")

    # Optional: image-level presence metrics
    if show_image_level:
        y_true_img, y_pred_img = evaluate_image_presence(gt_dir, pred_dir)
        img = image_level_metrics(y_true_img, y_pred_img)
        print("Image-level (object present vs not):")
        print(f"  Acc: {fmt(img['accuracy'])}  Prec: {fmt(img['precision'])}  Rec: {fmt(img['recall'])}  F1: {fmt(img['f1'])}")
        print(f"  Confusion matrix [rows true 0/1, cols pred 0/1]:\n{img['confusion_matrix']}\n")


def load_class_names(path):
    """
    Optional helper: one class name per line, index = class_id.
    e.g., coco_classes.txt.
    """
    if path is None or not os.path.exists(path):
        return None
    names = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            names[i] = line.strip()
    return names


# ---------------------------------------------------------
#  Main
# ---------------------------------------------------------
if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(_script_dir))
    gt_dir = os.path.join(_project_dir, "car_dataset_labels", "ground_truth_labels")
    # pred_dir = os.path.join(_project_dir, "car_dataset_labels", "attention_gridv1_labels")  # change as needed
    pred_dir = os.path.join(_project_dir, "agv2_car_dataset_labels")  # change as needed

    # Optional: class names (one per line). If you don't have it, keep None.
    class_names = load_class_names(None)  # e.g., "./coco_classes.txt"

    print_paper_report(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        show_image_level=False,   # papers usually emphasize mAP/AP, not image presence
        class_names=class_names
    )
