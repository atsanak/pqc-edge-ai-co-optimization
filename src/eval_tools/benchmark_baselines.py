"""
benchmark_baselines.py

Compare inference methods against a full-frame YOLO11s reference:
  1. yolo_baseline   — full-frame YOLO11s at native resolution, every frame
  2. sahi            — Slicing Aided Hyper Inference (Akyon et al., ICIP 2022)
                       requires:  pip install sahi
  3. frame_skip_yolo — full-frame YOLO11s run every N frames; cached predictions
                       reused on skipped frames (naive temporal caching baseline)

Supports both UA-DETRAC (vehicles) and MOT17 (pedestrians).
Choose the dataset, sequences, and per-method options via interactive menus.

Usage:
    cd <project_root>
    python -m eval_tools.benchmark_baselines
    OR
    cd eval_tools && python benchmark_baselines.py
"""

# ── Phase 0: stdlib + project path ──────────────────────────────────
import configparser
import csv
import glob
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_ROOT)

# ── Phase 1: GPU selection (before any CUDA library) ────────────────
# Only prompt when this file is run as a standalone script. When it's
# imported as a library (e.g. by run_split_inference_benchmark.py), the
# caller is responsible for picking the device so the user doesn't see
# the prompt twice.
from gpu_utils import select_gpu
if __name__ == "__main__":
    _PHYSICAL_GPU_INDEX = select_gpu()
else:
    _PHYSICAL_GPU_INDEX = 0  # caller has (or will) set CUDA_VISIBLE_DEVICES

from cuda_preload import preload_cuda_libs
preload_cuda_libs()

# ── Phase 2: heavy imports ──────────────────────────────────────────
import numpy as np
from PIL import Image

from object_clfs.heavy_yolo_classifier import save_yolo_txt, HeavyYoloClassifier
from eval_tools.performance_eval.system_monitor import SystemMonitor
from eval_tools.performance_eval.evalv2 import evaluate_detection, evaluate_map_sweep

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
#  SAHI availability check
# ============================================================
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _HAS_SAHI = True
except ImportError:
    _HAS_SAHI = False
    print("\n  [WARN] SAHI not installed.  SAHI method will be SKIPPED.")
    print("         Install with:  pip install sahi\n")


# ============================================================
#  TOP-LEVEL CONFIG  (edit these)
# ============================================================

_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "benchmark_baselines_output")

# ── YOLO weight options ───────────────────────────────────────
# Presented as a menu at startup; user picks one that applies to all methods.
YOLO_WEIGHT_OPTIONS = [
    "yolo11s.pt",   # default — same as AG benchmark
    "yolo11m.pt",   # medium  — "bigger model" comparison
    "yolo11l.pt",   # large
]

# ── Confidence threshold options ─────────────────────────────
# Lower = more detections (higher recall, lower precision).
CONF_THRESHOLD_OPTIONS = [0.25, 0.15, 0.35]

# ── AG per-scene grid config (loaded from JSON at runtime) ───
# Each scene may have a different rows×cols grid from the grid search.
# These are loaded once from the JSON files and looked up per sequence.
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "configs")
_DETRAC_SCENE_CONFIGS_PATH = os.path.join(_CONFIG_DIR, "ua_detrac_scene_configs.json")
_MOT17_SCENE_CONFIGS_PATH  = os.path.join(_CONFIG_DIR, "mot17_scene_configs.json")

def _load_scene_grid_configs() -> Dict[str, Tuple[int, int]]:
    """Return {seq_name: (rows, cols)} for all known scenes."""
    import json
    grid: Dict[str, Tuple[int, int]] = {}
    for path in (_DETRAC_SCENE_CONFIGS_PATH, _MOT17_SCENE_CONFIGS_PATH):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for name, cfg in data.get("sequences", {}).items():
            grid[name] = (int(cfg["rows"]), int(cfg["cols"]))
    return grid

_SCENE_GRID_CONFIGS: Dict[str, Tuple[int, int]] = _load_scene_grid_configs()

# ── SAHI settings ────────────────────────────────────────────
# Slice size is derived automatically per scene from the AG grid config JSON:
#   slice_h = img_h // ag_rows,  slice_w = img_w // ag_cols
# This ensures SAHI tiles match exactly one AG tile — a fair comparison.
# Only overlap is a user-selectable option.
# 0.25 matches the overlap used in the original SAHI paper (Akyon et al., ICIP 2022).
# Note: the library default is 0.2, but the paper explicitly evaluated at 25%.
SAHI_OVERLAP_OPTIONS     = [0.25, 0.2, 0.1]
SAHI_POSTPROCESS_TYPE    = "GREEDYNMM"   # "GREEDYNMM" | "NMM" | "UNIONMERGE"
SAHI_POSTPROCESS_THRESH  = 0.5

# ── Frame-skipping YOLO settings ─────────────────────────────
# YRI = YOLO Run Interval: YOLO runs every N frames; predictions cached in between.
# Multiple values run as separate experiments so you get a sweep in one go.
FRAME_SKIP_YRI_OPTIONS = [2, 3, 5]   # e.g. [2, 3, 5] runs three experiments

# ── Evaluation ───────────────────────────────────────────────
CLASS_AGNOSTIC_EVAL      = True
EVALUATE_UNIQUE_OBJECTS  = True
UNIQUE_IOU_THRESHOLD     = 0.50


# ============================================================
#  DATASET CONSTANTS  (populated at runtime after dataset choice)
# ============================================================

# UA-DETRAC
_DETRAC_ROOT         = os.path.join(_PROJECT_ROOT, "ua_detrac", "content",
                                    "UA-DETRAC", "DETRAC_Upload")
_DETRAC_IMAGES_DIR   = os.path.join(_DETRAC_ROOT, "images")
_DETRAC_LABELS_DIR   = os.path.join(_DETRAC_ROOT, "labels")
_DETRAC_ID_LABELS_DIR= os.path.join(_DETRAC_ROOT, "labels_with_ids")
_DETRAC_SPLIT        = "val"
_DETRAC_IMG_W        = 960
_DETRAC_IMG_H        = 540
_DETRAC_KEEP_IDS     = {2, 5, 7, 3}                  # car, bus, truck, motorbike
_DETRAC_COCO_MAP_AG  = {2: 0, 5: 0, 7: 0, 3: 0}     # class-agnostic: all → 0

# MOT17
_MOT17_ROOT              = os.path.join(_PROJECT_ROOT, "mot17", "MOT17")
_MOT17_SPLIT             = "train"
_MOT17_PREFERRED_VARIANT = "FRCNN"
_MOT17_KEEP_IDS          = {0}          # person
_MOT17_COCO_MAP_AG       = {0: 0}
_MOT17_PEDESTRIAN_CLS    = {1, 2}       # in gt.txt column 8
_MOT17_MIN_VIS           = 0.10
_MOT17_SEQ_META: Dict[str, Dict] = {
    "MOT17-02": {"camera": "static",  "fps": 30},
    "MOT17-04": {"camera": "static",  "fps": 30},
    "MOT17-05": {"camera": "moving",  "fps": 14},
    "MOT17-09": {"camera": "static",  "fps": 30},
    "MOT17-10": {"camera": "moving",  "fps": 30},
    "MOT17-11": {"camera": "moving",  "fps": 30},
    "MOT17-13": {"camera": "moving",  "fps": 25},
}


# ============================================================
#  SEQUENCE DATACLASSES
# ============================================================

@dataclass
class Sequence:
    """Unified sequence description used across both datasets."""
    seq_name:   str
    img_paths:  List[str]
    img_width:  int
    img_height: int
    ag_rows:    int = 3   # AG grid rows for this scene (from JSON config)
    ag_cols:    int = 5   # AG grid cols for this scene (from JSON config)
    # dataset-specific extras (None when not applicable)
    gt_csv_path:   Optional[str] = None   # MOT17 only
    camera:        Optional[str] = None   # MOT17 only ("static"/"moving")
    detrac_split:  Optional[str] = None   # UA-DETRAC split label


# ============================================================
#  DATASET SELECTION
# ============================================================

def choose_dataset() -> str:
    print("\n" + "=" * 60)
    print("  Select benchmark dataset")
    print("=" * 60)
    print("  [1] UA-DETRAC  (vehicle detection, 960×540)")
    print("  [2] MOT17      (pedestrian detection, 1920×1080+)")
    while True:
        raw = input("\nDataset [1/2] (default 1): ").strip() or "1"
        if raw in ("1", "2"):
            return "ua_detrac" if raw == "1" else "mot17"
        print("  Please enter 1 or 2.")


# ============================================================
#  OPTIONS MENUS
# ============================================================

def _pick_from_list(prompt: str, options: list, default: int = 0) -> object:
    """Generic numbered menu, returns chosen item."""
    print(f"\n  {prompt}")
    for i, opt in enumerate(options):
        marker = " (default)" if i == default else ""
        print(f"    [{i+1}] {opt}{marker}")
    while True:
        raw = input(f"  Choice [1-{len(options)}] (default {default+1}): ").strip()
        if raw == "":
            return options[default]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def choose_options() -> dict:
    """Interactive menu for all per-run options.  Returns a config dict."""
    print("\n" + "=" * 60)
    print("  Run options")
    print("=" * 60)

    yolo_weight = _pick_from_list(
        "YOLO weight  (applies to all methods):",
        YOLO_WEIGHT_OPTIONS, default=0,
    )
    min_conf = _pick_from_list(
        "Confidence threshold  (lower = more detections):",
        CONF_THRESHOLD_OPTIONS, default=0,
    )

    print("\n  — SAHI options —")
    print(f"  Slice size: auto-computed per scene from ua_detrac_scene_configs.json / mot17_scene_configs.json")
    print(f"              slice_h = img_h ÷ ag_rows,  slice_w = img_w ÷ ag_cols  (one AG tile per SAHI slice)")
    sahi_overlap = _pick_from_list(
        "SAHI overlap ratio:",
        SAHI_OVERLAP_OPTIONS, default=0,
    )

    print("\n  — Frame-skip YOLO options —")
    print("  YRI values to benchmark (multiple = one run per value):")
    for i, v in enumerate(FRAME_SKIP_YRI_OPTIONS):
        print(f"    [{i+1}] YRI={v}")
    print(f"    [0] ALL  (default)")
    raw = input("  Choice [0 for all, or comma-separated numbers]: ").strip()
    if raw == "" or raw == "0":
        yri_values = list(FRAME_SKIP_YRI_OPTIONS)
    else:
        chosen = []
        for part in raw.split(","):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(FRAME_SKIP_YRI_OPTIONS):
                    chosen.append(FRAME_SKIP_YRI_OPTIONS[idx])
            except ValueError:
                pass
        yri_values = chosen if chosen else list(FRAME_SKIP_YRI_OPTIONS)

    return {
        "yolo_weight":   yolo_weight,
        "min_conf":      min_conf,
        "sahi_overlap":  sahi_overlap,
        "yri_values":    yri_values,
    }


# ============================================================
#  SEQUENCE DISCOVERY — UA-DETRAC
# ============================================================

def _discover_detrac() -> Dict[str, Sequence]:
    img_dir = os.path.join(_DETRAC_IMAGES_DIR, _DETRAC_SPLIT)
    if not os.path.isdir(img_dir):
        print(f"  [ERROR] UA-DETRAC image dir not found: {img_dir}")
        return {}
    all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    seq_map: Dict[str, List[str]] = defaultdict(list)
    for p in all_imgs:
        m = re.match(r"(MVI_\d+)_img\d+\.jpg", os.path.basename(p))
        if m:
            seq_map[m.group(1)].append(p)
    result = {}
    for name, paths in sorted(seq_map.items()):
        ag_rows, ag_cols = _SCENE_GRID_CONFIGS.get(name, (3, 5))
        result[name] = Sequence(
            seq_name=name, img_paths=sorted(paths),
            img_width=_DETRAC_IMG_W, img_height=_DETRAC_IMG_H,
            ag_rows=ag_rows, ag_cols=ag_cols,
            detrac_split=_DETRAC_SPLIT,
        )
    return result


# ============================================================
#  SEQUENCE DISCOVERY — MOT17
# ============================================================

def _parse_seqinfo(ini_path: str) -> dict:
    out = {"imWidth": 1920, "imHeight": 1080, "frameRate": 30}
    if not os.path.exists(ini_path):
        return out
    cp = configparser.ConfigParser()
    cp.read(ini_path)
    if cp.has_section("Sequence"):
        s = cp["Sequence"]
        out["imWidth"]   = int(s.get("imWidth",   out["imWidth"]))
        out["imHeight"]  = int(s.get("imHeight",  out["imHeight"]))
        out["frameRate"] = int(s.get("frameRate", out["frameRate"]))
    return out


def _discover_mot17() -> Dict[str, Sequence]:
    split_dir = os.path.join(_MOT17_ROOT, _MOT17_SPLIT)
    if not os.path.isdir(split_dir):
        print(f"  [ERROR] MOT17 split dir not found: {split_dir}")
        return {}
    by_base: Dict[str, Dict[str, str]] = defaultdict(dict)
    for d in sorted(os.listdir(split_dir)):
        m = re.match(r"^(MOT17-\d+)-(DPM|FRCNN|SDP)$", d)
        if m:
            by_base[m.group(1)][m.group(2)] = os.path.join(split_dir, d)
    result = {}
    for base_name, variants in sorted(by_base.items()):
        for variant in (_MOT17_PREFERRED_VARIANT, "SDP", "FRCNN", "DPM"):
            if variant in variants:
                chosen_dir = variants[variant]
                break
        else:
            continue
        img_dir   = os.path.join(chosen_dir, "img1")
        gt_path   = os.path.join(chosen_dir, "gt", "gt.txt")
        seqinfo   = _parse_seqinfo(os.path.join(chosen_dir, "seqinfo.ini"))
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if not img_paths:
            continue
        meta = _MOT17_SEQ_META.get(base_name, {})
        ag_rows, ag_cols = _SCENE_GRID_CONFIGS.get(base_name, (3, 5))
        result[base_name] = Sequence(
            seq_name   = base_name,
            img_paths  = img_paths,
            img_width  = seqinfo["imWidth"],
            img_height = seqinfo["imHeight"],
            ag_rows    = ag_rows,
            ag_cols    = ag_cols,
            gt_csv_path = gt_path,
            camera      = meta.get("camera", "unknown"),
        )
    return result


def discover_sequences(dataset: str) -> Dict[str, Sequence]:
    if dataset == "ua_detrac":
        return _discover_detrac()
    return _discover_mot17()


# ============================================================
#  INTERACTIVE SEQUENCE PICKER
# ============================================================

def select_sequences_interactive(seq_map: Dict[str, Sequence]) -> List[Sequence]:
    seq_names = list(seq_map.keys())
    n = len(seq_names)
    print(f"\n{'=' * 65}")
    print(f"  Available sequences  ({n} total)")
    print("=" * 65)
    for i, name in enumerate(seq_names):
        s = seq_map[name]
        cam = f"  cam={s.camera}" if s.camera else ""
        print(f"  [{i+1:>3}]  {name:<16} {len(s.img_paths):>6} frames  "
              f"{s.img_width}×{s.img_height}{cam}")
    print(f"\n  [  0] ALL sequences")
    print("  Enter numbers, ranges (e.g. 1-5), or 0 for all.")
    while True:
        raw = input("\nSequence(s) (default 1): ").strip() or "1"
        if raw == "0":
            return list(seq_map.values())
        chosen: List[int] = []
        valid = True
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                lr = part.split("-", 1)
                try:
                    a, b = int(lr[0]), int(lr[1])
                    chosen.extend(range(a, b + 1))
                except ValueError:
                    valid = False; break
            else:
                try:
                    chosen.append(int(part))
                except ValueError:
                    valid = False; break
        if not valid or any(c < 1 or c > n for c in chosen):
            print(f"  Invalid.  Enter numbers 1–{n}.")
            continue
        chosen = sorted(set(chosen))
        result = [seq_map[seq_names[i - 1]] for i in chosen]
        print(f"  Selected: {', '.join(s.seq_name for s in result)}")
        return result


# ============================================================
#  GT LABEL PREPARATION
# ============================================================

def _frame_id_from_path(img_path: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def _load_mot17_gt(seq: Sequence) -> Dict[int, List[Tuple[int,float,float,float,float]]]:
    """Parse MOT17 gt.txt → {frame_id: [(track_id, x1, y1, x2, y2), ...]}."""
    by_frame: Dict[int, list] = defaultdict(list)
    if not seq.gt_csv_path or not os.path.exists(seq.gt_csv_path):
        return by_frame
    with open(seq.gt_csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = int(float(parts[6]))
                cls  = int(float(parts[7]))
                vis  = float(parts[8])
            except ValueError:
                continue
            if conf != 1 or cls not in _MOT17_PEDESTRIAN_CLS or vis < _MOT17_MIN_VIS:
                continue
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(seq.img_width),  x + w)
            y2 = min(float(seq.img_height), y + h)
            if x2 > x1 and y2 > y1:
                by_frame[frame_id].append((track_id, x1, y1, x2, y2))
    return by_frame


def prepare_gt_labels(seq: Sequence, gt_dir: str, dataset: str):
    """Write per-frame YOLO-format GT labels for evaluation."""
    os.makedirs(gt_dir, exist_ok=True)

    if dataset == "mot17":
        gt_by_frame = _load_mot17_gt(seq)
        for img_path in seq.img_paths:
            frame_id = _frame_id_from_path(img_path)
            out_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            dst = os.path.join(gt_dir, out_name)
            lines: List[str] = []
            for _, x1, y1, x2, y2 in gt_by_frame.get(frame_id or -1, []):
                cx = (x1 + x2) * 0.5 / seq.img_width
                cy = (y1 + y2) * 0.5 / seq.img_height
                bw = (x2 - x1) / seq.img_width
                bh = (y2 - y1) / seq.img_height
                if bw > 0 and bh > 0:
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(dst, "w") as f:
                f.write("\n".join(lines))

    else:  # ua_detrac
        lbl_dir = os.path.join(_DETRAC_LABELS_DIR, _DETRAC_SPLIT)
        for img_path in seq.img_paths:
            base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            src  = os.path.join(lbl_dir, base)
            dst  = os.path.join(gt_dir, base)
            if not os.path.exists(src):
                open(dst, "w").close()
                continue
            with open(src) as f:
                lines = f.readlines()
            out = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                if CLASS_AGNOSTIC_EVAL:
                    parts[0] = "0"
                out.append(" ".join(parts))
            with open(dst, "w") as f:
                f.write("\n".join(out))


# ============================================================
#  EVALUATION HELPERS
# ============================================================

def evaluate_sequence(pred_dir: str, gt_dir: str) -> dict:
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
    if not gt_files:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "TP": 0, "FP": 0, "FN": 0,
                "mAP_50": 0.0, "mAP_75": 0.0, "mAP_50_95": 0.0}
    det = evaluate_detection(gt_dir, pred_dir, iou_thresh=0.50,
                              class_agnostic=CLASS_AGNOSTIC_EVAL)
    ious = np.arange(0.50, 0.96, 0.05)
    sweep, _, _ = evaluate_map_sweep(gt_dir, pred_dir, ious,
                                     class_agnostic=CLASS_AGNOSTIC_EVAL)
    return {
        "precision": det["precision"],
        "recall":    det["recall"],
        "f1":        det["f1"],
        "TP":        det["TP"],
        "FP":        det["FP"],
        "FN":        det["FN"],
        "mAP_50":    sweep[0.50]["mAP"],
        "mAP_75":    sweep[0.75]["mAP"],
        "mAP_50_95": float(np.mean([sweep[t]["mAP"] for t in sweep])),
    }


def aggregate_results(results: List[dict]) -> dict:
    """Per-scene arithmetic-mean aggregation — same recipe for every method
    (AG, Baseline, SAHI) so tables compare apples to apples. Each scene
    contributes equally regardless of its length."""
    n = len(results)
    if n == 0:
        return {}
    keys = ["mAP_50", "mAP_75", "mAP_50_95", "precision", "recall", "f1"]
    agg  = {k: sum(r[k] for r in results) / n for k in keys}
    agg["TP"] = sum(r["TP"] for r in results)
    agg["FP"] = sum(r["FP"] for r in results)
    agg["FN"] = sum(r["FN"] for r in results)

    def _mean(key):
        xs = [r[key] for r in results if r.get(key) is not None]
        return sum(xs) / len(xs) if xs else None

    agg["mean_fps"]    = _mean("fps") or 0.0
    agg["gpu_power_w"] = _mean("gpu_power_w")
    agg["gpu_util"]    = _mean("gpu_util_pct")
    return agg


def _iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _norm_xywh_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w / 2) * W;  y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W;  y2 = (cy + h / 2) * H
    return x1, y1, x2, y2


def evaluate_unique_objects_detrac(pred_dir: str, seq: Sequence,
                                   iou_thresh: float = UNIQUE_IOU_THRESHOLD) -> dict:
    """UA-DETRAC unique vehicle recall using labels_with_ids."""
    id_lbl_dir = os.path.join(_DETRAC_ID_LABELS_DIR, _DETRAC_SPLIT)
    if not os.path.isdir(id_lbl_dir):
        return {"total_unique": 0, "detected_unique": 0, "missed_unique": 0, "unique_recall": 0.0}

    all_ids: set = set()
    gt_with_ids: Dict[str, list] = {}
    W, H = seq.img_width, seq.img_height

    for img_path in seq.img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        fpath = os.path.join(id_lbl_dir, base)
        boxes = []
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cx, cy, w, h = map(float, parts[1:5])
                        tid = int(parts[5])
                        boxes.append((_norm_xywh_to_xyxy(cx, cy, w, h, W, H), tid))
                        all_ids.add(tid)
        gt_with_ids[base] = boxes

    if not all_ids:
        return {"total_unique": 0, "detected_unique": 0, "missed_unique": 0, "unique_recall": 0.0}

    detected_ids: set = set()
    for img_path in seq.img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        gt_boxes = gt_with_ids.get(base, [])
        if not gt_boxes:
            continue
        pred_path = os.path.join(pred_dir, base)
        if not os.path.exists(pred_path):
            continue
        pred_boxes = []
        with open(pred_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    pred_boxes.append((_norm_xywh_to_xyxy(cx, cy, bw, bh, W, H), conf))
        if not pred_boxes:
            continue
        pred_boxes.sort(key=lambda x: x[1], reverse=True)
        matched_gt: set = set()
        for pred_box, _ in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for gi, (gt_box, _) in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                score = _iou_xyxy(pred_box, gt_box)
                if score > best_iou:
                    best_iou, best_idx = score, gi
            if best_iou >= iou_thresh and best_idx >= 0:
                matched_gt.add(best_idx)
                detected_ids.add(gt_boxes[best_idx][1])

    total    = len(all_ids)
    detected = len(detected_ids)
    return {"total_unique": total, "detected_unique": detected,
            "missed_unique": total - detected,
            "unique_recall": detected / total if total > 0 else 0.0}


def evaluate_unique_objects_mot17(pred_dir: str, seq: Sequence,
                                  iou_thresh: float = UNIQUE_IOU_THRESHOLD) -> dict:
    """MOT17 unique pedestrian recall from gt.txt track IDs."""
    gt_by_frame = _load_mot17_gt(seq)
    W, H = seq.img_width, seq.img_height

    all_ids: set = set()
    for boxes in gt_by_frame.values():
        for tid, *_ in boxes:
            all_ids.add(tid)

    if not all_ids:
        return {"total_unique": 0, "detected_unique": 0, "missed_unique": 0, "unique_recall": 0.0}

    detected_ids: set = set()
    for img_path in seq.img_paths:
        frame_id = _frame_id_from_path(img_path)
        gt_boxes = gt_by_frame.get(frame_id or -1, [])
        if not gt_boxes:
            continue
        base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        pred_path = os.path.join(pred_dir, base)
        if not os.path.exists(pred_path):
            continue
        pred_boxes = []
        with open(pred_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    pred_boxes.append((_norm_xywh_to_xyxy(cx, cy, bw, bh, W, H), conf))
        if not pred_boxes:
            continue
        pred_boxes.sort(key=lambda x: x[1], reverse=True)
        gt_xyxy_tid = [((tid, x1, y1, x2, y2)) for tid, x1, y1, x2, y2 in gt_boxes]
        matched_gt: set = set()
        for pred_box, _ in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for gi, (tid, x1, y1, x2, y2) in enumerate(gt_xyxy_tid):
                if gi in matched_gt:
                    continue
                score = _iou_xyxy(pred_box, (x1, y1, x2, y2))
                if score > best_iou:
                    best_iou, best_idx = score, gi
            if best_iou >= iou_thresh and best_idx >= 0:
                matched_gt.add(best_idx)
                detected_ids.add(gt_xyxy_tid[best_idx][0])

    total    = len(all_ids)
    detected = len(detected_ids)
    return {"total_unique": total, "detected_unique": detected,
            "missed_unique": total - detected,
            "unique_recall": detected / total if total > 0 else 0.0}


def evaluate_unique_objects(pred_dir: str, seq: Sequence, dataset: str) -> dict:
    if dataset == "mot17":
        return evaluate_unique_objects_mot17(pred_dir, seq)
    return evaluate_unique_objects_detrac(pred_dir, seq)


def _extract_system_metrics_from_samples(samples: List[dict]) -> dict:
    def _avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    return {
        "cpu_pct":    _avg([x["cpu_proc"]    for x in samples]),
        "ram_mb":     _avg([x["ram_proc_mb"] for x in samples]),
        "gpu_util":   _avg([x["gpu_util"]    for x in samples]),
        "gpu_mem_mb": _avg([x["gpu_mem_mb"]  for x in samples]),
        "gpu_power_w": _avg([x["gpu_power_w"] for x in samples]),
        # Temperatures (Jetson tegrastats; nvml on x86). Optional per-sample.
        "gpu_temp_c": _avg([x.get("gpu_temp_c") for x in samples]),
        "cpu_temp_c": _avg([x.get("cpu_temp_c") for x in samples]),
        # Sample count drives `tegrastats_samples` in the master row.
        "n_samples": len(samples),
    }


def _monitor_sample_count(monitor: SystemMonitor) -> int:
    lock = getattr(monitor, "_lock", None)
    if lock is not None:
        with lock:
            return len(monitor.samples)
    return len(monitor.samples)


def _extract_sequence_system_metrics(monitor: SystemMonitor, start_idx: int) -> dict:
    lock = getattr(monitor, "_lock", None)
    if lock is not None:
        with lock:
            subset = list(monitor.samples[start_idx:])
    else:
        subset = list(monitor.samples[start_idx:])
    return _extract_system_metrics_from_samples(subset)


def extract_system_metrics(monitor: SystemMonitor) -> dict:
    lock = getattr(monitor, "_lock", None)
    if lock is not None:
        with lock:
            snapshot = list(monitor.samples)
    else:
        snapshot = list(monitor.samples)
    return _extract_system_metrics_from_samples(snapshot)


# ============================================================
#  SAVE PREDICTIONS  (xyxy absolute pixels → YOLO normalized txt)
# ============================================================

def save_predictions_xyxy(
    img_pil: Image.Image,
    boxes_xyxy: np.ndarray,       # (N, 4) in absolute pixels [x1,y1,x2,y2]
    classes: np.ndarray,           # (N,) COCO class IDs
    scores: np.ndarray,            # (N,) confidences
    pred_dir: str,
    image_path: str,
    keep_coco_ids: set,
    coco_to_local: dict,
    min_conf: float = 0.25,
):
    """Convert absolute-pixel xyxy detections to normalised YOLO txt."""
    W, H = img_pil.size
    stem = os.path.splitext(os.path.basename(image_path))[0]
    dst  = os.path.join(pred_dir, stem + ".txt")
    lines = []
    if boxes_xyxy is not None and len(boxes_xyxy) > 0:
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            cls_id = int(classes[i]) if classes is not None else 0
            if keep_coco_ids and cls_id not in keep_coco_ids:
                continue
            conf = float(scores[i]) if scores is not None else 1.0
            if conf < min_conf:
                continue
            local_cls = coco_to_local.get(cls_id, cls_id) if coco_to_local else cls_id
            cx = (x1 + x2) * 0.5 / W
            cy = (y1 + y2) * 0.5 / H
            bw = abs(x2 - x1) / W
            bh = abs(y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{local_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")
    with open(dst, "w") as f:
        f.write("\n".join(lines))


def _effective_imgsz_for_sequence(seq: Sequence, dataset: str) -> int:
    """Match stress-test behavior: MOT17 uses per-sequence native resolution."""
    if dataset == "mot17":
        return max(seq.img_width, seq.img_height)
    return max(_DETRAC_IMG_W, _DETRAC_IMG_H)


# ============================================================
#  METHOD 1 — YOLO BASELINE (full-frame)
# ============================================================

def run_yolo_baseline(sequences: List[Sequence], dataset: str,
                      output_base: str, opts: dict) -> dict:
    keep_ids = _DETRAC_KEEP_IDS if dataset == "ua_detrac" else _MOT17_KEEP_IDS
    coco_map = _DETRAC_COCO_MAP_AG if dataset == "ua_detrac" else _MOT17_COCO_MAP_AG
    method   = "yolo_baseline"
    min_conf = opts["min_conf"]

    monitor = SystemMonitor()
    monitor.start()
    total_frames = 0; total_time = 0.0; seq_results = []

    for seq in sequences:
        n = len(seq.img_paths)
        total_frames += n
        seq_sample_start = _monitor_sample_count(monitor)

        # Stress-test parity: MOT17 baseline uses sequence-native imgsz.
        seq_imgsz = _effective_imgsz_for_sequence(seq, dataset)
        clf = HeavyYoloClassifier(weight=opts["yolo_weight"], imgsz=seq_imgsz)

        pred_dir = os.path.join(output_base, "predicted_labels", method, seq.seq_name)
        gt_dir   = os.path.join(output_base, "gt_labels", seq.seq_name)
        os.makedirs(pred_dir, exist_ok=True)
        prepare_gt_labels(seq, gt_dir, dataset)

        print(f"\n  [{method}] {seq.seq_name}  ({n} frames, "
              f"{seq.img_width}x{seq.img_height}, imgsz={seq_imgsz})")
        t0 = time.time()
        for i, img_path in enumerate(seq.img_paths):
            if i % 200 == 0 or i == n - 1:
                print(f"    Frame {i+1}/{n}")
            frame = Image.open(img_path).convert("RGB")
            _, boxes, classes, scores = clf.predict_image(frame)
            save_yolo_txt(img_pil=frame, boxes_xywh=boxes, classes=classes,
                          scores=scores, label_dir=pred_dir, image_path=img_path,
                          keep_coco_ids=keep_ids, coco_to_local=coco_map,
                          min_conf=min_conf)
        seq_time = time.time() - t0
        total_time += seq_time
        seq_sys_m = _extract_sequence_system_metrics(monitor, seq_sample_start)
        metrics = evaluate_sequence(pred_dir, gt_dir)
        metrics.update({
            "seq_name": seq.seq_name,
            "fps": n / seq_time if seq_time > 0 else 0,
            "gpu_power_w": seq_sys_m.get("gpu_power_w"),
            "gpu_util_pct": seq_sys_m.get("gpu_util"),
        })
        if EVALUATE_UNIQUE_OBJECTS:
            metrics.update(evaluate_unique_objects(pred_dir, seq, dataset))
        _print_seq_metrics(metrics, method)
        seq_results.append(metrics)

    monitor.stop()
    agg = aggregate_results(seq_results)
    agg.update({
        "method": method,
        "seq_results": seq_results,
    })
    if EVALUATE_UNIQUE_OBJECTS:
        agg.update(_aggregate_unique(seq_results))
    return agg


# ============================================================
#  METHOD 2 — SAHI
# ============================================================

def run_sahi(sequences: List[Sequence], dataset: str,
             output_base: str, opts: dict) -> dict:
    if not _HAS_SAHI:
        print("\n  [SKIP] SAHI not installed.  pip install sahi\n")
        return {"method": "sahi", "mean_fps": 0.0, "skipped": True}

    keep_ids = _DETRAC_KEEP_IDS if dataset == "ua_detrac" else _MOT17_KEEP_IDS
    coco_map = _DETRAC_COCO_MAP_AG if dataset == "ua_detrac" else _MOT17_COCO_MAP_AG
    min_conf  = opts["min_conf"]
    overlap   = opts["sahi_overlap"]
    method    = f"sahi_ag_equiv_ov{int(overlap*100)}"

    device = f"cuda:{_PHYSICAL_GPU_INDEX}" if _PHYSICAL_GPU_INDEX is not None else "cpu"

    # Keep one SAHI model per inference size so each sequence uses its native scale,
    # while avoiding repeated model construction for scenes that share resolution.
    det_model_cache: Dict[int, object] = {}
    image_size_arg_supported: Optional[bool] = None

    def _get_sahi_model(seq_imgsz: int):
        nonlocal image_size_arg_supported
        cached = det_model_cache.get(seq_imgsz)
        if cached is not None:
            return cached

        common_kwargs = dict(
            model_type="ultralytics",
            model_path=opts["yolo_weight"],
            confidence_threshold=min_conf,
            device=device,
        )

        if image_size_arg_supported is not False:
            try:
                model = AutoDetectionModel.from_pretrained(
                    **common_kwargs,
                    image_size=seq_imgsz,
                )
                image_size_arg_supported = True
            except TypeError:
                # Older SAHI versions may not expose image_size in from_pretrained.
                if image_size_arg_supported is None:
                    print("  [WARN] SAHI AutoDetectionModel.from_pretrained() "
                          "does not accept image_size; falling back to SAHI defaults.")
                image_size_arg_supported = False
                model = AutoDetectionModel.from_pretrained(**common_kwargs)
        else:
            model = AutoDetectionModel.from_pretrained(**common_kwargs)

        # Apply sequence-specific detector size when the model exposes it,
        # including older SAHI versions where from_pretrained() may not
        # accept image_size directly.
        if hasattr(model, "image_size"):
            try:
                model.image_size = seq_imgsz
            except Exception:
                pass

        det_model_cache[seq_imgsz] = model
        return model

    monitor = SystemMonitor()
    monitor.start()
    total_frames = 0; total_time = 0.0; seq_results = []

    for seq in sequences:
        n = len(seq.img_paths)
        total_frames += n
        seq_sample_start = _monitor_sample_count(monitor)

        # Slice size matches one AG tile for this scene's resolution and grid.
        # ag_rows/ag_cols come from ua_detrac_scene_configs.json / mot17_scene_configs.json
        # so they are per-scene (not static).
        slice_h = seq.img_height // seq.ag_rows
        slice_w = seq.img_width  // seq.ag_cols

        # Fair-comparison imgsz: SAHI runs YOLO per slice, so the detector's
        # inference resolution must match the slice's native max dim — the same
        # imgsz AG uses on each tile (attention_gridv2.py: imgsz = max(tile.size)).
        # Using the full-frame max dim here would silently upscale each slice,
        # giving SAHI a higher inference resolution than AG on the same content.
        seq_imgsz = max(slice_h, slice_w)
        det_model = _get_sahi_model(seq_imgsz)

        pred_dir = os.path.join(output_base, "predicted_labels", method, seq.seq_name)
        gt_dir   = os.path.join(output_base, "gt_labels", seq.seq_name)
        os.makedirs(pred_dir, exist_ok=True)
        prepare_gt_labels(seq, gt_dir, dataset)

        print(f"\n  [{method}] {seq.seq_name}  ({n} frames, "
              f"model_imgsz={seq_imgsz}, "
              f"slice={slice_w}×{slice_h} [={seq.img_width}÷{seq.ag_cols} × {seq.img_height}÷{seq.ag_rows}], "
              f"overlap={overlap})")
        t0 = time.time()

        for i, img_path in enumerate(seq.img_paths):
            if i % 200 == 0 or i == n - 1:
                print(f"    Frame {i+1}/{n}")
            frame_pil = Image.open(img_path).convert("RGB")

            result = get_sliced_prediction(
                image                       = frame_pil,
                detection_model             = det_model,
                slice_height                = slice_h,
                slice_width                 = slice_w,
                overlap_height_ratio        = overlap,
                overlap_width_ratio         = overlap,
                postprocess_type            = SAHI_POSTPROCESS_TYPE,
                postprocess_match_threshold = SAHI_POSTPROCESS_THRESH,
                verbose                     = 0,
            )

            boxes_xyxy, classes_arr, scores_arr = _sahi_result_to_arrays(result)
            save_predictions_xyxy(
                img_pil=frame_pil, boxes_xyxy=boxes_xyxy,
                classes=classes_arr, scores=scores_arr,
                pred_dir=pred_dir, image_path=img_path,
                keep_coco_ids=keep_ids, coco_to_local=coco_map,
                min_conf=min_conf,
            )

        seq_time = time.time() - t0
        total_time += seq_time
        seq_sys_m = _extract_sequence_system_metrics(monitor, seq_sample_start)
        metrics = evaluate_sequence(pred_dir, gt_dir)
        metrics.update({
            "seq_name": seq.seq_name,
            "fps": n / seq_time if seq_time > 0 else 0,
            "gpu_power_w": seq_sys_m.get("gpu_power_w"),
            "gpu_util_pct": seq_sys_m.get("gpu_util"),
        })
        if EVALUATE_UNIQUE_OBJECTS:
            metrics.update(evaluate_unique_objects(pred_dir, seq, dataset))
        _print_seq_metrics(metrics, method)
        seq_results.append(metrics)

    monitor.stop()
    agg = aggregate_results(seq_results)
    agg.update({
        "method": method,
        "seq_results": seq_results,
    })
    if EVALUATE_UNIQUE_OBJECTS:
        agg.update(_aggregate_unique(seq_results))
    return agg


def _sahi_result_to_arrays(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract xyxy, class IDs and scores from a SAHI PredictionResult."""
    boxes, classes, scores = [], [], []
    for pred in result.object_prediction_list:
        bb = pred.bbox
        boxes.append([bb.minx, bb.miny, bb.maxx, bb.maxy])
        classes.append(pred.category.id)
        scores.append(pred.score.value)
    if boxes:
        return (np.array(boxes, dtype=float),
                np.array(classes, dtype=int),
                np.array(scores,  dtype=float))
    return (np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0))


# ============================================================
#  METHOD 3 — FRAME-SKIPPING YOLO
# ============================================================

def run_frame_skip_yolo(sequences: List[Sequence], dataset: str,
                        output_base: str, opts: dict, yri: int) -> dict:
    """Full-frame YOLO11s run every `yri` frames; cached predictions reused
    on skipped frames.  This is the simplest possible temporal caching
    baseline — the naive alternative to AG's tile-based selective inference.

    On a YOLO frame: run YOLO on the full frame, save predictions, update cache.
    On a skipped frame: write the cached predictions as-is to the output dir.
    Time is measured wall-clock including cache writes, so FPS reflects real
    achievable throughput (cache write is negligible).
    """
    keep_ids = _DETRAC_KEEP_IDS if dataset == "ua_detrac" else _MOT17_KEEP_IDS
    coco_map = _DETRAC_COCO_MAP_AG if dataset == "ua_detrac" else _MOT17_COCO_MAP_AG
    min_conf = opts["min_conf"]
    method   = f"frame_skip_yri{yri}"

    monitor = SystemMonitor()
    monitor.start()
    total_frames = 0; total_time = 0.0; seq_results = []
    total_yolo_calls = 0; total_cache_hits = 0

    for seq in sequences:
        n = len(seq.img_paths)
        total_frames += n
        seq_sample_start = _monitor_sample_count(monitor)

        # Stress-test parity: MOT17 baseline uses sequence-native imgsz.
        seq_imgsz = _effective_imgsz_for_sequence(seq, dataset)
        clf = HeavyYoloClassifier(weight=opts["yolo_weight"], imgsz=seq_imgsz)

        pred_dir = os.path.join(output_base, "predicted_labels", method, seq.seq_name)
        gt_dir   = os.path.join(output_base, "gt_labels", seq.seq_name)
        os.makedirs(pred_dir, exist_ok=True)
        prepare_gt_labels(seq, gt_dir, dataset)

        print(f"\n  [{method}] {seq.seq_name}  ({n} frames, YRI={yri}, "
              f"{seq.img_width}x{seq.img_height}, imgsz={seq_imgsz})")
        t0 = time.time()

        # Cache: list of txt lines from the last YOLO run, reused on skip frames.
        cached_lines: List[str] = []
        seq_yolo_calls = 0; seq_cache_hits = 0

        for i, img_path in enumerate(seq.img_paths):
            if i % 200 == 0 or i == n - 1:
                yolo_pct = seq_yolo_calls / max(i, 1) * 100
                print(f"    Frame {i+1}/{n}  (YOLO calls so far: "
                      f"{seq_yolo_calls}, {yolo_pct:.0f}%)")

            stem = os.path.splitext(os.path.basename(img_path))[0]
            dst  = os.path.join(pred_dir, stem + ".txt")

            is_yolo_frame = (i % yri == 0)

            if is_yolo_frame:
                frame = Image.open(img_path).convert("RGB")
                _, boxes, classes, scores = clf.predict_image(frame)

                # Build and write the prediction lines
                lines = _build_yolo_txt_lines(
                    frame, boxes, classes, scores,
                    keep_coco_ids=keep_ids, coco_to_local=coco_map,
                    min_conf=min_conf,
                )
                cached_lines = lines
                seq_yolo_calls += 1
            else:
                # Skipped frame — reuse last cached predictions unchanged.
                # (Objects haven't moved since the last YOLO call.)
                lines = cached_lines
                seq_cache_hits += 1

            with open(dst, "w") as f:
                f.write("\n".join(lines))

        seq_time = time.time() - t0
        total_time += seq_time
        total_yolo_calls += seq_yolo_calls
        total_cache_hits += seq_cache_hits
        seq_sys_m = _extract_sequence_system_metrics(monitor, seq_sample_start)

        metrics = evaluate_sequence(pred_dir, gt_dir)
        metrics.update({
            "seq_name":    seq.seq_name,
            "fps":         n / seq_time if seq_time > 0 else 0,
            "yolo_calls":  seq_yolo_calls,
            "cache_hits":  seq_cache_hits,
            "gpu_power_w": seq_sys_m.get("gpu_power_w"),
            "gpu_util_pct": seq_sys_m.get("gpu_util"),
        })
        if EVALUATE_UNIQUE_OBJECTS:
            metrics.update(evaluate_unique_objects(pred_dir, seq, dataset))
        _print_seq_metrics(metrics, method)
        seq_results.append(metrics)

    monitor.stop()
    agg = aggregate_results(seq_results)
    agg.update({
        "method":        method,
        "yri":           yri,
        "seq_results":   seq_results,
        "total_yolo_calls": total_yolo_calls,
        "total_cache_hits": total_cache_hits,
        "yolo_call_pct": total_yolo_calls / total_frames * 100 if total_frames > 0 else 0,
    })
    if EVALUATE_UNIQUE_OBJECTS:
        agg.update(_aggregate_unique(seq_results))
    print(f"\n  Frame-skip stats: {total_yolo_calls} YOLO calls "
          f"({agg['yolo_call_pct']:.0f}%), {total_cache_hits} cache reuses")
    return agg


def _build_yolo_txt_lines(
    img_pil: Image.Image,
    boxes_xywh,
    classes,
    scores,
    keep_coco_ids: set,
    coco_to_local: dict,
    min_conf: float,
) -> List[str]:
    """Build YOLO-format txt lines from a prediction without writing to disk."""
    W, H = img_pil.size
    lines = []
    if boxes_xywh is None or len(boxes_xywh) == 0:
        return lines
    for i, (x1, y1, w, h) in enumerate(boxes_xywh):
        cls_id = int(classes[i]) if classes is not None else 0
        if keep_coco_ids and cls_id not in keep_coco_ids:
            continue
        conf = float(scores[i]) if scores is not None else 1.0
        if conf < min_conf:
            continue
        local_cls = coco_to_local.get(cls_id, cls_id) if coco_to_local else cls_id
        cx = (x1 + w / 2) / W
        cy = (y1 + h / 2) / H
        bw = w / W
        bh = h / H
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{local_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")
    return lines


# ============================================================
#  DISPLAY HELPERS
# ============================================================

def _print_seq_metrics(m: dict, method: str):
    fps_str = f"{m['fps']:.1f} FPS" if "fps" in m else ""
    map_str = f"mAP@50={m['mAP_50']:.4f}"
    prf_str = f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
    uniq_str = ""
    if "detected_unique" in m:
        uniq_str = (f" | Unique {m['detected_unique']}/{m['total_unique']} "
                    f"({m['unique_recall']:.1%})")
    print(f"    {fps_str}  {map_str}  {prf_str}{uniq_str}")


def _aggregate_unique(results: List[dict]) -> dict:
    total    = sum(r.get("total_unique", 0)    for r in results)
    detected = sum(r.get("detected_unique", 0) for r in results)
    return {"total_unique": total, "detected_unique": detected,
            "missed_unique": total - detected,
            "unique_recall": detected / total if total > 0 else 0.0}


def print_comparison_table(method_results: List[dict]):
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  COMPARISON SUMMARY")
    print(sep)
    cols = ["method", "mean_fps", "mAP_50", "mAP_75", "precision", "recall",
            "f1", "unique_recall", "gpu_power_w", "gpu_util"]
    header = (f"  {'Method':<20} {'FPS':>6} {'mAP50':>7} {'mAP75':>7} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'UniqR':>7} "
              f"{'Pwr(W)':>8} {'GPU%':>6}")
    print(header)
    print("  " + "-" * 86)
    for r in method_results:
        if r.get("skipped"):
            print(f"  {r['method']:<20}  (SKIPPED)")
            continue
        pw  = f"{r['gpu_power_w']:.1f}" if r.get("gpu_power_w") is not None else "N/A"
        gpu = f"{r['gpu_util']:.1f}"    if r.get("gpu_util")    is not None else "N/A"
        ur  = f"{r.get('unique_recall', 0.0):.4f}"
        print(f"  {r['method']:<20} {r['mean_fps']:>6.1f} "
              f"{r.get('mAP_50',0):>7.4f} {r.get('mAP_75',0):>7.4f} "
              f"{r.get('precision',0):>7.4f} {r.get('recall',0):>7.4f} "
              f"{r.get('f1',0):>7.4f} {ur:>7} "
              f"{pw:>8} {gpu:>6}")
    print(sep)


# ============================================================
#  CSV EXPORT
# ============================================================

def save_csv(method_results: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Aggregate CSV
    agg_path = os.path.join(out_dir, "comparison_summary.csv")
    fieldnames = ["method", "mean_fps", "mAP_50", "mAP_75", "mAP_50_95",
                  "precision", "recall", "f1", "unique_recall",
                  "total_unique", "detected_unique",
                  "TP", "FP", "FN", "gpu_power_w", "gpu_util"]
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in method_results:
            if r.get("skipped"):
                continue
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\n  Saved summary CSV: {agg_path}")

    # Per-sequence CSV
    seq_path = os.path.join(out_dir, "per_sequence_results.csv")
    seq_fields = ["method", "seq_name", "fps", "mAP_50", "precision", "recall",
                  "f1", "unique_recall", "total_unique", "detected_unique",
                  "gpu_power_w", "gpu_util_pct"]
    with open(seq_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seq_fields, extrasaction="ignore")
        writer.writeheader()
        for r in method_results:
            if r.get("skipped"):
                continue
            method_name = r["method"]
            for sr in r.get("seq_results", []):
                row = {k: sr.get(k, "") for k in seq_fields}
                row["method"] = method_name
                writer.writerow(row)
    print(f"  Saved per-sequence CSV: {seq_path}")


# ============================================================
#  VISUALIZATIONS
# ============================================================

_METHOD_COLORS = {
    "yolo_baseline": "#e74c3c",
    "sahi":          "#3498db",
    # frame_skip variants get shades of green
    "frame_skip_yri2": "#27ae60",
    "frame_skip_yri3": "#2ecc71",
    "frame_skip_yri5": "#a9dfbf",
}

def _method_color(method: str) -> str:
    if method in _METHOD_COLORS:
        return _METHOD_COLORS[method]
    if method.startswith("sahi"):
        return "#3498db"
    if method.startswith("frame_skip"):
        return "#2ecc71"
    return "#95a5a6"

def _bar_chart(ax, methods, values, title, ylabel, higher_better=True):
    colors = [_method_color(m) for m in methods]
    bars   = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticklabels(methods, fontsize=8, rotation=15, ha="right")
    for bar, val in zip(bars, values):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(v for v in values if v is not None) * 1.2 + 1e-6)


def generate_visualizations(method_results: List[dict], out_dir: str):
    valid = [r for r in method_results if not r.get("skipped")]
    if not valid:
        return
    os.makedirs(out_dir, exist_ok=True)

    methods = [r["method"] for r in valid]

    def _vals(key):
        return [r.get(key) or 0.0 for r in valid]

    # ── Figure 1: Throughput, mAP, Power ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Method Comparison — Throughput, Accuracy, Efficiency",
                 fontsize=12, fontweight="bold")
    _bar_chart(axes[0], methods, _vals("mean_fps"),    "Throughput",     "FPS")
    _bar_chart(axes[1], methods, _vals("mAP_50"),      "mAP@0.50",       "mAP")
    _bar_chart(axes[2], methods, _vals("gpu_power_w"), "Power (GPU)", "Watts",
               higher_better=False)
    plt.tight_layout()
    path = os.path.join(out_dir, "throughput_accuracy_power.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── Figure 2: Precision / Recall / F1 / Unique Recall ──────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Method Comparison — Detection Quality",
                 fontsize=12, fontweight="bold")
    _bar_chart(axes[0], methods, _vals("precision"),     "Precision",       "")
    _bar_chart(axes[1], methods, _vals("recall"),        "Recall",          "")
    _bar_chart(axes[2], methods, _vals("f1"),            "F1 Score",        "")
    _bar_chart(axes[3], methods, _vals("unique_recall"), "Unique-Obj Recall","")
    plt.tight_layout()
    path = os.path.join(out_dir, "precision_recall_f1_unique.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── Figure 3: mAP sweep (50 / 75 / 50:95) ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("mAP at Different IoU Thresholds",
                 fontsize=12, fontweight="bold")
    _bar_chart(axes[0], methods, _vals("mAP_50"),    "mAP@0.50",     "mAP")
    _bar_chart(axes[1], methods, _vals("mAP_75"),    "mAP@0.75",     "mAP")
    _bar_chart(axes[2], methods, _vals("mAP_50_95"), "mAP@0.50:0.95","mAP")
    plt.tight_layout()
    path = os.path.join(out_dir, "map_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── Figure 4: Precision–Recall scatter ──────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in valid:
        color = _method_color(r["method"])
        ax.scatter(r.get("recall", 0), r.get("precision", 0),
                   color=color, s=120, zorder=3, label=r["method"])
        ax.annotate(r["method"],
                    (r.get("recall", 0), r.get("precision", 0)),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Recall",    fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision vs. Recall",  fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(out_dir, "precision_recall_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── Figure 5: Per-sequence FPS ──────────────────────────────────
    seq_names = []
    for r in valid:
        for sr in r.get("seq_results", []):
            if sr["seq_name"] not in seq_names:
                seq_names.append(sr["seq_name"])

    if len(seq_names) > 1:
        x = np.arange(len(seq_names))
        width = 0.8 / max(len(valid), 1)
        fig, ax = plt.subplots(figsize=(max(10, len(seq_names) * 1.5), 5))
        for mi, r in enumerate(valid):
            fps_per_seq = {sr["seq_name"]: sr.get("fps", 0) for sr in r.get("seq_results", [])}
            vals = [fps_per_seq.get(s, 0) for s in seq_names]
            offset = (mi - len(valid) / 2 + 0.5) * width
            color  = _method_color(r["method"])
            ax.bar(x + offset, vals, width=width * 0.9, color=color,
                   label=r["method"], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("FPS", fontsize=10)
        ax.set_title("Per-Sequence Throughput by Method", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "per_sequence_fps.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # ── Figure 6: Per-sequence mAP ──────────────────────────────────
    if len(seq_names) > 1:
        fig, ax = plt.subplots(figsize=(max(10, len(seq_names) * 1.5), 5))
        for mi, r in enumerate(valid):
            map_per_seq = {sr["seq_name"]: sr.get("mAP_50", 0) for sr in r.get("seq_results", [])}
            vals   = [map_per_seq.get(s, 0) for s in seq_names]
            offset = (mi - len(valid) / 2 + 0.5) * width
            color  = _method_color(r["method"])
            ax.bar(x + offset, vals, width=width * 0.9, color=color,
                   label=r["method"], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mAP@0.50", fontsize=10)
        ax.set_title("Per-Sequence mAP@0.50 by Method", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "per_sequence_map.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    # ── Dataset selection ────────────────────────────────────────────
    dataset = choose_dataset()
    print(f"\n  Dataset: {dataset.upper()}")

    # ── Sequence discovery ───────────────────────────────────────────
    seq_map = discover_sequences(dataset)
    if not seq_map:
        print("  [ERROR] No sequences found. Check dataset paths.")
        return
    sequences = select_sequences_interactive(seq_map)
    if not sequences:
        print("  No sequences selected.")
        return

    # ── Method options ───────────────────────────────────────────────
    opts = choose_options()

    dataset_out = os.path.join(OUTPUT_DIR, dataset)
    seq_label   = "_".join(s.seq_name for s in sequences[:3])
    if len(sequences) > 3:
        seq_label += f"_and{len(sequences)-3}more"
    weight_tag  = opts["yolo_weight"].replace(".pt", "")
    run_out     = os.path.join(dataset_out, f"{seq_label}_{weight_tag}")
    plots_dir   = os.path.join(run_out, "plots")
    results_dir = os.path.join(run_out, "results")
    os.makedirs(run_out, exist_ok=True)

    n_steps = 1 + (1 if _HAS_SAHI else 0) + len(opts["yri_values"])
    sahi_tag = "SKIP — pip install sahi" if not _HAS_SAHI else \
               f"slice=auto per scene (ag_rows×ag_cols from JSON), overlap={opts['sahi_overlap']}"
    print(f"\n  Output    : {run_out}")
    print(f"  Weight    : {opts['yolo_weight']}   conf≥{opts['min_conf']}")
    print(f"  SAHI      : {sahi_tag}")
    print(f"  Frame-skip: YRI values = {opts['yri_values']}")
    print(f"  Sequences : {[s.seq_name for s in sequences]}\n")

    # ── Run methods ──────────────────────────────────────────────────
    all_results: List[dict] = []
    step = 1

    print("\n" + "=" * 70)
    print(f"  [{step}/{n_steps}] YOLO11s full-frame baseline")
    print("=" * 70)
    step += 1
    bl_result = run_yolo_baseline(sequences, dataset, run_out, opts)
    all_results.append(bl_result)

    print("\n" + "=" * 70)
    print(f"  [{step}/{n_steps}] SAHI (Slicing Aided Hyper Inference)")
    print("=" * 70)
    step += 1
    sahi_result = run_sahi(sequences, dataset, run_out, opts)
    all_results.append(sahi_result)

    for yri in opts["yri_values"]:
        print("\n" + "=" * 70)
        print(f"  [{step}/{n_steps}] Frame-skipping YOLO  (YRI={yri})")
        print("=" * 70)
        step += 1
        fs_result = run_frame_skip_yolo(sequences, dataset, run_out, opts, yri=yri)
        all_results.append(fs_result)

    # ── Results ──────────────────────────────────────────────────────
    print_comparison_table(all_results)
    save_csv(all_results, results_dir)

    print("\n  Generating visualizations …")
    generate_visualizations(all_results, plots_dir)

    print(f"\n  Done.  All output saved to: {run_out}")


if __name__ == "__main__":
    main()
