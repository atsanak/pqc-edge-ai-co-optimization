"""
generate_detrac_tracking_ids.py

Generate per-frame YOLO-format labels enriched with official tracking IDs
from the UA-DETRAC XML annotations.

Primary source: DETRAC-Test-Annotations-XML (official ground truth with
target IDs).  Falls back to IoU-based frame-to-frame linking if the XMLs
are not found.

Output format (one file per frame, same naming as existing labels):
    class x_center y_center w h track_id

Also produces a summary JSON with per-sequence unique object counts.

Usage:
    PYTHONPATH=src python -m eval_tools.generate_detrac_tracking_ids
"""

import os
import re
import sys
import json
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETRAC_ROOT   = os.path.join(_PROJECT_ROOT, "ua_detrac", "content",
                              "UA-DETRAC", "DETRAC_Upload")
LABELS_DIR    = os.path.join(DETRAC_ROOT, "labels")
IMAGES_DIR    = os.path.join(DETRAC_ROOT, "images")
SPLIT         = "val"

# Official XML annotations directory (primary source)
XML_ANNOT_DIR = os.path.join(_PROJECT_ROOT, "DETRAC-Annos",
                              "DETRAC-Test-Annotations-XML")

# Output directory for labels enriched with tracking IDs
OUTPUT_DIR    = os.path.join(DETRAC_ROOT, "labels_with_ids")

# IoU threshold for fallback linking (only used when XMLs not available)
IOU_LINK_THRESHOLD = 0.20

# COCO class mapping for vehicle types found in XML annotations
VEHICLE_TYPE_TO_CLASS = {
    "car": 0,
    "bus": 1,
    "van": 2,
    "others": 2,
}
# Class to use when class-agnostic (all vehicles -> 0)
CLASS_AGNOSTIC = True


# ---------------------------------------------------------------------------
#  XML PARSING
# ---------------------------------------------------------------------------

def get_image_size(seq_name: str, split: str) -> Tuple[int, int]:
    """Get (width, height) of the first image in a sequence."""
    img_dir = os.path.join(IMAGES_DIR, split)
    pattern = os.path.join(img_dir, f"{seq_name}_img00001.jpg")
    matches = glob.glob(pattern)
    if matches:
        import cv2
        img = cv2.imread(matches[0])
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    # Fallback: try to read from existing labels to infer
    return 960, 540  # UA-DETRAC default resolution


def parse_xml_annotations(
    xml_path: str,
    seq_name: str,
    img_w: int,
    img_h: int,
    class_agnostic: bool = CLASS_AGNOSTIC,
) -> Tuple[Dict[str, List[Tuple[int, float, float, float, float, int]]], int, Set[int]]:
    """Parse an official UA-DETRAC XML annotation file.

    Returns:
        frame_data: {label_filename: [(cls, cx_norm, cy_norm, w_norm, h_norm, track_id), ...]}
        n_unique: number of unique tracking IDs
        all_ids: set of all unique tracking IDs
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_ids: Set[int] = set()
    frame_data: Dict[str, List] = {}

    for frame_elem in root.findall("frame"):
        frame_num = int(frame_elem.attrib["num"])
        fname = f"{seq_name}_img{frame_num:05d}.txt"

        targets = []
        for target in frame_elem.findall(".//target"):
            tid = int(target.attrib["id"])
            all_ids.add(tid)

            box_elem = target.find("box")
            left = float(box_elem.attrib["left"])
            top = float(box_elem.attrib["top"])
            w = float(box_elem.attrib["width"])
            h = float(box_elem.attrib["height"])

            # Determine class
            attr_elem = target.find("attribute")
            if class_agnostic:
                cls = 0
            elif attr_elem is not None:
                vtype = attr_elem.attrib.get("vehicle_type", "car").lower()
                cls = VEHICLE_TYPE_TO_CLASS.get(vtype, 0)
            else:
                cls = 0

            # Convert to YOLO normalized format
            cx_norm = (left + w / 2) / img_w
            cy_norm = (top + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clamp to [0, 1]
            cx_norm = max(0.0, min(1.0, cx_norm))
            cy_norm = max(0.0, min(1.0, cy_norm))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            targets.append((cls, cx_norm, cy_norm, w_norm, h_norm, tid))

        frame_data[fname] = targets

    return frame_data, len(all_ids), all_ids


# ---------------------------------------------------------------------------
#  IoU FALLBACK TRACKER (used when XMLs are not available)
# ---------------------------------------------------------------------------

def xywh_to_xyxy(cx: float, cy: float, w: float, h: float):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_yolo_label(filepath: str) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def assign_tracking_ids_iou(
    label_files: List[str],
    iou_threshold: float = IOU_LINK_THRESHOLD,
) -> Tuple[Dict[str, List[Tuple[int, float, float, float, float, int]]], int]:
    next_id = 1
    prev_tracks: List[Tuple[int, float, float, float, float, int]] = []
    results: Dict[str, List] = {}

    for lbl_path in label_files:
        fname = os.path.basename(lbl_path)
        cur_boxes = parse_yolo_label(lbl_path)

        if not cur_boxes:
            results[fname] = []
            continue

        if not prev_tracks:
            cur_tracks = []
            for (cls, cx, cy, w, h) in cur_boxes:
                cur_tracks.append((cls, cx, cy, w, h, next_id))
                next_id += 1
            results[fname] = cur_tracks
            prev_tracks = cur_tracks
            continue

        prev_xyxy = [xywh_to_xyxy(t[1], t[2], t[3], t[4]) for t in prev_tracks]
        cur_xyxy = [xywh_to_xyxy(b[1], b[2], b[3], b[4]) for b in cur_boxes]

        iou_matrix = []
        for i, pa in enumerate(prev_xyxy):
            for j, cb in enumerate(cur_xyxy):
                score = iou(pa, cb)
                if score >= iou_threshold:
                    iou_matrix.append((score, i, j))

        iou_matrix.sort(reverse=True, key=lambda x: x[0])
        matched_prev = set()
        matched_cur = set()
        cur_tracks = [None] * len(cur_boxes)

        for score, pi, cj in iou_matrix:
            if pi in matched_prev or cj in matched_cur:
                continue
            cls, cx, cy, w, h = cur_boxes[cj]
            cur_tracks[cj] = (cls, cx, cy, w, h, prev_tracks[pi][5])
            matched_prev.add(pi)
            matched_cur.add(cj)

        for j in range(len(cur_boxes)):
            if cur_tracks[j] is None:
                cls, cx, cy, w, h = cur_boxes[j]
                cur_tracks[j] = (cls, cx, cy, w, h, next_id)
                next_id += 1

        results[fname] = cur_tracks
        prev_tracks = cur_tracks

    n_unique = next_id - 1
    return results, n_unique


# ---------------------------------------------------------------------------
#  SEQUENCE DISCOVERY
# ---------------------------------------------------------------------------

def discover_sequences(split: str) -> Dict[str, List[str]]:
    """Return {seq_name: [sorted label paths]} for the given split."""
    lbl_dir = os.path.join(LABELS_DIR, split)
    if not os.path.isdir(lbl_dir):
        print(f"  [ERROR] Label directory not found: {lbl_dir}")
        return {}

    all_labels = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
    seq_map: Dict[str, List[str]] = defaultdict(list)
    for lbl_path in all_labels:
        fname = os.path.basename(lbl_path)
        m = re.match(r"(MVI_\\d+)_img\\d+\\.txt", fname)
        if not m:
            m = re.match(r"(MVI_\d+)_img\d+\.txt", fname)
        if m:
            seq_map[m.group(1)].append(lbl_path)
    return dict(sorted(seq_map.items()))


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  UA-DETRAC  —  Generate Tracking IDs for GT Labels")
    print("=" * 65)

    seq_map = discover_sequences(SPLIT)
    if not seq_map:
        print(f"No sequences found for split '{SPLIT}'.")
        sys.exit(1)

    # Check if official XML annotations are available
    use_xml = os.path.isdir(XML_ANNOT_DIR)
    if use_xml:
        xml_files = glob.glob(os.path.join(XML_ANNOT_DIR, "*.xml"))
        xml_by_name = {os.path.splitext(os.path.basename(f))[0]: f
                       for f in xml_files}
        print(f"  Source: Official XML annotations ({len(xml_files)} files)")
    else:
        xml_by_name = {}
        print(f"  Source: IoU-based fallback (XML not found at {XML_ANNOT_DIR})")

    print(f"  Split: {SPLIT}  |  Sequences: {len(seq_map)}")

    out_dir = os.path.join(OUTPUT_DIR, SPLIT)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    total_unique = 0
    xml_used = 0
    iou_used = 0

    for seq_name, label_files in seq_map.items():
        xml_path = xml_by_name.get(seq_name)

        if xml_path and os.path.exists(xml_path):
            # Use official XML annotations
            img_w, img_h = get_image_size(seq_name, SPLIT)
            frame_data, n_unique, all_ids = parse_xml_annotations(
                xml_path, seq_name, img_w, img_h, CLASS_AGNOSTIC)
            xml_used += 1
            source = "XML"

            # Also produce entries for frames that exist in labels but
            # might not be in the XML (e.g., frames with no targets)
            for lbl_path in label_files:
                fname = os.path.basename(lbl_path)
                if fname not in frame_data:
                    frame_data[fname] = []
        else:
            # Fallback to IoU-based tracking
            frame_data, n_unique = assign_tracking_ids_iou(
                label_files, IOU_LINK_THRESHOLD)
            iou_used += 1
            source = "IoU"

        total_unique += n_unique

        # Write enriched labels
        for fname, tracks in frame_data.items():
            out_path = os.path.join(out_dir, fname)
            with open(out_path, "w") as f:
                for (cls, cx, cy, w, h, tid) in tracks:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {tid}\n")

        summary[seq_name] = {
            "n_frames": len(label_files),
            "n_unique_objects": n_unique,
            "source": source,
        }
        print(f"  {seq_name}: {len(label_files)} frames, "
              f"{n_unique} unique objects  [{source}]")

    # Write summary JSON
    summary_path = os.path.join(OUTPUT_DIR, f"{SPLIT}_tracking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total unique objects across {len(seq_map)} sequences: {total_unique}")
    print(f"  XML-sourced: {xml_used}  |  IoU-fallback: {iou_used}")
    print(f"  Labels with IDs saved to: {out_dir}")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
