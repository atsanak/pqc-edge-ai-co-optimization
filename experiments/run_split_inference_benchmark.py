"""
Unified interactive benchmark for the four inference scenarios:

    [1] Edge  + AG        — AttentionGrid runs everything locally on the Jetson.
    [2] Cloud + AG        — Saliency on edge; YOLO on the ground-station server over TLS 1.3.
    [3] Edge  + baseline  — Full-frame YOLO on the edge, no AG.
    [4] Cloud + baseline  — Full-frame YOLO sent whole-frame to the ground-station server.

For each scene, the script measures:
  - mAP@50, Precision, Recall, F1
  - Unique-object recall (UniqR)
  - Edge GPU power (W), GPU util (%), CPU (%), RAM (MB)
  - Server GPU power / util / energy (cloud modes only — pulled from /stats/window)
  - End-to-end FPS

Outputs (under runs/<timestamp>_<mode>_<dataset>/):
  - per_sequence_results.csv   per-scene metrics
  - summary.csv                aggregate (arithmetic mean across scenes)
  - run_config.json            what was run, with which knobs
  - predicted_labels/...       YOLO-format predictions per frame (intermediate,
                               deleted after summary.csv is written unless
                               --keep-labels is passed)
  - gt_labels/...              ground-truth YOLO labels mirrored from dataset
                               (same cleanup policy as predicted_labels)

Run it from the repo root:

    PYTHONPATH=src python experiments/run_split_inference_benchmark.py
"""

# ─────────────────────────── Phase 0: stdlib + path setup ───────────────────────────
import os
import sys
import csv
import json
import time
import shutil
import argparse
import datetime
from dataclasses import asdict
from typing import List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)

# ─────────────────────────── Phase 1: GPU / device selection ────────────────────────
# Same picker the rest of the repo uses. Handles CUDA / MPS / CPU.
from gpu_utils import select_gpu

# ─────────────────────────── Phase 2: heavy imports (after GPU pick) ────────────────
# NOTE: we DON'T call select_gpu() at import-time, because for cloud modes
# the edge doesn't need a GPU at all. The runner asks for a mode first,
# then selects a device only if needed.
import numpy as np                                                  # noqa: E402
from PIL import Image                                                # noqa: E402

# Reuse the existing infrastructure that benchmark_baselines.py already
# tested and tuned (Sequence dataclass, discover_sequences, GT prep,
# per-sequence evaluation, unique-object eval, scene config loader).
from eval_tools.benchmark_baselines import (                         # noqa: E402
    Sequence,
    discover_sequences,
    select_sequences_interactive,
    prepare_gt_labels,
    evaluate_sequence,
    evaluate_unique_objects,
    _DETRAC_KEEP_IDS,
    _DETRAC_COCO_MAP_AG,
    _MOT17_KEEP_IDS,
    _MOT17_COCO_MAP_AG,
    _effective_imgsz_for_sequence,
    _extract_sequence_system_metrics,
    _monitor_sample_count,
)
from object_clfs.heavy_yolo_classifier import (                      # noqa: E402
    HeavyYoloClassifier,
    save_yolo_txt,
)
from eval_tools.performance_eval.system_monitor import SystemMonitor # noqa: E402


# ─────────────────────────── menu helpers ────────────────────────────────

MODE_CHOICES = [
    ("edge_ag",       "Edge  + AG        — saliency + YOLO on this Jetson"),
    ("cloud_ag",      "Cloud + AG        — saliency here, YOLO on ground-station over TLS"),
    ("edge_baseline", "Edge  + baseline  — full-frame YOLO on this Jetson, no AG"),
    ("cloud_baseline","Cloud + baseline  — full-frame YOLO sent whole to ground-station"),
]
DATASET_CHOICES = ["ua_detrac", "mot17"]


def _menu_pick(prompt: str, options: list, default_idx: int = 0) -> int:
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = " (default)" if i == default_idx else ""
        print(f"  [{i+1}] {opt}{marker}")
    while True:
        raw = input(f"Choice [1-{len(options)}] (default {default_idx+1}): ").strip()
        if raw == "":
            return default_idx
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  Please enter 1..{len(options)}.")


def _ask_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


# ── Transport-security picker (cloud modes only) ───────────────────────
#
# Curated short list of (kem, sig) pairs designed for coverage of the NIST
# PQC ecosystem in a small number of cells, plus an automated sweep path and
# a full liboqs registry escape hatch.

# Imported lazily inside the picker so the rest of the script still
# imports even on hosts without liboqs.

def _pick_crypto_mode_interactive(default_kem: str, default_sig: str
                                  ) -> tuple[str, str, str, str, "str | None"]:
    """Return ``(crypto_mode, kem, sig, sweep_profile, pairs_file)``.

    - ``crypto_mode`` is one of ``tls13`` / ``classical`` / ``pqc`` /
      ``SWEEP_SENTINEL`` / ``FULL_SUITE_SENTINEL``.
    - When ``crypto_mode == SWEEP_SENTINEL`` the caller must iterate over
      every pair returned by ``pqc_sweep.expand_profile(sweep_profile)`` if
      ``pairs_file`` is ``None``, or over the pairs loaded from
      ``pairs_file`` if it is set (resume mode).
    - When ``crypto_mode == FULL_SUITE_SENTINEL`` the caller runs one
      classical pass then the full paper_curated PQC sweep without further
      prompts.
    - In every other branch ``sweep_profile`` is the empty string and
      ``pairs_file`` is ``None``.
    """
    from network import pqc_sweep as ps

    print("\nTransport security:")
    print("  [1] TLS 1.3 only               — raw HTTPS, no app-layer auth")
    print("                                   (TLS itself uses ECDHE+ECDSA but the")
    print("                                    JPEG payload rides the channel as-is)")
    print("  [2] Classical hybrid            — same protocol shape as PQC but")
    print("                                   ECDH+ECDSA app-layer crypto only")
    print("                                   (correct cost-delta baseline for PQC)")
    print("  [3] PQC hybrid                  — ML-KEM + ML-DSA / Falcon / SLH-DSA")
    print("                                   per-call sign+encrypt over TLS 1.3")
    print("  [4] FULL SUITE (no prompts)     — runs [2] classical once, then all")
    print("                                   paper-novelty PQC pairs automatically")
    print("  [5] RESUME from file            — re-run only the (KEM, SIG) pairs")
    print("                                   listed in a file (no classical pass).")
    print("                                   Default file: missing_pairs.txt")
    raw = input("Choice [1-5] (default 1): ").strip()
    if raw in ("", "1"):
        return "tls13", default_kem, default_sig, "", None
    if raw == "2":
        return "classical", "CLASSICAL", "CLASSICAL", "", None
    if raw == "4":
        # Pre-flight: count runnable pairs so the user knows what they signed up for.
        pairs_all = ps.expand_profile("paper_curated")
        runnable, skipped = ps.runnable_pairs(pairs_all)
        print(f"\n  FULL SUITE: 1 classical pass + {len(runnable)} PQC pairs "
              f"({len(skipped)} skipped / not in this liboqs build).")
        if not runnable:
            print("  [WARN] no runnable PQC pairs — only classical will run.")
        return ps.FULL_SUITE_SENTINEL, "", "", "paper_curated", None
    if raw == "5":
        default_path = os.path.join(_PROJECT_ROOT, "missing_pairs.txt")
        default_label = ("missing_pairs.txt" if os.path.isfile(default_path)
                         else "<no default found>")
        path = _ask_text("Path to pairs file", default_label)
        # Allow the user to accept the default by hitting Enter.
        if path in ("missing_pairs.txt", "<no default found>"):
            path = default_path
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            print(f"  [WARN] {path!r} not found; falling back to TLS 1.3.")
            return "tls13", default_kem, default_sig, "", None
        try:
            pairs = _load_pqc_pairs_file(path)
        except SystemExit as e:
            print(f"  [WARN] {e}; falling back to TLS 1.3.")
            return "tls13", default_kem, default_sig, "", None
        runnable, skipped = ps.runnable_pairs(pairs)
        print(f"\n  RESUME: {len(pairs)} pair(s) in file, "
              f"{len(runnable)} runnable on this liboqs build, "
              f"{len(skipped)} skipped.")
        if not runnable:
            print("  [WARN] no runnable pairs in the file; falling back to TLS 1.3.")
            return "tls13", default_kem, default_sig, "", None
        return ps.SWEEP_SENTINEL, "", "", "", path
    if raw != "3":
        print("  [WARN] unrecognised choice; falling back to TLS 1.3.")
        return "tls13", default_kem, default_sig, "", None

    # PQC: curated max-coverage list + auto sweep + expert path
    n_curated = len(ps.MAX_COVERAGE_PAIRS)
    print(f"\nPQC scheme pair ({n_curated} curated pairs for max PQC-family coverage):")
    for i, (kem, sig, label) in enumerate(ps.MAX_COVERAGE_PAIRS, start=1):
        marker = "*" if (kem == default_kem and sig == default_sig) else " "
        print(f"  [{i:2d}]{marker} {label}")
    auto_idx   = n_curated + 1
    expert_idx = n_curated + 2
    print(f"  [{auto_idx}]  AUTO SWEEP   — run a named sweep profile")
    print(f"  [{expert_idx}]  Browse full liboqs registry (expert)")
    pick = input(f"Choice [1-{expert_idx}] "
                 f"(default ML-KEM-768 + ML-DSA-44): ").strip()

    if pick == "":
        return "pqc", "ML-KEM-768", "ML-DSA-44", "", None
    try:
        i = int(pick)
    except ValueError:
        print("  [WARN] not a number; using ML-KEM-768 + ML-DSA-44.")
        return "pqc", "ML-KEM-768", "ML-DSA-44", "", None
    if 1 <= i <= n_curated:
        kem, sig, _ = ps.MAX_COVERAGE_PAIRS[i - 1]
        return "pqc", kem, sig, "", None

    # AUTO SWEEP — pick a sub-profile
    if i == auto_idx:
        print("\nSweep profile:")
        print(f"  [1] max_coverage       — {n_curated} curated pairs above")
        print("  [2] paper_curated      — public artifact sweep")
        print("  [3] standard           — 6 KEMs × 7 sigs  (≈ 42 pairs)")
        print("  [4] extended           — 12 KEMs × 19 sigs (≈ 228 pairs)")
        print("  [5] paper_curated      — same as option 2")
        print("  [6] stress             — high-cost cells only (≈ 8 pairs)")
        ans = input("Choice [1-6] (default 2): ").strip() or "2"
        prof_map = {"1": "max_coverage", "2": "paper_curated",
                    "3": "standard",     "4": "extended",
                    "5": "paper_curated", "6": "stress"}
        prof = prof_map.get(ans, "paper_curated")
        pairs = ps.expand_profile(prof)
        runnable, skipped = ps.runnable_pairs(pairs)
        print(f"  → profile '{prof}': {len(pairs)} pairs total, "
              f"{len(runnable)} runnable on this liboqs build, "
              f"{len(skipped)} skipped (missing from liboqs).")
        if not runnable:
            print("  [WARN] no runnable pairs; falling back to single ML-KEM-768 + ML-DSA-44.")
            return "pqc", "ML-KEM-768", "ML-DSA-44", "", None
        return ps.SWEEP_SENTINEL, "", "", prof, None

    # Expert path
    try:
        from network.pqc_crypto import list_available_kems, list_available_signatures
        all_kems = list_available_kems()
        all_sigs = list_available_signatures()
    except Exception as exc:
        print(f"  [WARN] could not enumerate liboqs registry: {exc}. "
              "Falling back to ML-KEM-768 + ML-DSA-44.")
        return "pqc", "ML-KEM-768", "ML-DSA-44", "", None

    print(f"\n  liboqs lists {len(all_kems)} KEMs.")
    print("  " + ", ".join(all_kems))
    kem = input("Enter KEM name (default ML-KEM-768): ").strip() or "ML-KEM-768"
    print(f"\n  liboqs lists {len(all_sigs)} signature schemes.")
    print("  " + ", ".join(all_sigs[:60]) + (" ..." if len(all_sigs) > 60 else ""))
    sig = input("Enter SIG name (default ML-DSA-44): ").strip() or "ML-DSA-44"
    return "pqc", kem, sig, "", None


def _ask_yn(prompt: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    if raw == "":
        return default
    return raw.startswith("y")


# ─────────────────────────── per-frame pipelines ───────────────────────────

def _coco_keep_and_map(dataset: str):
    if dataset == "ua_detrac":
        return _DETRAC_KEEP_IDS, _DETRAC_COCO_MAP_AG
    return _MOT17_KEEP_IDS, _MOT17_COCO_MAP_AG


def _load_scene_configs(dataset: str) -> dict:
    """Per-scene AG configs (grid + V2 params). Used by AG modes only."""
    cfg_path = os.path.join(_PROJECT_ROOT, "configs", f"{dataset}_scene_configs.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing scene config: {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _merge_scene_v2_params(json_cfg: dict, seq_name: str):
    """Return (rows, cols, merged_v2_params) for one scene."""
    defaults = dict(json_cfg.get("default_v2_params", {}))
    if seq_name in json_cfg.get("sequences", {}):
        scfg = json_cfg["sequences"][seq_name]
        rows = scfg.get("rows", 3)
        cols = scfg.get("cols", 5)
        defaults.update(scfg.get("v2_params", {}))
    else:
        rows, cols = 3, 5
    return rows, cols, defaults


def _build_ag(rows, cols, params, inference_mode, remote_url, remote_cafile,
              crypto_mode: str = "tls13",
              pqc_kem: str = "ML-KEM-768",
              pqc_sig: str = "ML-DSA-44"):
    """Construct AttentionGridV2 in either local or remote inference mode."""
    from attention_gridv2 import AttentionGrid as AttentionGridV2
    return AttentionGridV2(
        rows=rows, cols=cols,
        # Saliency / tiling
        saliency_method=params.get("saliency_method", "frame_diff"),
        saliency_measurement=params.get("saliency_measurement", "pixel_count"),
        saliency_scale=params.get("saliency_scale", 0.5),
        enable_tile_combination=params.get("enable_tile_combination", True),
        max_combined_tiles=params.get("max_combined_tiles", 4),
        recheck_threshold=params.get("recheck_threshold", 3),
        enable_recheck_tile=params.get("enable_recheck_tile", False),
        fullframe_every=params.get("fullframe_every", 30),
        # YOLO
        yolo_weight=params.get("yolo_weight", "yolo11s.pt"),
        use_finetuned=params.get("use_finetuned", False),
        # Prediction fusion / LKT
        prediction_fusion=params.get("prediction_fusion", True),
        fusion_edge_margin_pct=params.get("fusion_edge_margin_pct", 0.02),
        enable_lkt_tracking=params.get("enable_lkt_tracking", False),
        # Class filter
        class_filter=params.get("class_filter", None),
        # Saliency suppression
        enable_saliency_suppression=params.get("enable_saliency_suppression", True),
        saliency_suppression_rate=params.get("saliency_suppression_rate", 2.0),
        saliency_suppression_decay=params.get("saliency_suppression_decay", 0.05),
        yolo_run_interval=params.get("yolo_run_interval", 1),
        # ── Split-inference knobs ──
        inference_mode=inference_mode,            # "local" or "remote"
        remote_url=remote_url,
        remote_cafile=remote_cafile,
        crypto_mode=crypto_mode,
        pqc_kem_scheme=pqc_kem,
        pqc_sig_scheme=pqc_sig,
    )


# ─────────────────────────── per-scene drivers ───────────────────────────

def _run_scene_ag(seq: Sequence, dataset: str, mode_key: str,
                  json_cfg: dict, out_dir: str,
                  remote_url: Optional[str], remote_cafile: Optional[str],
                  monitor: SystemMonitor, min_conf: float,
                  remote_detector_holder: dict,
                  crypto_mode: str = "tls13",
                  pqc_kem: str = "ML-KEM-768",
                  pqc_sig: str = "ML-DSA-44") -> dict:
    """Run one scene through AttentionGridV2 (mode = edge_ag OR cloud_ag)."""
    keep_ids, coco_map = _coco_keep_and_map(dataset)
    rows, cols, params = _merge_scene_v2_params(json_cfg, seq.seq_name)
    inference_mode = "remote" if mode_key == "cloud_ag" else "local"

    pred_dir = os.path.join(out_dir, "predicted_labels", seq.seq_name)
    gt_dir   = os.path.join(out_dir, "gt_labels",         seq.seq_name)
    os.makedirs(pred_dir, exist_ok=True)
    prepare_gt_labels(seq, gt_dir, dataset)

    print(f"\n  [AG · {inference_mode}] {seq.seq_name}  "
          f"grid={rows}x{cols}  ({len(seq.img_paths)} frames)")

    ag = _build_ag(rows, cols, params, inference_mode, remote_url, remote_cafile,
                   crypto_mode=crypto_mode, pqc_kem=pqc_kem, pqc_sig=pqc_sig)

    # Capture the RemoteDetector for later stats summarisation (cloud_ag).
    if inference_mode == "remote":
        # AG instantiates the RemoteDetector lazily on first frame; we'll grab
        # it after the first inference call below.
        pass

    seq_sample_start = _monitor_sample_count(monitor)
    t0 = time.time()
    for idx, img_path in enumerate(seq.img_paths):
        frame = Image.open(img_path).convert("RGB")
        _, boxes_xywh, classes, scores = ag.process_frame(frame)
        save_yolo_txt(
            img_pil=frame, boxes_xywh=boxes_xywh,
            classes=classes, scores=scores,
            label_dir=pred_dir, image_path=img_path,
            keep_coco_ids=keep_ids, coco_to_local=coco_map,
            min_conf=min_conf,
        )
        if idx % 200 == 0:
            print(f"    frame {idx+1}/{len(seq.img_paths)}")

    seq_time = time.time() - t0

    # If we ran in remote mode, fish out the RemoteDetector instance(s)
    # AG cached internally so we can record bandwidth + RTT stats.
    network_stats = None
    if inference_mode == "remote":
        try:
            rds = [d for d in ag.yolo_clfs.values()
                   if d.__class__.__name__ in ("RemoteDetector", "PQCRemoteDetector")]
            if rds:
                # Sum across all RemoteDetector imgsz caches (usually 1).
                summary = {
                    "num_requests": 0,
                    "total_upload_MB": 0.0,
                    "total_download_MB": 0.0,
                    "total_rtt_ms": 0.0,
                    "total_server_infer_ms": 0.0,
                }
                for d in rds:
                    s = d.stats_summary()
                    summary["num_requests"] += d.num_requests
                    summary["total_upload_MB"]    += s["total_upload_MB"]
                    summary["total_download_MB"]  += s["total_download_MB"]
                    summary["total_rtt_ms"]       += d.total_network_rtt_ms
                    summary["total_server_infer_ms"] += d.total_server_infer_ms
                network_stats = summary
                # Hand the first RemoteDetector to the caller so it can
                # open a server stats window around the scene.
                remote_detector_holder["det"] = rds[0]
                # Stash ALL per-imgsz remote detectors so PQC stats can be
                # aggregated across them (single-detector reporting was the
                # cause of the "PQC columns are all zero" issue when AG
                # cached multiple supertile sizes).
                remote_detector_holder["all_dets"] = rds
        except Exception as e:  # pragma: no cover
            print(f"  [WARN] couldn't summarise remote stats: {e}")

    edge_sys = _extract_sequence_system_metrics(monitor, seq_sample_start)

    metrics = evaluate_sequence(pred_dir, gt_dir)
    metrics.update(evaluate_unique_objects(pred_dir, seq, dataset))
    metrics.update({
        "seq_name": seq.seq_name,
        "fps":           len(seq.img_paths) / seq_time if seq_time > 0 else 0.0,
        "edge_gpu_power_w": edge_sys.get("gpu_power_w"),
        "edge_gpu_util":    edge_sys.get("gpu_util"),
        "edge_cpu_pct":     edge_sys.get("cpu_pct"),
        "edge_ram_mb":      edge_sys.get("ram_mb"),
        "edge_gpu_temp_c":  edge_sys.get("gpu_temp_c"),
        "edge_cpu_temp_c":  edge_sys.get("cpu_temp_c"),
        "edge_sample_count": edge_sys.get("n_samples", 0),
        "ag_grid":   f"{rows}x{cols}",
        "n_frames":  len(seq.img_paths),
        "seq_time":  seq_time,
        "mode":      mode_key,
    })
    if network_stats:
        metrics["net_num_requests"]      = network_stats["num_requests"]
        metrics["net_upload_MB"]         = network_stats["total_upload_MB"]
        metrics["net_download_MB"]       = network_stats["total_download_MB"]
        metrics["net_avg_rtt_ms"]        = (network_stats["total_rtt_ms"]
                                            / max(network_stats["num_requests"], 1))
        metrics["net_avg_server_ms"]     = (network_stats["total_server_infer_ms"]
                                            / max(network_stats["num_requests"], 1))
    # Pull PQC metrics from every remote detector AG cached (one per imgsz).
    # The helper gracefully fills sentinel values when no PQC detector
    # touched the scene.
    all_dets = remote_detector_holder.get("all_dets")
    if all_dets is None:
        det_one = remote_detector_holder.get("det")
        all_dets = [det_one] if det_one is not None else []
    metrics.update(_pqc_metrics_from_detectors(all_dets, crypto_mode))
    return metrics


def _run_scene_local_baseline(seq: Sequence, dataset: str,
                              out_dir: str, monitor: SystemMonitor,
                              min_conf: float, yolo_weight: str) -> dict:
    """Full-frame YOLO on the edge, no AG (mode = edge_baseline)."""
    keep_ids, coco_map = _coco_keep_and_map(dataset)
    seq_imgsz = _effective_imgsz_for_sequence(seq, dataset)
    clf = HeavyYoloClassifier(weight=yolo_weight, imgsz=seq_imgsz)

    pred_dir = os.path.join(out_dir, "predicted_labels", seq.seq_name)
    gt_dir   = os.path.join(out_dir, "gt_labels",         seq.seq_name)
    os.makedirs(pred_dir, exist_ok=True)
    prepare_gt_labels(seq, gt_dir, dataset)

    print(f"\n  [baseline · local] {seq.seq_name}  imgsz={seq_imgsz}  "
          f"({len(seq.img_paths)} frames)")

    seq_sample_start = _monitor_sample_count(monitor)
    t0 = time.time()
    for idx, img_path in enumerate(seq.img_paths):
        frame = Image.open(img_path).convert("RGB")
        _, boxes_xywh, classes, scores = clf.predict_image(frame)
        save_yolo_txt(
            img_pil=frame, boxes_xywh=boxes_xywh,
            classes=classes, scores=scores,
            label_dir=pred_dir, image_path=img_path,
            keep_coco_ids=keep_ids, coco_to_local=coco_map,
            min_conf=min_conf,
        )
        if idx % 200 == 0:
            print(f"    frame {idx+1}/{len(seq.img_paths)}")
    seq_time = time.time() - t0

    edge_sys = _extract_sequence_system_metrics(monitor, seq_sample_start)
    metrics = evaluate_sequence(pred_dir, gt_dir)
    metrics.update(evaluate_unique_objects(pred_dir, seq, dataset))
    metrics.update({
        "seq_name": seq.seq_name,
        "fps":           len(seq.img_paths) / seq_time if seq_time > 0 else 0.0,
        "edge_gpu_power_w": edge_sys.get("gpu_power_w"),
        "edge_gpu_util":    edge_sys.get("gpu_util"),
        "edge_cpu_pct":     edge_sys.get("cpu_pct"),
        "edge_ram_mb":      edge_sys.get("ram_mb"),
        "edge_gpu_temp_c":  edge_sys.get("gpu_temp_c"),
        "edge_cpu_temp_c":  edge_sys.get("cpu_temp_c"),
        "edge_sample_count": edge_sys.get("n_samples", 0),
        "ag_grid":   None,
        "n_frames":  len(seq.img_paths),
        "seq_time":  seq_time,
        "mode":      "edge_baseline",
    })
    return metrics


def _run_scene_remote_baseline(seq: Sequence, dataset: str,
                               remote_url: str, remote_cafile: Optional[str],
                               out_dir: str, monitor: SystemMonitor,
                               min_conf: float,
                               remote_detector_holder: dict,
                               crypto_mode: str = "tls13",
                               pqc_kem: str = "ML-KEM-768",
                               pqc_sig: str = "ML-DSA-44") -> dict:
    """Send each FULL frame to the ground-station server (mode = cloud_baseline)."""
    keep_ids, coco_map = _coco_keep_and_map(dataset)
    seq_imgsz = _effective_imgsz_for_sequence(seq, dataset)
    if crypto_mode in ("pqc", "classical"):
        from network.pqc_client import PQCRemoteDetector
        det = PQCRemoteDetector(
            url=remote_url,
            crypto_mode=crypto_mode,
            kem_scheme=pqc_kem,
            sig_scheme=pqc_sig,
            imgsz=seq_imgsz,
            cafile=remote_cafile,
            sequence_name=seq.seq_name,
        )
    else:
        from network.client import RemoteDetector
        det = RemoteDetector(
            url=remote_url,
            imgsz=seq_imgsz,
            cafile=remote_cafile,
            sequence_name=seq.seq_name,
        )
    remote_detector_holder["det"] = det

    pred_dir = os.path.join(out_dir, "predicted_labels", seq.seq_name)
    gt_dir   = os.path.join(out_dir, "gt_labels",         seq.seq_name)
    os.makedirs(pred_dir, exist_ok=True)
    prepare_gt_labels(seq, gt_dir, dataset)

    print(f"\n  [baseline · cloud] {seq.seq_name}  imgsz={seq_imgsz}  "
          f"({len(seq.img_paths)} frames)")

    seq_sample_start = _monitor_sample_count(monitor)
    t0 = time.time()
    for idx, img_path in enumerate(seq.img_paths):
        frame = Image.open(img_path).convert("RGB")
        det.set_next_call_context(
            crop_bounds=(0, 0, frame.size[0], frame.size[1]),
            frame_size=(frame.size[0], frame.size[1]),
            frame_id=idx,
        )
        _, boxes_xywh, classes, scores = det.predict_image(frame)
        save_yolo_txt(
            img_pil=frame, boxes_xywh=boxes_xywh,
            classes=classes, scores=scores,
            label_dir=pred_dir, image_path=img_path,
            keep_coco_ids=keep_ids, coco_to_local=coco_map,
            min_conf=min_conf,
        )
        if idx % 200 == 0:
            print(f"    frame {idx+1}/{len(seq.img_paths)}")
    seq_time = time.time() - t0

    edge_sys = _extract_sequence_system_metrics(monitor, seq_sample_start)
    metrics = evaluate_sequence(pred_dir, gt_dir)
    metrics.update(evaluate_unique_objects(pred_dir, seq, dataset))
    s = det.stats_summary()
    metrics.update({
        "seq_name": seq.seq_name,
        "fps":           len(seq.img_paths) / seq_time if seq_time > 0 else 0.0,
        "edge_gpu_power_w": edge_sys.get("gpu_power_w"),
        "edge_gpu_util":    edge_sys.get("gpu_util"),
        "edge_cpu_pct":     edge_sys.get("cpu_pct"),
        "edge_ram_mb":      edge_sys.get("ram_mb"),
        "edge_gpu_temp_c":  edge_sys.get("gpu_temp_c"),
        "edge_cpu_temp_c":  edge_sys.get("cpu_temp_c"),
        "edge_sample_count": edge_sys.get("n_samples", 0),
        "ag_grid":   None,
        "n_frames":  len(seq.img_paths),
        "seq_time":  seq_time,
        "mode":      "cloud_baseline",
        "net_num_requests":  det.num_requests,
        "net_upload_MB":     s["total_upload_MB"],
        "net_download_MB":   s["total_download_MB"],
        "net_avg_rtt_ms":    s["avg_rtt_ms"],
        "net_avg_server_ms": s["avg_server_inference_ms"],
    })
    metrics.update(_pqc_metrics_from_detector(det, crypto_mode))
    return metrics


def _pqc_metrics_from_detectors(detectors, crypto_mode: str,
                                sysmon_window: dict | None = None,
                                baseline: dict | None = None,
                                baseline_throughput_mbps: float | None = None) -> dict:
    """Compute the PQC CSV columns from one *or more* PQCRemoteDetector
    instances. AG caches one detector per ``imgsz``, so a single cloud_ag
    scene can carry 2-3 detectors that each handshook once and processed a
    subset of frames; we aggregate them so every column reflects the *whole*
    scene rather than the slice handled by ``rds[0]``.

    Always emits ``crypto_mode``, ``kem_scheme``, ``sig_scheme`` so the CSV
    has uniform columns across runs (TLS-1.3 rows get sentinel values).
    """
    base_keys = [
        "kem_scheme", "sig_scheme",
        "handshake_ms_client", "handshake_ms_server", "handshake_rtt_ms",
        # phase averages
        "avg_sign_ec_ms", "avg_sign_pq_ms",
        "avg_encrypt_ms", "avg_decrypt_ms",
        "avg_verify_ec_ms", "avg_verify_pq_ms",
        "avg_server_decrypt_ms", "avg_server_verify_ec_ms",
        "avg_server_verify_pq_ms", "avg_server_sign_ec_ms",
        "avg_server_sign_pq_ms", "avg_server_encrypt_ms",
        # bytes
        "pre_crypto_upload_MB", "pre_crypto_download_MB",
        "crypto_expansion_ratio",
        # distribution shape on the dominant phases
        "sign_pq_p50_ms", "sign_pq_p95_ms", "sign_pq_p99_ms",
        "sign_pq_cv_pct", "sign_pq_skewness", "sign_pq_excess_kurtosis",
        "sign_pq_tail_ratio_p99_p50", "sign_pq_determinism_class",
        "encrypt_p50_ms", "encrypt_p95_ms", "encrypt_p99_ms",
        "encrypt_cv_pct", "encrypt_determinism_class",
        "rtt_p50_ms", "rtt_p95_ms", "rtt_p99_ms",
        "rtt_cv_pct", "rtt_jitter_pct", "rtt_jitter_class",
        # bottleneck attribution
        "sign_pct", "encrypt_pct", "network_pct",
        "dominant_phase", "sign_to_encrypt_ratio", "sign_to_network_ratio",
        # throughput
        "throughput_mbps", "goodput_mbps",
        "chunk_rate_hz", "bandwidth_overhead_pct",
    ]
    out: dict = {"crypto_mode": crypto_mode, "kem_scheme": "NONE", "sig_scheme": "NONE"}
    for k in base_keys:
        out.setdefault(k, None)

    # Normalise to a list, drop None entries and non-PQC detectors.
    if detectors is None:
        return out
    if not isinstance(detectors, (list, tuple)):
        detectors = [detectors]
    pqc_dets = [d for d in detectors
                if d is not None
                and d.__class__.__name__ == "PQCRemoteDetector"]
    if not pqc_dets:
        return out

    from network.pqc_client import PQCRemoteDetector
    agg = PQCRemoteDetector.aggregate_stats(pqc_dets)
    # The legacy *_ms / per-phase averages stay first so the canonical
    # master row below can overwrite them.

    n_req = max(agg.get("num_requests", 0), 1)

    # Helpers to pull <phase>__<stat> out of the aggregate.
    def avg_of(phase: str) -> float:
        return agg.get(f"{phase}__mean", 0.0) or 0.0

    out.update({
        "crypto_mode":             crypto_mode,
        "kem_scheme":              agg["kem_scheme"],
        "sig_scheme":              agg["sig_scheme"],
        "handshake_ms_client":     agg["handshake_ms_client"],
        "handshake_ms_server":     agg["handshake_ms_server"],
        "handshake_rtt_ms":        agg["handshake_rtt_ms"],

        "avg_sign_ec_ms":          avg_of("sign_ec"),
        "avg_sign_pq_ms":          avg_of("sign_pq"),
        "avg_encrypt_ms":          avg_of("encrypt"),
        "avg_decrypt_ms":          avg_of("decrypt"),
        "avg_verify_ec_ms":        avg_of("verify_ec"),
        "avg_verify_pq_ms":        avg_of("verify_pq"),
        "avg_server_decrypt_ms":   avg_of("server_decrypt"),
        "avg_server_verify_ec_ms": avg_of("server_verify_ec"),
        "avg_server_verify_pq_ms": avg_of("server_verify_pq"),
        "avg_server_sign_ec_ms":   avg_of("server_sign_ec"),
        "avg_server_sign_pq_ms":   avg_of("server_sign_pq"),
        "avg_server_encrypt_ms":   avg_of("server_encrypt"),

        "pre_crypto_upload_MB":    agg["pre_crypto_upload_MB"],
        "pre_crypto_download_MB":  agg["pre_crypto_download_MB"],
        "crypto_expansion_ratio":  agg["crypto_expansion_ratio_up"],

        # distribution shape on dominant phases
        "sign_pq_p50_ms":              agg.get("sign_pq__p50"),
        "sign_pq_p95_ms":              agg.get("sign_pq__p95"),
        "sign_pq_p99_ms":              agg.get("sign_pq__p99"),
        "sign_pq_cv_pct":              agg.get("sign_pq__cv_pct"),
        "sign_pq_skewness":            agg.get("sign_pq__skewness"),
        "sign_pq_excess_kurtosis":     agg.get("sign_pq__excess_kurtosis"),
        "sign_pq_tail_ratio_p99_p50":  agg.get("sign_pq__tail_ratio_p99_p50"),
        "sign_pq_determinism_class":   agg.get("sign_pq__determinism_class"),
        "encrypt_p50_ms":              agg.get("encrypt__p50"),
        "encrypt_p95_ms":              agg.get("encrypt__p95"),
        "encrypt_p99_ms":              agg.get("encrypt__p99"),
        "encrypt_cv_pct":              agg.get("encrypt__cv_pct"),
        "encrypt_determinism_class":   agg.get("encrypt__determinism_class"),
        "rtt_p50_ms":                  agg.get("rtt__p50"),
        "rtt_p95_ms":                  agg.get("rtt__p95"),
        "rtt_p99_ms":                  agg.get("rtt__p99"),
        "rtt_cv_pct":                  agg.get("rtt__cv_pct"),
        "rtt_jitter_pct":              agg.get("rtt__jitter_pct"),
        "rtt_jitter_class":            agg.get("rtt__jitter_class"),

        # bottleneck attribution
        "sign_pct":                    agg["sign_pct"],
        "encrypt_pct":                 agg["encrypt_pct"],
        "network_pct":                 agg["network_pct"],
        "dominant_phase":              agg["dominant_phase"],
        "sign_to_encrypt_ratio":       agg["sign_to_encrypt_ratio"],
        "sign_to_network_ratio":       agg["sign_to_network_ratio"],

        # throughput
        "throughput_mbps":             agg["throughput_mbps"],
        "goodput_mbps":                agg["goodput_mbps"],
        "chunk_rate_hz":               agg["chunk_rate_hz"],
        "bandwidth_overhead_pct":      agg["bandwidth_overhead_pct"],
    })
    # If the caller passed an explicit baseline_throughput_mbps it overrides
    # whatever was in `baseline`.  Either form ends up as a dict with at least
    # a "throughput_mbps" key, which is what build_master_row consumes.
    if baseline_throughput_mbps is not None and baseline_throughput_mbps > 0:
        baseline = {"throughput_mbps": float(baseline_throughput_mbps)}
    # Master row last: flattened transport columns are authoritative.
    master_row = PQCRemoteDetector.build_master_row(
        pqc_dets, sysmon_window=sysmon_window, baseline=baseline
    )
    out.update(master_row)
    return out


def _pqc_metrics_from_detector(det, crypto_mode: str) -> dict:
    """Back-compat wrapper for the single-detector callsite."""
    return _pqc_metrics_from_detectors([det], crypto_mode)


# ─────────────────────────── master orchestration ───────────────────────

def _run_scene_with_server_window(scene_fn, *, mode_key: str, seq: Sequence,
                                  remote_detector_holder: dict, **kwargs) -> dict:
    """Wrap a scene-runner in /stats/window/begin + /stats/window/end on the server."""
    is_cloud = mode_key in ("cloud_ag", "cloud_baseline")
    server_tag = None
    server_stats = None

    # For cloud_baseline we know the detector exists up-front; for cloud_ag
    # it's created lazily inside AGv2 on the first inference call. We can't
    # open the window with a non-existent client, so for cloud_ag we briefly
    # ping the server through a transient RemoteDetector first.
    if is_cloud:
        from network.client import RemoteDetector
        # 60s timeout (default 15s was tight: each scene opens a *fresh*
        # TLS-1.3 + HTTP/2 handshake just to fire one POST to /stats/window/
        # begin, and we'd rather pay an occasional slow handshake than lose
        # the server-side telemetry row). One retry on timeout absorbs the
        # transient "cold start" stalls that surface over SSH tunnels.
        bootstrap = RemoteDetector(
            url=kwargs["remote_url"],
            cafile=kwargs.get("remote_cafile"),
            timeout_s=60.0,
        )
        server_tag = None
        last_err: Exception | None = None
        for attempt in (1, 2):
            try:
                server_tag = bootstrap.begin_server_window(
                    tag=f"{mode_key}__{seq.seq_name}"
                )
                break
            except Exception as e:
                last_err = e
                if attempt == 1:
                    print(f"  [WARN] stats-window handshake slow, retrying once: {e}")
                    continue
                print(f"  [WARN] could not open server stats window: {e}")
        bootstrap.close()
        del last_err

    # Re-inject the args every scene-fn signature expects.  Drop kwargs that
    # the scene-fn itself doesn't accept (we use them only in the post-scene
    # aggregation step below).
    scene_kwargs = {k: v for k, v in kwargs.items()
                    if k != "baseline_throughput_mbps"}
    scene_kwargs["seq"] = seq
    if "mode_key" in scene_fn.__code__.co_varnames:
        scene_kwargs["mode_key"] = mode_key
    if "remote_detector_holder" in scene_fn.__code__.co_varnames:
        scene_kwargs["remote_detector_holder"] = remote_detector_holder

    metrics = scene_fn(**scene_kwargs)

    if is_cloud and server_tag is not None:
        try:
            # Reuse the actual RemoteDetector if AG/baseline left one behind;
            # otherwise stand up a one-off client to close the window.
            det = remote_detector_holder.get("det")
            if det is None:
                from network.client import RemoteDetector
                det = RemoteDetector(
                    url=kwargs["remote_url"],
                    cafile=kwargs.get("remote_cafile"),
                )
                server_stats = det.end_server_window(server_tag)
                det.close()
            else:
                server_stats = det.end_server_window(server_tag)
        except Exception as e:
            print(f"  [WARN] could not close server stats window: {e}")

    if server_stats:
        metrics["server_gpu_power_w"] = server_stats.get("gpu_power_w")
        metrics["server_gpu_util"]    = server_stats.get("gpu_util")
        metrics["server_cpu_pct"]     = server_stats.get("cpu_pct")
        metrics["server_ram_mb"]      = server_stats.get("ram_mb")
        metrics["server_wall_clock_s"] = server_stats.get("wall_clock_s")
        metrics["server_gpu_energy_j"] = server_stats.get("gpu_energy_j")

    # Re-run PQC aggregation now that we have a final SystemMonitor window.
    # Energy/power/temp/RAM in the paper refer to the EDGE (Jetson) — pull
    # from edge_sys-derived columns we already stamped onto `metrics`, NOT
    # from the ground-station's window.
    pqc_mode = kwargs.get("crypto_mode", "tls13")
    if pqc_mode in ("pqc", "classical"):
        edge_window = {
            "gpu_power_w":  metrics.get("edge_gpu_power_w") or 0.0,
            "gpu_util":     metrics.get("edge_gpu_util") or 0.0,
            "cpu_pct":      metrics.get("edge_cpu_pct") or 0.0,
            "ram_mb":       metrics.get("edge_ram_mb") or 0.0,
            "temp_gpu_c":   metrics.get("edge_gpu_temp_c") or 0.0,
            "temp_cpu_c":   metrics.get("edge_cpu_temp_c") or 0.0,
            "n_samples":    metrics.get("edge_sample_count") or 0,
            # Wall-clock for the scene drives energy_j = P * t when nvml
            # didn't report a direct energy counter (the Jetson tegrastats
            # path is power-only).
            "wall_clock_s": metrics.get("seq_time") or 0.0,
        }
        all_dets = remote_detector_holder.get("all_dets") \
            or ([remote_detector_holder["det"]]
                if remote_detector_holder.get("det") is not None else [])
        metrics.update(_pqc_metrics_from_detectors(
            all_dets, pqc_mode, sysmon_window=edge_window,
            baseline_throughput_mbps=kwargs.get("baseline_throughput_mbps"),
        ))
    return metrics


def _cleanup_label_dirs(out_dir: str) -> None:
    """Delete predicted_labels/ and gt_labels/ under ``out_dir``.

    Called once per (KEM, signature) pair after the per-pair CSVs are written.
    The label dumps are intermediate artifacts consumed only by the evaluator;
    after per_sequence_results.csv and summary.csv exist they are dead weight
    and can fill the disk on a full PQC sweep (Jetson ran out of space at the
    ML-KEM-768__ML-DSA-65 pair during the cloud_ag UA-DETRAC sweep).

    Errors during deletion are reported but never raised: a failed cleanup
    must not invalidate an otherwise-complete benchmark run.
    """
    freed_total = 0
    for sub in ("predicted_labels", "gt_labels"):
        target = os.path.join(out_dir, sub)
        if not os.path.isdir(target):
            continue
        try:
            freed = _dir_size_bytes(target)
        except OSError:
            freed = 0
        try:
            shutil.rmtree(target)
            freed_total += freed
        except OSError as e:
            print(f"  [WARN] could not remove {target}: {e}")
    if freed_total > 0:
        gb = freed_total / (1024 ** 3)
        if gb >= 1.0:
            size_str = f"{gb:,.2f} GB"
        else:
            mb = freed_total / (1024 ** 2)
            size_str = f"{mb:,.1f} MB" if mb >= 0.1 else f"{freed_total/1024:,.1f} KB"
        print(f"  [cleanup] reclaimed {size_str} from {out_dir}")


def _dir_size_bytes(path: str) -> int:
    """Sum of file sizes under ``path``. Ignores unreadable entries."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _write_per_sequence_csv(rows: list[dict], path: str):
    if not rows:
        return
    # union of keys, with a stable preferred order up front
    preferred = [
        "mode", "seq_name", "n_frames", "seq_time", "fps", "ag_grid",
        "mAP_50", "mAP_75", "mAP_50_95",
        "precision", "recall", "f1",
        "unique_recall", "total_unique", "detected_unique",
        "TP", "FP", "FN",
        "edge_gpu_power_w", "edge_gpu_util", "edge_cpu_pct", "edge_ram_mb",
        "server_gpu_power_w", "server_gpu_util", "server_cpu_pct",
        "server_ram_mb", "server_wall_clock_s", "server_gpu_energy_j",
        "net_num_requests", "net_upload_MB", "net_download_MB",
        "net_avg_rtt_ms", "net_avg_server_ms",
        # ── PQC / classical-hybrid columns ──
        "crypto_mode", "kem_scheme", "sig_scheme",
        "handshake_ms_client", "handshake_ms_server", "handshake_rtt_ms",
        "avg_sign_ec_ms", "avg_sign_pq_ms",
        "avg_encrypt_ms", "avg_decrypt_ms",
        "avg_verify_ec_ms", "avg_verify_pq_ms",
        "avg_server_decrypt_ms", "avg_server_verify_ec_ms", "avg_server_verify_pq_ms",
        "avg_server_sign_ec_ms", "avg_server_sign_pq_ms", "avg_server_encrypt_ms",
        "pre_crypto_upload_MB", "pre_crypto_download_MB", "crypto_expansion_ratio",
        # Distribution shape on dominant phases.
        "sign_pq_p50_ms", "sign_pq_p95_ms", "sign_pq_p99_ms",
        "sign_pq_cv_pct", "sign_pq_skewness", "sign_pq_excess_kurtosis",
        "sign_pq_tail_ratio_p99_p50", "sign_pq_determinism_class",
        "encrypt_p50_ms", "encrypt_p95_ms", "encrypt_p99_ms",
        "encrypt_cv_pct", "encrypt_determinism_class",
        "rtt_p50_ms", "rtt_p95_ms", "rtt_p99_ms",
        "rtt_cv_pct", "rtt_jitter_pct", "rtt_jitter_class",
        # Amdahl-style bottleneck attribution
        "sign_pct", "encrypt_pct", "network_pct", "dominant_phase",
        "sign_to_encrypt_ratio", "sign_to_network_ratio",
        # Operational throughput
        "throughput_mbps", "goodput_mbps", "chunk_rate_hz", "bandwidth_overhead_pct",
        # Per-phase transport statistics in seconds.
        # sign / verify / encrypt / decrypt / network_send / network_recv
        # Each phase emits: chunk_count, mean_s, stddev_s, min_s, max_s,
        # p50_s, p95_s, p99_s, mad_s, cv_pct, jitter_s, jitter_pct,
        # jitter_class, expansion_ratio, q1_s, q3_s, iqr_s, tukey_lower_s,
        # tukey_upper_s, outlier_count, outlier_pct, skewness,
        # excess_kurtosis, trimmed_mean_10pct_s, geometric_mean_s,
        # tail_ratio_p99_p50, tail_ratio_p95_p50, determinism_class
        *(f"sign_{c}" for c in [
            "chunk_count","mean_s","stddev_s","min_s","max_s","p50_s","p95_s","p99_s",
            "mad_s","cv_pct","jitter_s","jitter_pct","jitter_class","expansion_ratio",
            "q1_s","q3_s","iqr_s","tukey_lower_s","tukey_upper_s",
            "outlier_count","outlier_pct","skewness","excess_kurtosis",
            "trimmed_mean_10pct_s","geometric_mean_s",
            "tail_ratio_p99_p50","tail_ratio_p95_p50","determinism_class",
        ]),
        *(f"verify_{c}" for c in [
            "chunk_count","mean_s","stddev_s","min_s","max_s","p50_s","p95_s","p99_s",
            "mad_s","cv_pct","jitter_s","jitter_pct","jitter_class","expansion_ratio",
            "q1_s","q3_s","iqr_s","tukey_lower_s","tukey_upper_s",
            "outlier_count","outlier_pct","skewness","excess_kurtosis",
            "trimmed_mean_10pct_s","geometric_mean_s",
            "tail_ratio_p99_p50","tail_ratio_p95_p50","determinism_class",
        ]),
        *(f"encrypt_{c}" for c in [
            "chunk_count","mean_s","stddev_s","min_s","max_s","p50_s","p95_s","p99_s",
            "mad_s","cv_pct","jitter_s","jitter_pct","jitter_class","expansion_ratio",
            "q1_s","q3_s","iqr_s","tukey_lower_s","tukey_upper_s",
            "outlier_count","outlier_pct","skewness","excess_kurtosis",
            "trimmed_mean_10pct_s","geometric_mean_s",
            "tail_ratio_p99_p50","tail_ratio_p95_p50","determinism_class",
        ]),
        *(f"decrypt_{c}" for c in [
            "chunk_count","mean_s","stddev_s","min_s","max_s","p50_s","p95_s","p99_s",
            "mad_s","cv_pct","jitter_s","jitter_pct","jitter_class","expansion_ratio",
            "q1_s","q3_s","iqr_s","tukey_lower_s","tukey_upper_s",
            "outlier_count","outlier_pct","skewness","excess_kurtosis",
            "trimmed_mean_10pct_s","geometric_mean_s",
            "tail_ratio_p99_p50","tail_ratio_p95_p50","determinism_class",
        ]),
        *(f"network_send_{c}" for c in [
            "chunk_count","mean_s","stddev_s","min_s","max_s","p50_s","p95_s","p99_s",
            "mad_s","cv_pct","jitter_s","jitter_pct","jitter_class","expansion_ratio",
            "q1_s","q3_s","iqr_s","tukey_lower_s","tukey_upper_s",
            "outlier_count","outlier_pct","skewness","excess_kurtosis",
            "trimmed_mean_10pct_s","geometric_mean_s",
            "tail_ratio_p99_p50","tail_ratio_p95_p50","determinism_class",
        ]),
        *(f"network_recv_{c}" for c in [
            "chunk_count","mean_s","stddev_s","min_s","max_s","p50_s","p95_s","p99_s",
            "mad_s","cv_pct","jitter_s","jitter_pct","jitter_class","expansion_ratio",
            "q1_s","q3_s","iqr_s","tukey_lower_s","tukey_upper_s",
            "outlier_count","outlier_pct","skewness","excess_kurtosis",
            "trimmed_mean_10pct_s","geometric_mean_s",
            "tail_ratio_p99_p50","tail_ratio_p95_p50","determinism_class",
        ]),
        # Verification + hash check
        "verification_status",
        "shared_secret_match", "signature_failure_count", "decrypt_failure_count",
        "client_payload_sha256", "server_payload_sha256", "payload_sha256_match",
        # Client-side session block
        "client_session_setup_ms",
        "client_tls_handshake_time", "client_key_exchange_time", "client_hybrid_key_generation_time",
        "client_signing_time", "client_signature_verification_time",
        "client_encryption_time", "client_decryption_time",
        "client_total_signing_time", "client_total_verifying_time",
        "client_total_encryption_time", "client_total_decryption_time",
        "client_total_time_to_sent_or_receive",
        "client_total_wall_time_s", "client_num_chunks", "client_chunk_size_bytes",
        "client_raw_payload_bytes", "client_on_wire_bytes",
        "client_throughput_mbps_binary", "client_goodput_mbps_binary",
        "client_chunk_rate_hz", "client_signature_rate_hz", "client_bandwidth_overhead_pct",
        "client_sign_pct", "client_encrypt_pct", "client_network_pct", "client_dominant_phase",
        "client_sign_to_encrypt_ratio", "client_sign_to_network_ratio",
        "client_user_cpu_s", "client_sys_cpu_s", "client_peak_rss_mb",
        "client_vol_ctxs", "client_invol_ctxs", "client_minor_faults", "client_major_faults",
        "client_cpu_bound_score",
        "client_tcp_connect_time_s", "client_ttfb_s", "client_ttlb_s",
        "client_vs_baseline_throughput_ratio", "client_break_even_mb",
        "client_per_mb_penalty_ms", "client_amort_throughput_ratio",
        # Server-side session block
        "server_session_setup_ms",
        "server_tls_handshake_time", "server_key_exchange_time", "server_hybrid_key_generation_time",
        "server_signing_time", "server_signature_verification_time",
        "server_encryption_time", "server_decryption_time",
        "server_total_signing_time", "server_total_verifying_time",
        "server_total_encryption_time", "server_total_decryption_time",
        "server_total_time_to_sent_or_receive",
        "server_total_wall_time_s", "server_num_chunks", "server_chunk_size_bytes",
        "server_raw_payload_bytes", "server_on_wire_bytes",
        "server_throughput_mbps_binary", "server_goodput_mbps_binary",
        "server_chunk_rate_hz", "server_signature_rate_hz", "server_bandwidth_overhead_pct",
        "server_sign_pct", "server_encrypt_pct", "server_network_pct", "server_dominant_phase",
        "server_sign_to_encrypt_ratio", "server_sign_to_network_ratio",
        "server_user_cpu_s", "server_sys_cpu_s", "server_peak_rss_mb",
        "server_vol_ctxs", "server_invol_ctxs", "server_minor_faults", "server_major_faults",
        "server_cpu_bound_score",
        "server_tcp_connect_time_s", "server_ttfb_s", "server_ttlb_s",
        "server_vs_baseline_throughput_ratio", "server_break_even_mb",
        "server_per_mb_penalty_ms", "server_amort_throughput_ratio",
        # Sizes
        "size_chunk_size", "size_original_msg_size",
        "size_signed_msg_length", "size_encrypted_msg_length",
        "size_hybrid_aes_key_length", "size_classical_signature_length",
        "size_pqc_signature_length", "size_classic_encryption_sharedsecret_length",
        "size_pqc_encryption_sharedsecret_length",
        # Transfer aggregates
        "transfer_payload_bytes", "transfer_chunk_count",
        "transfer_hybrid_setup_bytes", "transfer_per_file_control_bytes",
        "transfer_protected_data_bytes", "transfer_total_app_bytes",
        "transfer_payload_goodput_mbps_decimal", "transfer_protected_goodput_mbps_decimal",
        "transfer_app_expansion_ratio_total",
        # Tegrastats / energy
        "tegrastats_samples", "avg_power_w", "energy_j",
        "energy_per_payload_mb_j", "energy_per_protected_mb_j",
        "ram_used_mb_mean", "cpu_pct_mean", "gpu_pct_mean",
        "temp_cpu_c_mean", "temp_gpu_c_mean",
        # RTT aggregates
        "rtt_count", "rtt_mean_ms", "rtt_stddev_ms", "rtt_jitter_ms", "rtt_jitter_class",
        # Network gap columns (documented as "not captured")
        "tcp_connect_time_s", "tcp_retransmits", "ttfb_s", "ttlb_s",
        # Baseline / amortization (top-level, populated if --baseline-throughput given)
        "vs_baseline_throughput_ratio", "break_even_mb",
        "per_mb_penalty_ms", "throughput_ratio_amort",
    ]
    all_keys = list(preferred)
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_summary_csv(per_seq: list[dict], path: str):
    """Arithmetic mean of numeric columns across all scenes."""
    if not per_seq:
        return
    NUMERIC_KEYS = [
        "fps", "mAP_50", "mAP_75", "mAP_50_95",
        "precision", "recall", "f1", "unique_recall",
        "edge_gpu_power_w", "edge_gpu_util", "edge_cpu_pct", "edge_ram_mb",
        "server_gpu_power_w", "server_gpu_util", "server_cpu_pct",
        "server_ram_mb", "server_gpu_energy_j",
        "net_upload_MB", "net_download_MB",
        "net_avg_rtt_ms", "net_avg_server_ms",
        # ── PQC numerics ──
        "handshake_ms_client", "handshake_ms_server", "handshake_rtt_ms",
        "avg_sign_ec_ms", "avg_sign_pq_ms",
        "avg_encrypt_ms", "avg_decrypt_ms",
        "avg_verify_ec_ms", "avg_verify_pq_ms",
        "avg_server_decrypt_ms", "avg_server_verify_ec_ms", "avg_server_verify_pq_ms",
        "avg_server_sign_ec_ms", "avg_server_sign_pq_ms", "avg_server_encrypt_ms",
        "pre_crypto_upload_MB", "pre_crypto_download_MB", "crypto_expansion_ratio",
        "sign_pq_p50_ms", "sign_pq_p95_ms", "sign_pq_p99_ms",
        "sign_pq_cv_pct", "sign_pq_skewness", "sign_pq_excess_kurtosis",
        "sign_pq_tail_ratio_p99_p50",
        "encrypt_p50_ms", "encrypt_p95_ms", "encrypt_p99_ms", "encrypt_cv_pct",
        "rtt_p50_ms", "rtt_p95_ms", "rtt_p99_ms", "rtt_cv_pct", "rtt_jitter_pct",
        "sign_pct", "encrypt_pct", "network_pct",
        "sign_to_encrypt_ratio", "sign_to_network_ratio",
        "throughput_mbps", "goodput_mbps", "chunk_rate_hz", "bandwidth_overhead_pct",
    ]
    # Stamp the crypto identity into the summary too (first row sets it).
    _agg_meta = {
        "crypto_mode": per_seq[0].get("crypto_mode", "n/a"),
        "kem_scheme":  per_seq[0].get("kem_scheme",  "NONE"),
        "sig_scheme":  per_seq[0].get("sig_scheme",  "NONE"),
    }
    agg: dict = {"mode": per_seq[0]["mode"], "n_sequences": len(per_seq), **_agg_meta}
    for k in NUMERIC_KEYS:
        vals = [r.get(k) for r in per_seq if isinstance(r.get(k), (int, float))]
        agg[k] = (sum(vals) / len(vals)) if vals else None
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)


def _load_scenes_file(path: str) -> list[str]:
    """Parse a scene-name list from ``path``.

    One scene name per line. Whitespace is stripped; '#' starts a comment.
    Returns names in file order; duplicates are preserved.
    """
    if not os.path.isfile(path):
        sys.exit(f"--scenes-file: {path!r} is not a file.")
    names: list[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if line:
                names.append(line)
    return names


def _missing_scenes_from_run(run_dir: str, all_scene_names: list[str]) -> list[str]:
    """Return scenes that appear in *zero* of the per-pair CSVs under ``run_dir``.

    The harness writes per-pair ``per_sequence_results.csv`` files under
    ``<run_dir>/<pair_label>/``. A scene is treated as "missing" only if it is
    absent from every one of those CSVs (i.e., systematic skip, not a per-pair
    quirk). The classical_baseline subdir is included in the scan.
    """
    if not os.path.isdir(run_dir):
        sys.exit(f"--missing-scenes-from: {run_dir!r} is not a directory.")
    seen: set[str] = set()
    for entry in sorted(os.listdir(run_dir)):
        pdir = os.path.join(run_dir, entry)
        if not os.path.isdir(pdir):
            continue
        csv_path = os.path.join(pdir, "per_sequence_results.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = row.get("seq_name", "").strip()
                if name:
                    seen.add(name)
    return [s for s in all_scene_names if s not in seen]


def _list_run_dirs_for(dataset: str) -> list[str]:
    """Recent ``runs/*_<dataset>_*`` directories, newest last."""
    runs_root = os.path.join(_PROJECT_ROOT, "runs")
    if not os.path.isdir(runs_root):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(runs_root)):
        full = os.path.join(runs_root, name)
        if os.path.isdir(full) and f"_{dataset}_" in name:
            out.append(full)
    return out


def _pick_scene_filter_interactive(seq_map: "dict",
                                   dataset: str,
                                   ) -> "list":
    """Menu wrapper around the existing scene picker.

    Adds a top-level option to resume MISSING scenes from a previous run
    directory. Returns a list of Sequence objects (the same shape that
    ``select_sequences_interactive`` returns).
    """
    seqs_all = list(seq_map.values())
    name_to_seq = {s.seq_name: s for s in seqs_all}

    options = [
        "Run all sequences (default)",
        "Pick a subset interactively",
        "Resume MISSING sequences from a previous run directory",
        "Load scene names from a file",
    ]
    idx = _menu_pick("Sequence selection:", options, default_idx=0)

    if idx == 0:
        return seqs_all
    if idx == 1:
        return select_sequences_interactive(seq_map)

    if idx == 2:
        run_dirs = _list_run_dirs_for(dataset)
        if not run_dirs:
            print(f"  (no previous run dirs found under runs/ for {dataset!r}; "
                  f"falling back to 'all')")
            return seqs_all
        labels = [os.path.basename(d) for d in run_dirs] + ["(cancel, use all)"]
        ridx = _menu_pick("Pick a previous run directory:", labels,
                          default_idx=len(labels) - 2)
        if ridx == len(labels) - 1:
            return seqs_all
        chosen = run_dirs[ridx]
        missing = _missing_scenes_from_run(chosen, list(name_to_seq.keys()))
        if not missing:
            print(f"  Nothing missing in {os.path.basename(chosen)}; "
                  f"all {len(name_to_seq)} scenes already present. "
                  f"Falling back to 'all'.")
            return seqs_all
        print(f"\n  {len(missing)} scene(s) missing from "
              f"{os.path.basename(chosen)}:")
        for s in missing:
            print(f"    - {s}")
        if not _ask_yn("Run these missing scenes only?", default=True):
            return seqs_all
        return [name_to_seq[s] for s in missing if s in name_to_seq]

    # idx == 3: scenes file
    path = input("  Path to scenes file (one scene name per line): ").strip()
    if not path:
        print("  (empty path; falling back to 'all')")
        return seqs_all
    names = _load_scenes_file(path)
    picked = [name_to_seq[s] for s in names if s in name_to_seq]
    missing = [s for s in names if s not in name_to_seq]
    if missing:
        print(f"  [warn] {len(missing)} name(s) not in this dataset, skipped: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not picked:
        print("  (no valid scenes from file; falling back to 'all')")
        return seqs_all
    return picked


def _load_pqc_pairs_file(path: str) -> list[tuple[str, str]]:
    """Parse a (KEM, SIG) pair list from ``path``.

    One pair per line. KEM and SIG are separated by tab, comma, or whitespace.
    Lines starting with '#' or empty lines are ignored. Examples::

        ML-KEM-1024   ML-DSA-87
        ML-KEM-768,SLH_DSA_PURE_SHA2_128F
        # comment line

    Returns the pairs in file order; duplicates and ordering are the caller's
    responsibility.
    """
    if not os.path.isfile(path):
        sys.exit(f"--pqc-pairs-file: {path!r} is not a file.")
    pairs: list[tuple[str, str]] = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # Allow tab, comma, or any whitespace as the field separator.
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) != 2:
                sys.exit(f"--pqc-pairs-file {path!r}, line {lineno}: "
                         f"expected 'KEM SIG', got {raw!r}")
            pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Split-inference benchmark for AttentionGrid.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES,
                        help="Skip the dataset menu and use this one.")
    parser.add_argument("--mode", choices=[m[0] for m in MODE_CHOICES],
                        help="Skip the mode menu and use this one.")
    parser.add_argument("--remote-url", default=None,
                        help='Cloud modes: server URL (e.g. "https://GROUND_STATION_IP:8443/infer").')
    parser.add_argument("--remote-cafile", default=None,
                        help="Cloud modes: path to the server's self-signed cert.")
    parser.add_argument("--crypto-mode", default=None,
                        choices=["tls13", "classical", "pqc"],
                        help=("Transport security for cloud modes. "
                              "'tls13' (default): existing /infer over TLS 1.3 only. "
                              "'classical': application-layer ECDH+ECDSA over TLS 1.3. "
                              "'pqc': hybrid ECDH+ML-KEM + ECDSA+ML-DSA over TLS 1.3."))
    parser.add_argument("--pqc-kem", default="ML-KEM-768",
                        help="PQC KEM scheme (only used when --crypto-mode pqc). "
                             "e.g. ML-KEM-512 / ML-KEM-768 / ML-KEM-1024 / "
                             "HQC-128 / FrodoKEM-976-AES / Classic-McEliece-348864.")
    parser.add_argument("--pqc-sig", default="ML-DSA-44",
                        help="PQC signature scheme (only used when --crypto-mode pqc). "
                             "e.g. ML-DSA-44 / ML-DSA-65 / ML-DSA-87 / "
                             "Falcon-512 / Falcon-1024 / SPHINCS+-SHA2-128f-simple.")
    parser.add_argument("--pqc-sweep", action="store_true",
                        help="Run the full PQC sweep — every (KEM, SIG) pair "
                             "from --pqc-sweep-profile, one full benchmark per pair. "
                             "Overrides --pqc-kem / --pqc-sig.")
    parser.add_argument("--pqc-sweep-profile", default="paper_curated",
                        choices=["max_coverage", "standard", "extended",
                                 "paper_curated", "stress"],
                        help="Which sweep profile to expand when --pqc-sweep is set. "
                             "max_coverage=8 curated, standard~42, extended~228, "
                             "paper_curated=public artifact sweep, stress~8 high-cost cells.")
    parser.add_argument("--pqc-pairs-file", default=None,
                        help="Resume mode: run ONLY the (KEM, SIG) pairs listed "
                             "in this file. One pair per line; KEM and SIG are "
                             "separated by tab, comma, or whitespace; '#' "
                             "starts a comment. Skips the classical baseline. "
                             "Overrides --pqc-kem / --pqc-sig / --pqc-sweep / "
                             "--crypto-mode. Use this to re-run only the pairs "
                             "that an interrupted full-suite missed.")
    parser.add_argument("--baseline-throughput-mbps", type=float, default=None,
                        help="Throughput in Mbps (binary) from a reference "
                             "TLS-1.3 run on the same hardware.  When supplied, "
                             "the master row gets vs_baseline_throughput_ratio, "
                             "break_even_mb, per_mb_penalty_ms and "
                             "throughput_ratio_amort populated for both the "
                             "client and the server side.")
    parser.add_argument("--weight", default="yolo11s.pt",
                        help="YOLO weight name (default yolo11s.pt).")
    parser.add_argument("--min-conf", type=float, default=0.25,
                        help="Confidence threshold for predictions.")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Cap on number of scenes (debug / smoke-test).")
    parser.add_argument("--max-frames-per-seq", type=int, default=None,
                        help="Cap on frames per scene (debug / smoke-test).")
    parser.add_argument("--scenes-file", default=None,
                        help="Run ONLY the scenes listed in this file. "
                             "One scene name per line; '#' starts a comment. "
                             "Resolved against the dataset's discovered scenes; "
                             "unknown names are skipped with a warning. "
                             "Overrides --max-sequences.")
    parser.add_argument("--missing-scenes-from", default=None,
                        help="Run ONLY the scenes that are absent from every "
                             "per-pair per_sequence_results.csv under the given "
                             "previous run directory. Use this to fill the "
                             "gaps left by a buggy or interrupted run without "
                             "re-running the full sweep. Overrides "
                             "--max-sequences and --scenes-file.")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip ALL menus; requires --dataset, --mode, and cloud args.")
    parser.add_argument("--keep-labels", action="store_true",
                        help="Keep per-scene predicted_labels/ and gt_labels/ "
                             "after evaluation. Default is to delete them once "
                             "per_sequence_results.csv and summary.csv are "
                             "written, since label dumps can be tens of GB on a "
                             "full PQC sweep.")
    args = parser.parse_args()

    print("\n" + "=" * 66)
    print("  AttentionGrid split-inference benchmark")
    print("=" * 66)

    # ── Dataset menu ──
    if args.dataset:
        dataset = args.dataset
    elif args.non_interactive:
        sys.exit("--non-interactive requires --dataset")
    else:
        idx = _menu_pick("Dataset:", ["UA-DETRAC", "MOT17"], default_idx=0)
        dataset = DATASET_CHOICES[idx]

    # ── Mode menu ──
    if args.mode:
        mode_key = args.mode
    elif args.non_interactive:
        sys.exit("--non-interactive requires --mode")
    else:
        idx = _menu_pick("Inference mode:",
                         [m[1] for m in MODE_CHOICES], default_idx=0)
        mode_key = MODE_CHOICES[idx][0]

    needs_local_gpu  = mode_key in ("edge_ag", "edge_baseline", "cloud_ag")
    needs_remote     = mode_key in ("cloud_ag", "cloud_baseline")
    # cloud_ag still computes saliency + caches on the edge; the *YOLO* call
    # is remote but everything else runs locally, so we still pick a device
    # for the AG bookkeeping (which uses NumPy/OpenCV but no GPU).
    # For pure cloud_baseline the edge does NO ML compute; we still call
    # select_gpu() to keep CUDA_VISIBLE_DEVICES consistent.

    if args.non_interactive:
        # In non-interactive mode we never prompt for a GPU. Default to
        # the first visible device (Jetson has just one anyway).
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    elif needs_local_gpu:
        select_gpu()
    else:
        # cloud_baseline doesn't need a local GPU, but we still let the user
        # pick one in case they want to monitor specific telemetry.
        if _ask_yn(
            "cloud_baseline: select a local GPU for monitoring?", default=False
        ):
            select_gpu()

    # ── Remote server settings ──
    remote_url    = args.remote_url
    remote_cafile = args.remote_cafile
    if needs_remote:
        if remote_url is None:
            remote_url = _ask_text(
                "Server URL", "https://127.0.0.1:8443/infer"
            )
        if remote_cafile is None:
            default_ca = os.path.join(_SRC_ROOT, "network", "certs", "server.crt")
            remote_cafile = _ask_text("Server cert (cafile)", default_ca)
            if not os.path.isfile(remote_cafile):
                print(f"  [WARN] cert {remote_cafile} not found — will use system CAs.")
                remote_cafile = None

    # ── Transport-security selection (cloud modes only) ──
    from network.pqc_sweep import SWEEP_SENTINEL, FULL_SUITE_SENTINEL

    crypto_mode    = args.crypto_mode
    pqc_kem        = args.pqc_kem
    pqc_sig        = args.pqc_sig
    sweep_profile  = ""
    explicit_pairs: list[tuple[str, str]] | None = None
    explicit_pairs_source: str | None = None  # path of the file we read

    if args.pqc_pairs_file:
        # Resume mode: caller supplies the (KEM, SIG) pairs explicitly.
        explicit_pairs = _load_pqc_pairs_file(args.pqc_pairs_file)
        explicit_pairs_source = os.path.abspath(args.pqc_pairs_file)
        if not explicit_pairs:
            sys.exit(f"--pqc-pairs-file {args.pqc_pairs_file!r} is empty.")
        crypto_mode = SWEEP_SENTINEL  # uses sweep layout (per-pair dirs, no classical)
        print(f"\n  RESUME MODE — {len(explicit_pairs)} explicit pair(s) from "
              f"{explicit_pairs_source}")
    elif args.pqc_sweep:
        # CLI shortcut to bypass the picker for scripted sweeps.
        crypto_mode = SWEEP_SENTINEL
        sweep_profile = args.pqc_sweep_profile

    if needs_remote and crypto_mode is None:
        if args.non_interactive:
            crypto_mode = "tls13"
        else:
            (crypto_mode, pqc_kem, pqc_sig,
             sweep_profile, picked_pairs_file) = _pick_crypto_mode_interactive(
                args.pqc_kem, args.pqc_sig
            )
            if picked_pairs_file is not None:
                explicit_pairs = _load_pqc_pairs_file(picked_pairs_file)
                explicit_pairs_source = picked_pairs_file
                print(f"\n  RESUME MODE — loaded {len(explicit_pairs)} pair(s) "
                      f"from {picked_pairs_file}")
    elif crypto_mode is None:
        crypto_mode = "tls13"   # not used for non-cloud modes

    # ── Sequence discovery + selection ──
    seq_map = discover_sequences(dataset)
    if not seq_map:
        sys.exit(f"No sequences discovered for {dataset!r}.")
    all_scene_names = list(seq_map.keys())

    # --missing-scenes-from beats --scenes-file beats --max-sequences beats
    # the interactive menu, in that precedence order.
    if args.missing_scenes_from:
        missing = _missing_scenes_from_run(args.missing_scenes_from,
                                           all_scene_names)
        if not missing:
            sys.exit(f"--missing-scenes-from: every dataset scene is already "
                     f"present in {args.missing_scenes_from!r}. Nothing to run.")
        seqs = [seq_map[s] for s in missing]
        print(f"  Resume-missing: {len(seqs)} scene(s) from "
              f"{os.path.basename(args.missing_scenes_from)}:")
        for s in missing:
            print(f"    - {s}")
    elif args.scenes_file:
        names = _load_scenes_file(args.scenes_file)
        seqs = [seq_map[s] for s in names if s in seq_map]
        unknown = [s for s in names if s not in seq_map]
        if unknown:
            print(f"  [warn] --scenes-file: {len(unknown)} unknown name(s) "
                  f"skipped: {unknown[:5]}{'...' if len(unknown) > 5 else ''}")
        if not seqs:
            sys.exit(f"--scenes-file: no valid scenes found in "
                     f"{args.scenes_file!r}.")
        print(f"  --scenes-file: {len(seqs)} scene(s) loaded.")
    elif args.non_interactive or args.max_sequences is not None:
        seqs = list(seq_map.values())
        if args.max_sequences is not None:
            seqs = seqs[: args.max_sequences]
    else:
        seqs = _pick_scene_filter_interactive(seq_map, dataset)

    # Smoke-test convenience: cap frames per scene.
    if args.max_frames_per_seq is not None and args.max_frames_per_seq > 0:
        for s in seqs:
            s.img_paths = s.img_paths[: args.max_frames_per_seq]
        print(f"  Frame cap: first {args.max_frames_per_seq} frame(s) per scene.")

    print(f"\n  Will run {len(seqs)} sequence(s) on {dataset} in mode {mode_key}.")

    # ── Resolve the pair list to run ──
    #   - Single-pair runs: one entry (the picker's choice / CLI flags).
    #   - Sweep runs: every (kem, sig) pair from the requested profile,
    #     filtered to those liboqs can actually instantiate.
    #   - Full-suite: classical pseudo-pair first, then the paper_curated set.
    from network import pqc_sweep as ps
    is_sweep      = (crypto_mode == SWEEP_SENTINEL)
    is_full_suite = (crypto_mode == FULL_SUITE_SENTINEL)

    if explicit_pairs is not None:
        # Resume mode bypasses profile expansion entirely.
        pairs, skipped = ps.runnable_pairs(explicit_pairs)
        if not pairs:
            sys.exit("--pqc-pairs-file had no runnable pairs "
                     "in the current liboqs build "
                     f"({len(skipped)} skipped: {skipped}).")
        print(f"  {len(pairs)} runnable pair(s), {len(skipped)} skipped.")
    elif is_full_suite:
        pqc_pairs_all = ps.expand_profile(sweep_profile or "paper_curated")
        pqc_pairs, skipped = ps.runnable_pairs(pqc_pairs_all)
        # Prepend the classical pass as a synthetic "pair"
        pairs = [ps.CLASSICAL_PAIR] + pqc_pairs
        print(f"\n  FULL SUITE MODE — 1 classical pass + "
              f"{len(pqc_pairs)} PQC pairs ({len(skipped)} skipped).")
    elif is_sweep:
        pairs_all = ps.expand_profile(sweep_profile or "paper_curated")
        pairs, skipped = ps.runnable_pairs(pairs_all)
        if not pairs:
            sys.exit(f"sweep profile {sweep_profile!r} has no runnable pairs "
                     "in the current liboqs build")
        print(f"\n  SWEEP MODE — profile={sweep_profile!r}, "
              f"{len(pairs)} runnable pair(s), {len(skipped)} skipped.")
    else:
        pairs = [(pqc_kem, pqc_sig)]

    # ── Output dir (root) ──
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if explicit_pairs is not None:
        _suffix = "_resume"
    elif is_full_suite:
        _suffix = "_full_suite"
    elif is_sweep:
        _suffix = "_sweep"
    else:
        _suffix = ""
    run_tag = f"{mode_key}_{dataset}{_suffix}"
    out_dir_root = os.path.join(_PROJECT_ROOT, "runs", f"{ts}_{run_tag}")
    os.makedirs(out_dir_root, exist_ok=True)
    print(f"  Output dir:  {out_dir_root}")

    # Top-level config (covers the entire sweep)
    with open(os.path.join(out_dir_root, "run_config.json"), "w") as f:
        json.dump({
            "dataset":         dataset,
            "mode":            mode_key,
            "remote_url":      remote_url,
            "remote_cafile":   remote_cafile,
            "weight":          args.weight,
            "min_conf":        args.min_conf,
            "sequences":       [s.seq_name for s in seqs],
            "timestamp":       ts,
            "crypto_mode":     ("pqc" if is_sweep or explicit_pairs is not None
                                else ("classical+pqc" if is_full_suite else crypto_mode)),
            "sweep":           is_sweep or is_full_suite,
            "sweep_profile":   sweep_profile if (is_sweep or is_full_suite) else None,
            "resume":          explicit_pairs is not None,
            "resume_source":   explicit_pairs_source,
            "pqc_pairs":       [{"kem": k, "sig": s} for k, s in pairs],
        }, f, indent=2)

    # ── Edge SystemMonitor (continuous, shared across pairs) ──
    monitor = SystemMonitor(interval_ms=100)
    monitor.start()
    print(f"  Edge monitor: {monitor.get_gpu_info_string()}")

    json_cfg = (_load_scene_configs(dataset)
                if mode_key in ("edge_ag", "cloud_ag") else None)

    # Aggregated per-pair-per-sequence rows for the campaign-level CSV
    all_pair_rows: list[dict] = []

    for pair_idx, (sweep_kem, sweep_sig) in enumerate(pairs, start=1):
        # Per-pair effective crypto settings
        is_classical_pair = (sweep_kem == ps.CLASSICAL_PAIR[0])
        if is_full_suite:
            if is_classical_pair:
                this_crypto_mode = "classical"
                this_kem, this_sig = "CLASSICAL", "CLASSICAL"
                pair_label = "classical_baseline"
            else:
                this_crypto_mode = "pqc"
                this_kem, this_sig = sweep_kem, sweep_sig
                pair_label = f"{sweep_kem}__{sweep_sig}".replace("/", "_").replace(" ", "_")
            out_dir = os.path.join(out_dir_root, pair_label)
        elif is_sweep:
            this_crypto_mode = "pqc"
            this_kem, this_sig = sweep_kem, sweep_sig
            pair_label = f"{sweep_kem}__{sweep_sig}".replace("/", "_").replace(" ", "_")
            out_dir = os.path.join(out_dir_root, pair_label)
        else:
            this_crypto_mode = crypto_mode
            this_kem, this_sig = pqc_kem, pqc_sig
            out_dir = out_dir_root
        os.makedirs(out_dir, exist_ok=True)

        if is_full_suite or is_sweep:
            print(f"\n[{pair_idx}/{len(pairs)}] PAIR  "
                  f"mode={this_crypto_mode}  kem={this_kem}  sig={this_sig}")

        # Per-pair run_config sits inside the per-pair directory so each
        # pair is self-describing for downstream analysis.
        with open(os.path.join(out_dir, "run_config.json"), "w") as f:
            json.dump({
                "dataset":       dataset,
                "mode":          mode_key,
                "remote_url":    remote_url,
                "remote_cafile": remote_cafile,
                "weight":        args.weight,
                "min_conf":      args.min_conf,
                "sequences":     [s.seq_name for s in seqs],
                "timestamp":     ts,
                "crypto_mode":   this_crypto_mode,
                "pqc_kem":       this_kem,
                "pqc_sig":       this_sig,
                "sweep_profile": sweep_profile if is_sweep else None,
            }, f, indent=2)

        per_seq: list[dict] = []
        for seq in seqs:
            remote_detector_holder: dict = {}
            try:
                if mode_key in ("edge_ag", "cloud_ag"):
                    m = _run_scene_with_server_window(
                        _run_scene_ag,
                        mode_key=mode_key, seq=seq,
                        remote_detector_holder=remote_detector_holder,
                        dataset=dataset, json_cfg=json_cfg, out_dir=out_dir,
                        remote_url=remote_url, remote_cafile=remote_cafile,
                        monitor=monitor, min_conf=args.min_conf,
                        crypto_mode=this_crypto_mode,
                        pqc_kem=this_kem, pqc_sig=this_sig,
                        baseline_throughput_mbps=args.baseline_throughput_mbps,
                    )
                elif mode_key == "edge_baseline":
                    m = _run_scene_local_baseline(
                        seq=seq, dataset=dataset, out_dir=out_dir,
                        monitor=monitor, min_conf=args.min_conf,
                        yolo_weight=args.weight,
                    )
                    m["crypto_mode"] = "n/a"
                elif mode_key == "cloud_baseline":
                    m = _run_scene_with_server_window(
                        _run_scene_remote_baseline,
                        mode_key=mode_key, seq=seq,
                        remote_detector_holder=remote_detector_holder,
                        dataset=dataset,
                        remote_url=remote_url, remote_cafile=remote_cafile,
                        out_dir=out_dir, monitor=monitor, min_conf=args.min_conf,
                        crypto_mode=this_crypto_mode,
                        pqc_kem=this_kem, pqc_sig=this_sig,
                        baseline_throughput_mbps=args.baseline_throughput_mbps,
                    )
                else:
                    raise ValueError(f"unknown mode {mode_key!r}")
            except Exception as e:
                print(f"  [ERROR] {seq.seq_name} ({this_kem}+{this_sig}) failed: {e}")
                continue

            # Stamp the pair identity onto the row so the campaign CSV is
            # disambiguatable by (kem, sig).
            m["kem_scheme"] = this_kem
            m["sig_scheme"] = this_sig
            m["sweep_pair_index"] = pair_idx
            per_seq.append(m)
            all_pair_rows.append(m)

            print(
                f"    => FPS={m.get('fps', 0):.2f}  "
                f"mAP@50={m.get('mAP_50', 0):.4f}  "
                f"P={m.get('precision', 0):.3f}  R={m.get('recall', 0):.3f}  "
                f"F1={m.get('f1', 0):.3f}  UniqR={m.get('unique_recall', 0):.3f}"
            )

        # Per-pair CSVs
        seq_csv  = os.path.join(out_dir, "per_sequence_results.csv")
        summ_csv = os.path.join(out_dir, "summary.csv")
        _write_per_sequence_csv(per_seq, seq_csv)
        _write_summary_csv(per_seq, summ_csv)

        # Reclaim disk: per-frame label dumps are only needed while the
        # evaluator runs. Once per_sequence_results.csv and summary.csv exist
        # they are redundant, and a full PQC sweep accumulates tens of GB of
        # them, which is what blew up the Jetson during the cloud_ag sweep.
        if not args.keep_labels:
            _cleanup_label_dirs(out_dir)

    monitor.stop()

    # ── Write the campaign-level master CSV (sweep / full-suite modes) ──
    if is_sweep or is_full_suite:
        master_csv = os.path.join(out_dir_root, "master_summary.csv")
        _write_per_sequence_csv(all_pair_rows, master_csv)
        print(f"\n  Wrote campaign master {master_csv}  "
              f"({len(all_pair_rows)} rows across {len(pairs)} pairs)")
        per_seq = all_pair_rows
        seq_csv = master_csv
        summ_csv = os.path.join(out_dir_root, "summary.csv")
        _write_summary_csv(all_pair_rows, summ_csv)
    else:
        seq_csv  = os.path.join(out_dir_root, "per_sequence_results.csv")
        summ_csv = os.path.join(out_dir_root, "summary.csv")
        print(f"\n  Wrote {seq_csv}")
        print(f"  Wrote {summ_csv}")

    # ── Final summary printout ──
    if per_seq:
        print("\n" + "=" * 66)
        print(f"  SUMMARY · mode={mode_key}  dataset={dataset}  scenes={len(per_seq)}")
        print("=" * 66)
        with open(summ_csv) as f:
            for line in f:
                print("    " + line.rstrip("\n"))


if __name__ == "__main__":
    main()
