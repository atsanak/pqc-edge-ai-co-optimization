"""
Universal timing utilities for AttentionGrid v1 and v2.
Provides a shared print_timing_summary function with consistent CSV output.
"""

import csv
from typing import Optional


def print_timing_summary(
    n_frames: int,
    total_time: float,
    saliency_tot: Optional[float] = None,
    yolo_tot: Optional[float] = None,
    tiler_tot: Optional[float] = None,
    overlay_tot: Optional[float] = None,
    imageio_tot: Optional[float] = None,
    merge_tiles_tot: Optional[float] = None,
    yolo_preds: Optional[int] = None,
    title: str = "Attention Grid — Timing Summary",
    model_name: str = "AttentionGrid",
    save_csv: Optional[str] = None,
):
    """
    Universal timing report + optional CSV (one tidy row).
    
    Parameters
    ----------
    n_frames : int
        Number of frames processed.
    total_time : float
        Total wall-clock time in seconds.
    saliency_tot : float, optional
        Total time spent on saliency computation (frame_diff, optical_flow, etc.).
    yolo_tot : float, optional
        Total time spent on YOLO inference (combines TinyYOLO + HeavyYOLO for v1).
    tiler_tot : float, optional
        Total time spent on image tiling operations.
    overlay_tot : float, optional
        Total time spent on overlay/visualization.
    imageio_tot : float, optional
        Total time spent on image I/O (open, convert, etc.).
    merge_tiles_tot : float, optional
        Total time spent on tile merging logic (v2 only).
    yolo_preds : int, optional
        Number of YOLO predictions made (for computing per-prediction average).
    title : str
        Title for the timing summary.
    model_name : str
        Name of the model/version for CSV output.
    save_csv : str, optional
        Path to CSV file. If provided, appends a row.
    """
    
    def _fmt(val: Optional[float]) -> str:
        return f"{val:>9.5f}s" if isinstance(val, (int, float)) else f"{'-':>9}"

    def _avg(val: Optional[float]) -> Optional[float]:
        if isinstance(val, (int, float)) and n_frames:
            return val / n_frames
        return None

    # Treat None as 0 for summing
    _s = saliency_tot or 0.0
    _y = yolo_tot or 0.0
    _t = tiler_tot or 0.0
    _o = overlay_tot or 0.0
    _i = imageio_tot or 0.0
    _m = merge_tiles_tot or 0.0

    known_sum = _s + _y + _t + _o + _i + _m
    other_tot = max(0.0, total_time - known_sum)

    # Averages (per frame)
    saliency_avg = _avg(saliency_tot)
    tiler_avg = _avg(tiler_tot)
    overlay_avg = _avg(overlay_tot)
    imageio_avg = _avg(imageio_tot)
    merge_avg = _avg(merge_tiles_tot)
    other_avg = _avg(other_tot)

    # YOLO average: per prediction, not per frame
    if isinstance(yolo_tot, (int, float)) and yolo_preds and yolo_preds > 0:
        yolo_avg = yolo_tot / float(yolo_preds)
    else:
        yolo_avg = None

    total_wo_io = total_time - (_i if isinstance(imageio_tot, (int, float)) else 0.0)
    fps = (n_frames / total_time) if (n_frames and total_time > 0) else None

    # ---- Console print ----
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)
    print(f"Frames processed: {n_frames}")
    if fps is not None:
        print(f"Approx FPS: {fps:.2f}")
    if yolo_preds is not None:
        print(f"YOLO predictions: {yolo_preds}")
    print("-" * 70)
    print(f"{'Saliency':<26} Total: {_fmt(saliency_tot)}   Average: {_fmt(saliency_avg)}")
    print(f"{'YOLO':<26} Total: {_fmt(yolo_tot)}   Average: {_fmt(yolo_avg)}")
    print(f"{'Merge Tiles':<26} Total: {_fmt(merge_tiles_tot)}   Average: {_fmt(merge_avg)}")
    print(f"{'SimpleImageTiler':<26} Total: {_fmt(tiler_tot)}   Average: {_fmt(tiler_avg)}")
    print(f"{'Overlay (grid/UI)':<26} Total: {_fmt(overlay_tot)}   Average: {_fmt(overlay_avg)}")
    print(f"{'Image I/O':<26} Total: {_fmt(imageio_tot)}   Average: {_fmt(imageio_avg)}")
    print("-" * 70)
    print(f"{'OTHER (prints, etc)':<26} Total: {_fmt(other_tot)}   Average: {_fmt(other_avg)}")
    print("-" * 70)
    print(f"{'TOTAL':<26} Total: {total_time:>9.5f}s")
    print("-" * 70)
    print(f"{'TOTAL w/o I/O':<26} Total: {total_wo_io:>9.5f}s")
    print("=" * 70)

    # ---- Optional CSV: one tidy row ----
    if save_csv:
        row = {
            "title": title,
            "model_name": model_name,
            "frames": n_frames,
            "fps": f"{fps:.4f}" if fps is not None else "-",
            "yolo_preds": yolo_preds if yolo_preds is not None else "-",
            "total_time_s": f"{total_time:.6f}",
            "saliency_total_s": f"{saliency_tot:.6f}" if isinstance(saliency_tot, (int, float)) else "-",
            "saliency_avg_s": f"{saliency_avg:.6f}" if isinstance(saliency_avg, (int, float)) else "-",
            "yolo_total_s": f"{yolo_tot:.6f}" if isinstance(yolo_tot, (int, float)) else "-",
            "yolo_avg_s": f"{yolo_avg:.6f}" if isinstance(yolo_avg, (int, float)) else "-",
            "merge_tiles_total_s": f"{merge_tiles_tot:.6f}" if isinstance(merge_tiles_tot, (int, float)) else "-",
            "merge_tiles_avg_s": f"{merge_avg:.6f}" if isinstance(merge_avg, (int, float)) else "-",
            "tiler_total_s": f"{tiler_tot:.6f}" if isinstance(tiler_tot, (int, float)) else "-",
            "tiler_avg_s": f"{tiler_avg:.6f}" if isinstance(tiler_avg, (int, float)) else "-",
            "overlay_total_s": f"{overlay_tot:.6f}" if isinstance(overlay_tot, (int, float)) else "-",
            "overlay_avg_s": f"{overlay_avg:.6f}" if isinstance(overlay_avg, (int, float)) else "-",
            "imageio_total_s": f"{imageio_tot:.6f}" if isinstance(imageio_tot, (int, float)) else "-",
            "imageio_avg_s": f"{imageio_avg:.6f}" if isinstance(imageio_avg, (int, float)) else "-",
            "other_total_s": f"{other_tot:.6f}",
            "other_avg_s": f"{other_avg:.6f}" if isinstance(other_avg, (int, float)) else "-",
            "total_wo_io_s": f"{total_wo_io:.6f}",
        }
        fieldnames = list(row.keys())
        try:
            need_header = False
            try:
                with open(save_csv, "r", newline="") as f:
                    need_header = (f.read(1) == "")
            except FileNotFoundError:
                need_header = True

            with open(save_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if need_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"[INFO] Timing saved to '{save_csv}'")
        except Exception as e:
            print(f"[WARN] Failed to write CSV '{save_csv}': {e}")
    print("=" * 70 + "\n")
