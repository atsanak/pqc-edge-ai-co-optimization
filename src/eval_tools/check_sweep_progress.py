#!/usr/bin/env python3
"""
Read-only progress check for a running run_split_inference_benchmark.py sweep.

Reads the top-level run_config.json to learn what pairs were planned and
counts per-pair subdirectories that have a finished summary.csv. Reports
done / remaining / current-pair / ETA without touching the benchmark.

Usage:
    PYTHONPATH=src python -m eval_tools.check_sweep_progress <run_dir>
    PYTHONPATH=src python -m eval_tools.check_sweep_progress runs/20260521_115251_cloud_baseline_ua_detrac_full_suite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _pair_label(kem: str, sig: str) -> str:
    """Match the harness convention for per-pair subdir names.

    The classical-anchor sentinel from network/pqc_sweep.py is the literal
    string "__CLASSICAL__" for both KEM and signature; the harness writes
    that pair into a subdir named "classical_baseline".
    """
    if kem == "__CLASSICAL__":
        return "classical_baseline"
    return f"{kem}__{sig}".replace("/", "_").replace(" ", "_")


def _fmt_dur(seconds: float) -> str:
    seconds = int(round(max(0.0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="Top-level sweep directory")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.is_file():
        print(f"[error] no run_config.json at {cfg_path}", file=sys.stderr)
        return 1

    with cfg_path.open() as f:
        cfg = json.load(f)

    pairs = cfg.get("pqc_pairs", [])
    if not pairs:
        print(f"[error] run_config has no 'pqc_pairs' list", file=sys.stderr)
        return 1

    planned_labels = [_pair_label(p["kem"], p["sig"]) for p in pairs]

    # A pair is "done" once its per-pair summary.csv exists. Track each pair's
    # 1-based position in the planned sweep order so the user can see which
    # index of the sweep is next up.
    done: list[tuple[int, str, float]] = []   # (idx, label, mtime)
    in_progress: list[tuple[int, str]] = []   # subdir exists, no summary.csv
    not_started: list[tuple[int, str]] = []
    for idx, lbl in enumerate(planned_labels, start=1):
        pdir = run_dir / lbl
        summary = pdir / "summary.csv"
        if summary.is_file():
            done.append((idx, lbl, summary.stat().st_mtime))
        elif pdir.is_dir():
            in_progress.append((idx, lbl))
        else:
            not_started.append((idx, lbl))

    n_total = len(planned_labels)
    n_done = len(done)
    n_remaining = n_total - n_done

    # ETA from per-pair durations (gap between successive summary.csv mtimes).
    eta_str = "n/a (need >=2 completed pairs)"
    if len(done) >= 2:
        done_sorted = sorted(done, key=lambda kv: kv[2])
        gaps = [b[2] - a[2] for a, b in zip(done_sorted, done_sorted[1:])]
        mean_gap = sum(gaps) / len(gaps)
        eta_s = mean_gap * n_remaining
        eta_str = (f"~{_fmt_dur(eta_s)} "
                   f"(mean {_fmt_dur(mean_gap)}/pair, n={len(gaps)})")

    # Header
    print(f"sweep:           {run_dir.name}")
    print(f"  mode:          {cfg.get('mode', '?')}")
    print(f"  dataset:       {cfg.get('dataset', '?')}")
    print(f"  profile:       {cfg.get('sweep_profile') or '(explicit)'}")
    print()
    print(f"progress:        {n_done}/{n_total}   "
          f"({100.0 * n_done / n_total:.1f}%)")
    print(f"  remaining:     {n_remaining}")
    print(f"  in progress:   {len(in_progress)}")
    print(f"  not started:   {len(not_started)}")
    print(f"  ETA:           {eta_str}")

    if done:
        last_idx, last_lbl, last_t = max(done, key=lambda kv: kv[2])
        age = time.time() - last_t
        print(f"  last done:     [{last_idx}/{n_total}] {last_lbl}  "
              f"({_fmt_dur(age)} ago)")

    # Detail: which pairs are left
    width = len(str(n_total))
    if in_progress:
        print()
        print(f"in progress ({len(in_progress)}):")
        for idx, lbl in in_progress:
            print(f"  [{idx:>{width}}/{n_total}] {lbl}")

    if not_started:
        print()
        print(f"not started ({len(not_started)}):")
        for idx, lbl in not_started:
            print(f"  [{idx:>{width}}/{n_total}] {lbl}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
