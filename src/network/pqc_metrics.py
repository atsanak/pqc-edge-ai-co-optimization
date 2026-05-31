"""
PQC metric helpers for transport timing, byte expansion, and resource data.

The output schema is intentionally stable because experiment summaries are
joined column-by-column across classical and PQC transport modes.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Iterable, List, Mapping, Optional

# rusage is Unix-only; graceful fallback elsewhere
try:
    import resource as _resource  # noqa: F401
    _RUSAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RUSAGE_AVAILABLE = False


# =============================================================================
# Hashing helpers.
# =============================================================================

def compute_sha256_hex(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()


def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Percentile + distribution-shape primitives.
# =============================================================================

def _percentile(sorted_data, p):
    """Linear-interpolation percentile on pre-sorted data (Python 3.6+)."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_data[f])
    return float(sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f]))


def _skewness(data, mean_val, stddev_val):
    """Fisher-Pearson sample skewness (bias-corrected G1)."""
    n = len(data)
    if n < 3 or stddev_val <= 0:
        return 0.0
    m3 = sum((x - mean_val) ** 3 for x in data) / n
    g1 = m3 / (stddev_val ** 3)
    return g1 * math.sqrt(n * (n - 1)) / (n - 2)


def _excess_kurtosis(data, mean_val, stddev_val):
    """Sample excess kurtosis (bias-corrected G2). 0 ≈ Gaussian tails."""
    n = len(data)
    if n < 4 or stddev_val <= 0:
        return 0.0
    m4 = sum((x - mean_val) ** 4 for x in data) / n
    g2 = m4 / (stddev_val ** 4) - 3.0
    return ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)


def _trimmed_mean(sorted_data, trim_fraction=0.10):
    """Symmetric trimmed mean. Default: drop 10% from each tail."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = int(n * trim_fraction)
    trimmed = sorted_data[k:n - k] if n - 2 * k > 0 else sorted_data
    return sum(trimmed) / len(trimmed) if trimmed else 0.0


def _geometric_mean(data):
    """Geometric mean of positive values. Returns 0.0 if any non-positive."""
    if not data or any(x <= 0 for x in data):
        return 0.0
    log_sum = sum(math.log(x) for x in data)
    return math.exp(log_sum / len(data))


# =============================================================================
# Chunk-level statistics.
# =============================================================================

def compute_chunk_statistics(times_array):
    """Compute publication-grade statistics from per-chunk timing samples.

    Returns a dict with: count, mean, stddev, min, max, p50, p95, p99,
    mad, cv_pct, jitter, jitter_pct, jitter_class.
    """
    n = len(times_array)
    if n == 0:
        return {k: 0.0 for k in [
            "count", "mean", "stddev", "min", "max",
            "p50", "p95", "p99", "mad", "cv_pct",
            "jitter", "jitter_pct", "jitter_class"
        ]}

    mean_val = statistics.mean(times_array)
    stddev_val = statistics.stdev(times_array) if n > 1 else 0.0
    min_val = min(times_array)
    max_val = max(times_array)

    sorted_data = sorted(times_array)
    p50 = _percentile(sorted_data, 50)
    p95 = _percentile(sorted_data, 95)
    p99 = _percentile(sorted_data, 99)

    median_val = statistics.median(times_array)
    mad_val = statistics.median([abs(x - median_val) for x in times_array])

    cv_pct = (100.0 * stddev_val / mean_val) if mean_val > 0 else 0.0

    jitter_val = max_val - min_val
    jitter_pct = (100.0 * jitter_val / mean_val) if mean_val > 0 else 0.0

    if jitter_pct < 1.0:
        jitter_class = "VERY_LOW"
    elif jitter_pct < 5.0:
        jitter_class = "LOW"
    elif jitter_pct < 15.0:
        jitter_class = "MODERATE"
    else:
        jitter_class = "HIGH"

    return {
        "count": n,
        "mean": mean_val,
        "stddev": stddev_val,
        "min": min_val,
        "max": max_val,
        "p50": p50, "p95": p95, "p99": p99,
        "mad": mad_val,
        "cv_pct": cv_pct,
        "jitter": jitter_val, "jitter_pct": jitter_pct,
        "jitter_class": jitter_class,
    }


def compute_tail_amplification(baseline_stats, stressed_stats):
    """Tail amplification factor — compares p99/p50 between two stat dicts."""
    base_ratio = (baseline_stats["p99"] / baseline_stats["p50"]
                  if baseline_stats["p50"] > 0 else 1.0)
    stress_ratio = (stressed_stats["p99"] / stressed_stats["p50"]
                    if stressed_stats["p50"] > 0 else 1.0)
    factor = stress_ratio / base_ratio if base_ratio > 0 else 1.0
    if factor < 0.5 or factor > 5.0:
        factor = 1.0
    return factor


def compute_expansion_ratio(chunk_size, encrypted_msg_len):
    """Expansion ratio of a single chunk after sign+encrypt."""
    return encrypted_msg_len / chunk_size if chunk_size > 0 else 0.0


def compute_energy_per_mb(power_watts, transfer_time_s, total_mb):
    """Energy per MB from tegrastats-style power readings (E = P·t)."""
    if total_mb <= 0 or transfer_time_s <= 0:
        return 0.0
    total_energy_j = power_watts * transfer_time_s
    return total_energy_j / total_mb


# =============================================================================
# Determinism class + advanced distribution stats.
# =============================================================================

def classify_timing_determinism(cv_pct, jitter_pct):
    """Fingerprint timing determinism of a PQC primitive.

      - ML-DSA (lattice, rejection with bounded norms): VERY_DETERMINISTIC
      - SLH-DSA (hash-based):                           VERY_DETERMINISTIC/DETERMINISTIC
      - FN-DSA (lattice, Gaussian sampling):            VARIABLE/HIGHLY_VARIABLE
    """
    if cv_pct < 0.5 and jitter_pct < 2.0:
        return "VERY_DETERMINISTIC"
    if cv_pct < 2.0 and jitter_pct < 10.0:
        return "DETERMINISTIC"
    if cv_pct < 10.0 and jitter_pct < 30.0:
        return "SEMI_VARIABLE"
    if cv_pct < 25.0:
        return "VARIABLE"
    return "HIGHLY_VARIABLE"


def compute_advanced_statistics(times_array, base_stats=None):
    """Distribution shape + tail metrics extending compute_chunk_statistics()."""
    n = len(times_array)
    if n == 0:
        empty = {k: 0.0 for k in [
            "q1", "q3", "iqr", "tukey_lower", "tukey_upper",
            "outlier_count", "outlier_pct", "skewness", "excess_kurtosis",
            "trimmed_mean_10pct", "geometric_mean",
            "tail_ratio_p99_p50", "tail_ratio_p95_p50",
        ]}
        empty["determinism_class"] = "UNKNOWN"
        return empty

    if base_stats is None:
        base_stats = compute_chunk_statistics(times_array)

    sorted_data = sorted(times_array)
    mean_val = base_stats["mean"]
    stddev_val = base_stats["stddev"]
    p50 = base_stats["p50"]
    p95 = base_stats["p95"]
    p99 = base_stats["p99"]
    cv_pct = base_stats["cv_pct"]
    jitter_pct = base_stats["jitter_pct"]

    q1 = _percentile(sorted_data, 25)
    q3 = _percentile(sorted_data, 75)
    iqr = q3 - q1
    tukey_lower = q1 - 1.5 * iqr
    tukey_upper = q3 + 1.5 * iqr
    outliers = [x for x in times_array if x < tukey_lower or x > tukey_upper]
    outlier_count = len(outliers)
    outlier_pct = (100.0 * outlier_count / n) if n > 0 else 0.0

    return {
        "q1": q1, "q3": q3, "iqr": iqr,
        "tukey_lower": tukey_lower, "tukey_upper": tukey_upper,
        "outlier_count": outlier_count, "outlier_pct": outlier_pct,
        "skewness": _skewness(times_array, mean_val, stddev_val),
        "excess_kurtosis": _excess_kurtosis(times_array, mean_val, stddev_val),
        "trimmed_mean_10pct": _trimmed_mean(sorted_data, 0.10),
        "geometric_mean": _geometric_mean(times_array),
        "tail_ratio_p99_p50": (p99 / p50) if p50 > 0 else 1.0,
        "tail_ratio_p95_p50": (p95 / p50) if p50 > 0 else 1.0,
        "determinism_class": classify_timing_determinism(cv_pct, jitter_pct),
    }


# =============================================================================
# Throughput / bottleneck attribution.
# =============================================================================

def compute_throughput_metrics(total_bytes_raw, total_bytes_on_wire,
                               total_time_s, num_chunks=0):
    """Throughput, goodput, chunk/signature rates, bandwidth overhead."""
    mb = 1024.0 * 1024.0
    if total_time_s <= 0:
        return {"throughput_mbps": 0.0, "goodput_mbps": 0.0,
                "chunk_rate_hz": 0.0, "signature_rate_hz": 0.0,
                "bandwidth_overhead_pct": 0.0}
    throughput_mbps = (total_bytes_on_wire / mb) / total_time_s
    goodput_mbps = (total_bytes_raw / mb) / total_time_s
    chunk_rate_hz = (num_chunks / total_time_s) if num_chunks > 0 else 0.0
    overhead_pct = (100.0 * (total_bytes_on_wire - total_bytes_raw) / total_bytes_raw
                    if total_bytes_raw > 0 else 0.0)
    return {"throughput_mbps": throughput_mbps,
            "goodput_mbps": goodput_mbps,
            "chunk_rate_hz": chunk_rate_hz,
            "signature_rate_hz": chunk_rate_hz,
            "bandwidth_overhead_pct": overhead_pct}


def compute_bottleneck_attribution(sign_time_s, encrypt_time_s, network_time_s):
    """Amdahl-style attribution of total time across sign/encrypt/network."""
    total = sign_time_s + encrypt_time_s + network_time_s
    if total <= 0:
        return {"sign_pct": 0.0, "encrypt_pct": 0.0, "network_pct": 0.0,
                "dominant_phase": "UNKNOWN",
                "sign_to_encrypt_ratio": 0.0, "sign_to_network_ratio": 0.0}
    sign_pct = 100.0 * sign_time_s / total
    encrypt_pct = 100.0 * encrypt_time_s / total
    network_pct = 100.0 * network_time_s / total
    phases = {"SIGNING": sign_pct, "ENCRYPTION": encrypt_pct, "NETWORK": network_pct}
    return {"sign_pct": sign_pct, "encrypt_pct": encrypt_pct, "network_pct": network_pct,
            "dominant_phase": max(phases, key=phases.get),
            "sign_to_encrypt_ratio": (sign_time_s / encrypt_time_s) if encrypt_time_s > 0 else 0.0,
            "sign_to_network_ratio": (sign_time_s / network_time_s) if network_time_s > 0 else 0.0}


def compute_vs_baseline(scheme_value, baseline_value, higher_is_better=True):
    """Slowdown / speedup multiplier and overhead percentage."""
    if baseline_value <= 0:
        return {"ratio": 0.0, "overhead_pct": 0.0, "slowdown": 0.0}
    ratio = scheme_value / baseline_value
    overhead_pct = 100.0 * (scheme_value - baseline_value) / baseline_value
    if higher_is_better:
        slowdown = (baseline_value / scheme_value) if scheme_value > 0 else float("inf")
    else:
        slowdown = ratio
    return {"ratio": ratio, "overhead_pct": overhead_pct, "slowdown": slowdown}


def compute_session_amortization(session_setup_ms, throughput_mbps,
                                 baseline_throughput_mbps):
    """Break-even transfer size (MB) before PQC setup is amortized."""
    if throughput_mbps <= 0 or baseline_throughput_mbps <= 0:
        return {"break_even_mb": float("inf"), "per_mb_penalty_ms": 0.0,
                "throughput_ratio": 0.0}
    if throughput_mbps >= baseline_throughput_mbps:
        return {"break_even_mb": 0.0, "per_mb_penalty_ms": 0.0,
                "throughput_ratio": throughput_mbps / baseline_throughput_mbps}
    per_mb_pqc_ms = 1000.0 / throughput_mbps
    per_mb_base_ms = 1000.0 / baseline_throughput_mbps
    per_mb_penalty_ms = per_mb_pqc_ms - per_mb_base_ms
    if per_mb_penalty_ms <= 0:
        return {"break_even_mb": 0.0, "per_mb_penalty_ms": 0.0,
                "throughput_ratio": throughput_mbps / baseline_throughput_mbps}
    return {"break_even_mb": session_setup_ms / per_mb_penalty_ms,
            "per_mb_penalty_ms": per_mb_penalty_ms,
            "throughput_ratio": throughput_mbps / baseline_throughput_mbps}


# =============================================================================
# Resource monitoring: rusage on Unix.
# =============================================================================

def get_resource_snapshot():
    """Capture current process resource usage (CPU sec / RSS / ctx / faults)."""
    if not _RUSAGE_AVAILABLE:
        return {"available": False, "user_cpu_s": 0.0, "sys_cpu_s": 0.0,
                "rss_mb": 0.0, "vol_ctxs": 0, "invol_ctxs": 0,
                "minor_faults": 0, "major_faults": 0}
    import sys as _sys
    r = _resource.getrusage(_resource.RUSAGE_SELF)
    if _sys.platform == "darwin":
        rss_mb = r.ru_maxrss / (1024.0 * 1024.0)
    else:
        rss_mb = r.ru_maxrss / 1024.0
    return {"available": True,
            "user_cpu_s": r.ru_utime, "sys_cpu_s": r.ru_stime,
            "rss_mb": rss_mb,
            "vol_ctxs": r.ru_nvcsw, "invol_ctxs": r.ru_nivcsw,
            "minor_faults": r.ru_minflt, "major_faults": r.ru_majflt}


def compute_resource_delta(before, after):
    """Delta between two resource snapshots; peak_rss_mb is the post snapshot."""
    if not (before.get("available") and after.get("available")):
        return {k: 0 for k in ["user_cpu_s", "sys_cpu_s", "vol_ctxs",
                               "invol_ctxs", "minor_faults", "major_faults"]} | {"peak_rss_mb": 0.0}
    return {"user_cpu_s":   after["user_cpu_s"]   - before["user_cpu_s"],
            "sys_cpu_s":    after["sys_cpu_s"]    - before["sys_cpu_s"],
            "vol_ctxs":     after["vol_ctxs"]     - before["vol_ctxs"],
            "invol_ctxs":   after["invol_ctxs"]   - before["invol_ctxs"],
            "minor_faults": after["minor_faults"] - before["minor_faults"],
            "major_faults": after["major_faults"] - before["major_faults"],
            "peak_rss_mb":  after["rss_mb"]}


def compute_cpu_bound_score(cpu_time_s, wall_time_s):
    """~1 = CPU-bound on one core, <1 = waiting on IO, >1 = multi-core."""
    if wall_time_s <= 0:
        return 0.0
    return cpu_time_s / wall_time_s


# =============================================================================
# Phase-aware row builder
# Maps the in-memory per-call data structures into stable summary columns.
# =============================================================================

# Phase to column-name prefix.
PHASE_PREFIX = {
    "signing":      "sign",
    "encryption":   "encrypt",
    "decryption":   "decrypt",
    "verification": "verify",
    "network_send": "network_send",
    "network_recv": "network_recv",
}


def _flatten_chunk_stats(times_s: List[float], prefix: str,
                        expansion_ratio: float = 0.0) -> dict:
    """Emit per-phase columns matching the chunk-stats schema:

        <prefix>_chunk_count, <prefix>_mean_s, <prefix>_stddev_s,
        <prefix>_min_s, <prefix>_max_s, <prefix>_p50_s, <prefix>_p95_s,
        <prefix>_p99_s, <prefix>_mad_s, <prefix>_cv_pct,
        <prefix>_jitter_s, <prefix>_jitter_pct, <prefix>_jitter_class,
        <prefix>_expansion_ratio
    """
    base = compute_chunk_statistics(times_s)
    return {
        f"{prefix}_chunk_count":    base["count"],
        f"{prefix}_mean_s":         base["mean"],
        f"{prefix}_stddev_s":       base["stddev"],
        f"{prefix}_min_s":          base["min"],
        f"{prefix}_max_s":          base["max"],
        f"{prefix}_p50_s":          base["p50"],
        f"{prefix}_p95_s":          base["p95"],
        f"{prefix}_p99_s":          base["p99"],
        f"{prefix}_mad_s":          base["mad"],
        f"{prefix}_cv_pct":         base["cv_pct"],
        f"{prefix}_jitter_s":       base["jitter"],
        f"{prefix}_jitter_pct":     base["jitter_pct"],
        f"{prefix}_jitter_class":   base["jitter_class"],
        f"{prefix}_expansion_ratio": expansion_ratio,
    }


def _flatten_advanced_stats(times_s: List[float], prefix: str) -> dict:
    """Emit per-phase columns matching the advanced chunk-stats schema:

        <prefix>_q1_s, <prefix>_q3_s, <prefix>_iqr_s,
        <prefix>_tukey_lower_s, <prefix>_tukey_upper_s,
        <prefix>_outlier_count, <prefix>_outlier_pct,
        <prefix>_skewness, <prefix>_excess_kurtosis,
        <prefix>_trimmed_mean_10pct_s, <prefix>_geometric_mean_s,
        <prefix>_tail_ratio_p99_p50, <prefix>_tail_ratio_p95_p50,
        <prefix>_determinism_class
    """
    adv = compute_advanced_statistics(times_s)
    return {
        f"{prefix}_q1_s":                  adv["q1"],
        f"{prefix}_q3_s":                  adv["q3"],
        f"{prefix}_iqr_s":                 adv["iqr"],
        f"{prefix}_tukey_lower_s":         adv["tukey_lower"],
        f"{prefix}_tukey_upper_s":         adv["tukey_upper"],
        f"{prefix}_outlier_count":         adv["outlier_count"],
        f"{prefix}_outlier_pct":           adv["outlier_pct"],
        f"{prefix}_skewness":              adv["skewness"],
        f"{prefix}_excess_kurtosis":       adv["excess_kurtosis"],
        f"{prefix}_trimmed_mean_10pct_s":  adv["trimmed_mean_10pct"],
        f"{prefix}_geometric_mean_s":      adv["geometric_mean"],
        f"{prefix}_tail_ratio_p99_p50":    adv["tail_ratio_p99_p50"],
        f"{prefix}_tail_ratio_p95_p50":    adv["tail_ratio_p95_p50"],
        f"{prefix}_determinism_class":     adv["determinism_class"],
    }


def build_phase_columns(times_s: List[float], phase: str,
                        expansion_ratio: float = 0.0) -> dict:
    """All chunk + advanced columns for one phase, in one dict."""
    prefix = PHASE_PREFIX.get(phase, phase)
    out: dict = {}
    out.update(_flatten_chunk_stats(times_s, prefix, expansion_ratio))
    out.update(_flatten_advanced_stats(times_s, prefix))
    return out


# =============================================================================
# Aggregation across multiple per-imgsz detectors
# =============================================================================

def merge_arrays(detectors: Iterable, attr_name: str) -> List[float]:
    out: List[float] = []
    for d in detectors:
        out.extend(getattr(d, attr_name, []) or [])
    return out
