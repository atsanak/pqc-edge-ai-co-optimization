"""Compat shim — historical name. All implementations now live in
``network.pqc_metrics`` (verbatim port from reference helper_functions.py
so master_summary columns join 1:1). New code should import from there
directly."""

from network.pqc_metrics import (  # noqa: F401
    _percentile, _skewness, _excess_kurtosis, _trimmed_mean, _geometric_mean,
    compute_chunk_statistics, compute_advanced_statistics,
    classify_timing_determinism, compute_tail_amplification,
    compute_throughput_metrics, compute_bottleneck_attribution,
    compute_vs_baseline, compute_session_amortization,
    get_resource_snapshot, compute_resource_delta, compute_cpu_bound_score,
    compute_sha256_hex, compute_file_sha256,
    compute_expansion_ratio, compute_energy_per_mb,
    merge_arrays as merge_time_arrays,
    build_phase_columns,
)
