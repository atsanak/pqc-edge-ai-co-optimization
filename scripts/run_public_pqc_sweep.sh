#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-ua_detrac}"
MODE="${MODE:-cloud_ag}"
REMOTE_URL="${REMOTE_URL:?Set REMOTE_URL, for example https://GROUND_STATION_IP:8443/infer}"
REMOTE_CAFILE="${REMOTE_CAFILE:-src/network/certs/server.crt}"
PQC_SWEEP_PROFILE="${PQC_SWEEP_PROFILE:-paper_curated}"
MAX_SEQUENCES="${MAX_SEQUENCES:-}"
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-}"
SCENES_FILE="${SCENES_FILE:-}"
MISSING_SCENES_FROM="${MISSING_SCENES_FROM:-}"
BASELINE_THROUGHPUT_MBPS="${BASELINE_THROUGHPUT_MBPS:-}"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"

args=(
  experiments/run_split_inference_benchmark.py
  --dataset "$DATASET"
  --mode "$MODE"
  --remote-url "$REMOTE_URL"
  --remote-cafile "$REMOTE_CAFILE"
  --pqc-sweep
  --pqc-sweep-profile "$PQC_SWEEP_PROFILE"
  --non-interactive
)

if [[ -n "$MAX_SEQUENCES" ]]; then
  args+=(--max-sequences "$MAX_SEQUENCES")
fi

if [[ -n "$MAX_FRAMES_PER_SEQ" ]]; then
  args+=(--max-frames-per-seq "$MAX_FRAMES_PER_SEQ")
fi

if [[ -n "$SCENES_FILE" ]]; then
  args+=(--scenes-file "$SCENES_FILE")
fi

if [[ -n "$MISSING_SCENES_FROM" ]]; then
  args+=(--missing-scenes-from "$MISSING_SCENES_FROM")
fi

if [[ -n "$BASELINE_THROUGHPUT_MBPS" ]]; then
  args+=(--baseline-throughput-mbps "$BASELINE_THROUGHPUT_MBPS")
fi

python "${args[@]}"
