#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-ua_detrac}"
MODE="${MODE:-cloud_ag}"
REMOTE_URL="${REMOTE_URL:?Set REMOTE_URL, for example https://GROUND_STATION_IP:8443/infer}"
REMOTE_CAFILE="${REMOTE_CAFILE:-src/network/certs/server.crt}"
CRYPTO_MODE="${CRYPTO_MODE:-pqc}"
PQC_KEM="${PQC_KEM:-ML-KEM-768}"
PQC_SIG="${PQC_SIG:-ML-DSA-65}"
MAX_SEQUENCES="${MAX_SEQUENCES:-1}"
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-}"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"

args=(
  experiments/run_split_inference_benchmark.py
  --dataset "$DATASET"
  --mode "$MODE"
  --remote-url "$REMOTE_URL"
  --remote-cafile "$REMOTE_CAFILE"
  --crypto-mode "$CRYPTO_MODE"
  --pqc-kem "$PQC_KEM"
  --pqc-sig "$PQC_SIG"
  --max-sequences "$MAX_SEQUENCES"
  --non-interactive
)

if [[ -n "$MAX_FRAMES_PER_SEQ" ]]; then
  args+=(--max-frames-per-seq "$MAX_FRAMES_PER_SEQ")
fi

python "${args[@]}"
