# Reproduction Notes

These notes expand the short README commands for reviewers who want to run the benchmark on Jetson Orin-class edge hardware with a separate ground-station inference server.

## Data Layout

Place datasets under the repository root:

```text
ua_detrac/content/UA-DETRAC/DETRAC_Upload/images/val/...
ua_detrac/content/UA-DETRAC/DETRAC_Upload/labels/val/...
ua_detrac/content/UA-DETRAC/DETRAC_Upload/labels_with_ids/val/...
mot17/MOT17/train/...
```

UA-DETRAC unique-object recall expects `labels_with_ids`. If those labels are not already present, generate them from official UA-DETRAC XML annotations:

```bash
PYTHONPATH=src python -m eval_tools.generate_detrac_tracking_ids
```

The generator expects official XML annotations under:

```text
DETRAC-Annos/DETRAC-Test-Annotations-XML/
```

## Ground-Station Server

Generate a local certificate on the ground station:

```bash
./src/network/generate_cert.sh GROUND_STATION_IP
```

Start the server:

```bash
PYTHONPATH=src python -m network.server \
  --host 0.0.0.0 \
  --port 8443 \
  --weight yolo11s.pt \
  --device cuda \
  --no-prompt
```

Copy `src/network/certs/server.crt` to the edge client and pass it as `--remote-cafile`.

## Edge Client

Smoke test one sequence:

```bash
REMOTE_URL=https://GROUND_STATION_IP:8443/infer \
REMOTE_CAFILE=src/network/certs/server.crt \
MAX_SEQUENCES=1 \
./scripts/run_jetson_orin_experiment.sh
```

Run the curated public PQC sweep:

```bash
REMOTE_URL=https://GROUND_STATION_IP:8443/infer \
REMOTE_CAFILE=src/network/certs/server.crt \
DATASET=ua_detrac \
MODE=cloud_ag \
./scripts/run_public_pqc_sweep.sh
```

Track a running sweep from another shell:

```bash
PYTHONPATH=src python -m eval_tools.check_sweep_progress runs/<timestamp>_<mode>_<dataset>_pqc_sweep_<profile>
```

## Public Result Verification

The released CSV can be checked without raw datasets:

```bash
python scripts/verify_public_results.py
python scripts/verify_public_results.py --print-selected
```

This verifies row counts, transport verification status, payload hash checks, failure counts, and the exact selected rows shown in `README.md`.

## Output Files

Single runs produce:

```text
runs/<timestamp>_<mode>_<dataset>/run_config.json
runs/<timestamp>_<mode>_<dataset>/per_sequence_results.csv
runs/<timestamp>_<mode>_<dataset>/summary.csv
```

Sweeps additionally produce one subdirectory per transport configuration and a campaign-level `master_summary.csv`.
