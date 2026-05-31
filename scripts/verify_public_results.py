#!/usr/bin/env python3
"""Verify the released public result summary.

This script intentionally checks exact CSV string values for the selected rows
reported in README.md. It does not recompute detector metrics from raw labels;
raw datasets and internal run directories are not part of this public release.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results" / "verified_summary.csv"

EXPECTED_COUNTS = {
    ("ua_detrac", "cloud_baseline"): 85,
    ("ua_detrac", "cloud_ag"): 85,
    ("mot17", "cloud_baseline"): 85,
    ("mot17", "cloud_ag"): 85,
}

REQUIRED_COLUMNS = {
    "dataset",
    "mode",
    "crypto_mode",
    "kem_scheme",
    "sig_scheme",
    "n_sequences",
    "total_frames",
    "all_verification_ok",
    "signature_failure_count_sum",
    "decrypt_failure_count_sum",
    "payload_sha256_match_all",
    "fps",
    "mAP_50",
    "unique_recall",
    "net_upload_MB",
    "net_avg_rtt_ms",
    "avg_sign_pq_ms",
}

SELECTED_ROWS = [
    {
        "dataset": "ua_detrac",
        "mode": "cloud_baseline",
        "crypto_mode": "classical",
        "kem_scheme": "CLASSICAL",
        "sig_scheme": "CLASSICAL",
        "n_sequences": "40",
        "total_frames": "56340",
        "fps": "14.575426082404524",
        "mAP_50": "0.7887680386454994",
        "unique_recall": "0.995519413376781",
        "net_upload_MB": "341.5926197052002",
        "net_avg_rtt_ms": "38.531959471702606",
        "avg_sign_pq_ms": "0.0",
        "all_verification_ok": "True",
    },
    {
        "dataset": "ua_detrac",
        "mode": "cloud_baseline",
        "crypto_mode": "pqc",
        "kem_scheme": "ML-KEM-768",
        "sig_scheme": "ML-DSA-65",
        "n_sequences": "40",
        "total_frames": "56340",
        "fps": "14.028069088272435",
        "mAP_50": "0.7887680386454994",
        "unique_recall": "0.995519413376781",
        "net_upload_MB": "349.4945358276367",
        "net_avg_rtt_ms": "39.89171083152946",
        "avg_sign_pq_ms": "2.821906350675323",
        "all_verification_ok": "True",
    },
    {
        "dataset": "ua_detrac",
        "mode": "cloud_ag",
        "crypto_mode": "classical",
        "kem_scheme": "CLASSICAL",
        "sig_scheme": "CLASSICAL",
        "n_sequences": "40",
        "total_frames": "56340",
        "fps": "26.60938899096889",
        "mAP_50": "0.7023915756083356",
        "unique_recall": "0.9950082440024337",
        "net_upload_MB": "83.27949299812317",
        "net_avg_rtt_ms": "31.333781755068173",
        "avg_sign_pq_ms": "0.0",
        "all_verification_ok": "True",
    },
    {
        "dataset": "ua_detrac",
        "mode": "cloud_ag",
        "crypto_mode": "pqc",
        "kem_scheme": "ML-KEM-768",
        "sig_scheme": "ML-DSA-65",
        "n_sequences": "40",
        "total_frames": "56340",
        "fps": "27.70964249435574",
        "mAP_50": "0.7023915756083356",
        "unique_recall": "0.9950082440024337",
        "net_upload_MB": "86.36703839302064",
        "net_avg_rtt_ms": "30.21689020118606",
        "avg_sign_pq_ms": "1.6222213314265477",
        "all_verification_ok": "True",
    },
    {
        "dataset": "mot17",
        "mode": "cloud_baseline",
        "crypto_mode": "classical",
        "kem_scheme": "CLASSICAL",
        "sig_scheme": "CLASSICAL",
        "n_sequences": "7",
        "total_frames": "5316",
        "fps": "7.7146948806774605",
        "mAP_50": "0.7380300020467594",
        "unique_recall": "0.9524993578714935",
        "net_upload_MB": "375.42105538504467",
        "net_avg_rtt_ms": "108.11250126131462",
        "avg_sign_pq_ms": "0.0",
        "all_verification_ok": "True",
    },
    {
        "dataset": "mot17",
        "mode": "cloud_baseline",
        "crypto_mode": "pqc",
        "kem_scheme": "ML-KEM-768",
        "sig_scheme": "ML-DSA-65",
        "n_sequences": "7",
        "total_frames": "5316",
        "fps": "7.62768436135504",
        "mAP_50": "0.7380300020467594",
        "unique_recall": "0.9524993578714935",
        "net_upload_MB": "379.6815414428711",
        "net_avg_rtt_ms": "108.79375112063074",
        "avg_sign_pq_ms": "2.9735494385733494",
        "all_verification_ok": "True",
    },
    {
        "dataset": "mot17",
        "mode": "cloud_ag",
        "crypto_mode": "classical",
        "kem_scheme": "CLASSICAL",
        "sig_scheme": "CLASSICAL",
        "n_sequences": "7",
        "total_frames": "5316",
        "fps": "11.37235775276736",
        "mAP_50": "0.6639733477363562",
        "unique_recall": "0.9334837951481472",
        "net_upload_MB": "142.3633279800415",
        "net_avg_rtt_ms": "72.14300010911329",
        "avg_sign_pq_ms": "0.0",
        "all_verification_ok": "True",
    },
    {
        "dataset": "mot17",
        "mode": "cloud_ag",
        "crypto_mode": "pqc",
        "kem_scheme": "ML-KEM-768",
        "sig_scheme": "ML-DSA-65",
        "n_sequences": "7",
        "total_frames": "5316",
        "fps": "11.457471058242273",
        "mAP_50": "0.6639733477363562",
        "unique_recall": "0.9334837951481472",
        "net_upload_MB": "144.95119789668493",
        "net_avg_rtt_ms": "73.59846595946367",
        "avg_sign_pq_ms": "2.1176792730235716",
        "all_verification_ok": "True",
    },
]


def selected_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["dataset"],
        row["mode"],
        row["crypto_mode"],
        row["kem_scheme"],
        row["sig_scheme"],
    )


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing CSV: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            fail(f"missing required columns: {sorted(missing)}")
        return list(reader)


def verify(path: Path) -> list[dict[str, str]]:
    rows = load_rows(path)
    if len(rows) != 340:
        fail(f"expected 340 rows, found {len(rows)}")

    counts = Counter((r["dataset"], r["mode"]) for r in rows)
    if counts != EXPECTED_COUNTS:
        fail(f"unexpected dataset/mode counts: {dict(counts)}")

    bad_verify = [r for r in rows if r["all_verification_ok"] != "True"]
    if bad_verify:
        fail(f"{len(bad_verify)} rows have all_verification_ok != True")

    sig_failures = sum(int(r["signature_failure_count_sum"]) for r in rows)
    decrypt_failures = sum(int(r["decrypt_failure_count_sum"]) for r in rows)
    if sig_failures or decrypt_failures:
        fail(f"failure counts are nonzero: signature={sig_failures}, decrypt={decrypt_failures}")

    bad_hash = [r for r in rows if r["payload_sha256_match_all"] != "True"]
    if bad_hash:
        fail(f"{len(bad_hash)} rows have payload_sha256_match_all != True")

    by_key = {selected_key(r): r for r in rows}
    for expected in SELECTED_ROWS:
        key = selected_key(expected)
        actual = by_key.get(key)
        if actual is None:
            fail(f"missing selected row: {key}")
        for col, expected_value in expected.items():
            if actual[col] != expected_value:
                fail(
                    f"selected row {key} column {col}: "
                    f"expected {expected_value!r}, found {actual[col]!r}"
                )

    return rows


def print_selected() -> None:
    columns = [
        "dataset",
        "mode",
        "crypto_mode",
        "kem_scheme",
        "sig_scheme",
        "n_sequences",
        "total_frames",
        "fps",
        "mAP_50",
        "unique_recall",
        "net_upload_MB",
        "net_avg_rtt_ms",
        "avg_sign_pq_ms",
        "all_verification_ok",
    ]
    print("\t".join(columns))
    for row in SELECTED_ROWS:
        print("\t".join(row[c] for c in columns))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--print-selected", action="store_true")
    args = parser.parse_args()

    rows = verify(args.csv)
    print(f"[OK] {args.csv}: {len(rows)} rows verified")
    print("[OK] dataset/mode counts: " + ", ".join(
        f"{dataset}/{mode}={count}"
        for (dataset, mode), count in sorted(EXPECTED_COUNTS.items())
    ))
    print("[OK] selected README rows match exact CSV values")
    if args.print_selected:
        print()
        print_selected()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
