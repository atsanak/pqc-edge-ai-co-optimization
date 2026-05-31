"""
Public PQC sweep definitions used by the split-inference benchmark.

The full artifact sweep is loaded from
``configs/pqc_paper_curated_pairs.csv``. It contains the 84 KEM/signature
pairs used for the released summary results and excludes experimental or
non-finalized families from the broader local research workspace.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PUBLIC_SWEEP_CSV = os.path.join(_REPO_ROOT, "configs", "pqc_paper_curated_pairs.csv")


MAX_COVERAGE_PAIRS: List[Tuple[str, str, str]] = [
    ("ML-KEM-512", "ML-DSA-44", "ML-KEM-512 + ML-DSA-44"),
    ("ML-KEM-768", "ML-DSA-65", "ML-KEM-768 + ML-DSA-65"),
    ("ML-KEM-1024", "ML-DSA-87", "ML-KEM-1024 + ML-DSA-87"),
    ("ML-KEM-768", "Falcon-512", "ML-KEM-768 + Falcon-512"),
    ("ML-KEM-768", "Falcon-1024", "ML-KEM-768 + Falcon-1024"),
    ("BIKE-L3", "ML-DSA-65", "BIKE-L3 + ML-DSA-65"),
    ("FrodoKEM-976-AES", "ML-DSA-65", "FrodoKEM-976-AES + ML-DSA-65"),
    ("FrodoKEM-976-AES", "Falcon-512", "FrodoKEM-976-AES + Falcon-512"),
]


def _load_pairs_csv(path: str = _PUBLIC_SWEEP_CSV) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kem = (row.get("kem") or "").strip()
                sig = (row.get("sig") or "").strip()
                if kem and sig:
                    pairs.append((kem, sig))
    except FileNotFoundError:
        pass
    return pairs


PAPER_CURATED_PAIRS: List[Tuple[str, str]] = _load_pairs_csv()

_PROFILE_STANDARD_KEMS = [
    "ML-KEM-512", "ML-KEM-768", "ML-KEM-1024",
    "HQC-128", "HQC-192", "HQC-256",
]
_PROFILE_STANDARD_SIGS = [
    "ML-DSA-44", "ML-DSA-65", "ML-DSA-87",
    "Falcon-512", "Falcon-1024",
    "SPHINCS+-SHA2-128f-simple", "SPHINCS+-SHA2-128s-simple",
]

_PROFILE_EXTENDED_KEMS = _PROFILE_STANDARD_KEMS + [
    "FrodoKEM-640-AES", "FrodoKEM-976-AES", "FrodoKEM-1344-AES",
    "BIKE-L1", "BIKE-L3", "BIKE-L5",
]
_PROFILE_EXTENDED_SIGS = _PROFILE_STANDARD_SIGS + [
    "Falcon-padded-512", "Falcon-padded-1024",
    "SPHINCS+-SHA2-192f-simple", "SPHINCS+-SHA2-192s-simple",
    "SPHINCS+-SHA2-256f-simple", "SPHINCS+-SHA2-256s-simple",
    "SPHINCS+-SHAKE-128f-simple", "SPHINCS+-SHAKE-128s-simple",
    "SPHINCS+-SHAKE-192f-simple", "SPHINCS+-SHAKE-192s-simple",
    "SPHINCS+-SHAKE-256f-simple", "SPHINCS+-SHAKE-256s-simple",
]

_PROFILE_STRESS_KEMS = ["ML-KEM-1024", "HQC-256"]
_PROFILE_STRESS_SIGS = [
    "ML-DSA-87", "Falcon-1024",
    "SPHINCS+-SHA2-256f-simple", "SPHINCS+-SHA2-256s-simple",
]


def expand_profile(name: str) -> List[Tuple[str, str]]:
    """Return the (kem, sig) pairs for a named profile."""
    name = name.strip().lower()
    if name == "paper_curated":
        return list(PAPER_CURATED_PAIRS)
    if name == "standard":
        return [(k, s) for k in _PROFILE_STANDARD_KEMS for s in _PROFILE_STANDARD_SIGS]
    if name == "extended":
        return [(k, s) for k in _PROFILE_EXTENDED_KEMS for s in _PROFILE_EXTENDED_SIGS]
    if name == "stress":
        return [(k, s) for k in _PROFILE_STRESS_KEMS for s in _PROFILE_STRESS_SIGS]
    if name == "max_coverage":
        return [(k, s) for (k, s, _label) in MAX_COVERAGE_PAIRS]
    raise ValueError(f"Unknown sweep profile: {name!r}")


PROFILE_NAMES = ("standard", "extended", "stress", "max_coverage", "paper_curated")


def runnable_pairs(pairs: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split a pair list by checking the active liboqs enabled mechanisms."""
    try:
        from network import pqc_crypto as pc
        kems = set(pc.list_available_kems())
        sigs = set(pc.list_available_signatures())
    except Exception:
        return list(pairs), []

    runnable: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    for kem, sig in pairs:
        kem_ok = False
        sig_ok = False
        try:
            pc.canonical_kem_name(kem, list(kems))
            kem_ok = True
        except Exception:
            pass
        try:
            pc.canonical_sig_name(sig, list(sigs))
            sig_ok = True
        except Exception:
            pass
        (runnable if kem_ok and sig_ok else skipped).append((kem, sig))
    return runnable, skipped


CLASSICAL_PAIR = ("__CLASSICAL__", "__CLASSICAL__")
SWEEP_SENTINEL = "__PQC_SWEEP__"
FULL_SUITE_SENTINEL = "__FULL_SUITE__"

