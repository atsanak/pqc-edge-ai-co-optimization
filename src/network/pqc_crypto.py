"""
Cryptographic primitives shared by the PQC client and server.

This is a deliberately small, self-contained module that mirrors the design
of the reference `helper_functions.py` (hybrid ECDH+ML-KEM key exchange,
HKDF-SHA3-512 derivation, AES-256-GCM bulk encryption, hybrid ECDSA+ML-DSA
signing) but is structured around the HTTP-style request/response pattern
this project uses instead of reference file-transfer socket protocol.

Three transport-security modes are supported, selected by string:

    crypto_mode='tls13'      — no application-layer crypto on top of TLS 1.3
                                (the existing /infer endpoint is used).
    crypto_mode='classical'  — application-layer ECDH (SECP256R1) +
                                ECDSA per-call signing; AES-256-GCM body
                                encryption. No post-quantum primitives.
                                Pass kem_scheme=sig_scheme='CLASSICAL'.
    crypto_mode='pqc'        — hybrid ECDH+ML-KEM key establishment,
                                hybrid ECDSA+ML-DSA per-call signing,
                                AES-256-GCM body encryption.

The PQC scheme names use the friendly liboqs spellings (e.g. 'ML-KEM-768',
'ML-DSA-44', 'Falcon-512'). Anything liboqs's enabled-mechanism list
accepts is accepted here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# `oqs` is the liboqs-python binding. It's a hard dependency for both
# 'classical' and 'pqc' modes (signing primitives flow through here either
# way). Defer the import so plain TLS 1.3 mode still works on machines
# without liboqs installed.
try:
    import oqs as _oqs
    _OQS_OK = True
    _OQS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # noqa: BLE001
    _oqs = None
    _OQS_OK = False
    _OQS_IMPORT_ERROR = exc


CLASSICAL_SENTINEL = "CLASSICAL"


def require_oqs() -> None:
    """Raise a clear error if oqs is missing. Call before PQC operations."""
    if not _OQS_OK:
        raise RuntimeError(
            "liboqs-python ('oqs') is not importable. Install it into "
            "this Python environment before running PQC modes:\n"
            "    pip install liboqs-python\n"
            f"(import error: {_OQS_IMPORT_ERROR!r})"
        )


# ───────────────────────── liboqs introspection ─────────────────────────

def list_available_kems() -> list[str]:
    require_oqs()
    return list(_oqs.get_enabled_kem_mechanisms())


def list_available_signatures() -> list[str]:
    require_oqs()
    return list(_oqs.get_enabled_sig_mechanisms())


# ───────────────────────── ECDH (classical leg) ─────────────────────────

def ecdh_generate() -> Tuple[ec.EllipticCurvePrivateKey, bytes]:
    """Returns (private_key, public_key_bytes_PEM)."""
    priv = ec.generate_private_key(ec.SECP256R1())
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub_bytes


def ecdh_shared(my_priv: ec.EllipticCurvePrivateKey, peer_pub_bytes: bytes) -> bytes:
    peer_pub = serialization.load_pem_public_key(peer_pub_bytes)
    return my_priv.exchange(ec.ECDH(), peer_pub)


# ─────────────────────── ECDSA (classical signing) ──────────────────────

def ecdsa_generate() -> Tuple[ec.EllipticCurvePrivateKey, bytes]:
    priv = ec.generate_private_key(ec.SECP256R1())
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub_bytes


def ecdsa_sign(priv: ec.EllipticCurvePrivateKey, msg: bytes) -> bytes:
    return priv.sign(msg, ec.ECDSA(hashes.SHA256()))


def ecdsa_verify(pub_bytes: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        pub = serialization.load_pem_public_key(pub_bytes)
        pub.verify(sig, msg, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:  # noqa: BLE001
        return False


# ─────────────────────────── KEM (post-quantum) ─────────────────────────
#
# liboqs's KeyEncapsulation holds the secret key inside the object so we
# return the object itself when the caller will need to decapsulate later.

class KemSession:
    """Wraps an oqs.KeyEncapsulation. Lifetime = handshake."""

    def __init__(self, scheme: str):
        require_oqs()
        self.scheme = scheme
        self._kem = _oqs.KeyEncapsulation(scheme)
        self.public_key: bytes = self._kem.generate_keypair()

    def decapsulate(self, ciphertext: bytes) -> bytes:
        return self._kem.decap_secret(ciphertext)

    def free(self) -> None:
        try:
            self._kem.free()
        except Exception:  # noqa: BLE001
            pass


def kem_encapsulate(scheme: str, peer_pub: bytes) -> Tuple[bytes, bytes]:
    """One-shot encapsulation against a peer's public key. Returns (ct, ss)."""
    require_oqs()
    with _oqs.KeyEncapsulation(scheme) as kem:
        ct, ss = kem.encap_secret(peer_pub)
    return ct, ss


# ────────────────────────── PQC signatures ───────────────────────────────

class SigSession:
    """Wraps an oqs.Signature. Holds the private key inside the object."""

    def __init__(self, scheme: str):
        require_oqs()
        self.scheme = scheme
        self._sig = _oqs.Signature(scheme)
        self.public_key: bytes = self._sig.generate_keypair()

    def sign(self, msg: bytes) -> bytes:
        return self._sig.sign(msg)

    def free(self) -> None:
        try:
            self._sig.free()
        except Exception:  # noqa: BLE001
            pass


def pqc_verify(scheme: str, pub: bytes, msg: bytes, sig: bytes) -> bool:
    require_oqs()
    try:
        with _oqs.Signature(scheme) as v:
            return bool(v.verify(msg, sig, pub))
    except Exception:  # noqa: BLE001
        return False


# ─────────────────────── HKDF-SHA3-512 derivation ───────────────────────

def derive_session_key(ec_secret: bytes, pq_secret: Optional[bytes]) -> bytes:
    """32-byte AES-256-GCM key. Matches reference helper_functions.kdf."""
    if pq_secret:
        info = b"Hybrid key"
        material = ec_secret + pq_secret
    else:
        info = b"Classical TLS1.3 ECDH-only baseline"
        material = ec_secret
    hkdf = HKDF(
        algorithm=hashes.SHA3_512(),
        length=32,
        salt=None,
        info=info,
    )
    return hkdf.derive(material)


# ─────────────────────────── AES-256-GCM ────────────────────────────────
#
# We use a 12-byte random nonce per call (NIST SP 800-38D recommended).
# The AESGCM object's encrypt() concatenates ciphertext||tag, but we keep
# nonce/ciphertext/tag separately to match reference wire schema.

@dataclass
class EncryptedBlob:
    nonce: bytes       # 12 B
    ciphertext: bytes  # without tag
    tag: bytes         # 16 B

    def to_wire(self) -> bytes:
        return json.dumps({
            "n": base64.b64encode(self.nonce).decode("ascii"),
            "c": base64.b64encode(self.ciphertext).decode("ascii"),
            "t": base64.b64encode(self.tag).decode("ascii"),
        }).encode("ascii")

    @classmethod
    def from_wire(cls, data: bytes) -> "EncryptedBlob":
        d = json.loads(data.decode("ascii"))
        return cls(
            nonce=base64.b64decode(d["n"]),
            ciphertext=base64.b64decode(d["c"]),
            tag=base64.b64decode(d["t"]),
        )


def aes_gcm_encrypt(key: bytes, plaintext: bytes) -> EncryptedBlob:
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ct_with_tag = aes.encrypt(nonce, plaintext, None)
    ct, tag = ct_with_tag[:-16], ct_with_tag[-16:]
    return EncryptedBlob(nonce=nonce, ciphertext=ct, tag=tag)


def aes_gcm_decrypt(key: bytes, blob: EncryptedBlob) -> bytes:
    aes = AESGCM(key)
    return aes.decrypt(blob.nonce, blob.ciphertext + blob.tag, None)


# ─────────────────── hybrid sign / verify on a payload ───────────────────

@dataclass
class SignedBlob:
    message: bytes
    classic_sig: bytes
    pqc_sig: bytes   # may be empty in classical mode

    def to_wire(self) -> bytes:
        return json.dumps({
            "m":   base64.b64encode(self.message).decode("ascii"),
            "cs":  base64.b64encode(self.classic_sig).decode("ascii"),
            "ps":  base64.b64encode(self.pqc_sig).decode("ascii"),
        }).encode("ascii")

    @classmethod
    def from_wire(cls, data: bytes) -> "SignedBlob":
        d = json.loads(data.decode("ascii"))
        return cls(
            message=base64.b64decode(d["m"]),
            classic_sig=base64.b64decode(d["cs"]),
            pqc_sig=base64.b64decode(d["ps"]),
        )


def hybrid_sign(
    ec_priv: ec.EllipticCurvePrivateKey,
    sig_session: Optional[SigSession],
    message: bytes,
) -> Tuple[bytes, bytes, float, float]:
    """Sign with ECDSA and (optionally) the bound PQC signature scheme.

    Returns ``(ecdsa_sig, pqc_sig, ecdsa_ms, pqc_ms)`` so the caller can
    record each leg's cost. ``pqc_sig`` is empty in classical mode.
    """
    t0 = time.perf_counter()
    ec_sig = ecdsa_sign(ec_priv, message)
    t1 = time.perf_counter()
    if sig_session is None:
        return ec_sig, b"", (t1 - t0) * 1000.0, 0.0
    pq_sig = sig_session.sign(message)
    t2 = time.perf_counter()
    return ec_sig, pq_sig, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def hybrid_verify(
    ec_pub_bytes: bytes,
    pqc_scheme: Optional[str],
    pqc_pub: bytes,
    blob: SignedBlob,
) -> Tuple[bool, float, float]:
    """Verify the hybrid signature. Returns (ok, ecdsa_ms, pqc_ms)."""
    t0 = time.perf_counter()
    ec_ok = ecdsa_verify(ec_pub_bytes, blob.message, blob.classic_sig)
    t1 = time.perf_counter()
    if not pqc_scheme or not pqc_pub:
        return ec_ok, (t1 - t0) * 1000.0, 0.0
    pq_ok = pqc_verify(pqc_scheme, pqc_pub, blob.message, blob.pqc_sig)
    t2 = time.perf_counter()
    return (ec_ok and pq_ok), (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


# ─────────────────────── scheme-name canonicalisation ───────────────────

def canonical_kem_name(name: str, available: list[str]) -> str:
    """Map common spellings of a KEM scheme to the liboqs canonical name."""
    if name.upper() == CLASSICAL_SENTINEL:
        return CLASSICAL_SENTINEL
    norm = name.strip().replace("_", "-")
    lc = norm.lower()
    avail_by_lc = {a.lower(): a for a in available}
    if lc in avail_by_lc:
        return avail_by_lc[lc]
    # tolerate 'kyber768' -> 'ML-KEM-768'
    if lc.startswith("kyber"):
        digits = "".join(ch for ch in lc if ch.isdigit())
        candidate = f"ml-kem-{digits}"
        if candidate in avail_by_lc:
            return avail_by_lc[candidate]
    raise ValueError(f"Unknown KEM scheme: {name!r}. Available: {available}")


def canonical_sig_name(name: str, available: list[str]) -> str:
    """Map common spellings of a signature scheme to the liboqs canonical name."""
    if name.upper() == CLASSICAL_SENTINEL:
        return CLASSICAL_SENTINEL
    norm = name.strip().replace("_", "-")
    lc = norm.lower()
    avail_by_lc = {a.lower(): a for a in available}
    if lc in avail_by_lc:
        return avail_by_lc[lc]
    # tolerate 'dilithium2/3/5' -> 'ML-DSA-44/65/87'
    dil = {"dilithium2": "ml-dsa-44",
           "dilithium3": "ml-dsa-65",
           "dilithium5": "ml-dsa-87"}
    if lc in dil and dil[lc] in avail_by_lc:
        return avail_by_lc[dil[lc]]
    raise ValueError(f"Unknown signature scheme: {name!r}. Available subset: "
                     f"{[a for a in available if a.startswith(('ML-DSA','Falcon','SPHINCS','SLH-DSA'))][:12]}")
