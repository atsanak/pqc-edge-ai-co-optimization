"""
Server-side PQC routes for the inference server.

This module exposes a FastAPI `APIRouter` that the existing
`network/server.py` mounts. It adds:

    POST /pqc/handshake     hybrid ECDH+(ML-)KEM key establishment
    POST /pqc/infer         signed + AES-256-GCM-encrypted inference call

Both endpoints reuse the existing YOLO detector cache exported by
`network.server` — there is exactly one YOLO instance per (weight, imgsz)
on the box regardless of whether requests arrive on /infer or /pqc/infer.
"""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from network import pqc_crypto as pc
from network import pqc_metrics as pm
from network.pqc_protocol import (
    HandshakeRequest, HandshakeResponse,
    PqcDetection, PqcInferRequest, PqcInferResponse,
    PQC_HANDSHAKE_PATH, PQC_INFER_PATH, SESSION_HEADER,
)


# ─────────────────────────── session state ──────────────────────────────

@dataclass
class _PqcSession:
    aes_key: bytes
    kem_scheme: str   # 'CLASSICAL' or e.g. 'ML-KEM-768'
    sig_scheme: str   # 'CLASSICAL' or e.g. 'ML-DSA-44'
    client_classic_sig_pub: bytes
    client_pqc_sig_pub: bytes
    server_sig_classic_priv: object  # cryptography EC private key
    server_sig_classic_pub: bytes
    server_sig_pqc_session: Optional[pc.SigSession] = None
    last_used: float = field(default_factory=time.time)
    n_calls: int = 0
    # --- per-session metric collection (mirror of client-side) ---
    t_session_start: float = field(default_factory=time.perf_counter)
    t_session_end:   float = 0.0
    rusage_before:   dict = field(default_factory=dict)
    rusage_after:    dict = field(default_factory=dict)
    # Per-call duration arrays (seconds)
    times_decrypt_s:   List[float] = field(default_factory=list)
    times_verify_ec_s: List[float] = field(default_factory=list)
    times_verify_pq_s: List[float] = field(default_factory=list)
    times_yolo_s:      List[float] = field(default_factory=list)
    times_sign_ec_s:   List[float] = field(default_factory=list)
    times_sign_pq_s:   List[float] = field(default_factory=list)
    times_encrypt_s:   List[float] = field(default_factory=list)
    # Per-call sizes
    bytes_in:    List[int] = field(default_factory=list)
    bytes_out:   List[int] = field(default_factory=list)
    bytes_plain_in:  List[int] = field(default_factory=list)
    bytes_plain_out: List[int] = field(default_factory=list)
    # Verification rolling state
    payload_sha256_running: object = None   # hashlib.sha256() instance
    signature_failure_count: int = 0
    decrypt_failure_count:   int = 0
    server_handshake_ms: float = 0.0
    # Handshake phase breakdown (seconds) so the client master row can
    # populate server_key_exchange_time / server_hybrid_key_generation_time.
    server_tls_handshake_time_s:         float = 0.0
    server_key_exchange_time_s:          float = 0.0
    server_hybrid_key_generation_time_s: float = 0.0


# Module-level session registry. Single-process server, so a plain dict +
# lock is fine. If the server ever runs with --workers >1 this needs Redis
# or similar; today it doesn't.
_SESSIONS: dict[str, _PqcSession] = {}
_SESSIONS_LOCK = threading.Lock()
_SESSION_TTL_S = 30 * 60   # 30 minutes


def _prune_old_sessions() -> None:
    now = time.time()
    with _SESSIONS_LOCK:
        stale = [sid for sid, s in _SESSIONS.items()
                 if now - s.last_used > _SESSION_TTL_S]
        for sid in stale:
            sess = _SESSIONS.pop(sid, None)
            if sess and sess.server_sig_pqc_session is not None:
                sess.server_sig_pqc_session.free()


# ─────────────────────── detector-cache delegation ──────────────────────
#
# The existing /infer endpoint owns the YOLO HeavyYoloClassifier cache in
# `network.server`. We import that function lazily so this module can be
# imported by other tools (tests, CLIs) without forcing a CUDA import.

def _get_detector(imgsz: int):
    from network.server import _get_detector as _impl  # local import
    return _impl(imgsz)


# ────────────────────────────── router ──────────────────────────────────

router = APIRouter()


@router.post(PQC_HANDSHAKE_PATH)
async def pqc_handshake(req: Request):
    t0 = time.perf_counter()
    body = await req.json()
    try:
        hreq = HandshakeRequest.from_dict(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad handshake JSON: {exc}") from exc

    kem_user = hreq.kem_scheme
    sig_user = hreq.sig_scheme

    # Resolve the scheme strings against liboqs's actual enabled-mechanism
    # lists. Classical-only mode bypasses oqs entirely.
    classical_mode = (kem_user.upper() == pc.CLASSICAL_SENTINEL
                      and sig_user.upper() == pc.CLASSICAL_SENTINEL)

    if classical_mode:
        kem_canon = pc.CLASSICAL_SENTINEL
        sig_canon = pc.CLASSICAL_SENTINEL
    else:
        try:
            pc.require_oqs()
            kem_canon = pc.canonical_kem_name(kem_user, pc.list_available_kems())
            sig_canon = pc.canonical_sig_name(sig_user, pc.list_available_signatures())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Unsupported scheme pair: {exc}") from exc

    # ---- Phase: server key-exchange (mirror of client_key_exchange_time) ---
    # Wraps the public-key swap + KEM ENCAPSULATE — the server-side analogue
    # of the client's decapsulate.  We bracket from when we start producing
    # the response material to when KEM operations are done.
    t_kex0 = time.perf_counter()
    s_ec_priv, s_ec_pub = pc.ecdh_generate()
    s_classic_sig_priv, s_classic_sig_pub = pc.ecdsa_generate()
    s_pqc_sig = None if classical_mode else pc.SigSession(sig_canon)

    # Encapsulate to the client's KEM pub (PQC mode only).
    if classical_mode:
        kem_ct = b""
        pq_secret = None
    else:
        client_kem_pub = base64.b64decode(hreq.client_kem_pub_b64)
        if not client_kem_pub:
            raise HTTPException(status_code=400,
                                detail="client_kem_pub_b64 required in PQC mode")
        kem_ct, pq_secret = pc.kem_encapsulate(kem_canon, client_kem_pub)
    server_key_exchange_time_s = time.perf_counter() - t_kex0

    # ---- Phase: server hybrid_key_generation_time --------------------------
    # ECDH derive + HKDF (matches reference helper_functions.kdf call).
    t_hkg0 = time.perf_counter()
    client_ec_pub_bytes = base64.b64decode(hreq.client_ecdh_pub_b64)
    ec_secret = pc.ecdh_shared(s_ec_priv, client_ec_pub_bytes)
    aes_key = pc.derive_session_key(ec_secret, pq_secret)
    server_hybrid_key_generation_time_s = time.perf_counter() - t_hkg0

    session_id = uuid.uuid4().hex
    handshake_ms = (time.perf_counter() - t0) * 1000.0
    # Server doesn't measure TLS-1.3 handshake itself (the ASGI app sees an
    # already-decrypted body), so server_tls_handshake_time mirrors the total
    # handshake wall-time minus the two PQC phases — i.e. JSON parsing + the
    # /pqc/handshake request overhead.
    server_tls_handshake_time_s = max(
        0.0,
        (handshake_ms / 1000.0)
        - server_key_exchange_time_s
        - server_hybrid_key_generation_time_s,
    )
    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = _PqcSession(
            aes_key=aes_key,
            kem_scheme=kem_canon,
            sig_scheme=sig_canon,
            client_classic_sig_pub=base64.b64decode(hreq.client_classic_sig_pub_b64),
            client_pqc_sig_pub=base64.b64decode(hreq.client_pqc_sig_pub_b64) if not classical_mode else b"",
            server_sig_classic_priv=s_classic_sig_priv,
            server_sig_classic_pub=s_classic_sig_pub,
            server_sig_pqc_session=s_pqc_sig,
            rusage_before=pm.get_resource_snapshot(),
            payload_sha256_running=hashlib.sha256(),
            server_handshake_ms=handshake_ms,
            server_tls_handshake_time_s=server_tls_handshake_time_s,
            server_key_exchange_time_s=server_key_exchange_time_s,
            server_hybrid_key_generation_time_s=server_hybrid_key_generation_time_s,
        )
    _prune_old_sessions()

    resp = HandshakeResponse(
        kem_scheme=kem_canon,
        sig_scheme=sig_canon,
        server_ecdh_pub_b64=base64.b64encode(s_ec_pub).decode("ascii"),
        kem_ciphertext_b64=base64.b64encode(kem_ct).decode("ascii"),
        server_classic_sig_pub_b64=base64.b64encode(s_classic_sig_pub).decode("ascii"),
        server_pqc_sig_pub_b64=(""
                                if s_pqc_sig is None
                                else base64.b64encode(s_pqc_sig.public_key).decode("ascii")),
        session_id=session_id,
        server_handshake_ms=handshake_ms,
    )
    return JSONResponse(resp.to_dict())


@router.post(PQC_INFER_PATH)
async def pqc_infer(req: Request,
                    x_pqc_session_id: str = Header(default="", alias=SESSION_HEADER)):
    if not x_pqc_session_id:
        raise HTTPException(status_code=400, detail=f"missing {SESSION_HEADER}")

    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(x_pqc_session_id)
        if sess is None:
            raise HTTPException(status_code=401, detail="unknown session")
        sess.last_used = time.time()
        sess.n_calls += 1

    classical_mode = (sess.sig_scheme == pc.CLASSICAL_SENTINEL)

    # Body = EncryptedBlob.to_wire() containing a SignedBlob containing an InferRequest JSON.
    raw_body = await req.body()
    bytes_in_wire = len(raw_body)

    t_dec0 = time.perf_counter()
    try:
        enc_blob = pc.EncryptedBlob.from_wire(raw_body)
        signed_wire = pc.aes_gcm_decrypt(sess.aes_key, enc_blob)
        signed_blob = pc.SignedBlob.from_wire(signed_wire)
    except Exception as exc:  # noqa: BLE001
        with _SESSIONS_LOCK:
            sess.decrypt_failure_count += 1
        raise HTTPException(status_code=400, detail=f"decrypt failed: {exc}") from exc
    t_dec1 = time.perf_counter()

    pqc_sig_scheme_for_verify = None if classical_mode else sess.sig_scheme
    pqc_pub_for_verify = b"" if classical_mode else sess.client_pqc_sig_pub
    ok, ec_v_ms, pq_v_ms = pc.hybrid_verify(
        sess.client_classic_sig_pub,
        pqc_sig_scheme_for_verify,
        pqc_pub_for_verify,
        signed_blob,
    )
    if not ok:
        with _SESSIONS_LOCK:
            sess.signature_failure_count += 1
        raise HTTPException(status_code=401, detail="signature verification failed")
    t_ver1 = time.perf_counter()

    # Roll the plaintext into the server's session-wide SHA-256 so we can
    # cross-check payload_sha256_match against the client's running hash.
    with _SESSIONS_LOCK:
        if sess.payload_sha256_running is not None:
            sess.payload_sha256_running.update(signed_blob.message)

    try:
        inf_req = PqcInferRequest.from_json(signed_blob.message.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad request JSON: {exc}") from exc

    # ---- Run YOLO via the existing detector cache ------------------------
    try:
        jpeg = base64.b64decode(inf_req.jpeg_b64)
        pil = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc

    det = _get_detector(int(inf_req.imgsz))
    det.conf = float(inf_req.conf)

    t_yolo0 = time.perf_counter()
    _, boxes_xywh, classes, scores = det.predict_image(pil)
    t_yolo1 = time.perf_counter()

    detections: list[PqcDetection] = []
    if boxes_xywh is not None:
        xywh = np.asarray(boxes_xywh, dtype=float)
        xyxy = xywh.copy()
        xyxy[:, 2] = xyxy[:, 0] + xywh[:, 2]
        xyxy[:, 3] = xyxy[:, 1] + xywh[:, 3]
        for i in range(xyxy.shape[0]):
            detections.append(PqcDetection(
                bbox_xyxy=(float(xyxy[i, 0]), float(xyxy[i, 1]),
                           float(xyxy[i, 2]), float(xyxy[i, 3])),
                cls=int(classes[i]),
                score=float(scores[i]),
            ))

    inf_resp = PqcInferResponse(
        detections=detections,
        server_inference_ms=(t_yolo1 - t_yolo0) * 1000.0,
        server_imgsz_used=int(inf_req.imgsz),
        frame_id=int(inf_req.frame_id),
        sequence=str(inf_req.sequence),
    )

    # ---- Sign + encrypt the response -------------------------------------
    resp_plain = inf_resp.to_json().encode("utf-8")
    t_sgn0 = time.perf_counter()
    ec_sig, pq_sig, ec_s_ms, pq_s_ms = pc.hybrid_sign(
        sess.server_sig_classic_priv,
        sess.server_sig_pqc_session,
        resp_plain,
    )
    t_sgn1 = time.perf_counter()
    signed_out = pc.SignedBlob(resp_plain, ec_sig, pq_sig)
    enc_out = pc.aes_gcm_encrypt(sess.aes_key, signed_out.to_wire())
    out_wire = enc_out.to_wire()
    t_enc1 = time.perf_counter()

    # Record per-call durations and bytes into the session's metric arrays
    # so /pqc/stats/<session_id> can return them later.
    with _SESSIONS_LOCK:
        sess.times_decrypt_s.append(t_dec1 - t_dec0)
        sess.times_verify_ec_s.append(ec_v_ms / 1000.0)
        sess.times_verify_pq_s.append(pq_v_ms / 1000.0)
        sess.times_yolo_s.append(t_yolo1 - t_yolo0)
        sess.times_sign_ec_s.append(ec_s_ms / 1000.0)
        sess.times_sign_pq_s.append(pq_s_ms / 1000.0)
        sess.times_encrypt_s.append(t_enc1 - t_sgn1)
        sess.bytes_in.append(bytes_in_wire)
        sess.bytes_out.append(len(out_wire))
        sess.bytes_plain_in.append(len(signed_blob.message))
        sess.bytes_plain_out.append(len(resp_plain))

    # Surface crypto cost as response headers so the client doesn't have to
    # measure them indirectly. The benchmark CSV columns consume these.
    headers = {
        "X-PQC-Server-Decrypt-Ms":    f"{(t_dec1 - t_dec0) * 1000.0:.3f}",
        "X-PQC-Server-Verify-Ec-Ms":  f"{ec_v_ms:.3f}",
        "X-PQC-Server-Verify-Pq-Ms":  f"{pq_v_ms:.3f}",
        "X-PQC-Server-Yolo-Ms":       f"{(t_yolo1 - t_yolo0) * 1000.0:.3f}",
        "X-PQC-Server-Sign-Ec-Ms":    f"{ec_s_ms:.3f}",
        "X-PQC-Server-Sign-Pq-Ms":    f"{pq_s_ms:.3f}",
        "X-PQC-Server-Encrypt-Ms":    f"{(t_enc1 - t_sgn1) * 1000.0:.3f}",
        "X-PQC-Wire-Bytes-In":        str(bytes_in_wire),
        "X-PQC-Wire-Bytes-Out":       str(len(out_wire)),
        "X-PQC-Plaintext-Bytes":      str(len(resp_plain)),
    }
    return Response(content=out_wire,
                    media_type="application/octet-stream",
                    headers=headers)


@router.get("/pqc/stats/{session_id}")
def pqc_session_stats(session_id: str):
    """Snapshot of the server-side per-session metric state.

    Used by the client to populate the ``server_*`` columns in the
    master row. The first call also finalises the rusage window. Subsequent
    calls return the same finalised snapshot until the session is pruned.
    """
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
        if not sess.rusage_after:
            sess.rusage_after = pm.get_resource_snapshot()
            sess.t_session_end = time.perf_counter()
        # Snapshot under the lock so concurrent /pqc/infer calls don't
        # mutate the lists while we iterate.
        snap = {
            "session_id":              session_id,
            "kem_scheme":              sess.kem_scheme,
            "sig_scheme":              sess.sig_scheme,
            "n_calls":                 sess.n_calls,
            "server_handshake_ms":     sess.server_handshake_ms,
            "server_tls_handshake_time_s":         sess.server_tls_handshake_time_s,
            "server_key_exchange_time_s":          sess.server_key_exchange_time_s,
            "server_hybrid_key_generation_time_s": sess.server_hybrid_key_generation_time_s,
            "wall_time_s":             max(0.0, sess.t_session_end - sess.t_session_start),
            "rusage_before":           dict(sess.rusage_before),
            "rusage_after":            dict(sess.rusage_after),
            "times_decrypt_s":         list(sess.times_decrypt_s),
            "times_verify_ec_s":       list(sess.times_verify_ec_s),
            "times_verify_pq_s":       list(sess.times_verify_pq_s),
            "times_yolo_s":            list(sess.times_yolo_s),
            "times_sign_ec_s":         list(sess.times_sign_ec_s),
            "times_sign_pq_s":         list(sess.times_sign_pq_s),
            "times_encrypt_s":         list(sess.times_encrypt_s),
            "bytes_in":                list(sess.bytes_in),
            "bytes_out":               list(sess.bytes_out),
            "bytes_plain_in":          list(sess.bytes_plain_in),
            "bytes_plain_out":         list(sess.bytes_plain_out),
            "signature_failure_count": sess.signature_failure_count,
            "decrypt_failure_count":   sess.decrypt_failure_count,
            "server_payload_sha256":   (sess.payload_sha256_running.hexdigest()
                                        if sess.payload_sha256_running is not None
                                        else ""),
        }
    return JSONResponse(snap)


@router.get("/pqc/sessions")
def list_sessions():
    """Debug helper: list active sessions (no secrets returned)."""
    _prune_old_sessions()
    with _SESSIONS_LOCK:
        out = [
            {"session_id": sid,
             "kem": s.kem_scheme, "sig": s.sig_scheme,
             "n_calls": s.n_calls,
             "age_s": time.time() - s.last_used}
            for sid, s in _SESSIONS.items()
        ]
    return {"sessions": out}
