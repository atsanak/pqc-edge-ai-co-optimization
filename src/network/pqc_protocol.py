"""
PQC wire protocol between the edge client and the inference server.

Sits on top of the existing TLS 1.3 channel — both endpoints already speak
TLS; the PQC layer adds hybrid key establishment (Phase 2) and per-call
hybrid signing + AES-GCM encryption (Phase 3) over the body of an
HTTP request. This matches the layering in the project protocol model

The two endpoints are:

    POST /pqc/handshake     (JSON in, JSON out)
        ``HandshakeRequest`` → ``HandshakeResponse``

    POST /pqc/infer         (binary in, binary out)
        Body = EncryptedBlob.to_wire() of a SignedBlob of an InferRequest JSON.
        Response body = same wrapping of an InferResponse JSON.
        Session id is carried in the ``X-PQC-Session-Id`` header.

This module defines only the *shapes* of those payloads. Crypto operations
live in `pqc_crypto.py`; the HTTP wiring is in `pqc_server.py` (router) and
`pqc_client.py` (PQCRemoteDetector).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, asdict
from typing import List, Tuple

from network.pqc_crypto import CLASSICAL_SENTINEL


# ─────────────────────── handshake (one round-trip) ──────────────────────
#
# Direction is client → server for the request, server → client for the
# response. After the handshake both sides have:
#   - a 32-byte AES-256-GCM session key
#   - the peer's signing public key (for verifying subsequent /pqc/infer)
#   - a session_id to use as a routing key on /pqc/infer requests

@dataclass
class HandshakeRequest:
    # The client picks the scheme pair. The server validates and accepts
    # (or fails) the pair before producing a response.
    kem_scheme: str            # e.g. 'ML-KEM-768' or 'CLASSICAL' for ECDH-only
    sig_scheme: str            # e.g. 'ML-DSA-44' or 'CLASSICAL' for ECDSA-only

    # Client's ephemeral classical-leg public key (SECP256R1, PEM).
    client_ecdh_pub_b64: str

    # Client's ephemeral KEM public key (empty in classical mode).
    client_kem_pub_b64: str

    # Client's signing public key (PEM for classical, raw bytes for PQC).
    client_classic_sig_pub_b64: str
    client_pqc_sig_pub_b64: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HandshakeRequest":
        return cls(**d)


@dataclass
class HandshakeResponse:
    # Echo of negotiated schemes (server may reject unsupported pairs).
    kem_scheme: str
    sig_scheme: str

    # Server's ephemeral classical-leg public key (SECP256R1, PEM).
    server_ecdh_pub_b64: str

    # Server-side KEM ciphertext encapsulated to the client's KEM pub.
    # Empty in classical mode.
    kem_ciphertext_b64: str

    # Server's signing public keys.
    server_classic_sig_pub_b64: str
    server_pqc_sig_pub_b64: str

    # Routing key for subsequent /pqc/infer calls.
    session_id: str

    # Server-side handshake CPU time, useful for cost attribution.
    server_handshake_ms: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HandshakeResponse":
        return cls(**d)


# ───────────────── /pqc/infer plaintext shapes (request) ────────────────
#
# The full HTTP request body is:
#   EncryptedBlob.to_wire()   (JSON: ciphertext/nonce/tag)
# whose plaintext is:
#   SignedBlob.to_wire()      (JSON: message/classic_sig/pqc_sig)
# whose ``message`` is:
#   PqcInferRequest.to_json().encode('utf-8')
#
# The JPEG sits inside the request as base64 — same expansion reference
# measures for their wire format (~2.74× per chunk).

@dataclass
class PqcInferRequest:
    imgsz: int
    conf: float
    crop_bounds: Tuple[int, int, int, int]
    frame_width: int
    frame_height: int
    frame_id: int
    sequence: str
    jpeg_b64: str

    def to_json(self) -> str:
        import json
        return json.dumps({
            "imgsz": self.imgsz,
            "conf": self.conf,
            "crop_bounds": list(self.crop_bounds),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frame_id": self.frame_id,
            "sequence": self.sequence,
            "jpeg_b64": self.jpeg_b64,
        })

    @classmethod
    def from_json(cls, raw: str) -> "PqcInferRequest":
        import json
        d = json.loads(raw)
        return cls(
            imgsz=int(d["imgsz"]),
            conf=float(d["conf"]),
            crop_bounds=tuple(d["crop_bounds"]),  # type: ignore[arg-type]
            frame_width=int(d["frame_width"]),
            frame_height=int(d["frame_height"]),
            frame_id=int(d.get("frame_id", 0)),
            sequence=str(d.get("sequence", "")),
            jpeg_b64=str(d["jpeg_b64"]),
        )


@dataclass
class PqcDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    cls: int
    score: float


@dataclass
class PqcInferResponse:
    detections: List[PqcDetection]
    server_inference_ms: float
    server_imgsz_used: int
    frame_id: int
    sequence: str

    def to_json(self) -> str:
        import json
        return json.dumps({
            "detections": [
                {"bbox_xyxy": list(d.bbox_xyxy), "cls": d.cls, "score": d.score}
                for d in self.detections
            ],
            "server_inference_ms": self.server_inference_ms,
            "server_imgsz_used": self.server_imgsz_used,
            "frame_id": self.frame_id,
            "sequence": self.sequence,
        })

    @classmethod
    def from_json(cls, raw: str) -> "PqcInferResponse":
        import json
        d = json.loads(raw)
        return cls(
            detections=[
                PqcDetection(
                    bbox_xyxy=tuple(x["bbox_xyxy"]),  # type: ignore[arg-type]
                    cls=int(x["cls"]),
                    score=float(x["score"]),
                )
                for x in d.get("detections", [])
            ],
            server_inference_ms=float(d.get("server_inference_ms", 0.0)),
            server_imgsz_used=int(d.get("server_imgsz_used", 0)),
            frame_id=int(d.get("frame_id", 0)),
            sequence=str(d.get("sequence", "")),
        )


# ───────────────────────── HTTP path conventions ────────────────────────

PQC_HANDSHAKE_PATH = "/pqc/handshake"
PQC_INFER_PATH = "/pqc/infer"

# Session id is shipped in an HTTP header so the encrypted body stays
# opaque to the transport layer.
SESSION_HEADER = "X-PQC-Session-Id"
