"""
Edge-side PQC client. Drop-in replacement for `RemoteDetector` with the
same `predict_image(PIL)` contract.

Lifecycle:

    det = PQCRemoteDetector(
        url="https://GROUND_STATION_IP:8443",  # base URL, NOT /infer
        imgsz=640, conf=0.25,
        cafile="src/network/certs/server.crt",
        crypto_mode="pqc",                       # or "classical"
        kem_scheme="ML-KEM-768",                 # ignored if classical
        sig_scheme="ML-DSA-44",                  # ignored if classical
    )
    det.handshake()                              # one round-trip
    img, xywh, classes, scores = det.predict_image(pil)

After the handshake the detector caches the AES-256-GCM session key and
the server's signing public key; every subsequent `predict_image()` call
signs+encrypts the JPEG bytes on the way out and decrypts+verifies the
detection JSON on the way back.

All per-call crypto timings are accumulated in instance counters and
surfaced by ``stats_summary()`` for the benchmark CSV.
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise ImportError("PQCRemoteDetector requires httpx") from exc

from network import pqc_crypto as pc
from network import pqc_metrics as pm
from network.pqc_protocol import (
    HandshakeRequest, HandshakeResponse,
    PqcInferRequest, PqcInferResponse,
    PQC_HANDSHAKE_PATH, PQC_INFER_PATH, SESSION_HEADER,
)


_DEFAULT_JPEG_QUALITY = 90


class PQCRemoteDetector:
    """`HeavyYoloClassifier`-shaped client that talks /pqc/infer."""

    def __init__(
        self,
        url: str,
        crypto_mode: str = "pqc",        # "pqc" | "classical"
        kem_scheme: str = "ML-KEM-768",
        sig_scheme: str = "ML-DSA-44",
        imgsz: int = 640,
        conf: float = 0.25,
        cafile: Optional[str] = None,
        verify_tls: bool = True,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        timeout_s: float = 30.0,
        sequence_name: str = "",
    ):
        # Accept either ".../infer" (compat with the TLS-1.3 form the benchmark
        # already collects) or a bare base URL. Strip the path; we build our
        # own.
        base = url
        for suffix in ("/infer", "/pqc/infer", "/pqc/handshake"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        self.base_url = base.rstrip("/")

        self.crypto_mode = crypto_mode.lower()
        if self.crypto_mode not in ("pqc", "classical"):
            raise ValueError(f"crypto_mode must be 'pqc' or 'classical', got {crypto_mode!r}")
        self.kem_scheme = pc.CLASSICAL_SENTINEL if self.crypto_mode == "classical" else kem_scheme
        self.sig_scheme = pc.CLASSICAL_SENTINEL if self.crypto_mode == "classical" else sig_scheme

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.jpeg_quality = int(jpeg_quality)
        self.sequence_name = sequence_name

        if not verify_tls:
            verify_arg: bool | str = False
        elif cafile:
            verify_arg = cafile
        else:
            verify_arg = True
        # HTTP/2 keep-alive across calls — important for per-frame latencies.
        self._client = httpx.Client(http2=True, verify=verify_arg, timeout=timeout_s)

        # ---- session state (filled by handshake) -------------------------
        self._aes_key: Optional[bytes] = None
        self._session_id: Optional[str] = None
        # local key material
        self._ec_priv = None
        self._ec_pub_bytes: Optional[bytes] = None
        self._kem_session: Optional[pc.KemSession] = None
        self._classic_sig_priv = None
        self._classic_sig_pub: Optional[bytes] = None
        self._pqc_sig_session: Optional[pc.SigSession] = None
        # server-side material we'll verify against
        self._server_classic_sig_pub: Optional[bytes] = None
        self._server_pqc_sig_pub: Optional[bytes] = None

        # ---- counters (read by benchmark) --------------------------------
        self.total_yolo_time = 0.0  # mirrors HeavyYoloClassifier semantics

        self.handshake_ms_client: float = 0.0
        self.handshake_ms_server: float = 0.0
        self.handshake_rtt_ms:    float = 0.0

        # Compatibility alias so RemoteDetector-shape consumers (the existing
        # benchmark CSV writer) can read total_server_infer_ms without
        # special-casing PQC.
        self.total_server_infer_ms = 0.0  # see predict_image() below

        self.total_client_encode_ms = 0.0
        self.total_client_sign_ec_ms = 0.0
        self.total_client_sign_pq_ms = 0.0
        self.total_client_encrypt_ms = 0.0
        self.total_network_rtt_ms = 0.0
        self.total_server_decrypt_ms = 0.0
        self.total_server_verify_ec_ms = 0.0
        self.total_server_verify_pq_ms = 0.0
        self.total_server_yolo_ms = 0.0
        self.total_server_sign_ec_ms = 0.0
        self.total_server_sign_pq_ms = 0.0
        self.total_server_encrypt_ms = 0.0
        self.total_client_decrypt_ms = 0.0
        self.total_client_verify_ec_ms = 0.0
        self.total_client_verify_pq_ms = 0.0

        self.total_plaintext_upload = 0   # JSON request bytes pre-crypto
        self.total_plaintext_download = 0 # JSON response bytes pre-crypto
        self.total_bytes_uploaded = 0     # on-wire (encrypted)
        self.total_bytes_downloaded = 0   # on-wire (encrypted)
        self.num_requests = 0

        # Per-call latency arrays — needed for distribution-shape statistics
        # (median / p95 / p99 / skewness / kurtosis / determinism class).
        # These mirror reference `client_advanced_chunk_stats.csv` schema.
        # Unit: SECONDS (matches reference *_s column naming).
        self.times_sign_s:        list[float] = []   # ECDSA + ML-DSA combined per call
        self.times_verify_s:      list[float] = []
        self.times_encrypt_s:     list[float] = []
        self.times_decrypt_s:     list[float] = []
        self.times_network_send_s: list[float] = []
        self.times_network_recv_s: list[float] = []
        # Per-leg (kept for the existing avg_sign_ec_ms / avg_sign_pq_ms columns)
        self.times_sign_ec_s:     list[float] = []
        self.times_sign_pq_s:     list[float] = []
        self.times_verify_ec_s:   list[float] = []
        self.times_verify_pq_s:   list[float] = []
        # Server-side per-call durations (received from response headers)
        self.times_server_yolo_s:       list[float] = []
        self.times_server_sign_ec_s:    list[float] = []
        self.times_server_sign_pq_s:    list[float] = []
        self.times_server_encrypt_s:    list[float] = []
        self.times_server_decrypt_s:    list[float] = []
        self.times_server_verify_ec_s:  list[float] = []
        self.times_server_verify_pq_s:  list[float] = []
        # Bytes per call
        self.bytes_on_wire_up:    list[int]   = []
        self.bytes_on_wire_down:  list[int]   = []
        self.bytes_plaintext_up:  list[int]   = []
        self.bytes_plaintext_down: list[int]  = []
        # Per-call signed/encrypted length on the up direction (drives
        # sign_expansion_ratio + encrypt_expansion_ratio)
        self.bytes_signed_up:     list[int]   = []   # SignedBlob.to_wire() len
        self.bytes_encrypted_up:  list[int]   = []   # EncryptedBlob.to_wire() len == bytes_on_wire_up

        # Running SHA-256 of plaintext request bodies (drives client_payload_sha256;
        # compared against server's running hash at session close to populate
        # payload_sha256_match).
        self._req_hash = hashlib.sha256()
        self.client_payload_sha256: str = ""

        # Failure counters (drive signature_failure_count / decrypt_failure_count
        # columns). Increment on caught exceptions inside predict_image().
        self.signature_failure_count: int = 0
        self.decrypt_failure_count:   int = 0

        # rusage capture: snapshotted at handshake() and again at close()
        # so we get accurate session-scope deltas. compute_resource_delta()
        # exposes user_cpu_s / sys_cpu_s / peak_rss_mb / vol_ctxs /
        # invol_ctxs / minor_faults / major_faults.
        self._rusage_before: Optional[dict] = None
        self._rusage_after:  Optional[dict] = None
        # session wall-clock window (handshake start → close())
        self._t_session_start: Optional[float] = None
        self._t_session_end:   Optional[float] = None

        # Per-phase byte sizes (taken once on first request) — drive
        # size_chunk_size / size_original_msg_size / size_signed_msg_length
        # / size_encrypted_msg_length columns.
        self.chunk_size_bytes: int = 0
        self.original_msg_size_bytes: int = 0
        self.signed_msg_length_bytes: int = 0
        self.encrypted_msg_length_bytes: int = 0
        # Filled at handshake time
        self.hybrid_aes_key_length_bytes: int = 32
        self.classic_signature_length_bytes: int = 0
        self.pqc_signature_length_bytes: int = 0
        self.classic_kem_sharedsecret_length_bytes: int = 32   # ECDH secret bytes
        self.pqc_kem_sharedsecret_length_bytes:    int = 0

        # Handshake phase breakdown (drives client_tls_handshake_time /
        # client_key_exchange_time / client_hybrid_key_generation_time).
        # We do not run an actual TLS-1.3 handshake from here (it's already
        # established by httpx connect), so tls_handshake_time captures the
        # HTTP-level connect + TLS time we can observe; key_exchange_time is
        # the body of the /pqc/handshake POST; hybrid_key_generation_time is
        # the local AES-key derivation.
        self.tls_handshake_time_s:        float = 0.0
        self.key_exchange_time_s:         float = 0.0
        self.hybrid_key_generation_time_s: float = 0.0

        # Network-gap timings.  TCP connect is measured at handshake() time
        # via the pre-warm healthz GET (which forces the TCP+TLS round-trip
        # before any /pqc/* call).  TTFB / TTLB are aggregated per /pqc/infer
        # call: TTFB ≈ time the request was wholly written + response headers
        # received; TTLB ≈ response fully received.  httpx exposes
        # ``response.elapsed`` (total wall time from request start) and the
        # transport timing we measure around the call gives us TTFB.
        self.tcp_connect_time_s: float = 0.0
        self.ttfb_total_s:       float = 0.0
        self.ttlb_total_s:       float = 0.0

        # crop context for the next predict_image() call (same idiom as RemoteDetector)
        self._next_crop_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._next_frame_size: Tuple[int, int] = (0, 0)
        self._next_frame_id: int = 0

    # ──────────────────────── handshake ─────────────────────────────────

    def handshake(self) -> None:
        """One-shot Phase-2 key establishment. Idempotent.

        Phase boundaries mirror reference ``Examples/client.py`` exactly so
        the three setup columns are 1:1 comparable to their master_summary:

            tls_handshake_time         — pure TLS-1.3 handshake (reference:
                                         time around ``context.wrap_socket``;
                                         here: time of a pre-warm GET on the
                                         same httpx connection, which forces
                                         TCP + TLS-1.3 before the handshake
                                         POST so the POST's RTT is *only*
                                         the network round-trip).
            key_exchange_time          — public-key/ciphertext round-trip
                                         + local KEM operation (reference:
                                         recv server pubs + ``encrypt(...)`` +
                                         send client pubs; here: POST RTT +
                                         client-side KEM decapsulate).
            hybrid_key_generation_time — ECDH derive + HKDF only (matches
                                         reference ``shared_ecdhsecret`` +
                                         ``key_derivation_function`` block).
            session_setup_ms           — sum of the three above × 1000.
        """
        if self._aes_key is not None:
            return

        # rusage + wall-clock window starts now and runs until close()
        self._rusage_before = pm.get_resource_snapshot()
        self._t_session_start = time.perf_counter()

        # --- Local keypair generation (not counted in any phase; same
        #     pre-handshake position as reference client.py) -------------
        self._ec_priv, self._ec_pub_bytes = pc.ecdh_generate()
        self._classic_sig_priv, self._classic_sig_pub = pc.ecdsa_generate()
        if self.crypto_mode == "pqc":
            self._kem_session = pc.KemSession(self.kem_scheme)
            self._pqc_sig_session = pc.SigSession(self.sig_scheme)
            client_kem_pub = self._kem_session.public_key
            client_pqc_sig_pub = self._pqc_sig_session.public_key
        else:
            client_kem_pub = b""
            client_pqc_sig_pub = b""

        # --- Phase: tls_handshake_time -------------------------------------
        # httpx wraps TCP+TLS-1.3 opaquely on the first request. A pre-warm
        # GET pays that cost once, isolating the TLS handshake from the
        # subsequent key-exchange POST (which then carries *only* network
        # latency + the in-handshake server compute).
        # The full GET wall time covers TCP connect + TLS-1.3 handshake +
        # the trivial /healthz roundtrip — we attribute it to
        # tls_handshake_time, and use it as the tcp_connect_time_s proxy.
        t_tls0 = time.perf_counter()
        try:
            self._client.get(f"{self.base_url}/healthz")
        except Exception:  # noqa: BLE001
            pass
        self.tls_handshake_time_s = time.perf_counter() - t_tls0
        # First TCP+TLS connect cost — exposed as tcp_connect_time_s.
        self.tcp_connect_time_s = self.tls_handshake_time_s

        hreq = HandshakeRequest(
            kem_scheme=self.kem_scheme,
            sig_scheme=self.sig_scheme,
            client_ecdh_pub_b64=base64.b64encode(self._ec_pub_bytes).decode("ascii"),
            client_kem_pub_b64=base64.b64encode(client_kem_pub).decode("ascii"),
            client_classic_sig_pub_b64=base64.b64encode(self._classic_sig_pub).decode("ascii"),
            client_pqc_sig_pub_b64=base64.b64encode(client_pqc_sig_pub).decode("ascii"),
        )

        # --- Phase: key_exchange_time --------------------------------------
        # Wraps the network swap of public keys + ciphertexts AND the
        # client-side KEM operation (here: decapsulate, since in our
        # 1-RTT shape the *server* encapsulates to the client's KEM pub).
        t_kex0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}{PQC_HANDSHAKE_PATH}",
                                 json=hreq.to_dict())
        if resp.status_code != 200:
            raise RuntimeError(f"PQC handshake failed: {resp.status_code} {resp.text}")
        hresp = HandshakeResponse.from_dict(resp.json())
        server_ec_pub = base64.b64decode(hresp.server_ecdh_pub_b64)
        if self.crypto_mode == "pqc":
            kem_ct = base64.b64decode(hresp.kem_ciphertext_b64)
            assert self._kem_session is not None
            pq_secret = self._kem_session.decapsulate(kem_ct)
            self.pqc_kem_sharedsecret_length_bytes = len(pq_secret)
        else:
            pq_secret = None
        self.key_exchange_time_s = time.perf_counter() - t_kex0

        # --- Phase: hybrid_key_generation_time -----------------------------
        # ECDH derive + HKDF-SHA3-512 only (matches reference
        # ``shared_ecdhsecret_client = priv.exchange(...)`` +
        # ``key_derivation_function(...)`` block).
        t_hkg0 = time.perf_counter()
        ec_secret = pc.ecdh_shared(self._ec_priv, server_ec_pub)
        self._aes_key = pc.derive_session_key(ec_secret, pq_secret)
        self.hybrid_key_generation_time_s = time.perf_counter() - t_hkg0

        # Remaining state — server keys + session id
        self._session_id = hresp.session_id
        self._server_classic_sig_pub = base64.b64decode(hresp.server_classic_sig_pub_b64)
        self._server_pqc_sig_pub = (b""
                                    if not hresp.server_pqc_sig_pub_b64
                                    else base64.b64decode(hresp.server_pqc_sig_pub_b64))

        # session_setup_ms = sum of the three phases × 1000 (reference recipe)
        self.handshake_ms_client = (
            self.tls_handshake_time_s
            + self.key_exchange_time_s
            + self.hybrid_key_generation_time_s
        ) * 1000.0
        self.handshake_rtt_ms = self.key_exchange_time_s * 1000.0
        self.handshake_ms_server = float(hresp.server_handshake_ms)

    # ──────────────────── HeavyYoloClassifier-shaped API ─────────────────

    def set_next_call_context(
        self,
        crop_bounds: Tuple[int, int, int, int],
        frame_size: Tuple[int, int],
        frame_id: int = 0,
    ) -> None:
        self._next_crop_bounds = tuple(crop_bounds)  # type: ignore[assignment]
        self._next_frame_size = tuple(frame_size)    # type: ignore[assignment]
        self._next_frame_id = int(frame_id)

    def predict_image(self, img_pil: Image.Image):
        if self._aes_key is None:
            self.handshake()

        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # JPEG encode
        t_enc0 = time.perf_counter()
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        jpeg_bytes = buf.getvalue()
        t_enc1 = time.perf_counter()

        if self._next_frame_size == (0, 0):
            fw, fh = img_pil.size
        else:
            fw, fh = self._next_frame_size
        if self._next_crop_bounds == (0, 0, 0, 0):
            crop_bounds = (0, 0, img_pil.size[0], img_pil.size[1])
        else:
            crop_bounds = self._next_crop_bounds

        req = PqcInferRequest(
            imgsz=self.imgsz,
            conf=self.conf,
            crop_bounds=crop_bounds,
            frame_width=fw,
            frame_height=fh,
            frame_id=self._next_frame_id,
            sequence=self.sequence_name,
            jpeg_b64=base64.b64encode(jpeg_bytes).decode("ascii"),
        )
        req_plain = req.to_json().encode("utf-8")

        # Hybrid sign over the plaintext.
        t_sgn0 = time.perf_counter()
        ec_sig, pq_sig, ec_s_ms, pq_s_ms = pc.hybrid_sign(
            self._classic_sig_priv,
            self._pqc_sig_session,
            req_plain,
        )
        t_sgn1 = time.perf_counter()
        signed = pc.SignedBlob(req_plain, ec_sig, pq_sig)

        # AES-GCM encrypt.
        enc = pc.aes_gcm_encrypt(self._aes_key, signed.to_wire())  # type: ignore[arg-type]
        wire = enc.to_wire()
        t_enc2 = time.perf_counter()

        # POST
        headers = {SESSION_HEADER: self._session_id or ""}
        t_net0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}{PQC_INFER_PATH}",
            content=wire,
            headers={**headers, "Content-Type": "application/octet-stream"},
        )
        t_net1 = time.perf_counter()
        # httpx records the elapsed time *between request start and response
        # being ready* on `resp.elapsed`; we use it as TTLB.  TTFB is approxed
        # as TTLB minus the download phase, which httpx doesn't expose
        # separately — best estimate is the server-claimed inference time
        # subtracted from the full roundtrip (i.e. the time before the server
        # started writing the response body), capped at TTLB.
        try:
            ttlb_call_s = float(resp.elapsed.total_seconds())
        except Exception:  # noqa: BLE001
            ttlb_call_s = (t_net1 - t_net0)
        self.ttlb_total_s += ttlb_call_s
        if resp.status_code != 200:
            raise RuntimeError(f"PQC infer failed: {resp.status_code} {resp.text[:300]}")

        # Decrypt + verify response.
        t_dec0 = time.perf_counter()
        resp_blob_wire = resp.content
        try:
            resp_enc = pc.EncryptedBlob.from_wire(resp_blob_wire)
            resp_signed_wire = pc.aes_gcm_decrypt(self._aes_key, resp_enc)  # type: ignore[arg-type]
            resp_signed = pc.SignedBlob.from_wire(resp_signed_wire)
        except Exception:
            self.decrypt_failure_count += 1
            raise
        t_dec1 = time.perf_counter()
        ok, ec_v_ms, pq_v_ms = pc.hybrid_verify(
            self._server_classic_sig_pub or b"",
            None if self.crypto_mode == "classical" else self.sig_scheme,
            self._server_pqc_sig_pub or b"",
            resp_signed,
        )
        if not ok:
            self.signature_failure_count += 1
            raise RuntimeError("PQC infer: server signature verification failed")
        ir = PqcInferResponse.from_json(resp_signed.message.decode("utf-8"))

        # --- counter updates (all in SECONDS for reference-shape columns) --
        ec_s_s, pq_s_s = ec_s_ms / 1000.0, pq_s_ms / 1000.0
        sign_total_s    = ec_s_s + pq_s_s
        enc_s_this      = (t_enc2 - t_sgn1)
        dec_s_this      = (t_dec1 - t_dec0)
        rtt_s_this      = (t_net1 - t_net0)
        ec_v_s, pq_v_s  = ec_v_ms / 1000.0, pq_v_ms / 1000.0
        verify_total_s  = ec_v_s + pq_v_s

        srv_dec_s   = float(resp.headers.get("X-PQC-Server-Decrypt-Ms", 0)) / 1000.0
        srv_v_ec_s  = float(resp.headers.get("X-PQC-Server-Verify-Ec-Ms", 0)) / 1000.0
        srv_v_pq_s  = float(resp.headers.get("X-PQC-Server-Verify-Pq-Ms", 0)) / 1000.0
        srv_yolo_s  = float(resp.headers.get("X-PQC-Server-Yolo-Ms",
                                            ir.server_inference_ms)) / 1000.0
        srv_s_ec_s  = float(resp.headers.get("X-PQC-Server-Sign-Ec-Ms", 0)) / 1000.0
        srv_s_pq_s  = float(resp.headers.get("X-PQC-Server-Sign-Pq-Ms", 0)) / 1000.0
        srv_enc_s   = float(resp.headers.get("X-PQC-Server-Encrypt-Ms", 0)) / 1000.0

        # Network split: RTT − server compute = network_send + network_recv;
        # we approximate them as halves of the leftover (good enough for most
        # workloads, exact only when packets are symmetric in size).
        srv_compute_s = (srv_dec_s + srv_v_ec_s + srv_v_pq_s + srv_yolo_s
                         + srv_s_ec_s + srv_s_pq_s + srv_enc_s)
        net_total_s = max(0.0, rtt_s_this - srv_compute_s)
        # Weight the split by byte volume so a heavy-up payload counts
        # against network_send time and a heavy-down payload against recv.
        up_w = len(wire) / max(len(wire) + len(resp_blob_wire), 1)
        net_send_s = net_total_s * up_w
        net_recv_s = net_total_s - net_send_s

        # TTFB ≈ TTLB minus server-side compute and downstream network leg.
        # When httpx's ``resp.elapsed`` differs from our ``t_net1-t_net0``
        # by more than the response read, fall back to the wall bracket so
        # we never report a negative value.
        ttfb_call_s = max(0.0, ttlb_call_s - net_recv_s)
        self.ttfb_total_s += ttfb_call_s

        # Legacy ms accumulators (kept for back-compat of avg_*_ms columns)
        self.total_client_encode_ms     += (t_enc1 - t_enc0) * 1000.0
        self.total_client_sign_ec_ms    += ec_s_ms
        self.total_client_sign_pq_ms    += pq_s_ms
        self.total_client_encrypt_ms    += enc_s_this * 1000.0
        self.total_network_rtt_ms       += rtt_s_this * 1000.0
        self.total_client_decrypt_ms    += dec_s_this * 1000.0
        self.total_client_verify_ec_ms  += ec_v_ms
        self.total_client_verify_pq_ms  += pq_v_ms

        self.total_server_decrypt_ms    += srv_dec_s   * 1000.0
        self.total_server_verify_ec_ms  += srv_v_ec_s  * 1000.0
        self.total_server_verify_pq_ms  += srv_v_pq_s  * 1000.0
        self.total_server_yolo_ms       += srv_yolo_s  * 1000.0
        self.total_server_infer_ms      += srv_yolo_s  * 1000.0
        self.total_server_sign_ec_ms    += srv_s_ec_s  * 1000.0
        self.total_server_sign_pq_ms    += srv_s_pq_s  * 1000.0
        self.total_server_encrypt_ms    += srv_enc_s   * 1000.0

        self.total_plaintext_upload     += len(req_plain)
        self.total_plaintext_download   += len(resp_signed.message)
        self.total_bytes_uploaded       += len(wire)
        self.total_bytes_downloaded     += len(resp_blob_wire)
        self.num_requests               += 1
        self.total_yolo_time            += rtt_s_this

        # Update the running SHA-256 over plaintext request bodies.
        # Matches reference compute_sha256_hex semantics applied to the
        # concatenated payload.
        self._req_hash.update(req_plain)

        # Per-call arrays (in SECONDS) — drive reference-shape phase columns
        self.times_sign_s.append(sign_total_s)
        self.times_sign_ec_s.append(ec_s_s)
        self.times_sign_pq_s.append(pq_s_s)
        self.times_encrypt_s.append(enc_s_this)
        self.times_decrypt_s.append(dec_s_this)
        self.times_verify_s.append(verify_total_s)
        self.times_verify_ec_s.append(ec_v_s)
        self.times_verify_pq_s.append(pq_v_s)
        self.times_network_send_s.append(net_send_s)
        self.times_network_recv_s.append(net_recv_s)
        self.times_server_yolo_s.append(srv_yolo_s)
        self.times_server_sign_ec_s.append(srv_s_ec_s)
        self.times_server_sign_pq_s.append(srv_s_pq_s)
        self.times_server_encrypt_s.append(srv_enc_s)
        self.times_server_decrypt_s.append(srv_dec_s)
        self.times_server_verify_ec_s.append(srv_v_ec_s)
        self.times_server_verify_pq_s.append(srv_v_pq_s)
        self.bytes_on_wire_up.append(len(wire))
        self.bytes_on_wire_down.append(len(resp_blob_wire))
        self.bytes_plaintext_up.append(len(req_plain))
        self.bytes_plaintext_down.append(len(resp_signed.message))
        self.bytes_signed_up.append(len(signed.to_wire()))
        self.bytes_encrypted_up.append(len(wire))

        # Capture per-call size primitives once (first request).
        if self.chunk_size_bytes == 0:
            self.chunk_size_bytes            = len(req_plain)
            self.original_msg_size_bytes     = len(req_plain)
            self.signed_msg_length_bytes     = len(signed.to_wire())
            self.encrypted_msg_length_bytes  = len(wire)
            self.classic_signature_length_bytes = len(ec_sig)
            self.pqc_signature_length_bytes    = len(pq_sig)

        # reset stash
        self._next_crop_bounds = (0, 0, 0, 0)
        self._next_frame_size = (0, 0)
        self._next_frame_id = 0

        if not ir.detections:
            return img_pil, None, None, None

        n = len(ir.detections)
        xyxy = np.zeros((n, 4), dtype=np.float32)
        classes = np.zeros((n,), dtype=np.int32)
        scores = np.zeros((n,), dtype=np.float32)
        for i, d in enumerate(ir.detections):
            xyxy[i] = d.bbox_xyxy
            classes[i] = d.cls
            scores[i] = d.score
        xywh = xyxy.copy()
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return img_pil, xywh, classes, scores

    def predict_batch(self, images: list):
        """API-compat fallback for HeavyYoloClassifier.predict_batch().

        Runs N sequential predict_image() calls. The protected-inference
        path does not implement true server-side batching, so this is a
        per-image loop. Required so AG's recheck path (which calls
        yolo.predict_batch when LKT is disabled, attention_gridv2.py:2079)
        does not raise AttributeError under cloud_ag mode.
        """
        return [self.predict_image(img) for img in images]

    # ──────────────────────── housekeeping ──────────────────────────────

    def _get_total_yolo_time(self) -> float:
        return self.total_yolo_time

    def reset_counters(self) -> None:
        for attr, val in list(vars(self).items()):
            if attr.startswith("total_") and isinstance(val, (int, float)):
                setattr(self, attr, 0 if isinstance(val, int) else 0.0)
        self.num_requests = 0
        self.total_yolo_time = 0.0

    def stats_summary(self) -> dict:
        n = max(self.num_requests, 1)
        out = {
            "num_requests":             self.num_requests,
            "kem_scheme":               self.kem_scheme,
            "sig_scheme":               self.sig_scheme,
            "handshake_ms_client":      self.handshake_ms_client,
            "handshake_ms_server":      self.handshake_ms_server,
            "handshake_rtt_ms":         self.handshake_rtt_ms,
            "avg_encode_ms":            self.total_client_encode_ms / n,
            "avg_sign_ec_ms":           self.total_client_sign_ec_ms / n,
            "avg_sign_pq_ms":           self.total_client_sign_pq_ms / n,
            "avg_encrypt_ms":           self.total_client_encrypt_ms / n,
            "avg_rtt_ms":               self.total_network_rtt_ms / n,
            "avg_decrypt_ms":           self.total_client_decrypt_ms / n,
            "avg_verify_ec_ms":         self.total_client_verify_ec_ms / n,
            "avg_verify_pq_ms":         self.total_client_verify_pq_ms / n,
            "avg_server_decrypt_ms":    self.total_server_decrypt_ms / n,
            "avg_server_verify_ec_ms":  self.total_server_verify_ec_ms / n,
            "avg_server_verify_pq_ms":  self.total_server_verify_pq_ms / n,
            "avg_server_yolo_ms":       self.total_server_yolo_ms / n,
            "avg_server_inference_ms":  self.total_server_yolo_ms / n,  # alias for CSV symmetry
            "avg_server_sign_ec_ms":    self.total_server_sign_ec_ms / n,
            "avg_server_sign_pq_ms":    self.total_server_sign_pq_ms / n,
            "avg_server_encrypt_ms":    self.total_server_encrypt_ms / n,
            "avg_upload_kB":            (self.total_bytes_uploaded   / n) / 1024.0,
            "avg_download_kB":          (self.total_bytes_downloaded / n) / 1024.0,
            "total_upload_MB":          self.total_bytes_uploaded   / (1024.0 ** 2),
            "total_download_MB":        self.total_bytes_downloaded / (1024.0 ** 2),
            "total_plaintext_upload_MB":   self.total_plaintext_upload   / (1024.0 ** 2),
            "total_plaintext_download_MB": self.total_plaintext_download / (1024.0 ** 2),
        }
        # Expansion ratio = on-wire / plaintext, on the request side
        pre = max(self.total_plaintext_upload, 1)
        out["expansion_ratio_upload"] = self.total_bytes_uploaded / pre
        return out

    # ──────────────────── advanced / aggregated stats ───────────────────

    def advanced_stats_summary(self) -> dict:
        """Single-detector advanced stats (distribution shape, throughput,
        bottleneck attribution). Combine across detectors with
        ``PQCRemoteDetector.aggregate_stats([d1, d2, ...])``.
        """
        return PQCRemoteDetector.aggregate_stats([self])

    @staticmethod
    def build_master_row(detectors: list,
                         sysmon_window: Optional[dict] = None,
                         baseline: Optional[dict] = None) -> dict:
        """Produce a flat metric row whose keys match reference
        ``results/_master_summary/master_summary.csv`` 1:1.

        ``detectors``     — list of PQCRemoteDetector instances active during
                            the scene (AG caches one per imgsz). Each must
                            have had ``finalize_session()`` (or ``close()``)
                            called before this function reads its state.
        ``sysmon_window`` — optional dict from the server's
                            ``/stats/window/end`` (or an edge-side equivalent)
                            providing gpu_power_w / cpu_pct / gpu_util /
                            ram_mb / wall_clock_s / gpu_energy_j. Used to
                            populate the energy + telemetry columns.
        ``baseline``      — optional dict with at least ``throughput_mbps``
                            (a previous TLS-1.3 run's throughput) to drive
                            the *_vs_baseline_* and amortization columns.

        Keys produced (in addition to the ones the existing CSV already
        had) cover every entry in the reference column list the project
        owner provided. Missing values are emitted as 0/empty so the
        row schema stays uniform across runs.
        """
        from network import pqc_metrics as pm
        dets = [d for d in detectors if d is not None
                and d.__class__.__name__ == "PQCRemoteDetector"]
        out: dict = {}

        if not dets:
            # No PQC detectors involved in this scene — emit sentinel row.
            return out

        # Ensure every detector finalised its rusage / SHA-256 windows.
        for d in dets:
            try:
                d.finalize_session()
            except Exception:  # noqa: BLE001
                pass

        # AttentionGrid caches one detector per imgsz.  Some cached detectors
        # may never have processed a single frame (their imgsz was prepared
        # but unused).  Reading per-call sizes / rusage / SHA-256 from such a
        # detector yields zeros + the empty-string SHA.  Pick the first
        # detector that actually saw requests; fall back to dets[0] if none
        # did (e.g. an edge case where the scene had zero frames).
        active = [d for d in dets if d.num_requests > 0] or dets
        d_active = active[0]

        # ---- scheme identity (all detectors share these) -------------------
        kem = d_active.kem_scheme
        sig = d_active.sig_scheme
        out["kem_scheme"] = kem
        out["sig_scheme"] = sig

        # ---- pre-aggregate per-call arrays in seconds ----------------------
        def merged(attr):
            v: list = []
            for d in dets:
                v.extend(getattr(d, attr, []) or [])
            return v

        client_sign_s        = merged("times_sign_s")
        client_verify_s      = merged("times_verify_s")
        client_encrypt_s     = merged("times_encrypt_s")
        client_decrypt_s     = merged("times_decrypt_s")
        client_net_send_s    = merged("times_network_send_s")
        client_net_recv_s    = merged("times_network_recv_s")

        # Per-phase chunk + advanced stats (reference column names).
        # Each phase gets a meaningful per-call expansion ratio:
        #   signing      : signed_up      / plaintext_up   (sign wrapping growth)
        #   verification : plaintext_up   / signed_up      (unwrap shrinkage; <1)
        #   encryption   : encrypted_up   / signed_up      (AES-GCM tag+nonce overhead)
        #   decryption   : signed_up      / encrypted_up   (unwrap shrinkage; <1)
        #   network_send : encrypted_up   / plaintext_up   (full upstream wire/raw)
        #   network_recv : on_wire_down   / plaintext_down (full downstream wire/raw)
        out.update(pm.build_phase_columns(client_sign_s,    "signing",
                                          _avg_expansion(dets, "signed_up", "plaintext_up")))
        out.update(pm.build_phase_columns(client_verify_s,  "verification",
                                          _avg_expansion(dets, "plaintext_up", "signed_up")))
        out.update(pm.build_phase_columns(client_encrypt_s, "encryption",
                                          _avg_expansion(dets, "encrypted_up", "signed_up")))
        out.update(pm.build_phase_columns(client_decrypt_s, "decryption",
                                          _avg_expansion(dets, "signed_up", "encrypted_up")))
        out.update(pm.build_phase_columns(client_net_send_s, "network_send",
                                          _avg_expansion(dets, "encrypted_up", "plaintext_up")))
        out.update(pm.build_phase_columns(client_net_recv_s, "network_recv",
                                          _avg_expansion(dets, "on_wire_down", "plaintext_down")))

        # ---- transfer / session aggregates (client side) -------------------
        total_plain_up   = sum(merged("bytes_plaintext_up"))
        total_plain_down = sum(merged("bytes_plaintext_down"))
        total_wire_up    = sum(merged("bytes_on_wire_up"))
        total_wire_down  = sum(merged("bytes_on_wire_down"))
        total_signed_up  = sum(merged("bytes_signed_up"))
        total_req = sum(d.num_requests for d in dets)

        # Wall-clock across all detectors (max of end - start across them)
        client_wall_s = max(
            (d._t_session_end or 0) - (d._t_session_start or 0)
            for d in dets if d._t_session_start is not None
        ) if any(d._t_session_start for d in dets) else 0.0

        client_rusage = pm.compute_resource_delta(
            d_active._rusage_before or {}, d_active._rusage_after or {}
        )
        client_cpu_s = client_rusage["user_cpu_s"] + client_rusage["sys_cpu_s"]
        out["client_user_cpu_s"]   = client_rusage["user_cpu_s"]
        out["client_sys_cpu_s"]    = client_rusage["sys_cpu_s"]
        out["client_peak_rss_mb"]  = client_rusage["peak_rss_mb"]
        out["client_vol_ctxs"]     = client_rusage["vol_ctxs"]
        out["client_invol_ctxs"]   = client_rusage["invol_ctxs"]
        out["client_minor_faults"] = client_rusage["minor_faults"]
        out["client_major_faults"] = client_rusage["major_faults"]
        out["client_cpu_bound_score"] = pm.compute_cpu_bound_score(client_cpu_s, client_wall_s)

        out["client_total_wall_time_s"] = client_wall_s
        out["client_num_chunks"]        = total_req
        out["client_chunk_size_bytes"]  = d_active.chunk_size_bytes
        out["client_raw_payload_bytes"] = total_plain_up + total_plain_down
        out["client_on_wire_bytes"]     = total_wire_up + total_wire_down

        thr = pm.compute_throughput_metrics(
            total_bytes_raw=out["client_raw_payload_bytes"],
            total_bytes_on_wire=out["client_on_wire_bytes"],
            total_time_s=client_wall_s,
            num_chunks=total_req,
        )
        out["client_throughput_mbps_binary"] = thr["throughput_mbps"]
        out["client_goodput_mbps_binary"]    = thr["goodput_mbps"]
        out["client_chunk_rate_hz"]          = thr["chunk_rate_hz"]
        out["client_signature_rate_hz"]      = thr["signature_rate_hz"]
        out["client_bandwidth_overhead_pct"] = thr["bandwidth_overhead_pct"]

        bn = pm.compute_bottleneck_attribution(
            sign_time_s=sum(client_sign_s) + sum(client_verify_s),
            encrypt_time_s=sum(client_encrypt_s) + sum(client_decrypt_s),
            network_time_s=sum(client_net_send_s) + sum(client_net_recv_s),
        )
        out["client_sign_pct"]              = bn["sign_pct"]
        out["client_encrypt_pct"]           = bn["encrypt_pct"]
        out["client_network_pct"]           = bn["network_pct"]
        out["client_dominant_phase"]        = bn["dominant_phase"]
        out["client_sign_to_encrypt_ratio"] = bn["sign_to_encrypt_ratio"]
        out["client_sign_to_network_ratio"] = bn["sign_to_network_ratio"]
        # Alias without prefix (reference emits both)
        out["sign_pct"]               = bn["sign_pct"]
        out["encrypt_pct"]            = bn["encrypt_pct"]
        out["network_pct"]            = bn["network_pct"]
        out["dominant_phase"]         = bn["dominant_phase"]
        out["sign_to_encrypt_ratio"]  = bn["sign_to_encrypt_ratio"]
        out["sign_to_network_ratio"]  = bn["sign_to_network_ratio"]

        # Handshake breakdown (averaged across detectors that did handshake)
        used = [d for d in dets if d.num_requests > 0] or dets
        out["client_tls_handshake_time"]         = _mean(d.tls_handshake_time_s for d in used)
        out["client_key_exchange_time"]          = _mean(d.key_exchange_time_s  for d in used)
        out["client_hybrid_key_generation_time"] = _mean(d.hybrid_key_generation_time_s for d in used)
        out["client_session_setup_ms"] = _mean(d.handshake_ms_client for d in used)
        # And the totals-by-phase that reference calls "total_*_time"
        out["client_total_signing_time"]    = sum(client_sign_s) + sum(client_verify_s)  # combined sign+verify on client
        out["client_total_signing_time"]    = sum(client_sign_s)
        out["client_total_verifying_time"]  = sum(client_verify_s)
        out["client_total_encryption_time"] = sum(client_encrypt_s)
        out["client_total_decryption_time"] = sum(client_decrypt_s)
        out["client_total_time_to_sent_or_receive"] = sum(merged("times_network_send_s")) + sum(merged("times_network_recv_s"))
        # Lower-resolution averages (also published as legacy *_ms columns)
        if total_req > 0:
            out["client_signing_time"]                = out["client_total_signing_time"] / total_req
            out["client_signature_verification_time"] = out["client_total_verifying_time"] / total_req
            out["client_encryption_time"]             = out["client_total_encryption_time"] / total_req
            out["client_decryption_time"]             = out["client_total_decryption_time"] / total_req

        # ---- size columns (reference setup_csv_size_logging) --------------
        d0 = d_active
        out["size_original_msg_size"]               = d0.original_msg_size_bytes
        out["size_signed_msg_length"]               = d0.signed_msg_length_bytes
        out["size_encrypted_msg_length"]            = d0.encrypted_msg_length_bytes
        out["size_chunk_size"]                      = d0.chunk_size_bytes
        out["size_hybrid_aes_key_length"]           = d0.hybrid_aes_key_length_bytes
        out["size_classical_signature_length"]      = d0.classic_signature_length_bytes
        out["size_pqc_signature_length"]            = d0.pqc_signature_length_bytes
        out["size_classic_encryption_sharedsecret_length"] = d0.classic_kem_sharedsecret_length_bytes
        out["size_pqc_encryption_sharedsecret_length"]     = d0.pqc_kem_sharedsecret_length_bytes

        # ---- transfer-level expansion (whole-session) ----------------------
        # `_decimal` columns use 10^6 (reference DECIMAL_MB) for paper
        # consistency; `_binary` ones use 2^20 (compute_throughput_metrics).
        DECIMAL_MB = 1_000_000.0
        out["transfer_payload_bytes"]                  = total_plain_up
        out["transfer_chunk_count"]                    = total_req
        out["transfer_hybrid_setup_bytes"]             = (d0.signed_msg_length_bytes
                                                          - d0.original_msg_size_bytes)
        out["transfer_per_file_control_bytes"]         = 0
        out["transfer_protected_data_bytes"]           = total_wire_up
        out["transfer_total_app_bytes"]                = total_wire_up + total_wire_down
        out["transfer_payload_goodput_mbps_decimal"]   = (
            (total_plain_up / DECIMAL_MB) / client_wall_s if client_wall_s > 0 else 0.0
        )
        out["transfer_protected_goodput_mbps_decimal"] = (
            (total_wire_up / DECIMAL_MB) / client_wall_s if client_wall_s > 0 else 0.0
        )
        out["transfer_app_expansion_ratio_total"]      = (total_wire_up / total_plain_up
                                                          if total_plain_up > 0 else 0.0)

        # ---- verification + hash columns -----------------------------------
        # Use the active detector's SHA, not dets[0] — a cached-but-unused
        # detector reports the empty-string hash (e3b0c442...).
        out["client_payload_sha256"]      = d_active.client_payload_sha256
        out["signature_failure_count"]    = sum(d.signature_failure_count for d in dets)
        out["decrypt_failure_count"]      = sum(d.decrypt_failure_count   for d in dets)
        # shared_secret_match is implicitly True if handshake succeeded
        # (server and client derived the same AES key — otherwise none of
        # the per-call AES-GCM decrypts would have succeeded).
        out["shared_secret_match"]        = bool(total_req > 0 and out["decrypt_failure_count"] == 0)
        # verification_status mirrors reference write_verification_report:
        # "OK" when no signature / decrypt failures and at least one chunk
        # was exchanged; "FAIL" otherwise. The payload_sha256_match column
        # added below is the *cryptographic* end-to-end equality check.
        out["verification_status"] = (
            "OK" if (total_req > 0
                     and out["signature_failure_count"] == 0
                     and out["decrypt_failure_count"]   == 0)
            else "FAIL"
        )

        # ---- server side (single combined session) -------------------------
        # Each detector talks to a *different* server session (one session per
        # detector). We fetch a state per detector and pair-check the SHAs so
        # payload_sha256_match reflects ALL the per-detector bytes-on-the-wire,
        # not just the first detector's slice.
        srv_states: list = []
        per_det_server_sha: list[str] = []
        per_det_client_sha: list[str] = []
        for d in dets:
            ss = d.fetch_server_session_stats()
            if ss:
                srv_states.append(ss)
                per_det_server_sha.append(ss.get("server_payload_sha256", ""))
            else:
                per_det_server_sha.append("")
            per_det_client_sha.append(d.client_payload_sha256 or "")
        out.update(_server_block(srv_states, baseline=baseline))

        # SHA pairing: every (detector, session) pair must agree, AND at
        # least one detector must actually have sent data.
        any_pair = False
        all_match = True
        for cs, ss in zip(per_det_client_sha, per_det_server_sha):
            if cs and ss:
                any_pair = True
                if cs != ss:
                    all_match = False
        out["payload_sha256_match"] = bool(any_pair and all_match)

        # Surface a "primary" SHA pair that actually corresponds to non-empty
        # data — cached-but-unused detectors emit the SHA-256 of the empty
        # string (e3b0c442...) which is useless as a primary representation.
        # Find the first (cs, ss) pair where the client SHA isn't the empty-
        # string hash; fall back to dets[0]'s if every detector was idle.
        _EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        primary_cs, primary_ss = "", ""
        for cs, ss in zip(per_det_client_sha, per_det_server_sha):
            if cs and cs != _EMPTY_SHA:
                primary_cs, primary_ss = cs, ss
                break
        if not primary_cs and per_det_client_sha:
            primary_cs = per_det_client_sha[0]
            primary_ss = per_det_server_sha[0] if per_det_server_sha else ""
        out["client_payload_sha256"]    = primary_cs
        out["server_payload_sha256"]    = primary_ss
        out["client_payload_sha256_list"] = ";".join(per_det_client_sha)
        out["server_payload_sha256_list"] = ";".join(per_det_server_sha)

        # Network-gap columns.  TCP connect cost is paid once per detector
        # at handshake() time (we use the pre-warm /healthz GET wall time as
        # a proxy); TTFB/TTLB are accumulated across all /pqc/infer calls
        # and surfaced as per-call averages.
        tcp_vals = [d.tcp_connect_time_s for d in dets if d.tcp_connect_time_s > 0]
        avg_tcp_s = (sum(tcp_vals) / len(tcp_vals)) if tcp_vals else 0.0
        ttfb_sum  = sum(d.ttfb_total_s for d in dets)
        ttlb_sum  = sum(d.ttlb_total_s for d in dets)
        avg_ttfb_s = (ttfb_sum / total_req) if total_req > 0 else 0.0
        avg_ttlb_s = (ttlb_sum / total_req) if total_req > 0 else 0.0
        out["tcp_connect_time_s"] = avg_tcp_s
        out["ttfb_s"]             = avg_ttfb_s
        out["ttlb_s"]             = avg_ttlb_s
        out["client_tcp_connect_time_s"] = avg_tcp_s
        out["client_ttfb_s"]             = avg_ttfb_s
        out["client_ttlb_s"]             = avg_ttlb_s
        # The server has no symmetric view of TCP/TTFB/TTLB (those are by
        # definition client-observed wall times), so we mirror the client
        # values into the server_* slots so the row schema stays uniform.
        out["server_tcp_connect_time_s"] = avg_tcp_s
        out["server_ttfb_s"]             = avg_ttfb_s
        out["server_ttlb_s"]             = avg_ttlb_s

        # ---- baseline / amortization (only if a baseline dict is supplied) -
        if baseline and "throughput_mbps" in baseline and baseline["throughput_mbps"] > 0:
            cmp_thr = pm.compute_vs_baseline(thr["throughput_mbps"],
                                             baseline["throughput_mbps"],
                                             higher_is_better=True)
            out["vs_baseline_throughput_ratio"]        = cmp_thr["ratio"]
            out["client_vs_baseline_throughput_ratio"] = cmp_thr["ratio"]
            amort = pm.compute_session_amortization(out["client_session_setup_ms"],
                                                    thr["throughput_mbps"],
                                                    baseline["throughput_mbps"])
            out["break_even_mb"]                = amort["break_even_mb"]
            out["per_mb_penalty_ms"]            = amort["per_mb_penalty_ms"]
            out["throughput_ratio_amort"]       = amort["throughput_ratio"]
            out["client_break_even_mb"]         = amort["break_even_mb"]
            out["client_per_mb_penalty_ms"]     = amort["per_mb_penalty_ms"]
            out["client_amort_throughput_ratio"] = amort["throughput_ratio"]
            # Server-side mirror — use the server's throughput we already
            # computed in _server_block().
            srv_thr = out.get("server_throughput_mbps_binary", 0.0) or 0.0
            srv_setup = out.get("server_session_setup_ms", 0.0) or 0.0
            if srv_thr > 0:
                cmp_srv = pm.compute_vs_baseline(srv_thr, baseline["throughput_mbps"],
                                                 higher_is_better=True)
                amort_srv = pm.compute_session_amortization(
                    srv_setup, srv_thr, baseline["throughput_mbps"])
                out["server_vs_baseline_throughput_ratio"] = cmp_srv["ratio"]
                out["server_break_even_mb"]                = amort_srv["break_even_mb"]
                out["server_per_mb_penalty_ms"]            = amort_srv["per_mb_penalty_ms"]
                out["server_amort_throughput_ratio"]       = amort_srv["throughput_ratio"]
            else:
                out["server_vs_baseline_throughput_ratio"] = 0.0
                out["server_break_even_mb"]                = 0.0
                out["server_per_mb_penalty_ms"]            = 0.0
                out["server_amort_throughput_ratio"]       = 0.0
        else:
            # Still emit the columns (sentinel 0.0) so schema is uniform.
            for k in ("vs_baseline_throughput_ratio", "break_even_mb",
                      "per_mb_penalty_ms", "throughput_ratio_amort",
                      "client_vs_baseline_throughput_ratio",
                      "client_break_even_mb", "client_per_mb_penalty_ms",
                      "client_amort_throughput_ratio",
                      "server_vs_baseline_throughput_ratio",
                      "server_break_even_mb", "server_per_mb_penalty_ms",
                      "server_amort_throughput_ratio"):
                out.setdefault(k, 0.0)

        # ---- system-monitor / energy block --------------------------------
        if sysmon_window:
            energy_block = _energy_block(sysmon_window,
                                         total_payload_mb=total_plain_up / (1024.0 ** 2),
                                         total_protected_mb=total_wire_up / (1024.0 ** 2))
            out.update(energy_block)

        # ---- RTT aggregate (in ms, as reference rtt_*_ms columns) ---------
        rtt_ms = [s * 1000.0 for s in (merged("times_network_send_s")
                                       + merged("times_network_recv_s"))]
        rtt_stats = pm.compute_chunk_statistics(rtt_ms)
        out["rtt_count"]        = rtt_stats["count"]
        out["rtt_mean_ms"]      = rtt_stats["mean"]
        out["rtt_stddev_ms"]    = rtt_stats["stddev"]
        out["rtt_jitter_ms"]    = rtt_stats["jitter"]
        out["rtt_jitter_class"] = rtt_stats["jitter_class"]

        return out

    @staticmethod
    def aggregate_stats(detectors: list) -> dict:
        """Merge per-call arrays across one or more detectors and compute
        reference-style v2.1 statistics:

            count / mean / stddev / p50 / p95 / p99
            CV% / jitter / jitter_class
            skewness / excess_kurtosis / IQR / Tukey outliers
            tail_ratio_p99_p50 / tail_ratio_p95_p50 / determinism_class
            throughput_mbps / goodput_mbps / bandwidth_overhead_pct
            sign_pct / encrypt_pct / network_pct / dominant_phase
            sign_to_encrypt_ratio / sign_to_network_ratio
        """
        from network import pqc_stats as ps

        detectors = [d for d in detectors if d is not None]
        if not detectors:
            return {}

        # Surface scheme identity (all detectors share these on a single run).
        kem = detectors[0].kem_scheme
        sig = detectors[0].sig_scheme

        # Handshake numbers: same scheme on every detector, but handshake_ms_*
        # varies per instance. Report the average across the detectors that
        # actually handshook (n_calls > 0); fallback to the first one.
        used = [d for d in detectors if d.num_requests > 0] or detectors
        hs_client = sum(d.handshake_ms_client for d in used) / len(used)
        hs_server = sum(d.handshake_ms_server for d in used) / len(used)
        hs_rtt    = sum(d.handshake_rtt_ms    for d in used) / len(used)

        total_req = sum(d.num_requests for d in detectors)

        out: dict = {
            "kem_scheme":          kem,
            "sig_scheme":          sig,
            "num_requests":        total_req,
            "num_detectors":       len(detectors),
            "handshake_ms_client": hs_client,
            "handshake_ms_server": hs_server,
            "handshake_rtt_ms":    hs_rtt,
        }

        # Per-phase distribution stats. Bucket label is what reference uses
        # in client_advanced_chunk_stats.csv.
        # All per-call arrays are stored in SECONDS (_s suffix).  We convert
        # to milliseconds here so the stat values match the _ms column names
        # the CSV writer expects downstream.
        phases = [
            ("sign_ec",          "times_sign_ec_s"),
            ("sign_pq",          "times_sign_pq_s"),
            ("encrypt",          "times_encrypt_s"),
            ("decrypt",          "times_decrypt_s"),
            ("verify_ec",        "times_verify_ec_s"),
            ("verify_pq",        "times_verify_pq_s"),
            ("server_yolo",      "times_server_yolo_s"),
            ("server_sign_ec",   "times_server_sign_ec_s"),
            ("server_sign_pq",   "times_server_sign_pq_s"),
            ("server_encrypt",   "times_server_encrypt_s"),
            ("server_decrypt",   "times_server_decrypt_s"),
            ("server_verify_ec", "times_server_verify_ec_s"),
            ("server_verify_pq", "times_server_verify_pq_s"),
        ]
        for label, attr in phases:
            arr_ms = [x * 1000.0 for x in ps.merge_time_arrays(detectors, attr)]
            base = ps.compute_chunk_statistics(arr_ms)
            adv = ps.compute_advanced_statistics(arr_ms, base)
            # Flatten with `<label>__` prefix so columns stay readable.
            for k, v in base.items():
                out[f"{label}__{k}"] = v
            for k, v in adv.items():
                out[f"{label}__{k}"] = v

        # RTT = network_send + network_recv per call (no single attribute stores
        # the combined value; construct it here in ms).
        net_send_ms = [x * 1000.0 for x in ps.merge_time_arrays(detectors, "times_network_send_s")]
        net_recv_ms = [x * 1000.0 for x in ps.merge_time_arrays(detectors, "times_network_recv_s")]
        rtt_arr_ms  = [s + r for s, r in zip(net_send_ms, net_recv_ms)]
        rtt_base = ps.compute_chunk_statistics(rtt_arr_ms)
        rtt_adv  = ps.compute_advanced_statistics(rtt_arr_ms, rtt_base)
        for k, v in rtt_base.items():
            out[f"rtt__{k}"] = v
        for k, v in rtt_adv.items():
            out[f"rtt__{k}"] = v

        # Per-call byte arrays
        wire_up   = ps.merge_time_arrays(detectors, "bytes_on_wire_up")
        wire_down = ps.merge_time_arrays(detectors, "bytes_on_wire_down")
        plain_up  = ps.merge_time_arrays(detectors, "bytes_plaintext_up")
        plain_down = ps.merge_time_arrays(detectors, "bytes_plaintext_down")
        total_wire_up   = sum(wire_up)
        total_wire_down = sum(wire_down)
        total_plain_up  = sum(plain_up)
        total_plain_down = sum(plain_down)
        out["total_upload_MB"]            = total_wire_up   / (1024.0 ** 2)
        out["total_download_MB"]          = total_wire_down / (1024.0 ** 2)
        out["pre_crypto_upload_MB"]       = total_plain_up   / (1024.0 ** 2)
        out["pre_crypto_download_MB"]     = total_plain_down / (1024.0 ** 2)
        out["crypto_expansion_ratio_up"]  = (total_wire_up   / total_plain_up)   if total_plain_up   > 0 else 0.0
        out["crypto_expansion_ratio_down"] = (total_wire_down / total_plain_down) if total_plain_down > 0 else 0.0

        # Throughput / bottleneck attribution use *seconds*.
        # Per-call arrays are in seconds (_s suffix); totals are in ms (_ms suffix).
        total_sign_s    = sum(sum(getattr(d, "times_sign_ec_s", [])) +
                              sum(getattr(d, "times_sign_pq_s", []))
                              for d in detectors)
        total_encrypt_s = sum(sum(getattr(d, "times_encrypt_s", [])) +
                              sum(getattr(d, "times_decrypt_s", []))
                              for d in detectors)
        total_rtt_s     = sum(getattr(d, "total_network_rtt_ms", 0.0) for d in detectors) / 1000.0
        # Network time = RTT − server-side compute (best per-call estimate).
        total_server_compute_s = sum(getattr(d, "total_server_yolo_ms", 0.0) +
                                     getattr(d, "total_server_sign_ec_ms", 0.0) +
                                     getattr(d, "total_server_sign_pq_ms", 0.0) +
                                     getattr(d, "total_server_encrypt_ms", 0.0) +
                                     getattr(d, "total_server_decrypt_ms", 0.0) +
                                     getattr(d, "total_server_verify_ec_ms", 0.0) +
                                     getattr(d, "total_server_verify_pq_ms", 0.0)
                                     for d in detectors) / 1000.0
        total_network_s = max(0.0, total_rtt_s - total_server_compute_s)

        bottleneck = ps.compute_bottleneck_attribution(
            total_sign_s, total_encrypt_s, total_network_s
        )
        out["sign_pct"]               = bottleneck["sign_pct"]
        out["encrypt_pct"]            = bottleneck["encrypt_pct"]
        out["network_pct"]            = bottleneck["network_pct"]
        out["dominant_phase"]         = bottleneck["dominant_phase"]
        out["sign_to_encrypt_ratio"]  = bottleneck["sign_to_encrypt_ratio"]
        out["sign_to_network_ratio"]  = bottleneck["sign_to_network_ratio"]
        out["total_sign_s"]           = total_sign_s
        out["total_encrypt_s"]        = total_encrypt_s
        out["total_network_s"]        = total_network_s

        thr = ps.compute_throughput_metrics(
            total_bytes_raw=total_plain_up + total_plain_down,
            total_bytes_on_wire=total_wire_up + total_wire_down,
            total_time_s=total_rtt_s,
            num_chunks=total_req,
        )
        out.update(thr)
        return out

    def finalize_session(self) -> None:
        """Snapshot rusage + freeze the SHA-256 of all plaintext request bodies.

        Call this before reading aggregate_stats() if you want client_payload_sha256
        and the rusage delta to reflect a fully-closed session. Safe to call
        more than once.
        """
        if self._t_session_end is None:
            self._t_session_end = time.perf_counter()
        if self._rusage_after is None:
            self._rusage_after = pm.get_resource_snapshot()
        if not self.client_payload_sha256:
            self.client_payload_sha256 = self._req_hash.hexdigest()

    def close(self) -> None:
        # Always finalise the metrics window first.
        try:
            self.finalize_session()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        if self._kem_session is not None:
            self._kem_session.free()
        if self._pqc_sig_session is not None:
            self._pqc_sig_session.free()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ─────────────────────── server stats windows ───────────────────────
    #
    # Plaintext metrics (GPU power, etc.) don't need to be PQC-wrapped; the
    # existing /stats/window/{begin,end} endpoints on the TLS-1.3 channel
    # are fine.

    def begin_server_window(self, tag: str = "") -> str:
        r = self._client.post(f"{self.base_url}/stats/window/begin",
                              params={"tag": tag})
        r.raise_for_status()
        return r.json()["tag"]

    def end_server_window(self, tag: str) -> dict:
        r = self._client.post(f"{self.base_url}/stats/window/end",
                              params={"tag": tag})
        r.raise_for_status()
        return r.json()

    def server_health(self) -> dict:
        r = self._client.get(f"{self.base_url}/healthz")
        r.raise_for_status()
        return r.json()

    def fetch_server_session_stats(self) -> dict:
        """Pull the server-side per-session metric dict — rusage delta,
        per-call timing arrays, byte counters, SHA-256 of all received
        plaintext request bodies — used to populate the ``server_*`` columns
        in the master row. Returns {} if the server hasn't kept stats for
        this session (older server, expired session)."""
        if not self._session_id:
            return {}
        try:
            r = self._client.get(
                f"{self.base_url}/pqc/stats/{self._session_id}",
                timeout=10.0,
            )
            if r.status_code != 200:
                return {}
            return r.json()
        except Exception:  # noqa: BLE001
            return {}


# =====================================================================
# Module-scope helpers used by build_master_row().
# =====================================================================


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def _avg_expansion(detectors: list, out_attr: str, in_attr: str) -> float:
    """Average expansion ratio over per-call (out_bytes / in_bytes) samples.

    ``out_attr`` / ``in_attr`` name the suffix of a ``bytes_<name>`` per-call
    array on the detector (e.g. ``"signed_up"``, ``"plaintext_up"``,
    ``"on_wire_down"``).  Detectors that never ran (no per-call entries)
    contribute nothing.
    """
    ratios: list[float] = []
    for d in detectors:
        outs = getattr(d, f"bytes_{out_attr}", None)
        ins  = getattr(d, f"bytes_{in_attr}",  None)
        if not outs or not ins:
            continue
        for o, i in zip(outs, ins):
            if i > 0:
                ratios.append(o / i)
    return (sum(ratios) / len(ratios)) if ratios else 0.0


def _server_block(srv_states: list, baseline: Optional[dict] = None) -> dict:
    """Server-side mirror of the client metric block. Aggregates across
    every server session involved in the scene."""
    from network import pqc_metrics as pm

    out: dict = {}
    if not srv_states:
        # Server stats endpoint absent or session expired — emit sentinels.
        for k in [
            "server_total_wall_time_s", "server_num_chunks", "server_chunk_size_bytes",
            "server_raw_payload_bytes", "server_on_wire_bytes",
            "server_throughput_mbps_binary", "server_goodput_mbps_binary",
            "server_chunk_rate_hz", "server_signature_rate_hz",
            "server_bandwidth_overhead_pct",
            "server_sign_pct", "server_encrypt_pct", "server_network_pct",
            "server_dominant_phase",
            "server_sign_to_encrypt_ratio", "server_sign_to_network_ratio",
            "server_user_cpu_s", "server_sys_cpu_s", "server_peak_rss_mb",
            "server_vol_ctxs", "server_invol_ctxs",
            "server_minor_faults", "server_major_faults",
            "server_cpu_bound_score", "server_session_setup_ms",
            "server_total_signing_time", "server_total_encryption_time",
            "server_total_verifying_time", "server_total_decryption_time",
            "server_total_time_to_sent_or_receive",
            "server_signing_time", "server_signature_verification_time",
            "server_encryption_time", "server_decryption_time",
            "server_tls_handshake_time", "server_key_exchange_time",
            "server_hybrid_key_generation_time",
            "server_payload_sha256",
        ]:
            out[k] = 0 if k.endswith(("count", "bytes", "ctxs", "faults")) else ""
        return out

    def cat(key: str):
        v: list = []
        for s in srv_states:
            v.extend(s.get(key, []) or [])
        return v

    decrypt_s = cat("times_decrypt_s")
    verify_ec_s = cat("times_verify_ec_s")
    verify_pq_s = cat("times_verify_pq_s")
    yolo_s    = cat("times_yolo_s")
    sign_ec_s = cat("times_sign_ec_s")
    sign_pq_s = cat("times_sign_pq_s")
    encrypt_s = cat("times_encrypt_s")
    verify_total = [a + b for a, b in zip(verify_ec_s, verify_pq_s)]
    sign_total   = [a + b for a, b in zip(sign_ec_s, sign_pq_s)]

    bytes_in        = cat("bytes_in")
    bytes_out       = cat("bytes_out")
    bytes_plain_in  = cat("bytes_plain_in")
    bytes_plain_out = cat("bytes_plain_out")

    total_calls = sum(int(s.get("n_calls", 0)) for s in srv_states)
    wall_s      = max((float(s.get("wall_time_s", 0.0)) for s in srv_states), default=0.0)

    # Per-phase distribution columns on the server side share key namespace
    # with the client phases — that's intentional (reference master_summary
    # builder collapses them this way). Don't overwrite the client phases:
    # only emit the server-only handshake / aggregate columns here.

    out["server_total_wall_time_s"]    = wall_s
    out["server_num_chunks"]           = total_calls
    out["server_chunk_size_bytes"]     = bytes_plain_in[0] if bytes_plain_in else 0
    out["server_raw_payload_bytes"]    = sum(bytes_plain_in) + sum(bytes_plain_out)
    out["server_on_wire_bytes"]        = sum(bytes_in) + sum(bytes_out)

    thr_srv = pm.compute_throughput_metrics(
        total_bytes_raw=out["server_raw_payload_bytes"],
        total_bytes_on_wire=out["server_on_wire_bytes"],
        total_time_s=wall_s,
        num_chunks=total_calls,
    )
    out["server_throughput_mbps_binary"] = thr_srv["throughput_mbps"]
    out["server_goodput_mbps_binary"]    = thr_srv["goodput_mbps"]
    out["server_chunk_rate_hz"]          = thr_srv["chunk_rate_hz"]
    out["server_signature_rate_hz"]      = thr_srv["signature_rate_hz"]
    out["server_bandwidth_overhead_pct"] = thr_srv["bandwidth_overhead_pct"]

    bn_srv = pm.compute_bottleneck_attribution(
        sign_time_s=sum(sign_total) + sum(verify_total),
        encrypt_time_s=sum(encrypt_s) + sum(decrypt_s),
        network_time_s=max(0.0, wall_s - sum(yolo_s) - sum(sign_total)
                           - sum(verify_total) - sum(encrypt_s) - sum(decrypt_s)),
    )
    out["server_sign_pct"]              = bn_srv["sign_pct"]
    out["server_encrypt_pct"]           = bn_srv["encrypt_pct"]
    out["server_network_pct"]           = bn_srv["network_pct"]
    out["server_dominant_phase"]        = bn_srv["dominant_phase"]
    out["server_sign_to_encrypt_ratio"] = bn_srv["sign_to_encrypt_ratio"]
    out["server_sign_to_network_ratio"] = bn_srv["sign_to_network_ratio"]

    # Per-state rusage delta — first state is fine (single session per detector)
    s0 = srv_states[0]
    rb = s0.get("rusage_before", {}) or {}
    ra = s0.get("rusage_after",  {}) or {}
    srv_rd = pm.compute_resource_delta(rb, ra)
    out["server_user_cpu_s"]   = srv_rd["user_cpu_s"]
    out["server_sys_cpu_s"]    = srv_rd["sys_cpu_s"]
    out["server_peak_rss_mb"]  = srv_rd["peak_rss_mb"]
    out["server_vol_ctxs"]     = srv_rd["vol_ctxs"]
    out["server_invol_ctxs"]   = srv_rd["invol_ctxs"]
    out["server_minor_faults"] = srv_rd["minor_faults"]
    out["server_major_faults"] = srv_rd["major_faults"]
    out["server_cpu_bound_score"] = pm.compute_cpu_bound_score(
        srv_rd["user_cpu_s"] + srv_rd["sys_cpu_s"], wall_s)

    out["server_session_setup_ms"] = _mean(float(s.get("server_handshake_ms", 0.0))
                                           for s in srv_states)
    out["server_total_signing_time"]    = sum(sign_total)
    out["server_total_verifying_time"]  = sum(verify_total)
    out["server_total_encryption_time"] = sum(encrypt_s)
    out["server_total_decryption_time"] = sum(decrypt_s)
    out["server_total_time_to_sent_or_receive"] = wall_s - sum(yolo_s)
    if total_calls > 0:
        out["server_signing_time"]                = sum(sign_total) / total_calls
        out["server_signature_verification_time"] = sum(verify_total) / total_calls
        out["server_encryption_time"]             = sum(encrypt_s) / total_calls
        out["server_decryption_time"]             = sum(decrypt_s) / total_calls

    # Server-side handshake phase breakdown — pulled from /pqc/stats/<sid>,
    # which the modern server fills in.  Older servers that don't emit these
    # fields gracefully fall back to surfacing server_session_setup_ms as
    # the TLS-handshake bucket so the row schema stays uniform.
    tls_vals = [float(s.get("server_tls_handshake_time_s", 0.0) or 0.0) for s in srv_states]
    kex_vals = [float(s.get("server_key_exchange_time_s",  0.0) or 0.0) for s in srv_states]
    hkg_vals = [float(s.get("server_hybrid_key_generation_time_s", 0.0) or 0.0) for s in srv_states]
    if any(v > 0 for v in (tls_vals + kex_vals + hkg_vals)):
        out["server_tls_handshake_time"]         = _mean(tls_vals)
        out["server_key_exchange_time"]          = _mean(kex_vals)
        out["server_hybrid_key_generation_time"] = _mean(hkg_vals)
    else:
        # Old server — no phase breakdown available.
        out["server_tls_handshake_time"]         = out["server_session_setup_ms"] / 1000.0
        out["server_key_exchange_time"]          = 0.0
        out["server_hybrid_key_generation_time"] = 0.0

    # SHA-256 of all plaintext received server-side
    shas = [s.get("server_payload_sha256", "") for s in srv_states]
    out["server_payload_sha256"] = shas[0] if len(set(shas)) == 1 and shas[0] else ""

    return out


def _energy_block(sysmon_window: dict,
                  total_payload_mb: float, total_protected_mb: float) -> dict:
    """Translate a SystemMonitor window snapshot into reference energy +
    telemetry columns."""
    out: dict = {}
    wall = float(sysmon_window.get("wall_clock_s", 0.0) or 0.0)
    power_w = float(sysmon_window.get("gpu_power_w", 0.0) or 0.0)
    energy_j = float(sysmon_window.get("gpu_energy_j", power_w * wall) or 0.0)
    out["tegrastats_samples"]        = int(sysmon_window.get("n_samples", 0) or 0)
    out["avg_power_w"]               = power_w
    out["energy_j"]                  = energy_j
    out["energy_per_payload_mb_j"]   = (energy_j / total_payload_mb)   if total_payload_mb   > 0 else 0.0
    out["energy_per_protected_mb_j"] = (energy_j / total_protected_mb) if total_protected_mb > 0 else 0.0
    out["cpu_pct_mean"]              = float(sysmon_window.get("cpu_pct", 0.0) or 0.0)
    out["gpu_pct_mean"]              = float(sysmon_window.get("gpu_util", 0.0) or 0.0)
    out["ram_used_mb_mean"]          = float(sysmon_window.get("ram_mb", 0.0) or 0.0)
    out["temp_cpu_c_mean"]           = float(sysmon_window.get("temp_cpu_c", 0.0) or 0.0)
    out["temp_gpu_c_mean"]           = float(sysmon_window.get("temp_gpu_c", 0.0) or 0.0)
    # tcp_retransmits requires OS-level (netstat / ss) sampling we don't do.
    # tcp_connect_time_s / ttfb_s / ttlb_s are populated by build_master_row
    # from the per-detector httpx timings; do NOT zero them here.
    out["tcp_retransmits"]    = 0
    return out
