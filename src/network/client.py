"""
Edge-side client for the inference server.

`RemoteDetector` is a drop-in replacement for `HeavyYoloClassifier`:
identical `predict_image(PIL.Image)` signature and identical return shape
``(annotated_pil, boxes_xywh, classes, scores)``. The only difference is
that the actual forward pass happens on a remote ground-station GPU/MPS
device, over TLS 1.3.

Usage in code:

    from network.client import RemoteDetector

    det = RemoteDetector(
        url="https://GROUND_STATION_IP:8443/infer",
        imgsz=640,
        cafile="src/network/certs/server.crt",
    )
    annotated_pil, boxes_xywh, classes, scores = det.predict_image(pil_img)

The class also tracks per-stage timing so the new benchmark scripts can
report ``client_encode_ms``, ``network_rtt_ms``, ``server_inference_ms``,
and total bytes-on-wire per call.
"""

from __future__ import annotations

import io
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import httpx
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RemoteDetector requires `httpx`. Add it to requirements.txt:\n"
        "    httpx[http2]\n"
    ) from e

from network.protocol import (
    InferMeta,
    InferResponse,
    DEFAULT_JPEG_QUALITY,
)


class RemoteDetector:
    """HeavyYoloClassifier-compatible front-end that posts to a remote server."""

    def __init__(
        self,
        url: str,
        imgsz: int = 640,
        conf: float = 0.25,
        cafile: Optional[str] = None,
        verify_tls: bool = True,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        timeout_s: float = 15.0,
        sequence_name: str = "",
    ):
        """
        Args:
            url:          Full POST URL, e.g. "https://GROUND_STATION_IP:8443/infer".
            imgsz:        Detector input size hint sent to the server.
            conf:         Confidence threshold sent to the server.
            cafile:       Path to the server's TLS certificate (for self-signed dev).
            verify_tls:   If False, skip cert validation (dev only).
            jpeg_quality: 1–100. 90 is a good detection-quality / size trade-off.
            timeout_s:    Per-request hard timeout.
            sequence_name: Echoed back from the server for logging.
        """
        self.url = url
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.jpeg_quality = int(jpeg_quality)
        self.sequence_name = sequence_name

        # Default to verifying the self-signed cert by trusting the provided
        # cafile. Setting verify_tls=False disables verification entirely
        # (only for local debugging).
        if not verify_tls:
            verify_arg: bool | str = False
        elif cafile:
            verify_arg = cafile
        else:
            verify_arg = True  # use system CA bundle

        # Persistent HTTP/2 client over TLS 1.3 — reuses a single
        # TLS connection across frames, which saves a full handshake on
        # every call. Critical for per-frame call rates ≥ a few Hz.
        self._client = httpx.Client(
            http2=True,
            verify=verify_arg,
            timeout=timeout_s,
        )

        # Running counters (read by benchmark scripts).
        self.total_yolo_time = 0.0           # mirrors HeavyYoloClassifier
        self.total_client_encode_ms = 0.0
        self.total_network_rtt_ms = 0.0      # request_sent → response_received
        self.total_server_infer_ms = 0.0
        self.total_bytes_uploaded = 0
        self.total_bytes_downloaded = 0
        self.num_requests = 0

        # The most recent crop bounds + frame size, set externally before
        # each predict_image() call (since the AGv2 API doesn't pass them in).
        # Defaults assume "whole image is the crop" (no-op remap), which makes
        # this safe to use as a remote BASELINE detector too.
        self._next_crop_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._next_frame_size: Tuple[int, int] = (0, 0)
        self._next_frame_id: int = 0

    # ────────────────── per-call metadata setters ──────────────────
    #
    # AGv2 calls `predict_image(pil)` with no extra context. To preserve
    # that interface we let the caller stash the metadata for the next
    # call via these setters. If the caller doesn't set them, we infer
    # crop_bounds from the PIL image's size (treating the crop AS the
    # whole frame — correct behaviour for the remote baseline).

    def set_next_call_context(
        self,
        crop_bounds: Tuple[int, int, int, int],
        frame_size: Tuple[int, int],
        frame_id: int = 0,
    ) -> None:
        self._next_crop_bounds = tuple(crop_bounds)  # type: ignore[assignment]
        self._next_frame_size = tuple(frame_size)    # type: ignore[assignment]
        self._next_frame_id = int(frame_id)

    # ────────────────── main entry point ──────────────────
    def predict_image(self, img_pil: Image.Image):
        """Drop-in replacement for HeavyYoloClassifier.predict_image()."""
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # JPEG encode
        t_enc0 = time.perf_counter()
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        jpeg_bytes = buf.getvalue()
        t_enc1 = time.perf_counter()

        # Resolve crop bounds (use caller-provided context if set, else self).
        if self._next_frame_size == (0, 0):
            fw, fh = img_pil.size
        else:
            fw, fh = self._next_frame_size
        if self._next_crop_bounds == (0, 0, 0, 0):
            crop_bounds = (0, 0, img_pil.size[0], img_pil.size[1])
        else:
            crop_bounds = self._next_crop_bounds

        meta = InferMeta(
            imgsz=self.imgsz,
            crop_bounds=crop_bounds,
            frame_width=fw,
            frame_height=fh,
            frame_id=self._next_frame_id,
            sequence=self.sequence_name,
            conf=self.conf,
        )

        # Network round-trip
        t_net0 = time.perf_counter()
        files = {"image": ("crop.jpg", jpeg_bytes, "image/jpeg")}
        data = {"meta": meta.to_json()}
        try:
            resp = self._client.post(self.url, files=files, data=data)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            # Surface network errors cleanly. AGv2 currently has no fallback
            # path; the calling benchmark should catch + degrade gracefully.
            raise RuntimeError(f"RemoteDetector request failed: {e}") from e
        t_net1 = time.perf_counter()

        # Parse
        ir = InferResponse.from_dict(resp.json())

        # Update timers / byte counters
        encode_ms = (t_enc1 - t_enc0) * 1000.0
        rtt_ms = (t_net1 - t_net0) * 1000.0
        self.total_client_encode_ms += encode_ms
        self.total_network_rtt_ms += rtt_ms
        self.total_server_infer_ms += ir.server_inference_ms
        self.total_bytes_uploaded += len(jpeg_bytes)
        self.total_bytes_downloaded += len(resp.content)
        self.num_requests += 1
        # Match HeavyYoloClassifier's "total inference time" semantics:
        # the time AGv2 attributes to "running YOLO" is now the
        # end-to-end round-trip, since that's what blocks the edge.
        self.total_yolo_time += (t_net1 - t_net0)

        # Reset per-call stash so the next call doesn't accidentally
        # reuse stale crop bounds.
        self._next_crop_bounds = (0, 0, 0, 0)
        self._next_frame_size = (0, 0)
        self._next_frame_id = 0

        # ── Build the return tuple in the HeavyYoloClassifier shape ──
        if not ir.detections:
            # Mimic HeavyYoloClassifier: return the original image as the
            # "annotated" frame (no boxes to draw locally without round-trip).
            return img_pil, None, None, None

        n = len(ir.detections)
        xyxy = np.zeros((n, 4), dtype=np.float32)
        classes = np.zeros((n,), dtype=np.int32)
        scores = np.zeros((n,), dtype=np.float32)
        for i, d in enumerate(ir.detections):
            xyxy[i] = d.bbox_xyxy
            classes[i] = d.cls
            scores[i] = d.score

        # xyxy → xywh to match HeavyYoloClassifier's contract
        xywh = xyxy.copy()
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

        # We deliberately do NOT round-trip the annotated image back from
        # the server (would 2× the bandwidth). Callers that need visual
        # output can draw boxes locally using a lightweight utility.
        return img_pil, xywh, classes, scores

    def predict_batch(self, images: list):
        """API-compat fallback for HeavyYoloClassifier.predict_batch().

        Runs N sequential predict_image() calls. The remote inference path
        does not implement true server-side batching, so this is a
        per-image loop. Required so AG's recheck path (which calls
        yolo.predict_batch when LKT is disabled, attention_gridv2.py:2079)
        does not raise AttributeError under cloud_ag mode.
        """
        return [self.predict_image(img) for img in images]

    # ────────────────── housekeeping ──────────────────
    def _get_total_yolo_time(self) -> float:
        """Mirror HeavyYoloClassifier's method (used by AGv2 for FPS calc)."""
        return self.total_yolo_time

    def reset_counters(self) -> None:
        self.total_yolo_time = 0.0
        self.total_client_encode_ms = 0.0
        self.total_network_rtt_ms = 0.0
        self.total_server_infer_ms = 0.0
        self.total_bytes_uploaded = 0
        self.total_bytes_downloaded = 0
        self.num_requests = 0

    def stats_summary(self) -> dict:
        n = max(self.num_requests, 1)
        return {
            "num_requests":          self.num_requests,
            "avg_encode_ms":         self.total_client_encode_ms / n,
            "avg_rtt_ms":            self.total_network_rtt_ms / n,
            "avg_server_inference_ms": self.total_server_infer_ms / n,
            "avg_upload_kB":         (self.total_bytes_uploaded   / n) / 1024.0,
            "avg_download_kB":       (self.total_bytes_downloaded / n) / 1024.0,
            "total_upload_MB":       self.total_bytes_uploaded   / (1024.0 ** 2),
            "total_download_MB":     self.total_bytes_downloaded / (1024.0 ** 2),
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ────────────────── server-side stat windows ──────────────────
    #
    # The server runs a continuous SystemMonitor. These helpers open and
    # close a "window" so the client can ask for aggregated power /
    # GPU-util / energy on the ground station between two events.

    def begin_server_window(self, tag: str = "") -> str:
        """Open a server-side measurement window. Returns the tag the server assigned."""
        base = self.url.rsplit("/", 1)[0]  # strip /infer
        r = self._client.post(f"{base}/stats/window/begin", params={"tag": tag})
        r.raise_for_status()
        return r.json()["tag"]

    def end_server_window(self, tag: str) -> dict:
        """Close the window and return aggregated server-side stats."""
        base = self.url.rsplit("/", 1)[0]
        r = self._client.post(f"{base}/stats/window/end", params={"tag": tag})
        r.raise_for_status()
        return r.json()

    def server_health(self) -> dict:
        """GET /healthz on the server."""
        base = self.url.rsplit("/", 1)[0]
        r = self._client.get(f"{base}/healthz")
        r.raise_for_status()
        return r.json()
