"""
Ground-station inference server.

Accepts JPEG-encoded image crops (a supertile from AttentionGrid OR a full
frame from the remote baseline) over a TLS 1.3 channel, runs YOLO once per
request, and returns the detections.

Run on the ground station (any host with a recent enough GPU OR Apple MPS):

    cd <project_root>
    ./src/network/generate_cert.sh GROUND_STATION_IP
    PYTHONPATH=src python -m network.server        # interactive GPU/MPS picker
    PYTHONPATH=src python -m network.server --host 0.0.0.0 --port 8443 --weight yolo11s.pt

Endpoint:
    POST /infer  (multipart/form-data: image=<jpeg>, meta=<json>)
        → JSON InferResponse

Health:
    GET /healthz                                   → {"ok": true, "device": "..."}
"""

# ── Phase 0: stdlib + project path ──────────────────────────────────
import os
import sys
import io
import time
import ssl
import json
import argparse
from typing import Optional

# Make the project root importable so we can reuse gpu_utils.py and
# object_clfs/heavy_yolo_classifier.py without copy-pasting them.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Phase 1: GPU / MPS / CPU picker (BEFORE any CUDA-aware import) ──
# This is the same interactive picker every other entry-point in the
# repo uses (main.py, benchmark_baselines.py, …). On macOS it offers
# MPS; on a Jetson or a Linux box with NVIDIA GPUs it offers CUDA;
# everywhere it offers a CPU fallback.
from gpu_utils import select_gpu


def _pick_device(non_interactive: bool, forced_device: Optional[str]) -> str:
    """Either prompt the user for a device, or take it from the CLI."""
    if forced_device is not None:
        # Caller has already picked. Honor it verbatim.
        # Acceptable values: "cuda", "mps", "cpu", "0", "1", …
        print(f"[server] Device forced via --device: {forced_device}")
        return forced_device
    if non_interactive:
        # Default order in non-interactive mode: cuda > mps > cpu.
        print("[server] --no-prompt: auto-selecting best available device.")
        return "auto"
    # Interactive path: existing repo-wide picker. Sets CUDA_VISIBLE_DEVICES.
    select_gpu()
    return "auto"


# ── Phase 2: CUDA-heavy imports (CUDA_VISIBLE_DEVICES now finalised) ──
import socket
import threading
import uuid
import numpy as np
from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from object_clfs.heavy_yolo_classifier import HeavyYoloClassifier
from eval_tools.performance_eval.system_monitor import SystemMonitor
from network.protocol import (
    InferMeta,
    InferResponse,
    Detection,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_ENDPOINT,
)


# ───────────────────── detector cache ─────────────────────
#
# AttentionGrid switches detector imgsz per supertile, so a single
# server instance may serve many different imgsz values. We cache one
# HeavyYoloClassifier per (weight, imgsz) pair, exactly like AGv2 does
# on the edge.

_DETECTOR_CACHE: dict[tuple[str, int], HeavyYoloClassifier] = {}
_WEIGHT_NAME: str = "yolo11s.pt"
_DEVICE: Optional[str | int] = None
_DEVICE_DISPLAY: str = "auto"


def _get_detector(imgsz: int) -> HeavyYoloClassifier:
    """Return a HeavyYoloClassifier for this imgsz, instantiating on first use."""
    key = (_WEIGHT_NAME, imgsz)
    if key not in _DETECTOR_CACHE:
        print(f"[server] Loading {_WEIGHT_NAME} @ imgsz={imgsz} on device={_DEVICE_DISPLAY}")
        _DETECTOR_CACHE[key] = HeavyYoloClassifier(
            weight=_WEIGHT_NAME,
            imgsz=imgsz,
            device=_DEVICE,
        )
    return _DETECTOR_CACHE[key]


# ───────────────────── server-side telemetry ─────────────────────
#
# A single SystemMonitor instance starts when the server boots and samples
# continuously. Clients can open "windows" (POST /stats/window/begin) and
# later close them (POST /stats/window/end?tag=...), receiving aggregated
# power / GPU-util / CPU stats for everything that happened between.
#
# This is how the edge measures GROUND-STATION energy for the journal
# comparison: open a window before each scene, close it after, and the
# server returns the aggregates that get joined with edge-side numbers.

_SYS_MONITOR: SystemMonitor | None = None
_WINDOWS: dict[str, dict] = {}        # tag → {"start_idx": int, "t0": float}
_WINDOWS_LOCK = threading.Lock()


def _monitor_sample_count() -> int:
    if _SYS_MONITOR is None:
        return 0
    lock = getattr(_SYS_MONITOR, "_lock", None)
    if lock is not None:
        with lock:
            return len(_SYS_MONITOR.samples)
    return len(_SYS_MONITOR.samples)


def _aggregate_window(start_idx: int) -> dict:
    """Slice samples from start_idx → end, aggregate the relevant scalars."""
    if _SYS_MONITOR is None:
        return {}
    lock = getattr(_SYS_MONITOR, "_lock", None)
    if lock is not None:
        with lock:
            subset = list(_SYS_MONITOR.samples[start_idx:])
    else:
        subset = list(_SYS_MONITOR.samples[start_idx:])

    def _avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    return {
        "n_samples":   len(subset),
        "cpu_pct":     _avg([x.get("cpu_proc")    for x in subset]),
        "ram_mb":      _avg([x.get("ram_proc_mb") for x in subset]),
        "gpu_util":    _avg([x.get("gpu_util")    for x in subset]),
        "gpu_mem_mb":  _avg([x.get("gpu_mem_mb")  for x in subset]),
        "gpu_power_w": _avg([x.get("gpu_power_w") for x in subset]),
    }


# ───────────────────── FastAPI app ─────────────────────

app = FastAPI(title="AttentionGrid inference server", version="0.1.0")

# Mount PQC routes (/pqc/handshake, /pqc/infer, /pqc/sessions). The router
# reuses _get_detector() above so YOLO models stay cached across all
# transport modes. Import is lazy so a host without liboqs can still serve
# plain TLS 1.3 requests via /infer.
try:
    from network.pqc_server import router as _pqc_router
    app.include_router(_pqc_router)
    _PQC_ROUTES_OK = True
    _PQC_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001
    _PQC_ROUTES_OK = False
    _PQC_IMPORT_ERROR = _exc
    print(f"[server] WARNING: PQC routes unavailable ({_exc!r}); "
          "only TLS-1.3 /infer will work.")


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "weight": _WEIGHT_NAME,
        "device": _DEVICE_DISPLAY,
        "loaded_imgsz_cache": sorted(int(k[1]) for k in _DETECTOR_CACHE.keys()),
        "monitor_active": _SYS_MONITOR is not None,
        "pqc_routes": _PQC_ROUTES_OK,
    }


@app.post("/stats/window/begin")
def stats_window_begin(tag: str = ""):
    """Open a server-side measurement window. Returns a tag the client uses to close it."""
    if _SYS_MONITOR is None:
        raise HTTPException(status_code=503, detail="SystemMonitor unavailable on this host")
    handle = tag or uuid.uuid4().hex[:12]
    with _WINDOWS_LOCK:
        _WINDOWS[handle] = {
            "start_idx": _monitor_sample_count(),
            "t0": time.time(),
        }
    return {"tag": handle}


@app.post("/stats/window/end")
def stats_window_end(tag: str):
    """Close a measurement window and return aggregated samples."""
    if _SYS_MONITOR is None:
        raise HTTPException(status_code=503, detail="SystemMonitor unavailable on this host")
    with _WINDOWS_LOCK:
        win = _WINDOWS.pop(tag, None)
    if win is None:
        raise HTTPException(status_code=404, detail=f"No open window with tag {tag!r}")
    agg = _aggregate_window(win["start_idx"])
    agg["wall_clock_s"] = time.time() - win["t0"]
    # Approximate energy on the server side as (avg_power_W * wall_clock_s).
    # This is an upper-bound estimate (assumes constant power); for finer
    # granularity, integrate per-sample power across the window.
    if agg.get("gpu_power_w") is not None and agg["wall_clock_s"] > 0:
        agg["gpu_energy_j"] = float(agg["gpu_power_w"]) * float(agg["wall_clock_s"])
    return {"tag": tag, **agg}


@app.get("/stats/now")
def stats_now():
    """Instant snapshot of the most recent sample."""
    if _SYS_MONITOR is None or not _SYS_MONITOR.samples:
        return {"available": False}
    last = _SYS_MONITOR.samples[-1]
    return {"available": True, **last}


@app.post(DEFAULT_ENDPOINT)
async def infer(
    image: UploadFile = File(...),
    meta: str = Form(...),
):
    """Run YOLO on the uploaded crop and return detections in CROP-LOCAL coords."""

    # ── Parse metadata ────────────────────────────────────────────
    try:
        m = InferMeta.from_json(meta)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad meta JSON: {exc}") from exc

    # ── Decode image ─────────────────────────────────────────────
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad image: {exc}") from exc

    # ── Run YOLO ─────────────────────────────────────────────────
    det = _get_detector(int(m.imgsz))
    # We override the cached classifier's conf each call (cheap) so the
    # edge can ask for different conf thresholds without invalidating
    # the model cache.
    det.conf = float(m.conf)

    t0 = time.perf_counter()
    _, boxes_xywh, classes, scores = det.predict_image(pil)
    t1 = time.perf_counter()

    detections: list[Detection] = []
    if boxes_xywh is not None:
        # xywh → xyxy in crop-local pixel space
        # (matches the rest of AG's coordinate convention upstream)
        boxes_xywh = np.asarray(boxes_xywh, dtype=float)
        xyxy = boxes_xywh.copy()
        xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
        xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
        for i in range(xyxy.shape[0]):
            detections.append(
                Detection(
                    bbox_xyxy=(float(xyxy[i, 0]), float(xyxy[i, 1]),
                               float(xyxy[i, 2]), float(xyxy[i, 3])),
                    cls=int(classes[i]),
                    score=float(scores[i]),
                )
            )

    resp = InferResponse(
        detections=detections,
        server_inference_ms=(t1 - t0) * 1000.0,
        server_imgsz_used=int(m.imgsz),
        frame_id=int(m.frame_id),
        sequence=str(m.sequence),
    )
    return JSONResponse(resp.to_dict())


# ───────────────────── CLI ─────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AG inference server (TLS 1.3)")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"Bind address (default {DEFAULT_HOST}).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"Bind port (default {DEFAULT_PORT}).")
    p.add_argument("--weight", default="yolo11s.pt",
                   help="YOLO weight name (default yolo11s.pt).")
    p.add_argument("--cert", default=os.path.join(_HERE, "certs", "server.crt"),
                   help="TLS certificate path.")
    p.add_argument("--key", default=os.path.join(_HERE, "certs", "server.key"),
                   help="TLS private key path.")
    p.add_argument("--device", default=None,
                   help=("Device override: 'cuda', 'mps', 'cpu', or an integer "
                         "GPU index. If omitted, the interactive selector runs."))
    p.add_argument("--no-prompt", action="store_true",
                   help="Skip interactive device prompt and auto-pick the best.")
    p.add_argument("--no-tls", action="store_true",
                   help=("Disable TLS — serve over plain HTTP on the same port. "
                         "Use only for local debugging."))
    return p.parse_args()


def _resolve_device_arg(forced: Optional[str]) -> Optional[str | int]:
    """Translate a CLI --device string into the value HeavyYoloClassifier expects."""
    if forced is None:
        return None  # let HeavyYoloClassifier auto-detect
    s = forced.strip().lower()
    if s in ("cpu", "mps"):
        return s
    if s == "cuda":
        return 0
    if s == "auto":
        return None
    if s.isdigit():
        return int(s)
    raise ValueError(f"Unrecognised --device value: {forced}")


def _client_facing_ipv4s(bind_host: str) -> list[str]:
    """Return IPv4 addresses a remote client should dial.

    If the user bound to a specific address, just echo it back.
    For 0.0.0.0 (or empty), enumerate all interfaces and drop loopback
    and docker bridges so the printed URLs are ones a Jetson can actually use.
    """
    if bind_host and bind_host not in ("0.0.0.0", "::", ""):
        return [bind_host]

    candidates: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = info[4][0]
            if ip and ip not in candidates:
                candidates.append(ip)
    except socket.gaierror:
        pass

    # Fallback: ask the kernel which source IP it would pick for an outside dst.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and ip not in candidates:
                candidates.append(ip)
    except OSError:
        pass

    def _keep(ip: str) -> bool:
        if ip.startswith("127."):
            return False
        octets = ip.split(".")
        if len(octets) == 4 and octets[0] == "172" and octets[1] == "17":
            return False
        return True

    filtered = [ip for ip in candidates if _keep(ip)]
    return filtered or candidates or ["127.0.0.1"]


def _describe_device(device_val) -> str:
    if device_val is None:
        if torch.cuda.is_available():
            return "cuda:0 (auto)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (auto)"
        return "cpu (auto)"
    return str(device_val)


def main():
    global _WEIGHT_NAME, _DEVICE, _DEVICE_DISPLAY, _SYS_MONITOR

    args = _parse_args()
    _WEIGHT_NAME = args.weight

    # Device picker (interactive unless overridden).
    _pick_device(non_interactive=args.no_prompt, forced_device=args.device)
    _DEVICE = _resolve_device_arg(args.device)
    _DEVICE_DISPLAY = _describe_device(_DEVICE)

    # Spin up the continuous SystemMonitor *before* loading YOLO so
    # warm-up samples don't show as part of any benchmark window.
    try:
        _SYS_MONITOR = SystemMonitor(interval_ms=100)
        _SYS_MONITOR.start()
        print(f"[server] SystemMonitor active — {_SYS_MONITOR.get_gpu_info_string()}")
    except Exception as e:  # pragma: no cover
        print(f"[server] SystemMonitor unavailable: {e}")
        _SYS_MONITOR = None

    # Pre-warm the default imgsz so the first request isn't slow.
    _get_detector(640)

    # TLS context (TLS 1.3 only) unless --no-tls explicitly set.
    ssl_kwargs = {}
    if not args.no_tls:
        if not os.path.isfile(args.cert) or not os.path.isfile(args.key):
            print("[server] ERROR: cert/key not found.\n"
                  f"  cert={args.cert}\n  key ={args.key}\n"
                  "Run ./src/network/generate_cert.sh first, or pass --no-tls for "
                  "plaintext debugging.")
            sys.exit(2)
        ssl_kwargs = dict(
            ssl_certfile=args.cert,
            ssl_keyfile=args.key,
            ssl_version=ssl.PROTOCOL_TLS_SERVER,
        )
        scheme = "https"
    else:
        scheme = "http"

    client_ips = _client_facing_ipv4s(args.host)
    primary_url = f"{scheme}://{client_ips[0]}:{args.port}{DEFAULT_ENDPOINT}"

    print()
    print("=" * 64)
    print(f"  AG inference server")
    print(f"    weight : {_WEIGHT_NAME}")
    print(f"    device : {_DEVICE_DISPLAY}")
    print(f"    bind   : {scheme}://{args.host}:{args.port}{DEFAULT_ENDPOINT}")
    print(f"    health : {scheme}://{args.host}:{args.port}/healthz")
    print("-" * 64)
    print(f"  Paste this at the client's 'Server URL' prompt:")
    print(f"    {primary_url}")
    if len(client_ips) > 1:
        print(f"  Other reachable addresses on this host:")
        for ip in client_ips[1:]:
            print(f"    {scheme}://{ip}:{args.port}{DEFAULT_ENDPOINT}")
    print("=" * 64)
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
