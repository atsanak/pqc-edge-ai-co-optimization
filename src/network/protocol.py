"""
Wire protocol for the split-inference link between the edge (AttentionGrid)
and the ground-station inference server.

The request is sent as ``multipart/form-data`` with two parts:

    image: bytes      JPEG-encoded crop (a supertile OR a full frame).
    meta:  JSON       InferMeta serialised as JSON.

The response is JSON containing an InferResponse.

Both ends import this module so the shapes stay in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import json


# ───────────────────────── request side ──────────────────────────

@dataclass
class InferMeta:
    """Metadata accompanying the JPEG image in an /infer POST."""

    # Detector hint: max(width, height) of the crop. The server passes this
    # to YOLO's ``imgsz=`` argument so AG's variable-resolution behaviour
    # is preserved end-to-end.
    imgsz: int

    # Absolute pixel coordinates of the crop inside the original frame:
    #   (x_offset, y_offset, crop_width, crop_height).
    # If the client is sending a full frame, this should be
    # (0, 0, frame_width, frame_height) and the edge skips the remap step.
    crop_bounds: Tuple[int, int, int, int]

    # Original frame dimensions (so the server has full context if it ever
    # needs it; today it's only used for sanity checking).
    frame_width: int
    frame_height: int

    # Logging / debug aids.
    frame_id: int = 0
    sequence: str = ""

    # Confidence threshold the edge would have used locally. The server
    # passes this through to YOLO so behaviour matches AG-local.
    conf: float = 0.25

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "InferMeta":
        d = json.loads(raw)
        # crop_bounds may arrive as a list from JSON — coerce to tuple
        d["crop_bounds"] = tuple(d["crop_bounds"])
        return cls(**d)


# ───────────────────────── response side ─────────────────────────

@dataclass
class Detection:
    """A single bounding box returned by the server.

    Coordinates are in CROP-LOCAL pixel space (not global frame coords).
    The edge client remaps to global using ``InferMeta.crop_bounds``.
    """
    bbox_xyxy: Tuple[float, float, float, float]
    cls: int
    score: float


@dataclass
class InferResponse:
    detections: List[Detection] = field(default_factory=list)

    # Server-side timing (so the edge can subtract from RTT to get
    # network-only latency).
    server_inference_ms: float = 0.0
    server_imgsz_used: int = 0

    # Echoed from the request for debugging.
    frame_id: int = 0
    sequence: str = ""

    def to_dict(self) -> dict:
        return {
            "detections": [
                {"bbox_xyxy": list(d.bbox_xyxy), "cls": d.cls, "score": d.score}
                for d in self.detections
            ],
            "server_inference_ms": self.server_inference_ms,
            "server_imgsz_used": self.server_imgsz_used,
            "frame_id": self.frame_id,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InferResponse":
        return cls(
            detections=[
                Detection(
                    bbox_xyxy=tuple(x["bbox_xyxy"]),
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


# ───────────────────── default endpoint settings ─────────────────

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8443
DEFAULT_ENDPOINT = "/infer"
DEFAULT_JPEG_QUALITY = 90  # used by the client when encoding crops
