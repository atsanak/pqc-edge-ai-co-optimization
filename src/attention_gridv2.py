
import cv2
from PIL import Image
import numpy as np
from collections import deque, Counter, defaultdict
import os
import csv
import time



from img_tiler import SimpleImageTiler
from saliency_methods.frame_diff import FrameDiff
from saliency_methods.frame_diff_kmeans import FrameDiffKMeans, DEFAULT_KMEANS_FD_PARAMS
from saliency_methods.optical_flow import OpticalFlow
from saliency_methods.temporal_phase import TemporalPhaseSaliency, DEFAULT_TPS_PARAMS
from saliency_methods.hybrid_saliency import HybridSaliency, DEFAULT_HYBRID_PARAMS

# Optional deep learning models — only load if dependencies are installed
try:
    from saliency_methods.u2net_saliency import U2NetSaliency, DEFAULT_U2_PARAMS
except (ImportError, ModuleNotFoundError):
    U2NetSaliency = None
    DEFAULT_U2_PARAMS = {}

try:
    from saliency_methods.deva_saliency import DevaSaliency, DEFAULT_DEVA_PARAMS
except (ImportError, ModuleNotFoundError):
    DevaSaliency = None
    DEFAULT_DEVA_PARAMS = {}

try:
    from saliency_methods.inspyrenet_saliency import InSPyReNetSaliency, DEFAULT_INSPYRENET_PARAMS
except (ImportError, ModuleNotFoundError):
    InSPyReNetSaliency = None
    DEFAULT_INSPYRENET_PARAMS = {}

try:
    from saliency_methods.poolnet_saliency import PoolNetSaliency, DEFAULT_POOLNET_PARAMS
except (ImportError, ModuleNotFoundError):
    PoolNetSaliency = None
    DEFAULT_POOLNET_PARAMS = {}

from object_clfs.heavy_yolo_classifier import HeavyYoloClassifier # YOLOV11


class TrackedObject:
    """Instance-level tracked object with unique ID for LKT tracking."""
    __slots__ = (
        'obj_id', 'class_id', 'confidence', 'bbox', 'feature_points',
        'velocity', 'saliency_footprint', 'current_tile', 'source_tile',
        'last_yolo_frame', 'frames_since_yolo', 'active',
    )

    def __init__(self, obj_id: int, class_id: int, confidence: float,
                 bbox: np.ndarray, current_tile: tuple, last_yolo_frame: int,
                 feature_points=None, saliency_footprint: float = 0.0):
        self.obj_id = obj_id
        self.class_id = int(class_id)
        self.confidence = float(confidence)
        self.bbox = np.array(bbox, dtype=np.float32).ravel()[:4]  # (4,) xywh global
        self.feature_points = feature_points  # (K, 1, 2) float32 or None
        self.velocity = (0.0, 0.0)  # (dx, dy) smoothed
        self.saliency_footprint = float(saliency_footprint)  # baseline saliency inside bbox at detection time
        self.current_tile = current_tile  # (r, c)
        self.source_tile = current_tile   # (r, c) previous tile (for direction check)
        self.last_yolo_frame = int(last_yolo_frame)
        self.frames_since_yolo = 0
        self.active = True


class AttentionGrid:
    def __init__(self,
                 rows: int = 3,
                 cols: int = 3,
                 enable_recheck_tile: bool = True,
                 recheck_threshold: int = 3,
                 saliency_method: str = "frame_diff",
                 saliency_measurement: str = "pixel_count",
                 enable_tile_combination: bool = True,
                 max_combined_tiles: int = 4,
                 yolo_weight: str = "yolo11s.pt",
                 merge_add_tile_motion_pct: float = 0.00,  # 0.0 => current behavior (always expand)
                 use_finetuned: bool = False,  # Use fine-tuned YOLO weights
                 enable_tile_memory: bool = False,  # Enable tile movement/object frequency memory
                 device: str = None,  # "cuda:0", "cuda:1", "mps", "cpu", or None (auto)
                 yolo_run_interval: int = 1,  # Run YOLO every N frames (1 = every frame, 5 = every 5th). Between runs, cached/LKT-tracked detections are used.
                 fullframe_every: int = 0,  # Run YOLO on full frame every N frames to refresh caches (0 = disabled)
                 saliency_scale: float = 1.0,  # Downscale factor for saliency (0.5 = half res, faster)
                 prediction_fusion: bool = True,  # Fuse cut-off predictions across neighboring tiles
                 fusion_edge_margin_pct: float = 0.02,  # Edge margin as % of tile dim for fusion (default 2%)
                 enable_lkt_tracking: bool = False,  # Lucas-Kanade-Tomasi optical flow tracking between YOLO runs
                 lkt_min_points: int = 4,  # Minimum tracked points per bbox; object is dropped if below this
                 lkt_quality_level: float = 0.01,  # Shi-Tomasi corner quality threshold
                 lkt_max_corners: int = 15,  # Max feature points per bounding box
                 lkt_iou_match_threshold: float = 0.3,  # IoU threshold for matching YOLO detections to tracked objects
                 lkt_max_drift_frames: int = 60,  # Max frames an object can be tracked without YOLO before expiry
                 lkt_velocity_smoothing: float = 0.5,  # EMA factor for velocity smoothing (higher = more responsive)
                 class_filter: dict = None,  # e.g. {"car": True, "person": True} — only keep these classes
                 enable_saliency_suppression: bool = True,  # Dampen saliency for tiles that repeatedly show high motion but no detections
                 saliency_suppression_rate: float = 1.0,  # How much dampening to add per high-saliency miss (higher = suppress faster)
                 saliency_suppression_decay: float = 0.05,  # Per-frame decay of dampening (lower = longer suppression)
                 enable_attention_priority: bool = False,  # Prioritise stale salient tiles over recently-visited ones
                 attention_stale_threshold: int = 5,  # Frames since last visit before a salient tile is considered "stale"
                 # ── Split-inference (edge → ground-station) options ───────────
                 inference_mode: str = "local",   # "local": run YOLO on this device. "remote": POST crops to a ground-station server over TLS.
                 remote_url: str = "https://127.0.0.1:8443/infer",  # used only when inference_mode == "remote"
                 remote_cafile: str = None,       # path to server's TLS cert (self-signed dev)
                 remote_verify_tls: bool = True,  # set False to skip cert verification (debug only)
                 remote_jpeg_quality: int = 90,   # 1-100. Higher = larger payload, lossless trend.
                 remote_timeout_s: float = 15.0,  # per-request hard timeout
                 # ── Transport-security selection ─────────────────────────────
                 # "tls13"     = existing /infer endpoint, TLS 1.3 only (default)
                 # "classical" = application-layer ECDH+ECDSA over TLS 1.3
                 # "pqc"       = hybrid ECDH+ML-KEM + ECDSA+ML-DSA over TLS 1.3
                 crypto_mode: str = "tls13",
                 pqc_kem_scheme: str = "ML-KEM-768",
                 pqc_sig_scheme: str = "ML-DSA-44",
                 **kwargs,  # Accept and ignore removed params (blacklist_ttl, motion_unblacklist_pct, merge_overlapping_bboxes)
                 ):
        
        self.rows = rows
        self.cols = cols
        self.enable_recheck_tile = enable_recheck_tile
        self.recheck_threshold = recheck_threshold
        self.saliency_method = saliency_method
        # Backward-compat: treat legacy "mask" as "pixel_count"
        if saliency_measurement == "mask":
            saliency_measurement = "pixel_count"
        self.saliency_measurement = saliency_measurement
        self.enable_tile_combination = enable_tile_combination
        self.max_combined_tiles = max_combined_tiles
        self.yolo_weight = yolo_weight
        self.merge_add_tile_motion_pct = float(merge_add_tile_motion_pct)
        self.use_finetuned = use_finetuned
        self.enable_tile_memory = enable_tile_memory
        self.device = device  # propagated to HeavyYoloClassifier
        self.yolo_run_interval = max(1, int(yolo_run_interval))
        self.fullframe_every = int(fullframe_every)  # 0 = disabled
        self.saliency_scale = float(max(0.1, min(1.0, saliency_scale)))  # clamp to [0.1, 1.0]
        self.prediction_fusion = prediction_fusion
        self.fusion_edge_margin_pct = float(fusion_edge_margin_pct)
        self.enable_lkt_tracking = enable_lkt_tracking
        self.lkt_min_points = int(lkt_min_points)
        self.lkt_quality_level = float(lkt_quality_level)
        self.lkt_max_corners = int(lkt_max_corners)
        self.lkt_iou_match_threshold = float(lkt_iou_match_threshold)
        self.lkt_max_drift_frames = int(lkt_max_drift_frames)
        self.lkt_velocity_smoothing = float(max(0.0, min(1.0, lkt_velocity_smoothing)))
        self.class_filter = class_filter  # kept as-is; resolved to IDs after coco_classes loaded
        self.enable_saliency_suppression = enable_saliency_suppression
        self.saliency_suppression_rate = float(saliency_suppression_rate)
        self.saliency_suppression_decay = float(saliency_suppression_decay)
        self.enable_attention_priority = enable_attention_priority
        self.attention_stale_threshold = max(1, int(attention_stale_threshold))

        # ── Split-inference settings (only used when inference_mode == "remote") ──
        inference_mode = (inference_mode or "local").strip().lower()
        if inference_mode not in ("local", "remote"):
            raise ValueError(
                f"inference_mode must be 'local' or 'remote', got {inference_mode!r}"
            )
        self.inference_mode = inference_mode
        self.remote_url = remote_url
        self.remote_cafile = remote_cafile
        self.remote_verify_tls = bool(remote_verify_tls)
        self.remote_jpeg_quality = int(remote_jpeg_quality)
        self.remote_timeout_s = float(remote_timeout_s)

        crypto_mode = (crypto_mode or "tls13").strip().lower()
        if crypto_mode not in ("tls13", "classical", "pqc"):
            raise ValueError(
                f"crypto_mode must be 'tls13', 'classical', or 'pqc', got {crypto_mode!r}"
            )
        self.crypto_mode = crypto_mode
        self.pqc_kem_scheme = pqc_kem_scheme
        self.pqc_sig_scheme = pqc_sig_scheme


        # Load COCO class names for labeling
        self.coco_classes = self._load_coco_classes()

        # Build allowed class-ID set from class_filter dict
        # class_filter can be: {"car": True, "person": True} or ["car", "person"]
        self._allowed_class_ids = None  # None means "keep all"
        if self.class_filter is not None and self.coco_classes is not None:
            names = set()
            if isinstance(self.class_filter, dict):
                names = {k.lower() for k, v in self.class_filter.items() if v}
            elif isinstance(self.class_filter, (list, tuple, set)):
                names = {str(n).lower() for n in self.class_filter}
            if names:
                self._allowed_class_ids = set()
                for idx, cname in enumerate(self.coco_classes):
                    if cname.lower() in names:
                        self._allowed_class_ids.add(idx)

        self.first_run = True
        self.yolo_ran_this_frame = False  # set per frame; used by visualization
        self.tile_width = None
        self.tile_height = None
        self.max_dim = None # Max dimension among width and height
        self.total_area = None # Total area of the tile


        self.max_merge_memory = 300
        self.yolo_clfs: dict[int, HeavyYoloClassifier] = {}
        self.tiler = SimpleImageTiler(rows=self.rows, cols=self.cols)
        self.frame_memory = deque(maxlen=250) # maxlen dermines how many previous frames to store
        self.frame_counter = 0
        self.tile_array = np.empty((self.rows, self.cols), dtype=object)
        # merged_tiles_memory: dict(frame_num: {"merged_tiles_idx_map": np.ndarray, "merged_tiles_img": PIL.Image.Image})
        self.merged_tiles_memory = {}

        # Per-tile cache (predictions stay on a tile until tile is revisited)
        # We persist DETECTIONS ONLY (global coords), NOT annotated pixels.
        # (r,c) -> {"boxes": (N,4) xywh global, "classes": (N,), "scores": (N,)}
        # N: # of detected objects on that tile
        self.tile_det_cache = {
            (r, c): {
                "boxes": np.zeros((0, 4), dtype=np.float32), # global xywh pixel coords
                "classes": np.zeros((0,), dtype=np.int64), # class ids (N objects in tile -> N class ids)
                "scores": np.zeros((0,), dtype=np.float32), # confidence scores (N objects in tile -> N scores)
            }
            for r in range(self.rows) for c in range(self.cols)
        }


        # Track last time each tile was visited. Contains integer frame counters (self.frame_counter).
        # type: dict[tuple[int, int], int]
        self.tile_last_visited = {(r, c): 0 for r in range(self.rows) for c in range(self.cols)}

        # For visualization: which tiles were rechecked on THIS frame (blue highlight)
        self.rechecked_tiles_current = set()

        # ---- Saliency suppression: per-tile dampening for "high saliency, no objects" tiles ----
        # Dampening value accumulates on misses, decays every frame.
        # Effective saliency = raw_score / (1 + dampening)
        self._saliency_dampening = {
            (r, c): 0.0 for r in range(self.rows) for c in range(self.cols)
        }

        # ---- Attention priority: stale salient tiles get seed priority ----
        # No extra state needed — uses self.tile_last_visited which already exists.

        # ---- Timing accumulators ----
        self.total_saliency_time = 0.0   # Frame differencing / optical flow etc.
        self.total_yolo_time = 0.0       # All YOLO inference (merged + single tile)
        self.total_merge_tiles_time = 0.0  # _merge_tiles_based_on_scoremap
        self.total_tiler_time = 0.0      # Image tiling operations
        self.total_overlay_time = 0.0    # reconstruct_grid_image + visualize_overlay
        self.total_lkt_time = 0.0        # Lucas-Kanade-Tomasi tracking time
        self.lkt_track_count = 0         # Number of frames where LKT was used instead of YOLO
        self.yolo_preds = 0              # Number of YOLO forward passes

        # ---- Tile Memory (movement frequency & object frequency per tile) ----
        # Only initialized if enable_tile_memory=True to save memory
        if self.enable_tile_memory:
            # tile_movement_memory: tracks how often each tile had movement
            # (r, c) -> {"movement_count": int, "total_movement_area": float}
            self.tile_movement_memory = {
                (r, c): {
                    "movement_count": 0,       # Number of frames where tile had movement
                    "total_movement_area": 0.0,  # Cumulative movement area (pixels)
                }
                for r in range(self.rows) for c in range(self.cols)
            }
            
            # tile_object_memory: tracks frequency of each object class detected per tile
            # (r, c) -> Counter({class_id: count, ...})
            self.tile_object_memory = {
                (r, c): Counter()
                for r in range(self.rows) for c in range(self.cols)
            }
            
            # Tile memory enabled (prints removed for performance)
        else:
            self.tile_movement_memory = None
            self.tile_object_memory = None

        # ---- LKT Tracking State (object-centric) ----
        # Tracked objects registry: obj_id -> TrackedObject
        self._tracked_objects: dict[int, TrackedObject] = {}
        self._next_obj_id: int = 1
        # Per-tile index of active object IDs for fast lookup
        self._tile_object_ids: dict[tuple, set] = {
            (r, c): set() for r in range(self.rows) for c in range(self.cols)
        }
        # Previous frame as grayscale (set each frame for LK flow)
        self._lkt_prev_gray = None

        # LK optical flow parameters (pyramidal Lucas-Kanade)
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Shi-Tomasi feature extraction parameters
        self._lkt_feature_params = dict(
            maxCorners=self.lkt_max_corners,
            qualityLevel=self.lkt_quality_level,
            minDistance=7,
            blockSize=7,
        )


    def _load_coco_classes(self):
        """Load COCO class names from file."""
        classes_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "object_clfs",
            "coco_classes.txt",
        )
        try:
            with open(classes_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]
            return classes
        except FileNotFoundError:
            print("[WARN] coco_classes.txt not found, using class indices")
            return None

    def pil_to_bgr(self, img: Image.Image) -> np.ndarray:
        """Convert PIL RGB image to OpenCV BGR uint8 numpy array."""
        if not isinstance(img, Image.Image):
            raise TypeError(f"pil_to_bgr expects PIL.Image.Image, got {type(img)}")
        rgb = np.array(img)  # (H,W,3) RGB
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ---- Tile Memory Methods ----
    def _update_tile_movement_memory(self, tiles_score_map: np.ndarray):
        """
        Update tile_movement_memory based on saliency scores.
        Called after computing tiles_score_map for each frame.
        """
        if not self.enable_tile_memory or self.tile_movement_memory is None:
            return
        
        for r in range(self.rows):
            for c in range(self.cols):
                movement_area = float(tiles_score_map[r, c])
                if movement_area > 0:
                    self.tile_movement_memory[(r, c)]["movement_count"] += 1
                    self.tile_movement_memory[(r, c)]["total_movement_area"] += movement_area

    def _update_tile_object_memory(self, tile_r: int, tile_c: int, classes: np.ndarray):
        """
        Update tile_object_memory with detected object classes.
        Called after YOLO detections are assigned to a tile.
        """
        if not self.enable_tile_memory or self.tile_object_memory is None:
            return
        
        if classes is not None and len(classes) > 0:
            self.tile_object_memory[(tile_r, tile_c)].update(classes.tolist())

    def print_tile_memory_summary(self):
        """
        Print summary of tile movement and object frequency memory.
        """
        if not self.enable_tile_memory:
            print("[Tile Memory] DISABLED - no data collected")
            return
        
        print("\n" + "=" * 60)
        print("  TILE MEMORY SUMMARY")
        print("=" * 60)
        
        # Movement frequency
        print("\n--- Movement Frequency per Tile ---")
        print(f"{'Tile (r,c)':<12} {'Movement Count':<18} {'Total Area':<15}")
        print("-" * 45)
        for r in range(self.rows):
            for c in range(self.cols):
                mem = self.tile_movement_memory[(r, c)]
                if mem["movement_count"] > 0:
                    print(f"({r},{c})        {mem['movement_count']:<18} {mem['total_movement_area']:<15.0f}")
        
        # Object frequency per tile
        print("\n--- Object Frequency per Tile ---")
        for r in range(self.rows):
            for c in range(self.cols):
                obj_counts = self.tile_object_memory[(r, c)]
                if obj_counts:
                    print(f"\nTile ({r},{c}):")
                    for class_id, count in obj_counts.most_common():
                        class_name = self.coco_classes[class_id] if self.coco_classes and class_id < len(self.coco_classes) else f"class_{class_id}"
                        print(f"  {class_name}: {count}")
        
        print("\n" + "=" * 60)

    def get_tile_memory(self):
        """
        Return tile memory data structures for external use.
        Returns (tile_movement_memory, tile_object_memory) or (None, None) if disabled.
        """
        return self.tile_movement_memory, self.tile_object_memory


    # =========================================================================
    # SALIENCY SUPPRESSION
    # =========================================================================

    def _decay_saliency_suppression(self):
        """
        Reduce all per-tile dampening values each frame so that suppression
        naturally fades over time.  A tile suppressed with dampening D will
        recover to 0 in roughly ``D / decay`` frames.
        """
        if not self.enable_saliency_suppression:
            return
        decay = self.saliency_suppression_decay
        for key in self._saliency_dampening:
            if self._saliency_dampening[key] > 0.0:
                self._saliency_dampening[key] = max(
                    0.0, self._saliency_dampening[key] - decay
                )

    def _apply_saliency_suppression(self, tiles_score_map: np.ndarray):
        """
        Modify ``tiles_score_map`` in-place by dividing each tile's score
        by ``(1 + dampening)``.  A dampening of 0 leaves the score unchanged;
        a dampening of 1 halves the score; a dampening of 3 quarters it; etc.
        """
        if not self.enable_saliency_suppression:
            return
        for r in range(self.rows):
            for c in range(self.cols):
                d = self._saliency_dampening.get((r, c), 0.0)
                if d > 0.0:
                    tiles_score_map[r, c] = float(tiles_score_map[r, c]) / (1.0 + d)

    # ---- Attention priority ------------------------------------------------
    def _is_edge_tile(self, r: int, c: int) -> bool:
        """True if tile (r, c) sits on the grid boundary."""
        return r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1

    def _pick_priority_seed(self, tiles_score_map: np.ndarray):
        """
        Choose the seed tile for rectangle expansion.

        Normal behaviour (enable_attention_priority=False):
            Returns None → _merge_tiles_based_on_scoremap uses its default
            argmax seed.

        Priority behaviour (enable_attention_priority=True):
            Only considers EDGE tiles (tiles touching the grid boundary) —
            these are where new objects enter the frame.

            1. Collect edge tiles that are salient (score > 0).
            2. Among those, find "stale" ones: not visited in the last
               attention_stale_threshold frames.
            3. If any stale salient edge tiles exist, return the one with
               the highest saliency score as the seed override.
            4. Otherwise return None (fall back to global argmax).

        Interior tiles are never overridden — they always compete via
        normal argmax. This keeps the system focused on the strongest
        signal unless a neglected edge tile has new movement (likely a
        new object entering the frame).

        Returns (row, col) or None.
        """
        if not self.enable_attention_priority:
            return None

        threshold = self.attention_stale_threshold
        best_seed = None
        best_score = -1.0

        for r in range(self.rows):
            for c in range(self.cols):
                if not self._is_edge_tile(r, c):
                    continue
                score = float(tiles_score_map[r, c])
                if score <= 0:
                    continue
                frames_since = self.frame_counter - self.tile_last_visited.get((r, c), 0)
                if frames_since >= threshold and score > best_score:
                    best_score = score
                    best_seed = (r, c)

        return best_seed

    def _update_saliency_suppression_after_yolo(
        self,
        rmin: int, rmax: int, cmin: int, cmax: int,
        raw_tiles_score_map: np.ndarray,
    ):
        """
        After YOLO runs on the saliency region [rmin..rmax, cmin..cmax],
        update the per-tile dampening:

        - **No detections in tile + tile had meaningful raw saliency** →
          increase dampening by ``saliency_suppression_rate``.  The tile is
          "noisy" (lots of motion but nothing useful).
        - **Detections found in tile** → reset dampening to 0.  The tile has
          real objects; it should keep its full saliency priority.

        ``raw_tiles_score_map`` must be the **original** score map produced by
        ``_select_tiles_from_mask`` BEFORE any suppression penalty has been
        applied.  This ensures the "meaningful saliency" check
        reflects actual motion, not an already-dampened value.

        "Meaningful saliency" is defined as saliency > 1% of tile area.  This
        avoids penalising tiles with trivial residual pixel noise.
        """
        if not self.enable_saliency_suppression:
            return

        # Threshold: 1% of tile pixel area
        tile_area = 1.0
        if self.total_area is not None:
            tile_area = float(self.total_area)
        min_saliency = 0.01 * tile_area

        rate = self.saliency_suppression_rate

        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                det = self.tile_det_cache.get((r, c))
                has_det = (det is not None and
                           det.get("boxes") is not None and
                           len(det["boxes"]) > 0)
                if has_det:
                    # Objects found → reset dampening (this tile is valuable)
                    self._saliency_dampening[(r, c)] = 0.0
                else:
                    # No objects — only penalise if tile had meaningful saliency
                    saliency = float(raw_tiles_score_map[r, c])
                    if saliency >= min_saliency:
                        self._saliency_dampening[(r, c)] = (
                            self._saliency_dampening.get((r, c), 0.0) + rate
                        )

    def _reset_saliency_suppression(self):
        """Reset all dampening (called on full-frame refresh)."""
        for key in self._saliency_dampening:
            self._saliency_dampening[key] = 0.0

    def _run_fullframe_refresh(self, frame: Image.Image):
        """
        Run YOLO on the full frame and refresh ALL tile caches.
        This catches objects in tiles that saliency missed.
        """
        frame_w, frame_h = frame.size
        frame_imgsz = max(frame_w, frame_h)
        yolo_frame = self._get_yolo(frame_imgsz)

        t_yolo_start = time.perf_counter()
        _, boxes, classes, scores = yolo_frame.predict_image(frame)
        self.total_yolo_time += (time.perf_counter() - t_yolo_start)
        self.yolo_preds += 1
        self.yolo_ran_this_frame = True

        boxes_global = np.array(boxes, dtype=np.float32) if boxes is not None else np.zeros((0, 4), dtype=np.float32)
        classes_arr = np.array(classes, dtype=np.int64) if classes is not None else np.zeros((0,), dtype=np.int64)
        scores_arr = np.array(scores, dtype=np.float32) if scores is not None else np.zeros((0,), dtype=np.float32)

        # Apply class filter before caching
        boxes_global, classes_arr, scores_arr = self._filter_classes_from_arrays(
            boxes_global, classes_arr, scores_arr)

        # Clear all tile caches
        for r in range(self.rows):
            for c in range(self.cols):
                self.tile_det_cache[(r, c)] = {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "classes": np.zeros((0,), dtype=np.int64),
                    "scores": np.zeros((0,), dtype=np.float32),
                }
                self.tile_last_visited[(r, c)] = self.frame_counter

        # Assign detections to tiles
        assign = self._assign_boxes_to_tiles(boxes_global, classes_arr, scores_arr)
        for (r, c), idxs in assign.items():
            self.tile_det_cache[(r, c)] = {
                "boxes": boxes_global[idxs],
                "classes": classes_arr[idxs],
                "scores": scores_arr[idxs],
            }
            self._update_tile_object_memory(r, c, classes_arr[idxs])

        # LKT: clear all tracked objects and re-create from full-frame YOLO results
        if self.enable_lkt_tracking:
            curr_gray = cv2.cvtColor(self.pil_to_bgr(frame), cv2.COLOR_BGR2GRAY)
            self._lkt_clear_all()
            self._lkt_create_objects_from_detections(
                boxes_global, classes_arr, scores_arr, curr_gray)
            self._lkt_prev_gray = curr_gray

    def _downscale_for_saliency(self, bgr: np.ndarray) -> np.ndarray:
        """Downscale a BGR image for cheaper saliency computation."""
        if self.saliency_scale >= 1.0:
            return bgr
        h, w = bgr.shape[:2]
        new_w = max(1, int(w * self.saliency_scale))
        new_h = max(1, int(h * self.saliency_scale))
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _upscale_mask(self, mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Upscale a saliency mask back to original resolution."""
        if self.saliency_scale >= 1.0:
            return mask
        return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


    def init_yolo_per_tile(self):
        first_tile = self.tile_array[0, 0]
        width, height = first_tile.size
        max_dim = max(width, height)

        self.max_dim = max_dim
        self.tile_width = width
        self.tile_height = height
        self.total_area = self.tile_width * self.tile_height

        self._get_yolo(max_dim)  # Initialize and cache HeavyYoloClassifier for this imgsz
    
    def _build_tile_array(self) -> np.ndarray:
        for r in range(self.rows):
            for c in range(self.cols):
                self.tile_array[r, c] = self.tiler.get_tile(r, c)     # PIL.Image.Image

    def _get_yolo(self, imgsz: int):
        """Return a cached detector for this imgsz.

        Returns ``HeavyYoloClassifier`` when ``inference_mode='local'`` and
        ``RemoteDetector`` when ``inference_mode='remote'``. Both share the
        ``predict_image(PIL)`` contract that the rest of AGv2 uses.
        """
        if imgsz in self.yolo_clfs:
            return self.yolo_clfs[imgsz]

        if self.inference_mode == "remote":
            # Lazy import so the local path stays dependency-free if httpx
            # (and, for PQC, liboqs) is not installed.
            if self.crypto_mode in ("pqc", "classical"):
                from network.pqc_client import PQCRemoteDetector
                det = PQCRemoteDetector(
                    url=self.remote_url,
                    crypto_mode=self.crypto_mode,
                    kem_scheme=self.pqc_kem_scheme,
                    sig_scheme=self.pqc_sig_scheme,
                    imgsz=imgsz,
                    conf=0.25,
                    cafile=self.remote_cafile,
                    verify_tls=self.remote_verify_tls,
                    jpeg_quality=self.remote_jpeg_quality,
                    timeout_s=self.remote_timeout_s,
                )
            else:
                from network.client import RemoteDetector
                det = RemoteDetector(
                    url=self.remote_url,
                    imgsz=imgsz,
                    conf=0.25,  # AGv2 today doesn't expose per-call conf; match HeavyYolo default
                    cafile=self.remote_cafile,
                    verify_tls=self.remote_verify_tls,
                    jpeg_quality=self.remote_jpeg_quality,
                    timeout_s=self.remote_timeout_s,
                )
        else:
            det = HeavyYoloClassifier(
                weight=self.yolo_weight,
                imgsz=imgsz,
                use_finetuned=self.use_finetuned,
                device=self.device,
            )

        self.yolo_clfs[imgsz] = det
        return det
    

    def update_frame_memory(self, frame: Image.Image, annotated, boxes, classes, scores):
        self.frame_memory.append(
            {
            "frame": frame,
            "annotated": annotated,
            "boxes": boxes,
            "classes": classes,
            "scores": scores
            })
        
    def _select_tiles_from_mask(self, mask: np.ndarray):
        """
        Select tiles based on saliency mask.
        Args:
          mask: np.ndarray (H, W), binary mask where salient regions are marked.
        Returns:
            tiles_score_map: array of [rows, cols] indicating saliency scores per tile.
        """

        tiles_score_map = np.zeros((self.rows, self.cols), dtype=float)

        if mask is None:
            return tiles_score_map

        if mask.ndim == 3:
            mask = mask[:, :, 0]  # take first channel

        mask_h, mask_w = mask.shape

        tile_h = mask_h // self.rows
        tile_w = mask_w // self.cols

        for r in range(self.rows):
            for c in range(self.cols):
                y0 = r * tile_h
                y1 = (r + 1) * tile_h if r < self.rows - 1 else mask_h
                x0 = c * tile_w
                x1 = (c + 1) * tile_w if c < self.cols - 1 else mask_w

                tile_mask = mask[y0:y1, x0:x1]

                salient_pixel_count = np.sum(tile_mask > 0)  # Count non-zero pixels
                tiles_score_map[r, c] = salient_pixel_count

    
        return tiles_score_map
    
    def _merge_tiles_based_on_scoremap(self, tiles_score_map: np.ndarray,
                                       seed_override: tuple = None):
        """
        Brute-force rectangle selection around seed (argmax tile), for k=1..max_combined_tiles.
        Chooses the rectangle with the maximum TOTAL saliency (sum of tile scores),
        among all rectangles of exact area k that include the seed.

        NEW: expansion gate based on motion % in *newly added tiles*.
        - Only accept a larger rectangle if every newly added tile has motion >=
        (self.merge_add_tile_motion_pct * tile_area).
        - If self.merge_add_tile_motion_pct == 0.0 => behaves exactly like before.

        """

        rows, cols = self.rows, self.cols
        max_tiles = max(1, int(self.max_combined_tiles))
        merged_tiles_idx_map = np.zeros((rows, cols), dtype=bool)

        # --- ensure tile area known ---
        if self.total_area is None:
            t = self.tile_array[0, 0]
            if t is not None:
                w, h = t.size
                self.total_area = int(w) * int(h)

        tile_area = float(self.total_area) if self.total_area is not None else 1.0

        # Motion threshold per tile (in "saliency pixels" units)
        add_thr = float(getattr(self, "merge_add_tile_motion_pct", 0.0)) * tile_area

        if tiles_score_map is None or tiles_score_map.size == 0:
            r0, c0 = rows // 2, cols // 2
            merged_tiles_idx_map[r0, c0] = True
            return self.tile_array[r0, c0], merged_tiles_idx_map

        # If everything is zero -> pick a deterministic tile
        if float(np.max(tiles_score_map)) <= 0.0:
            candidates = [(r, c) for r in range(rows) for c in range(cols)]
            r0, c0 = candidates[len(candidates) // 2]
            merged_tiles_idx_map[r0, c0] = True
            return self.tile_array[r0, c0], merged_tiles_idx_map

        # Seed tile = override (priority) or argmax saliency
        if seed_override is not None:
            sr, sc = int(seed_override[0]), int(seed_override[1])
        else:
            sr, sc = np.unravel_index(np.argmax(tiles_score_map, axis=None), tiles_score_map.shape)
            sr, sc = int(sr), int(sc)

        def factor_pairs(k: int):
            out = []
            for h in range(1, k + 1):
                if (k % h) == 0:
                    out.append((h, k // h))
            return out

        def rect_score(rmin, rmax, cmin, cmax) -> float:
            return float(np.sum(tiles_score_map[rmin:rmax + 1, cmin:cmax + 1]))

        def rect_set(rmin, rmax, cmin, cmax):
            return {(rr, cc) for rr in range(rmin, rmax + 1) for cc in range(cmin, cmax + 1)}

        def expansion_ok(prev_rect, new_rect) -> bool:
            """
            Gate expansion based on newly added tiles.
            If add_thr == 0 => always ok (current behavior).
            """
            if add_thr <= 0.0 or prev_rect is None:
                return True

            prmin, prmax, pcmin, pcmax = prev_rect
            nrmin, nrmax, ncmin, ncmax = new_rect

            prev_tiles = rect_set(prmin, prmax, pcmin, pcmax)
            new_tiles = rect_set(nrmin, nrmax, ncmin, ncmax)

            added = new_tiles - prev_tiles
            if len(added) == 0:
                return True

            # Require each *added* tile to have enough motion.
            # (Alternative: require at least one added tile to pass — but you asked "only add if movement exists",
            # and per-tile requirement is the strictest/cleanest.)
            for (rr, cc) in added:
                if float(tiles_score_map[rr, cc]) < add_thr:
                    return False
            return True

        # Baseline: k=1
        best_rect = (sr, sr, sc, sc)
        best_score = rect_score(sr, sr, sc, sc)

        # Try k=2..max_tiles
        for k in range(2, max_tiles + 1):
            cand_best_score = -1e18
            cand_best_rect = None

            for (h, w) in factor_pairs(k):
                # placements that include seed and stay in bounds
                rmin_lo = max(0, sr - (h - 1))
                rmin_hi = min(sr, rows - h)
                cmin_lo = max(0, sc - (w - 1))
                cmin_hi = min(sc, cols - w)

                for rmin in range(rmin_lo, rmin_hi + 1):
                    rmax = rmin + h - 1
                    for cmin in range(cmin_lo, cmin_hi + 1):
                        cmax = cmin + w - 1

                        # must include seed
                        if not (rmin <= sr <= rmax and cmin <= sc <= cmax):
                            continue

                        s = rect_score(rmin, rmax, cmin, cmax)
                        if s > cand_best_score:
                            cand_best_score = s
                            cand_best_rect = (rmin, rmax, cmin, cmax)

            if cand_best_rect is None:
                break

            # NEW: expansion gate
            if not expansion_ok(best_rect, cand_best_rect):
                # stop expanding at the previous best_rect
                break

            # accept expansion
            best_rect = cand_best_rect
            best_score = cand_best_score

        # Build idx map from best_rect
        rmin, rmax, cmin, cmax = best_rect
        merged_tiles_idx_map[rmin:rmax + 1, cmin:cmax + 1] = True

        # Stitch tiles into merged image
        sample_tile = self.tile_array[rmin, cmin]
        tile_w, tile_h = sample_tile.size

        out_w = (cmax - cmin + 1) * tile_w
        out_h = (rmax - rmin + 1) * tile_h
        merged_img = Image.new("RGB", (out_w, out_h))

        for rr in range(rmin, rmax + 1):
            for cc in range(cmin, cmax + 1):
                tile = self.tile_array[rr, cc]
                x = (cc - cmin) * tile_w
                y = (rr - rmin) * tile_h
                merged_img.paste(tile, (x, y))

        self.max_dim = max(merged_img.size)
        return merged_img, merged_tiles_idx_map


    # =========================================================================
    # LUCAS-KANADE-TOMASI (LKT) OBJECT TRACKING METHODS
    # =========================================================================

    def _lkt_extract_features_for_object(self, obj: TrackedObject, gray: np.ndarray):
        """
        Extract Shi-Tomasi corner features inside an object's bbox.
        Updates obj.feature_points in-place.
        """
        bx, by, bw, bh = obj.bbox
        h_img, w_img = gray.shape[:2]

        x0 = max(0, int(round(float(bx))))
        y0 = max(0, int(round(float(by))))
        x1 = min(w_img, int(round(float(bx) + float(bw))))
        y1 = min(h_img, int(round(float(by) + float(bh))))

        if x1 <= x0 or y1 <= y0:
            obj.feature_points = None
            return

        roi_crop = gray[y0:y1, x0:x1]
        pts = cv2.goodFeaturesToTrack(roi_crop, **self._lkt_feature_params)
        if pts is not None and len(pts) > 0:
            pts = pts.astype(np.float32)
            pts[:, 0, 0] += x0
            pts[:, 0, 1] += y0
            obj.feature_points = pts
        else:
            obj.feature_points = None

    def _lkt_compute_object_saliency(self, obj: TrackedObject, mask: np.ndarray) -> float:
        """
        Compute saliency (white pixel count) inside an object's bbox from the
        saliency mask. Used to set the baseline saliency_footprint.
        """
        if mask is None:
            return 0.0
        bx, by, bw, bh = obj.bbox
        h_m, w_m = mask.shape[:2]
        x0 = max(0, int(round(float(bx))))
        y0 = max(0, int(round(float(by))))
        x1 = min(w_m, int(round(float(bx) + float(bw))))
        y1 = min(h_m, int(round(float(by) + float(bh))))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = mask[y0:y1, x0:x1]
        if roi.ndim == 3:
            roi = roi[:, :, 0]
        return float(np.sum(roi > 0))

    def _lkt_create_objects_from_detections(
        self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
        gray: np.ndarray, mask: np.ndarray = None,
    ) -> list:
        """
        Create TrackedObjects from YOLO detections. Extract features and
        compute baseline saliency footprint for each.
        Returns list of new obj_ids.
        """
        if boxes is None or len(boxes) == 0:
            return []

        tw = int(self.tile_width)
        th = int(self.tile_height)
        new_ids = []

        for i in range(len(boxes)):
            bx, by, bw, bh = boxes[i]
            cx = float(bx) + float(bw) / 2.0
            cy = float(by) + float(bh) / 2.0
            c = max(0, min(self.cols - 1, int(cx // tw)))
            r = max(0, min(self.rows - 1, int(cy // th)))

            obj = TrackedObject(
                obj_id=self._next_obj_id,
                class_id=int(classes[i]),
                confidence=float(scores[i]),
                bbox=np.array([float(bx), float(by), float(bw), float(bh)], dtype=np.float32),
                current_tile=(r, c),
                last_yolo_frame=self.frame_counter,
            )

            # Extract feature points
            self._lkt_extract_features_for_object(obj, gray)

            # Compute baseline saliency footprint
            if mask is not None:
                obj.saliency_footprint = self._lkt_compute_object_saliency(obj, mask)

            self._tracked_objects[self._next_obj_id] = obj
            self._tile_object_ids[(r, c)].add(self._next_obj_id)
            new_ids.append(self._next_obj_id)
            self._next_obj_id += 1

        return new_ids

    def _lkt_track_all_objects(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        """
        Track ALL active objects using a single batched calcOpticalFlowPyrLK call.
        Updates each object's bbox, velocity, and feature_points.
        Objects that lose too many feature points (< lkt_min_points) are
        deactivated (dropped) rather than triggering a YOLO call.
        """
        h_img, w_img = curr_gray.shape[:2]

        # ---- 1) Collect all valid keypoints across all objects ----
        obj_jobs = []     # (obj_id, n_points)  — only objects with valid points
        all_pts_list = []
        drop_objs = []    # obj_ids where points are missing/insufficient → deactivate

        for obj_id, obj in self._tracked_objects.items():
            if not obj.active:
                continue
            if obj.feature_points is None or len(obj.feature_points) < self.lkt_min_points:
                drop_objs.append(obj_id)
                continue
            obj_jobs.append((obj_id, len(obj.feature_points)))
            all_pts_list.append(obj.feature_points)

        # Deactivate objects with insufficient feature points
        for obj_id in drop_objs:
            obj = self._tracked_objects[obj_id]
            obj.active = False
            self._tile_object_ids.get(obj.current_tile, set()).discard(obj_id)

        if not all_pts_list:
            return

        # ---- 2) Single batched optical flow call ----
        batched_old = np.concatenate(all_pts_list, axis=0)  # (total_K, 1, 2)
        batched_new, batched_st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, batched_old, None, **self._lk_params
        )

        # ---- 3) Split results back per-object ----
        offset = 0
        alpha = self.lkt_velocity_smoothing

        for (obj_id, n_pts) in obj_jobs:
            obj = self._tracked_objects[obj_id]
            box_new = batched_new[offset:offset + n_pts]
            box_st = batched_st[offset:offset + n_pts]
            box_old = batched_old[offset:offset + n_pts]
            offset += n_pts

            if box_new is None or box_st is None:
                obj.active = False
                self._tile_object_ids.get(obj.current_tile, set()).discard(obj_id)
                continue

            good_mask = box_st.flatten() == 1
            good_old = box_old[good_mask]
            good_new = box_new[good_mask]

            if len(good_new) < self.lkt_min_points:
                obj.active = False
                self._tile_object_ids.get(obj.current_tile, set()).discard(obj_id)
                continue

            # Median displacement
            dx = float(np.median(good_new[:, 0, 0] - good_old[:, 0, 0]))
            dy = float(np.median(good_new[:, 0, 1] - good_old[:, 0, 1]))

            # Update velocity (exponential moving average)
            old_vx, old_vy = obj.velocity
            obj.velocity = (
                alpha * dx + (1.0 - alpha) * old_vx,
                alpha * dy + (1.0 - alpha) * old_vy,
            )

            # Move bbox
            bx, by, bw, bh = obj.bbox
            new_bx = max(0.0, min(float(w_img) - float(bw), float(bx) + dx))
            new_by = max(0.0, min(float(h_img) - float(bh), float(by) + dy))
            obj.bbox = np.array([new_bx, new_by, float(bw), float(bh)], dtype=np.float32)

            obj.feature_points = good_new.reshape(-1, 1, 2).astype(np.float32)
            obj.frames_since_yolo += 1

    def _lkt_update_object_tiles(self):
        """
        Reassign objects to correct tiles based on current bbox center.
        Updates source_tile to remember where the object came from.
        """
        tw = int(self.tile_width)
        th = int(self.tile_height)

        for obj_id, obj in self._tracked_objects.items():
            if not obj.active:
                continue
            bx, by, bw, bh = obj.bbox
            cx = float(bx) + float(bw) / 2.0
            cy = float(by) + float(bh) / 2.0
            new_c = max(0, min(self.cols - 1, int(cx // tw)))
            new_r = max(0, min(self.rows - 1, int(cy // th)))
            new_tile = (new_r, new_c)

            if obj.current_tile != new_tile:
                old_tile = obj.current_tile
                self._tile_object_ids[old_tile].discard(obj_id)
                self._tile_object_ids[new_tile].add(obj_id)
                obj.source_tile = old_tile
                obj.current_tile = new_tile

    def _lkt_expire_stale_objects(self):
        """Remove objects that haven't been confirmed by YOLO for too long."""
        max_drift = self.lkt_max_drift_frames
        expired = []
        for obj_id, obj in self._tracked_objects.items():
            if not obj.active:
                expired.append(obj_id)
                continue
            if obj.frames_since_yolo > max_drift:
                obj.active = False
                self._tile_object_ids[obj.current_tile].discard(obj_id)
                expired.append(obj_id)

        for obj_id in expired:
            del self._tracked_objects[obj_id]

    def _lkt_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """IoU between two xywh boxes."""
        ax1, ay1 = float(box_a[0]), float(box_a[1])
        ax2, ay2 = ax1 + float(box_a[2]), ay1 + float(box_a[3])
        bx1, by1 = float(box_b[0]), float(box_b[1])
        bx2, by2 = bx1 + float(box_b[2]), by1 + float(box_b[3])

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = float(box_a[2]) * float(box_a[3])
        area_b = float(box_b[2]) * float(box_b[3])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _lkt_match_yolo_to_tracked(
        self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
        tiles_set: set, gray: np.ndarray, mask: np.ndarray = None,
    ):
        """
        Match YOLO detections to existing tracked objects via IoU.
        - Matched objects: update bbox, confidence, re-extract features, keep ID.
        - Unmatched detections: create new TrackedObjects.
        - Unmatched tracked objects in affected tiles: deactivate.
        """
        iou_thr = self.lkt_iou_match_threshold

        # Gather existing active objects in affected tiles
        existing = {}
        for tile in tiles_set:
            for obj_id in list(self._tile_object_ids.get(tile, set())):
                obj = self._tracked_objects.get(obj_id)
                if obj and obj.active:
                    existing[obj_id] = obj

        matched_det = set()
        matched_obj = set()

        if boxes is not None and len(boxes) > 0:
            # Greedy IoU matching (best match per detection)
            for i in range(len(boxes)):
                best_iou = 0.0
                best_id = None
                for obj_id, obj in existing.items():
                    if obj_id in matched_obj:
                        continue
                    if int(classes[i]) != obj.class_id:
                        continue
                    iou = self._lkt_iou(boxes[i], obj.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = obj_id

                if best_iou >= iou_thr and best_id is not None:
                    # Update existing object
                    obj = self._tracked_objects[best_id]
                    obj.bbox = np.array(boxes[i], dtype=np.float32).ravel()[:4]
                    obj.confidence = float(scores[i])
                    obj.last_yolo_frame = self.frame_counter
                    obj.frames_since_yolo = 0
                    self._lkt_extract_features_for_object(obj, gray)
                    if mask is not None:
                        obj.saliency_footprint = self._lkt_compute_object_saliency(obj, mask)
                    matched_det.add(i)
                    matched_obj.add(best_id)

        # Deactivate unmatched objects in these tiles
        for obj_id, obj in existing.items():
            if obj_id not in matched_obj:
                obj.active = False
                self._tile_object_ids[obj.current_tile].discard(obj_id)

        # Create new objects for unmatched detections
        if boxes is not None and len(boxes) > 0:
            unmatched_idx = [i for i in range(len(boxes)) if i not in matched_det]
            if unmatched_idx:
                um_boxes = boxes[unmatched_idx]
                um_classes = classes[unmatched_idx]
                um_scores = scores[unmatched_idx]
                self._lkt_create_objects_from_detections(
                    um_boxes, um_classes, um_scores, gray, mask)

        # Clean up deactivated objects
        to_del = [oid for oid, o in self._tracked_objects.items() if not o.active]
        for oid in to_del:
            del self._tracked_objects[oid]

    def _lkt_populate_tile_cache(self):
        """
        Populate tile_det_cache from tracked objects.
        Called after any LKT state change to keep the cache in sync.
        """
        # Clear all caches
        for r in range(self.rows):
            for c in range(self.cols):
                self.tile_det_cache[(r, c)] = {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "classes": np.zeros((0,), dtype=np.int64),
                    "scores": np.zeros((0,), dtype=np.float32),
                }

        # Group active objects by tile
        tile_data = defaultdict(lambda: ([], [], []))
        for obj_id, obj in self._tracked_objects.items():
            if not obj.active:
                continue
            boxes_l, cls_l, sc_l = tile_data[obj.current_tile]
            boxes_l.append(obj.bbox)
            cls_l.append(obj.class_id)
            sc_l.append(obj.confidence)

        for (r, c), (boxes_l, cls_l, sc_l) in tile_data.items():
            if boxes_l:
                self.tile_det_cache[(r, c)] = {
                    "boxes": np.array(boxes_l, dtype=np.float32).reshape(-1, 4),
                    "classes": np.array(cls_l, dtype=np.int64),
                    "scores": np.array(sc_l, dtype=np.float32),
                }

    def _lkt_clear_all(self):
        """Clear all tracked objects (used before full-frame refresh)."""
        self._tracked_objects.clear()
        for key in self._tile_object_ids:
            self._tile_object_ids[key] = set()

    def process_frame(self, frame: Image.Image):

        # 1) Tile current frame
        t_tiler_start = time.perf_counter()
        self.tiler.load(frame)
        self.tiler.split()
        self._build_tile_array()
        self.total_tiler_time += (time.perf_counter() - t_tiler_start)

        # Increment frame counter
        self.frame_counter += 1
        self.yolo_ran_this_frame = False

        # -----------------------------
        # FIRST RUN: run YOLO on full frame,
        # initialize DETECTION cache for ALL tiles
        # -----------------------------
        if self.first_run:
            self.first_run = False
            self.init_yolo_per_tile()

            # Mark all tiles visited on first run
            for r in range(self.rows):
                for c in range(self.cols):
                    self.tile_last_visited[(r, c)] = self.frame_counter
    
            # Getting the yolo classifier for full frame (1280x720 for car dataset)
            frame_w, frame_h = frame.size
            frame_imgsz = max(frame_w, frame_h)
            yolo_frame = self._get_yolo(frame_imgsz)

            t_yolo_start = time.perf_counter()
            _, boxes, classes, scores = yolo_frame.predict_image(frame)  # ignore annotated pixels
            self.total_yolo_time += (time.perf_counter() - t_yolo_start)
            self.yolo_preds += 1
            self.yolo_ran_this_frame = True

            # Convert to arrays (GLOBAL coords already)
            boxes_global = np.array(boxes, dtype=np.float32) if boxes is not None else np.zeros((0, 4), dtype=np.float32)
            classes_arr = np.array(classes, dtype=np.int64) if classes is not None else np.zeros((0,), dtype=np.int64)
            scores_arr  = np.array(scores, dtype=np.float32) if scores is not None else np.zeros((0,), dtype=np.float32)

            # Apply class filter before caching
            boxes_global, classes_arr, scores_arr = self._filter_classes_from_arrays(
                boxes_global, classes_arr, scores_arr)

            # Reset all tiles to empty on first frame
            for r in range(self.rows):
                for c in range(self.cols):
                    self.tile_det_cache[(r, c)] = {
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "classes": np.zeros((0,), dtype=np.int64),
                        "scores": np.zeros((0,), dtype=np.float32),
                    }

            # Assign detections to tiles and store
            assign = self._assign_boxes_to_tiles(boxes_global, classes_arr, scores_arr)
            for (r, c), idxs in assign.items():
                self.tile_det_cache[(r, c)] = {
                    "boxes": boxes_global[idxs],
                    "classes": classes_arr[idxs],
                    "scores": scores_arr[idxs],
                }
                # Update tile object memory with detected classes
                self._update_tile_object_memory(r, c, classes_arr[idxs])

            # Gather cached detections (drawing deferred to get_overlay_image)
            out_boxes, out_classes, out_scores = self._gather_all_cached_detections()
            self.update_frame_memory(frame, None, out_boxes, out_classes, out_scores)

            # LKT: create tracked objects from first-frame detections
            if self.enable_lkt_tracking:
                curr_gray = cv2.cvtColor(self.pil_to_bgr(frame), cv2.COLOR_BGR2GRAY)
                self._lkt_create_objects_from_detections(
                    boxes_global, classes_arr, scores_arr, curr_gray)
                self._lkt_prev_gray = curr_gray

            return None, out_boxes, out_classes, out_scores


        # -----------------------------
        # NORMAL RUN
        # -----------------------------

        # Periodic full-frame YOLO refresh
        if self.fullframe_every > 0 and (self.frame_counter % self.fullframe_every) == 0:
            self._run_fullframe_refresh(frame)
            self._reset_saliency_suppression()
            out_boxes, out_classes, out_scores = self._gather_all_cached_detections()
            self.update_frame_memory(frame, None, out_boxes, out_classes, out_scores)
            return None, out_boxes, out_classes, out_scores

        # ---- LKT: track ALL objects before saliency / YOLO ----
        curr_bgr = self.pil_to_bgr(frame)  # computed once, reused by LKT and saliency
        curr_gray_for_lkt = None
        if self.enable_lkt_tracking and self._lkt_prev_gray is not None:
            t_lkt_start = time.perf_counter()
            curr_gray_for_lkt = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
            self._lkt_track_all_objects(self._lkt_prev_gray, curr_gray_for_lkt)
            # Reassign objects that drifted across tile boundaries
            self._lkt_update_object_tiles()
            # Expire objects not confirmed by YOLO for too long
            self._lkt_expire_stale_objects()
            # Sync tile_det_cache with tracked object positions
            self._lkt_populate_tile_cache()
            self.total_lkt_time += (time.perf_counter() - t_lkt_start)
            self.lkt_track_count += 1

        t_saliency_start = time.perf_counter()
        prev_frame = self.frame_memory[-1]["frame"] if len(self.frame_memory) > 0 else frame
        prev_bgr = self.pil_to_bgr(prev_frame)

        # Downscale for cheaper saliency if saliency_scale < 1.0
        orig_h, orig_w = curr_bgr.shape[:2]
        prev_bgr_sal = self._downscale_for_saliency(prev_bgr)
        curr_bgr_sal = self._downscale_for_saliency(curr_bgr)

        if self.saliency_method == "frame_diff":
            saliency_obj = FrameDiff(frame1=prev_bgr_sal, frame2=curr_bgr_sal)
            merged_bboxes, total_area, mask = saliency_obj.auto_run(saliency_measurement=self.saliency_measurement)

        elif self.saliency_method == "frame_diff_kmeans":
            saliency_obj = FrameDiffKMeans(
                frame1=prev_bgr_sal,
                frame2=curr_bgr_sal,
                n_clusters=DEFAULT_KMEANS_FD_PARAMS["n_clusters"],
                min_cluster_size=DEFAULT_KMEANS_FD_PARAMS["min_cluster_size"],
                min_cluster_density=DEFAULT_KMEANS_FD_PARAMS["min_cluster_density"],
                use_minibatch=DEFAULT_KMEANS_FD_PARAMS["use_minibatch"],
                max_pixels_for_kmeans=DEFAULT_KMEANS_FD_PARAMS["max_pixels_for_kmeans"],
                morphology_close_size=DEFAULT_KMEANS_FD_PARAMS["morphology_close_size"],
                morphology_open_size=DEFAULT_KMEANS_FD_PARAMS["morphology_open_size"],
            )
            merged_bboxes, total_area, mask = saliency_obj.auto_run(saliency_measurement=self.saliency_measurement)

        elif self.saliency_method == "optical_flow":
            saliency_obj = OpticalFlow(frame1=prev_bgr_sal, frame2=curr_bgr_sal)
            merged_bboxes, total_area, mask = saliency_obj.auto_run(saliency_measurement=self.saliency_measurement)

        elif self.saliency_method == "temporal_phase":
            saliency_obj = TemporalPhaseSaliency(frame1=prev_bgr_sal, frame2=curr_bgr_sal)
            merged_bboxes, total_area, mask = saliency_obj.auto_run(
                spatial_sigma=DEFAULT_TPS_PARAMS["spatial_sigma"],
                threshold_k=DEFAULT_TPS_PARAMS["threshold_k"],
                morph_kernel_size=DEFAULT_TPS_PARAMS["morph_kernel_size"],
                open_iterations=DEFAULT_TPS_PARAMS["open_iterations"],
                close_iterations=DEFAULT_TPS_PARAMS["close_iterations"],
                min_area=DEFAULT_TPS_PARAMS["min_area"],
                nms_threshold=DEFAULT_TPS_PARAMS["nms_threshold"],
                saliency_measurement=self.saliency_measurement,
                plot=False
            )

        elif self.saliency_method == "u2":
            if U2NetSaliency is None:
                raise ImportError("U2NetSaliency not available. Install U-2-Net dependencies first.")
            saliency_obj = U2NetSaliency(
                frame1=prev_bgr,
                frame2=curr_bgr,
                model_name=DEFAULT_U2_PARAMS["model_name"],
                device=DEFAULT_U2_PARAMS["device"]
            )
            merged_bboxes, total_area, mask = saliency_obj.auto_run(
                plot=False,
                saliency_measurement=self.saliency_measurement,
                threshold=DEFAULT_U2_PARAMS["threshold"],
                kernel_size=DEFAULT_U2_PARAMS["kernel_size"],
                min_area=DEFAULT_U2_PARAMS["min_area"],
                max_area_ratio=DEFAULT_U2_PARAMS["max_area_ratio"],
                nms_threshold=DEFAULT_U2_PARAMS["nms_threshold"],
            )

        elif self.saliency_method == "deva":
            if DevaSaliency is None:
                raise ImportError("DevaSaliency not available. Install DEVA dependencies first.")
            saliency_obj = DevaSaliency(
                frame1=prev_bgr,
                frame2=curr_bgr,
                text_prompt=DEFAULT_DEVA_PARAMS["text_prompt"],
                device=DEFAULT_DEVA_PARAMS["device"],
                box_threshold=DEFAULT_DEVA_PARAMS["box_threshold"],
                text_threshold=DEFAULT_DEVA_PARAMS["text_threshold"],
            )
            merged_bboxes, total_area, mask = saliency_obj.auto_run(
                plot=False,
                saliency_measurement=self.saliency_measurement,
                threshold=DEFAULT_DEVA_PARAMS["threshold"],
                kernel_size=DEFAULT_DEVA_PARAMS["kernel_size"],
                min_area=DEFAULT_DEVA_PARAMS["min_area"],
                max_area_ratio=DEFAULT_DEVA_PARAMS["max_area_ratio"],
                nms_threshold=DEFAULT_DEVA_PARAMS["nms_threshold"],
            )

        elif self.saliency_method == "inspyrenet":
            if InSPyReNetSaliency is None:
                raise ImportError("InSPyReNetSaliency not available. Install InSPyReNet dependencies first.")
            saliency_obj = InSPyReNetSaliency(
                frame1=prev_bgr,
                frame2=curr_bgr,
                mode=DEFAULT_INSPYRENET_PARAMS["mode"],
                device=DEFAULT_INSPYRENET_PARAMS["device"],
                resize=DEFAULT_INSPYRENET_PARAMS["resize"],
                multi_scale=DEFAULT_INSPYRENET_PARAMS["multi_scale"],
                scales=DEFAULT_INSPYRENET_PARAMS["scales"],
                tile_size=DEFAULT_INSPYRENET_PARAMS["tile_size"],
                tile_overlap=DEFAULT_INSPYRENET_PARAMS["tile_overlap"],
            )
            merged_bboxes, total_area, mask = saliency_obj.auto_run(
                plot=False,
                saliency_measurement=self.saliency_measurement,
                threshold=DEFAULT_INSPYRENET_PARAMS["threshold"],
                kernel_size=DEFAULT_INSPYRENET_PARAMS["kernel_size"],
                min_area=DEFAULT_INSPYRENET_PARAMS["min_area"],
                max_area_ratio=DEFAULT_INSPYRENET_PARAMS["max_area_ratio"],
                nms_threshold=DEFAULT_INSPYRENET_PARAMS["nms_threshold"],
            )

        elif self.saliency_method == "poolnet":
            if PoolNetSaliency is None:
                raise ImportError("PoolNetSaliency not available. Install PoolNet dependencies first.")
            saliency_obj = PoolNetSaliency(
                frame1=prev_bgr,
                frame2=curr_bgr,
                backbone=DEFAULT_POOLNET_PARAMS["backbone"],
                device=DEFAULT_POOLNET_PARAMS["device"],
                input_size=DEFAULT_POOLNET_PARAMS["input_size"],
                normalize=DEFAULT_POOLNET_PARAMS["normalize"],
            )
            merged_bboxes, total_area, mask = saliency_obj.auto_run(
                plot=False,
                saliency_measurement=self.saliency_measurement,
                threshold=DEFAULT_POOLNET_PARAMS["threshold"],
                kernel_size=DEFAULT_POOLNET_PARAMS["kernel_size"],
                min_area=DEFAULT_POOLNET_PARAMS["min_area"],
                max_area_ratio=DEFAULT_POOLNET_PARAMS["max_area_ratio"],
                nms_threshold=DEFAULT_POOLNET_PARAMS["nms_threshold"],
            )

        elif self.saliency_method == "hybrid":
            saliency_obj = HybridSaliency(
                frame1=prev_bgr,
                frame2=curr_bgr,
                motion_weight=DEFAULT_HYBRID_PARAMS["motion_weight"],
                static_weight=DEFAULT_HYBRID_PARAMS["static_weight"],
                temporal_decay=DEFAULT_HYBRID_PARAMS["temporal_decay"],
                use_temporal_memory=DEFAULT_HYBRID_PARAMS["use_temporal_memory"],
                spectral_scale=DEFAULT_HYBRID_PARAMS["spectral_scale"],
                gaussian_blur=DEFAULT_HYBRID_PARAMS["gaussian_blur"],
            )
            merged_bboxes, total_area, mask = saliency_obj.auto_run(
                plot=False,
                saliency_measurement=self.saliency_measurement,
                threshold=DEFAULT_HYBRID_PARAMS["threshold"],
                kernel_size=DEFAULT_HYBRID_PARAMS["kernel_size"],
                open_iterations=DEFAULT_HYBRID_PARAMS["open_iterations"],
                min_area=DEFAULT_HYBRID_PARAMS["min_area"],
                max_area_ratio=DEFAULT_HYBRID_PARAMS["max_area_ratio"],
                nms_threshold=DEFAULT_HYBRID_PARAMS["nms_threshold"],
            )

        else:
            raise ValueError(f"Unknown saliency_method: {self.saliency_method}")

        self.total_saliency_time += (time.perf_counter() - t_saliency_start)

        # Upscale saliency mask back to original resolution if downscaled
        mask = self._upscale_mask(mask, orig_h, orig_w)

        tiles_score_map = self._select_tiles_from_mask(mask)

        # Keep a copy of raw saliency scores BEFORE any penalties/suppression
        # so that _update_saliency_suppression_after_yolo can judge "meaningful
        # saliency" from the original signal, not a dampened one.
        raw_tiles_score_map = tiles_score_map.copy()

        # Update tile movement memory (before penalties are applied)
        self._update_tile_movement_memory(tiles_score_map)

        # Saliency suppression: decay dampening then reduce scores for "noisy" tiles
        self._decay_saliency_suppression()
        self._apply_saliency_suppression(tiles_score_map)

        # Attention priority: pick stale salient tile as seed (if any)
        priority_seed = self._pick_priority_seed(tiles_score_map)

        # Determine visited region (idx_map + merged image)
        t_merge_start = time.perf_counter()
        if self.enable_tile_combination:
            merged_tiles_img, merged_tiles_idx_map = self._merge_tiles_based_on_scoremap(
                tiles_score_map, seed_override=priority_seed)
        else:
            if priority_seed is not None:
                max_row_idx, max_col_idx = priority_seed
            else:
                max_row_idx, max_col_idx = np.unravel_index(np.argmax(tiles_score_map, axis=None), tiles_score_map.shape)
            max_row_idx, max_col_idx = int(max_row_idx), int(max_col_idx)
            merged_tiles_idx_map = np.zeros((self.rows, self.cols), dtype=bool)
            merged_tiles_idx_map[max_row_idx, max_col_idx] = True
            merged_tiles_img = self.tile_array[max_row_idx, max_col_idx]
        self.total_merge_tiles_time += (time.perf_counter() - t_merge_start)

        # ---- YOLO run interval: skip YOLO on non-scheduled frames ----
        # Cached detections (and LKT tracking if enabled) bridge the gap.
        rect = self._idx_map_to_rect(merged_tiles_idx_map)
        is_yolo_frame = (self.yolo_run_interval <= 1
                         or (self.frame_counter % self.yolo_run_interval) == 0)
        skip_yolo_for_saliency = not is_yolo_frame

        if not skip_yolo_for_saliency:
            # Run YOLO on the visited region (original behaviour)
            imgsz = max(merged_tiles_img.size)
            yolo_clf = self._get_yolo(imgsz)

            t_yolo_start = time.perf_counter()
            _, boxes_local, classes, scores = yolo_clf.predict_image(merged_tiles_img)
            self.total_yolo_time += (time.perf_counter() - t_yolo_start)
            self.yolo_preds += 1
            self.yolo_ran_this_frame = True

            # Figure out tile-rectangle bounds
            if rect is None:
                # All tiles locked out — return persistent cached detections
                out_boxes, out_classes, out_scores = self._gather_all_cached_detections()
                self.update_frame_memory(frame, None, out_boxes, out_classes, out_scores)
                # LKT: update prev gray
                if self.enable_lkt_tracking and curr_gray_for_lkt is not None:
                    self._lkt_prev_gray = curr_gray_for_lkt
                return None, out_boxes, out_classes, out_scores

            rmin, rmax, cmin, cmax = rect
            tw, th = int(self.tile_width), int(self.tile_height)

            # Mark saliency-visited tiles as visited now + update YOLO timestamp
            for rr in range(rmin, rmax + 1):
                for cc in range(cmin, cmax + 1):
                    self.tile_last_visited[(rr, cc)] = self.frame_counter

            # ---- B) Convert local boxes -> GLOBAL boxes ----
            boxes_local = np.array(boxes_local, dtype=np.float32) if boxes_local is not None else np.zeros((0, 4), dtype=np.float32)
            classes_arr = np.array(classes, dtype=np.int64) if classes is not None else np.zeros((0,), dtype=np.int64)
            scores_arr  = np.array(scores, dtype=np.float32) if scores is not None else np.zeros((0,), dtype=np.float32)

            offset_x = cmin * tw
            offset_y = rmin * th

            boxes_global = boxes_local.copy()
            if len(boxes_global) > 0:
                boxes_global[:, 0] += offset_x
                boxes_global[:, 1] += offset_y

            # ---- C) Overwrite per-tile DET cache only for visited tiles ----
            # (this is what makes predictions persist until revisit)
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    self.tile_det_cache[(r, c)] = {
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "classes": np.zeros((0,), dtype=np.int64),
                        "scores": np.zeros((0,), dtype=np.float32),
                    }

            # Apply class filter before caching
            boxes_global, classes_arr, scores_arr = self._filter_classes_from_arrays(
                boxes_global, classes_arr, scores_arr)

            assign = self._assign_boxes_to_tiles(boxes_global, classes_arr, scores_arr)
            for (r, c), idxs in assign.items():
                if rmin <= r <= rmax and cmin <= c <= cmax:
                    self.tile_det_cache[(r, c)] = {
                        "boxes": boxes_global[idxs],
                        "classes": classes_arr[idxs],
                        "scores": scores_arr[idxs]
                    }
                    # Update tile object memory with detected classes
                    self._update_tile_object_memory(r, c, classes_arr[idxs])

            # LKT: match YOLO results to tracked objects + create new ones
            if self.enable_lkt_tracking:
                gray = curr_gray_for_lkt
                if gray is None:
                    gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
                    curr_gray_for_lkt = gray
                visited_tiles = {(rr, cc) for rr in range(rmin, rmax + 1)
                                 for cc in range(cmin, cmax + 1)}
                self._lkt_match_yolo_to_tracked(
                    boxes_global, classes_arr, scores_arr,
                    visited_tiles, gray, mask)
                self._lkt_populate_tile_cache()

            # Saliency suppression: update dampening based on YOLO results
            # Use raw (pre-suppression, pre-penalty) scores so the threshold
            # check reflects actual saliency, not an already-dampened value.
            self._update_saliency_suppression_after_yolo(
                rmin, rmax, cmin, cmax, raw_tiles_score_map)

        else:
            # skip_yolo_for_saliency is True — LKT already updated the boxes.
            # Still update saliency suppression using cached detections so that
            # dampening accumulates on noisy tiles even when YOLO is skipped.
            if rect is not None:
                rmin, rmax, cmin, cmax = rect
                self._update_saliency_suppression_after_yolo(
                    rmin, rmax, cmin, cmax, raw_tiles_score_map)

                # Mark saliency-selected tiles as visited even though YOLO was
                # skipped.  This prevents recheck from treating them as stale
                # during yolo_run_interval gaps.
                for rr in range(rmin, rmax + 1):
                    for cc in range(cmin, cmax + 1):
                        self.tile_last_visited[(rr, cc)] = self.frame_counter

        # Keep frame-based memory only for red highlighting
        self.merged_tiles_memory[self.frame_counter] = {"merged_tiles_idx_map": merged_tiles_idx_map}
        if len(self.merged_tiles_memory) > self.max_merge_memory:
            oldest_key = min(self.merged_tiles_memory.keys())
            del self.merged_tiles_memory[oldest_key]

        # -----------------------------
        # RECHECK SECTION

        # After saliency update, optionally recheck stale tiles (blue highlight)
        # Pass the current saliency rect so recheck skips tiles already covered.
        self._perform_recheck_if_needed(saliency_rect=rect)

        # -----------------------------

        # LKT: update previous gray frame
        if self.enable_lkt_tracking:
            if curr_gray_for_lkt is None:
                curr_gray_for_lkt = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
            self._lkt_prev_gray = curr_gray_for_lkt

        # ---- D) Return PERSISTENT outputs (all cached detections, no drawing) ----
        out_boxes, out_classes, out_scores = self._gather_all_cached_detections()
        self.update_frame_memory(frame, None, out_boxes, out_classes, out_scores)

        return None, out_boxes, out_classes, out_scores


    

    def _idx_map_to_rect(self, idx_map: np.ndarray):
        """
        Example input:
            [[False False False False]
            [False  True  True False]
            [False  True  True False]
            [False False False False]]
        Returns: 
            (1,2,1,2) - 1: rmin, 2: rmax, 1: cmin, 2: cmax

        If no True values, returns None.
        Args:
            idx_map: np.ndarray of shape (rows, cols), boolean
        Returns:
            rmin: minimum row index with True
            rmax: maximum row index with True
            cmin: minimum column index with True
            cmax: maximum column index with True
        """
        rows_idx, cols_idx = np.where(idx_map)
        if len(rows_idx) == 0:
            return None
        rmin, rmax = int(rows_idx.min()), int(rows_idx.max())
        cmin, cmax = int(cols_idx.min()), int(cols_idx.max())
        return rmin, rmax, cmin, cmax


    def _assign_boxes_to_tiles(self, boxes_xywh_global: np.ndarray, classes: np.ndarray, scores: np.ndarray):
        """
        Assign each detection to a tile based on bbox center (global coords).
        Updates self.tile_det_cache only for tiles that appear in the assignment dict.
        Returns: dict[(r,c)] -> list of indices belonging to that tile
        """
        assign = {}
        if boxes_xywh_global is None or len(boxes_xywh_global) == 0:
            return assign

        tw = int(self.tile_width)
        th = int(self.tile_height)

        for i, (x, y, w, h) in enumerate(boxes_xywh_global):
            cx = float(x) + float(w) / 2.0
            cy = float(y) + float(h) / 2.0
            c = int(cx // tw)
            r = int(cy // th)
            # clamp
            r = max(0, min(self.rows - 1, r))
            c = max(0, min(self.cols - 1, c))
            assign.setdefault((r, c), []).append(i)

        return assign


    def _gather_all_cached_detections(self):
        """
        Combine per-tile cached detections into single arrays to return.
        If prediction_fusion is enabled, fuse cut-off detections across tile boundaries.
        Returns (boxes, classes, scores) in global coords.
        """
        all_boxes = []
        all_classes = []
        all_scores = []
        all_tiles = []  # track tile origin for each detection (needed for fusion)

        for (r, c), det in self.tile_det_cache.items():
            b = det.get("boxes", None)
            cl = det.get("classes", None)
            sc = det.get("scores", None)
            if b is None or len(b) == 0:
                continue
            all_boxes.append(b)
            all_classes.append(cl)
            all_scores.append(sc)
            all_tiles.extend([(r, c)] * len(b))

        if len(all_boxes) == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.float32))

        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        classes = np.concatenate(all_classes, axis=0).astype(np.int64)
        scores = np.concatenate(all_scores, axis=0).astype(np.float32)

        if self.prediction_fusion:
            boxes, classes, scores = self._fuse_cross_tile_predictions(
                boxes, classes, scores, all_tiles
            )

        # Apply class filter (keep only allowed classes)
        boxes, classes, scores = self._filter_classes_from_arrays(boxes, classes, scores)

        return boxes, classes, scores


    def _fuse_cross_tile_predictions(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        tile_origins: list,
    ):
        """
        Fuse bounding boxes that are cut off at tile boundaries.

        Two detections are merged when ALL of the following hold:
          1. **Same class** — only same-class boxes are candidates.
          2. **Adjacent tiles** — the two boxes come from cardinal or diagonal
             neighbour tiles in the grid.
          3. **Both boxes touch the shared tile boundary** — each box's edge
             must be within ``fusion_edge_margin_pct * tile_dim`` pixels of the
             boundary that separates the two tiles.
          4. **Spatial overlap / proximity in global coords** — the boxes must
             genuinely overlap or nearly touch each other across the boundary.
             Specifically:
               - Along the axis *perpendicular* to the shared boundary, the gap
                 between the two boxes' nearest edges must be ≤ the edge margin.
                 (Boxes that are far apart in global space are NOT fused.)
               - Along the axis *parallel* to the shared boundary, the boxes
                 must overlap by at least ``fusion_min_parallel_overlap_pct``
                 of the smaller box's extent on that axis.  This prevents fusing
                 two unrelated objects that happen to sit at the same boundary
                 but at different vertical/horizontal positions.
          5. **Aspect-ratio sanity check** — the merged enclosing box's aspect
             ratio must not exceed ``fusion_max_aspect_ratio``.  Fusing a tall
             thin box with a wide flat one is almost always wrong.

        Uses union-find so an object spanning 3+ tiles is correctly grouped.
        Does NOT modify ``tile_det_cache``; operates purely on the flat arrays
        produced by ``_gather_all_cached_detections``.

        Args:
            boxes:   (N, 4) float32 — xywh in global pixel coords (top-left).
            classes: (N,) int64     — class IDs.
            scores:  (N,) float32   — confidence scores.
            tile_origins: list of (r, c) tuples, one per detection.

        Returns:
            (fused_boxes, fused_classes, fused_scores)
        """
        n = len(boxes)
        if n < 2 or self.tile_width is None or self.tile_height is None:
            return boxes, classes, scores

        tw = int(self.tile_width)
        th = int(self.tile_height)
        margin_x = self.fusion_edge_margin_pct * tw
        margin_y = self.fusion_edge_margin_pct * th

        # Tunable constants
        PARALLEL_OVERLAP_PCT = 0.25   # min overlap along shared-boundary axis
        MAX_ASPECT_RATIO     = 5.0    # reject merged box if AR exceeds this

        # ---- Pre-compute absolute box edges (x1, y1, x2, y2) ----
        x1 = boxes[:, 0].astype(np.float64)
        y1 = boxes[:, 1].astype(np.float64)
        x2 = x1 + boxes[:, 2].astype(np.float64)
        y2 = y1 + boxes[:, 3].astype(np.float64)

        # ---- Pre-compute which tile edges each detection touches ----
        touches_left   = np.zeros(n, dtype=bool)
        touches_right  = np.zeros(n, dtype=bool)
        touches_top    = np.zeros(n, dtype=bool)
        touches_bottom = np.zeros(n, dtype=bool)

        for i in range(n):
            r, c = tile_origins[i]
            tile_left   = c * tw
            tile_right  = (c + 1) * tw
            tile_top    = r * th
            tile_bottom = (r + 1) * th

            touches_left[i]   = x1[i] <= tile_left   + margin_x
            touches_right[i]  = x2[i] >= tile_right  - margin_x
            touches_top[i]    = y1[i] <= tile_top     + margin_y
            touches_bottom[i] = y2[i] >= tile_bottom  - margin_y

        # ---- Union-Find ----
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # ---- Index detections by tile ----
        from collections import defaultdict
        tile_dets = defaultdict(list)
        for i in range(n):
            tile_dets[tile_origins[i]].append(i)

        # ---- Helper: check parallel-axis overlap ----
        def _parallel_overlap_ok(ia, ib, axis):
            """
            Check that boxes ia and ib overlap sufficiently along *axis*.
            axis = 'h' → check horizontal overlap (x-axis)
            axis = 'v' → check vertical   overlap (y-axis)
            """
            if axis == 'h':
                lo_a, hi_a = x1[ia], x2[ia]
                lo_b, hi_b = x1[ib], x2[ib]
            else:
                lo_a, hi_a = y1[ia], y2[ia]
                lo_b, hi_b = y1[ib], y2[ib]

            overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
            if overlap <= 0:
                return False
            extent_a = hi_a - lo_a
            extent_b = hi_b - lo_b
            min_extent = min(extent_a, extent_b)
            if min_extent <= 0:
                return False
            return (overlap / min_extent) >= PARALLEL_OVERLAP_PCT

        # ---- Helper: check perpendicular-axis gap ----
        def _perp_gap_ok(ia, ib, axis, margin):
            """
            Check the gap between the two boxes along *axis* is <= margin.
            axis = 'h' → gap along x-axis (for a vertical boundary)
            axis = 'v' → gap along y-axis (for a horizontal boundary)
            """
            if axis == 'h':
                gap = max(0, max(x1[ia], x1[ib]) - min(x2[ia], x2[ib]))
            else:
                gap = max(0, max(y1[ia], y1[ib]) - min(y2[ia], y2[ib]))
            return gap <= margin

        # ---- Helper: aspect-ratio sanity ----
        def _aspect_ok(ia, ib):
            """Check that the union box wouldn't have an extreme aspect ratio."""
            ux1 = min(x1[ia], x1[ib])
            uy1 = min(y1[ia], y1[ib])
            ux2 = max(x2[ia], x2[ib])
            uy2 = max(y2[ia], y2[ib])
            uw = ux2 - ux1
            uh = uy2 - uy1
            if uw <= 0 or uh <= 0:
                return False
            ar = max(uw / uh, uh / uw)
            return ar <= MAX_ASPECT_RATIO

        # ---- Helper: full pairwise fusion check ----
        def _should_fuse(ia, ib, perp_axis, parallel_axis, perp_margin):
            if int(classes[ia]) != int(classes[ib]):
                return False
            if not _perp_gap_ok(ia, ib, perp_axis, perp_margin):
                return False
            if not _parallel_overlap_ok(ia, ib, parallel_axis):
                return False
            if not _aspect_ok(ia, ib):
                return False
            return True

        # ---- Cardinal neighbours ----
        # For a vertical boundary   (tiles side-by-side): perp = 'h', parallel = 'v'
        # For a horizontal boundary (tiles stacked):      perp = 'v', parallel = 'h'
        cardinal_cfg = {
            (0,  1): ("right",  "left",   'h', 'v', margin_x),
            (0, -1): ("left",   "right",  'h', 'v', margin_x),
            (1,  0): ("bottom", "top",    'v', 'h', margin_y),
            (-1, 0): ("top",    "bottom", 'v', 'h', margin_y),
        }
        edge_flag = {
            "left":   touches_left,
            "right":  touches_right,
            "top":    touches_top,
            "bottom": touches_bottom,
        }

        for (r, c), idx_a_list in tile_dets.items():
            for (dr, dc), (ea, eb, perp, para, perp_m) in cardinal_cfg.items():
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                idx_b_list = tile_dets.get((nr, nc), [])
                if not idx_b_list:
                    continue
                flag_a, flag_b = edge_flag[ea], edge_flag[eb]
                for ia in idx_a_list:
                    if not flag_a[ia]:
                        continue
                    for ib in idx_b_list:
                        if not flag_b[ib]:
                            continue
                        if _should_fuse(ia, ib, perp, para, perp_m):
                            union(ia, ib)

        # ---- Diagonal neighbours ----
        # Both edge pairs must be touched, and the boxes must be close in both axes.
        diagonal_cfg = {
            ( 1,  1): (("bottom", "right"), ("top",    "left")),
            ( 1, -1): (("bottom", "left"),  ("top",    "right")),
            (-1,  1): (("top",    "right"), ("bottom", "left")),
            (-1, -1): (("top",    "left"),  ("bottom", "right")),
        }

        for (r, c), idx_a_list in tile_dets.items():
            for (dr, dc), (edges_a, edges_b) in diagonal_cfg.items():
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                idx_b_list = tile_dets.get((nr, nc), [])
                if not idx_b_list:
                    continue
                fa0, fa1 = edge_flag[edges_a[0]], edge_flag[edges_a[1]]
                fb0, fb1 = edge_flag[edges_b[0]], edge_flag[edges_b[1]]
                for ia in idx_a_list:
                    if not (fa0[ia] and fa1[ia]):
                        continue
                    cls_a = int(classes[ia])
                    for ib in idx_b_list:
                        if not (fb0[ib] and fb1[ib]):
                            continue
                        if int(classes[ib]) != cls_a:
                            continue
                        # For diagonal: check gap in BOTH axes
                        if not _perp_gap_ok(ia, ib, 'h', margin_x):
                            continue
                        if not _perp_gap_ok(ia, ib, 'v', margin_y):
                            continue
                        if not _aspect_ok(ia, ib):
                            continue
                        union(ia, ib)

        # ---- Collect groups & merge ----
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        fused_boxes = []
        fused_classes = []
        fused_scores = []

        for root, members in groups.items():
            if len(members) == 1:
                idx = members[0]
                fused_boxes.append(boxes[idx])
                fused_classes.append(classes[idx])
                fused_scores.append(scores[idx])
            else:
                # Enclosing bounding box (xywh)
                mx1 = min(x1[m] for m in members)
                my1 = min(y1[m] for m in members)
                mx2 = max(x2[m] for m in members)
                my2 = max(y2[m] for m in members)
                merged_box = np.array(
                    [mx1, my1, mx2 - mx1, my2 - my1], dtype=np.float32
                )
                fused_boxes.append(merged_box)
                fused_classes.append(classes[members[0]])
                fused_scores.append(max(float(scores[m]) for m in members))

        fused_boxes = np.array(fused_boxes, dtype=np.float32).reshape(-1, 4)
        fused_classes = np.array(fused_classes, dtype=np.int64)
        fused_scores = np.array(fused_scores, dtype=np.float32)

        return fused_boxes, fused_classes, fused_scores


    def _filter_classes_from_arrays(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
    ):
        """
        Remove detections whose class ID is NOT in self._allowed_class_ids.
        If _allowed_class_ids is None (no filter configured), returns inputs unchanged.

        Args:
            boxes:   (N, 4) float32
            classes: (N,) int64
            scores:  (N,) float32

        Returns:
            (filtered_boxes, filtered_classes, filtered_scores)
        """
        if self._allowed_class_ids is None:
            return boxes, classes, scores
        if boxes is None or len(boxes) == 0:
            return boxes, classes, scores

        mask = np.array([int(c) in self._allowed_class_ids for c in classes], dtype=bool)
        return boxes[mask], classes[mask], scores[mask]


    def _perform_recheck_if_needed(self, saliency_rect=None):
        """
        Recheck tiles that are stale, BUT only if they currently have detections.

        Tiles inside saliency_rect are excluded — they are already covered by the
        main saliency YOLO pass (or will be on the next scheduled YOLO frame), so
        running recheck on them would be redundant.

        When LKT tracking is enabled, stale tiles are already being tracked
        frame-to-frame by Lucas-Kanade, so we skip YOLO and just reset
        the staleness counter.
        """
        self.rechecked_tiles_current = set()

        if not self.enable_recheck_tile:
            return
        if self.recheck_threshold is None or int(self.recheck_threshold) <= 0:
            return

        thr = int(self.recheck_threshold)

        # Build set of tiles already covered by the saliency region
        saliency_tiles = set()
        if saliency_rect is not None:
            rmin, rmax, cmin, cmax = saliency_rect
            for rr in range(rmin, rmax + 1):
                for cc in range(cmin, cmax + 1):
                    saliency_tiles.add((rr, cc))

        def tile_has_cached_dets(r: int, c: int) -> bool:
            det = self.tile_det_cache.get((r, c), None)
            if det is None:
                return False
            b = det.get("boxes", None)
            return (b is not None) and (len(b) > 0)

        stale = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in saliency_tiles:
                    continue  # already handled by saliency pass
                # recheck only if there are detections to "maintain"
                if not tile_has_cached_dets(r, c):
                    continue

                last = int(self.tile_last_visited.get((r, c), 0))
                if (self.frame_counter - last) >= thr:
                    stale.append((r, c))

        if not stale:
            return

        # ---- LKT shortcut: if LKT is enabled, these tiles are already being tracked ----
        # Simply mark them as visited so the staleness counter resets.  No YOLO needed.
        if self.enable_lkt_tracking:
            for (r, c) in stale:
                self.tile_last_visited[(r, c)] = self.frame_counter
                self.rechecked_tiles_current.add((r, c))
            return

        # ── Batch YOLO: collect all stale tile images, run ONE predict_batch ──
        tile_images = []
        tile_coords = []
        for (r, c) in stale:
            tile_images.append(self.tile_array[r, c])
            tile_coords.append((r, c))

        imgsz = max(tile_images[0].size)  # all single tiles share the same size
        yolo = self._get_yolo(imgsz)

        t_yolo_start = time.perf_counter()
        batch_results = yolo.predict_batch(tile_images)
        self.total_yolo_time += (time.perf_counter() - t_yolo_start)
        self.yolo_preds += len(tile_images)

        tw, th = int(self.tile_width), int(self.tile_height)

        for i, (r, c) in enumerate(tile_coords):
            _, boxes_local, classes, scores = batch_results[i]

            boxes_local = np.array(boxes_local, dtype=np.float32) if boxes_local is not None else np.zeros((0, 4), dtype=np.float32)
            classes_arr = np.array(classes, dtype=np.int64) if classes is not None else np.zeros((0,), dtype=np.int64)
            scores_arr  = np.array(scores, dtype=np.float32) if scores is not None else np.zeros((0,), dtype=np.float32)

            # Local -> global offsets
            off_x = c * tw
            off_y = r * th
            boxes_global = boxes_local.copy()
            if len(boxes_global) > 0:
                boxes_global[:, 0] += off_x
                boxes_global[:, 1] += off_y

            # Apply class filter before caching
            boxes_global, classes_arr, scores_arr = self._filter_classes_from_arrays(
                boxes_global, classes_arr, scores_arr)

            # Overwrite cache for this tile
            self.tile_det_cache[(r, c)] = {
                "boxes": boxes_global,
                "classes": classes_arr,
                "scores": scores_arr,
            }
            self._update_tile_object_memory(r, c, classes_arr)
            self.tile_last_visited[(r, c)] = self.frame_counter

            self.rechecked_tiles_current.add((r, c))




    def reconstruct_grid_image(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Build a LIVE annotated frame:
        - start from current frame (live pixels)
        - overlay cached detections (persistent boxes)
        When LKT is enabled, uses colour-coded boxes:
          Green  = YOLO-confirmed this frame
          Orange = LKT-tracked (no YOLO this frame)
        Labels include object ID and source tag (YOLO / LKT).
        Returns BGR for cv2.imshow.
        """

        # Start from live current frame
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"frame_rgb must be (H,W,3) RGB, got {frame_rgb.shape}")

        img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # ---- LKT-aware rendering: draw from tracked objects with colour coding ----
        if self.enable_lkt_tracking and self._tracked_objects:
            COLOR_YOLO = (0, 220, 0)      # green  — freshly YOLO-confirmed
            COLOR_LKT  = (0, 165, 255)    # orange — LKT-tracked

            for obj in self._tracked_objects.values():
                if not obj.active:
                    continue

                x, y, w, h = obj.bbox
                x1 = max(0, min(img_bgr.shape[1] - 1, int(round(float(x)))))
                y1 = max(0, min(img_bgr.shape[0] - 1, int(round(float(y)))))
                x2 = max(0, min(img_bgr.shape[1] - 1, int(round(float(x) + float(w)))))
                y2 = max(0, min(img_bgr.shape[0] - 1, int(round(float(y) + float(h)))))

                is_yolo = (obj.frames_since_yolo == 0)
                color = COLOR_YOLO if is_yolo else COLOR_LKT
                tag = "YOLO" if is_yolo else "LKT"

                # Rectangle
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

                # Label: #ID class conf TAG
                if self.coco_classes is not None and obj.class_id < len(self.coco_classes):
                    name = self.coco_classes[obj.class_id]
                else:
                    name = str(obj.class_id)
                label = f"#{obj.obj_id} {name} {obj.confidence:.2f} {tag}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            return img_bgr

        # ---- Fallback: non-LKT rendering (original yellow boxes) ----
        boxes, classes, scores = self._gather_all_cached_detections()

        if boxes is None or len(boxes) == 0:
            return img_bgr

        for (x, y, w, h), cls, sc in zip(boxes, classes, scores):
            x1 = int(round(float(x)))
            y1 = int(round(float(y)))
            x2 = int(round(float(x) + float(w)))
            y2 = int(round(float(y) + float(h)))

            # Clamp to image
            x1 = max(0, min(img_bgr.shape[1] - 1, x1))
            y1 = max(0, min(img_bgr.shape[0] - 1, y1))
            x2 = max(0, min(img_bgr.shape[1] - 1, x2))
            y2 = max(0, min(img_bgr.shape[0] - 1, y2))

            # Rectangle
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Label text
            if self.coco_classes is not None and int(cls) < len(self.coco_classes):
                name = self.coco_classes[int(cls)]
            else:
                name = str(int(cls))
            label = f"{name} {float(sc):.2f}"

            # Text background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 255), -1)
            cv2.putText(img_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return img_bgr




    def visualize_overlay(self, frame_rgb: np.ndarray, delay_ms: int = 1):
        """
        Live display showing:
        - live frame + persistent boxes
        - visible grid lines
        - attention region highlighted in red
        - rechecked tiles highlighted in blue (border)
        """
        grid_img = self.reconstruct_grid_image(frame_rgb)
        H, W = grid_img.shape[:2]

        x_edges = [int(round(i * W / self.cols)) for i in range(self.cols + 1)]
        y_edges = [int(round(i * H / self.rows)) for i in range(self.rows + 1)]

        # Saliency attention rectangle (drawn first so grid + blue go on top)
        # Bright red = YOLO ran on this region this frame
        # Dim grey   = saliency selected but YOLO skipped (cached detections used)
        mem = self.merged_tiles_memory.get(self.frame_counter)
        if mem is not None and "merged_tiles_idx_map" in mem:
            idx_map = mem["merged_tiles_idx_map"]
            rows_idx, cols_idx = np.where(idx_map)
            if len(rows_idx) > 0 and len(cols_idx) > 0:
                rmin, rmax = int(rows_idx.min()), int(rows_idx.max())
                cmin, cmax = int(cols_idx.min()), int(cols_idx.max())

                x0, x1 = x_edges[cmin], x_edges[cmax + 1]
                y0, y1 = y_edges[rmin], y_edges[rmax + 1]

                if self.yolo_ran_this_frame:
                    fill_color = (0, 0, 255)      # bright red (BGR)
                    border_color = (0, 0, 255)
                    alpha = 0.25
                else:
                    fill_color = (100, 100, 100)   # dim grey
                    border_color = (100, 100, 100)
                    alpha = 0.15

                overlay = grid_img.copy()
                cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), fill_color, thickness=-1)
                grid_img = cv2.addWeighted(overlay, alpha, grid_img, 1.0 - alpha, 0)

                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness=5)
                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), border_color, thickness=3)

        # Grid lines (shadow + green)
        for x in x_edges[1:-1]:
            cv2.line(grid_img, (x, 0), (x, H - 1), (0, 0, 0), thickness=3)
        for y in y_edges[1:-1]:
            cv2.line(grid_img, (0, y), (W - 1, y), (0, 0, 0), thickness=3)

        for x in x_edges[1:-1]:
            cv2.line(grid_img, (x, 0), (x, H - 1), (0, 255, 0), thickness=1)
        for y in y_edges[1:-1]:
            cv2.line(grid_img, (0, y), (W - 1, y), (0, 255, 0), thickness=1)

        # BLUE: rechecked tiles LAST so they are clearly visible on top of grid
        if hasattr(self, "rechecked_tiles_current"):
            for (r, c) in self.rechecked_tiles_current:
                x0, x1 = x_edges[c], x_edges[c + 1]
                y0, y1 = y_edges[r], y_edges[r + 1]
                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness=4)
                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), thickness=2)

        cv2.imshow("Attention Grid", grid_img)
        cv2.waitKey(delay_ms)

    def get_overlay_image(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Same rendering as visualize_overlay but returns the BGR image
        instead of calling cv2.imshow. Useful for headless / file-save mode.
        """
        grid_img = self.reconstruct_grid_image(frame_rgb)
        H, W = grid_img.shape[:2]

        x_edges = [int(round(i * W / self.cols)) for i in range(self.cols + 1)]
        y_edges = [int(round(i * H / self.rows)) for i in range(self.rows + 1)]

        # Saliency attention rectangle (drawn first so grid + blue go on top)
        # Bright red = YOLO ran on this region this frame
        # Dim grey   = saliency selected but YOLO skipped (cached detections used)
        mem = self.merged_tiles_memory.get(self.frame_counter)
        if mem is not None and "merged_tiles_idx_map" in mem:
            idx_map = mem["merged_tiles_idx_map"]
            rows_idx, cols_idx = np.where(idx_map)
            if len(rows_idx) > 0 and len(cols_idx) > 0:
                rmin, rmax = int(rows_idx.min()), int(rows_idx.max())
                cmin, cmax = int(cols_idx.min()), int(cols_idx.max())

                x0, x1 = x_edges[cmin], x_edges[cmax + 1]
                y0, y1 = y_edges[rmin], y_edges[rmax + 1]

                if self.yolo_ran_this_frame:
                    fill_color = (0, 0, 255)      # bright red (BGR)
                    border_color = (0, 0, 255)
                    alpha = 0.25
                else:
                    fill_color = (100, 100, 100)   # dim grey
                    border_color = (100, 100, 100)
                    alpha = 0.15

                overlay = grid_img.copy()
                cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), fill_color, thickness=-1)
                grid_img = cv2.addWeighted(overlay, alpha, grid_img, 1.0 - alpha, 0)

                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness=5)
                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), border_color, thickness=3)

        # Grid lines (shadow + green)
        for x in x_edges[1:-1]:
            cv2.line(grid_img, (x, 0), (x, H - 1), (0, 0, 0), thickness=3)
        for y in y_edges[1:-1]:
            cv2.line(grid_img, (0, y), (W - 1, y), (0, 0, 0), thickness=3)

        for x in x_edges[1:-1]:
            cv2.line(grid_img, (x, 0), (x, H - 1), (0, 255, 0), thickness=1)
        for y in y_edges[1:-1]:
            cv2.line(grid_img, (0, y), (W - 1, y), (0, 255, 0), thickness=1)

        # BLUE: rechecked tiles LAST so they are clearly visible on top of grid
        if hasattr(self, "rechecked_tiles_current"):
            for (r, c) in self.rechecked_tiles_current:
                x0, x1 = x_edges[c], x_edges[c + 1]
                y0, y1 = y_edges[r], y_edges[r + 1]
                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness=4)
                cv2.rectangle(grid_img, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), thickness=2)

        return grid_img


    # =========================================================================
    # TIMING GETTERS (compatible with print_timing_summary)
    # =========================================================================
    def _get_total_saliency_time(self) -> float:
        """Total time spent on saliency computation (frame_diff, etc.)."""
        return self.total_saliency_time

    def _get_total_diff_time(self) -> float:
        """Alias for saliency time (backward compatible with v1)."""
        return self.total_saliency_time

    def _get_combined_yolo_time(self) -> float:
        """Total time spent on all YOLO inference."""
        return self.total_yolo_time

    def _get_total_yolo_time(self) -> float:
        """Alias for combined yolo time."""
        return self.total_yolo_time

    def _get_total_merge_tiles_time(self) -> float:
        """Total time spent on tile merging logic."""
        return self.total_merge_tiles_time

    def _get_total_tiler_time(self) -> float:
        """Total time spent on image tiling operations."""
        return self.total_tiler_time

    def _get_total_overlay_time(self) -> float:
        """Total time spent on overlay/visualization."""
        return self.total_overlay_time

    def _get_total_heavy_yolo_time(self) -> float:
        """V2 only uses HeavyYOLO, so this is same as _get_total_yolo_time (for v1 compatibility)."""
        return self.total_yolo_time

    def _get_combined_yolo_preds(self) -> int:
        """Number of YOLO forward passes."""
        return self.yolo_preds

    def _get_yolo_counters(self):
        """Return (yolo_preds, heavy_yolo_preds) for compatibility (v2 only has one YOLO)."""
        return 0, self.yolo_preds

    def _get_total_lkt_time(self) -> float:
        """Total time spent on Lucas-Kanade-Tomasi tracking."""
        return self.total_lkt_time

    def _get_lkt_track_count(self) -> int:
        """Number of frames where LKT tracking was used."""
        return self.lkt_track_count


