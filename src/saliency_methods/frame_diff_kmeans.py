"""
Frame Difference + K-Means Clustering Saliency Method

This method combines traditional frame differencing with K-means clustering
to produce cleaner motion masks by removing isolated noise pixels.

The approach:
1. Compute frame difference mask (standard approach)
2. Extract spatial coordinates of foreground pixels
3. Apply K-means clustering to group pixels into spatial clusters
4. Remove clusters that are too small (likely noise) or too sparse
5. Reconstruct a cleaner mask from the remaining clusters

Author: Auto-generated for AttentionGrid project
"""

import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.ndimage import label as scipy_label
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from saliency_methods.frame_diff import FrameDiff, merge_direct_neighbors_xywh


# Default parameters for the K-means enhanced frame diff
DEFAULT_KMEANS_FD_PARAMS = {
    "n_clusters": 3,              # Number of K-means clusters (will be auto-adjusted)
    "min_cluster_size": 20,       # Minimum pixels per cluster to keep
    "min_cluster_density": 0.03,  # Min density (pixels in cluster / cluster bbox area)
    "use_minibatch": True,        # Use MiniBatchKMeans for speed
    "max_pixels_for_kmeans": 10000,  # Downsample if more pixels than this
    "morphology_close_size": 25,  # Closing kernel size - CRITICAL for filling edges (21-25 optimal)
    "morphology_open_size": 5,    # Opening kernel size after clustering
}


class FrameDiffKMeans:
    """
    Frame Difference with K-Means clustering for noise reduction.
    
    The basic FrameDiff can produce noisy masks with scattered pixels.
    This class applies K-means clustering to the foreground pixel coordinates
    to identify coherent object regions and filter out noise.
    """
    
    def __init__(self, frame1: np.ndarray, frame2: np.ndarray,
                 n_clusters: int = 5,
                 min_cluster_size: int = 50,
                 min_cluster_density: float = 0.1,
                 use_minibatch: bool = True,
                 max_pixels_for_kmeans: int = 10000,
                 morphology_close_size: int = 5,
                 morphology_open_size: int = 3):
        """
        Args:
            frame1: First frame (BGR numpy array)
            frame2: Second frame (BGR numpy array)
            n_clusters: Number of K-means clusters (auto-adjusted based on pixel count)
            min_cluster_size: Minimum number of pixels in a cluster to keep it
            min_cluster_density: Minimum density ratio (cluster_pixels / bbox_area)
            use_minibatch: Use MiniBatchKMeans for faster clustering
            max_pixels_for_kmeans: Downsample if more foreground pixels than this
            morphology_close_size: Size of closing kernel after reconstruction
            morphology_open_size: Size of opening kernel after reconstruction
        """
        self.frame1 = frame1
        self.frame2 = frame2
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.min_cluster_density = min_cluster_density
        self.use_minibatch = use_minibatch
        self.max_pixels_for_kmeans = max_pixels_for_kmeans
        self.morphology_close_size = morphology_close_size
        self.morphology_open_size = morphology_open_size
        
        # Will be populated during processing
        self.raw_mask = None        # Original frame diff mask
        self.clean_mask = None      # K-means cleaned mask
        self.cluster_labels = None  # Cluster assignments
        self.kept_clusters = None   # Which clusters were kept
        
        print("FrameDiffKMeans Initiated!")
    
    def _get_raw_frame_diff_mask(self) -> np.ndarray:
        """Get the raw frame difference mask using FrameDiff."""
        fd = FrameDiff(self.frame1, self.frame2)
        mask = fd.get_mask()
        mask = fd.build_binary_mask(mask)  # Ensure 0/255
        self.raw_mask = mask
        return mask
    
    def _extract_foreground_coords(self, mask: np.ndarray) -> np.ndarray:
        """Extract (y, x) coordinates of foreground pixels."""
        ys, xs = np.where(mask > 0)
        coords = np.column_stack([ys, xs])  # Shape: (N, 2)
        return coords
    
    def _apply_kmeans(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply K-means clustering to pixel coordinates.
        
        Returns:
            labels: Cluster label for each pixel
            centers: Cluster center coordinates
        """
        n_pixels = len(coords)
        
        if n_pixels == 0:
            return np.array([]), np.array([])
        
        # Adjust number of clusters based on pixel count
        actual_n_clusters = min(self.n_clusters, max(1, n_pixels // self.min_cluster_size))
        
        # Downsample if too many pixels
        if n_pixels > self.max_pixels_for_kmeans:
            sample_idx = np.random.choice(n_pixels, self.max_pixels_for_kmeans, replace=False)
            coords_sample = coords[sample_idx]
        else:
            coords_sample = coords
            sample_idx = None
        
        # Apply K-means
        if self.use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=actual_n_clusters,
                random_state=42,
                batch_size=min(1024, len(coords_sample)),
                n_init=3
            )
        else:
            kmeans = KMeans(
                n_clusters=actual_n_clusters,
                random_state=42,
                n_init=10
            )
        
        kmeans.fit(coords_sample)
        centers = kmeans.cluster_centers_
        
        # Predict labels for all pixels (not just sample)
        if sample_idx is not None:
            labels = kmeans.predict(coords)
        else:
            labels = kmeans.labels_
        
        return labels, centers
    
    def _filter_clusters(self, coords: np.ndarray, labels: np.ndarray,
                         mask_shape: Tuple[int, int]) -> List[int]:
        """
        Filter clusters based on size and density criteria.
        
        Returns:
            List of cluster indices to keep
        """
        unique_labels = np.unique(labels)
        kept_clusters = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_coords = coords[cluster_mask]
            cluster_size = len(cluster_coords)
            
            # Size check
            if cluster_size < self.min_cluster_size:
                continue
            
            # Density check: compute bounding box of cluster
            y_min, y_max = cluster_coords[:, 0].min(), cluster_coords[:, 0].max()
            x_min, x_max = cluster_coords[:, 1].min(), cluster_coords[:, 1].max()
            bbox_area = max(1, (y_max - y_min + 1) * (x_max - x_min + 1))
            density = cluster_size / bbox_area
            
            if density < self.min_cluster_density:
                continue
            
            kept_clusters.append(label)
        
        return kept_clusters
    
    def _reconstruct_mask(self, coords: np.ndarray, labels: np.ndarray,
                          kept_clusters: List[int], mask_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct a clean mask from kept clusters."""
        clean_mask = np.zeros(mask_shape, dtype=np.uint8)
        
        for label in kept_clusters:
            cluster_mask = labels == label
            cluster_coords = coords[cluster_mask]
            clean_mask[cluster_coords[:, 0], cluster_coords[:, 1]] = 255
        
        # Apply morphological operations to fill gaps and remove small artifacts
        if self.morphology_close_size > 0:
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.morphology_close_size, self.morphology_close_size)
            )
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
        
        if self.morphology_open_size > 0:
            kernel_open = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morphology_open_size, self.morphology_open_size)
            )
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)
        
        return clean_mask
    
    def get_mask(self) -> np.ndarray:
        """
        Compute the K-means enhanced frame difference mask.
        
        Returns:
            clean_mask: Binary mask (0/255) with noise removed
        """
        # Step 1: Get raw frame diff mask
        raw_mask = self._get_raw_frame_diff_mask()
        
        # Step 2: Extract foreground coordinates
        coords = self._extract_foreground_coords(raw_mask)
        
        if len(coords) == 0:
            self.clean_mask = np.zeros_like(raw_mask)
            return self.clean_mask
        
        # Step 3: Apply K-means clustering
        labels, centers = self._apply_kmeans(coords)
        self.cluster_labels = labels
        
        if len(labels) == 0:
            self.clean_mask = np.zeros_like(raw_mask)
            return self.clean_mask
        
        # Step 4: Filter clusters
        kept_clusters = self._filter_clusters(coords, labels, raw_mask.shape)
        self.kept_clusters = kept_clusters
        
        # Step 5: Reconstruct clean mask
        clean_mask = self._reconstruct_mask(coords, labels, kept_clusters, raw_mask.shape)
        self.clean_mask = clean_mask
        
        return clean_mask
    
    def get_raw_mask(self) -> np.ndarray:
        """Get the raw frame diff mask without K-means filtering."""
        if self.raw_mask is None:
            self._get_raw_frame_diff_mask()
        return self.raw_mask
    
    def get_contour_detections(self, mask: np.ndarray, thresh: int = 0, 
                                plot_bboxes: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding boxes from the mask (same interface as FrameDiff).
        """
        H, W = mask.shape[:2]
        min_area = max(100, int(0.0005 * W * H))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            a = w * h
            if a < min_area:
                continue
            
            roi = mask[y:y+h, x:x+w]
            fill_ratio = cv2.countNonZero(roi) / float(a)
            
            cnt_area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = max(1.0, cv2.contourArea(hull))
            solidity = cnt_area / hull_area
            
            ar = w / float(h + 1e-6)
            ok_ar = 0.2 <= ar <= 5.0
            
            if fill_ratio >= 0.15 and solidity >= 0.15 and ok_ar:
                detections.append([x, y, x+w, y+h, a])
        
        if len(detections) == 0:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)
        
        detections_array = np.array(detections, dtype=int)
        bboxes = detections_array[:, :4]
        scores = detections_array[:, -1].astype(float)
        
        return bboxes, scores
    
    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, 
                            threshold: float = 0.1) -> np.ndarray:
        """Apply NMS to bounding boxes (same interface as FrameDiff)."""
        if len(boxes) == 0:
            return boxes
        
        # Sort by score descending
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        
        keep = []
        used = np.zeros(len(boxes), dtype=bool)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            keep.append(i)
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                # Calculate IoU
                xi1 = max(boxes[i][0], boxes[j][0])
                yi1 = max(boxes[i][1], boxes[j][1])
                xi2 = min(boxes[i][2], boxes[j][2])
                yi2 = min(boxes[i][3], boxes[j][3])
                
                inter_w = max(0, xi2 - xi1)
                inter_h = max(0, yi2 - yi1)
                inter_area = inter_w * inter_h
                
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                union_area = area_i + area_j - inter_area
                
                iou = inter_area / max(union_area, 1e-6)
                
                if iou > threshold:
                    used[j] = True
        
        return boxes[keep]
    
    def total_box_area(self, boxes: np.ndarray) -> int:
        """Calculate total area of all boxes."""
        if boxes is None or len(boxes) == 0:
            return 0
        b = np.asarray(boxes, dtype=int)
        w = np.clip(b[:, 2] - b[:, 0], a_min=0, a_max=None)
        h = np.clip(b[:, 3] - b[:, 1], a_min=0, a_max=None)
        return int(np.sum(w * h))
    
    def auto_run(self, plot: bool = False, 
                 saliency_measurement: str = "bbox_area") -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
        """
        Main entry point - same interface as FrameDiff.auto_run()
        
        Args:
            plot: Whether to show visualization
            saliency_measurement: "bbox_area" or "pixel_count"
        
        Returns:
            (merged_bboxes, total_area, mask)
        """
        # Accept legacy alias "mask" → "pixel_count"
        if saliency_measurement == "mask":
            saliency_measurement = "pixel_count"

        # Get the K-means cleaned mask
        mask = self.get_mask()
        
        if plot:
            self.visualize_comparison()
        
        if saliency_measurement == "pixel_count":
            total_area = cv2.countNonZero(mask)
            merged_bboxes = None
        
        elif saliency_measurement == "bbox_area":
            bboxes, scores = self.get_contour_detections(mask)
            nms_bboxes = self.non_max_suppression(bboxes, scores, threshold=0.01)
            merged_bboxes = nms_bboxes
            total_area = self.total_box_area(merged_bboxes)
        
        else:
            raise ValueError(f"Unknown saliency_measurement: {saliency_measurement}")
        
        return merged_bboxes, total_area, mask
    
    def visualize_comparison(self, figsize: Tuple[int, int] = (15, 5)):
        """Visualize raw vs K-means cleaned mask."""
        if self.raw_mask is None:
            self._get_raw_frame_diff_mask()
        if self.clean_mask is None:
            self.get_mask()
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Frame 1
        frame1_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        axes[0].imshow(frame1_rgb)
        axes[0].set_title("Frame 1")
        axes[0].axis('off')
        
        # Frame 2
        frame2_rgb = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)
        axes[1].imshow(frame2_rgb)
        axes[1].set_title("Frame 2")
        axes[1].axis('off')
        
        # Raw mask
        axes[2].imshow(self.raw_mask, cmap='gray')
        raw_pixels = cv2.countNonZero(self.raw_mask)
        axes[2].set_title(f"Raw Frame Diff\n({raw_pixels:,} pixels)")
        axes[2].axis('off')
        
        # K-means cleaned mask
        axes[3].imshow(self.clean_mask, cmap='gray')
        clean_pixels = cv2.countNonZero(self.clean_mask)
        reduction = 100 * (1 - clean_pixels / max(1, raw_pixels))
        axes[3].set_title(f"K-Means Cleaned\n({clean_pixels:,} pixels, {reduction:.1f}% reduction)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()


def compare_masks_with_gt(frame1_path: str, frame2_path: str, gt_mask_path: str,
                          kmeans_params: dict = None, figsize: Tuple[int, int] = (20, 5)):
    """
    Compare raw frame diff, K-means cleaned, and ground truth masks.
    
    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame
        gt_mask_path: Path to ground truth mask
        kmeans_params: Optional dict of K-means parameters
        figsize: Figure size for visualization
    """
    # Load frames
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if frame1 is None or frame2 is None:
        raise ValueError(f"Could not load frames: {frame1_path}, {frame2_path}")
    if gt_mask is None:
        raise ValueError(f"Could not load GT mask: {gt_mask_path}")
    
    # Binarize GT mask (handle RGB gt masks)
    gt_mask_orig = cv2.imread(gt_mask_path)
    if gt_mask_orig is not None and len(gt_mask_orig.shape) == 3:
        # Convert to grayscale first
        gt_mask = cv2.cvtColor(gt_mask_orig, cv2.COLOR_BGR2GRAY)
    gt_mask = (gt_mask > 0).astype(np.uint8) * 255
    
    # Get raw frame diff mask
    fd = FrameDiff(frame1, frame2)
    raw_mask = fd.get_mask()
    raw_mask = fd.build_binary_mask(raw_mask)
    
    # Get K-means cleaned mask
    params = kmeans_params or DEFAULT_KMEANS_FD_PARAMS
    fd_kmeans = FrameDiffKMeans(frame1, frame2, **params)
    clean_mask = fd_kmeans.get_mask()
    
    # Calculate metrics
    def calc_iou(pred, gt):
        pred_bin = pred > 0
        gt_bin = gt > 0
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        return intersection / max(union, 1)
    
    def calc_precision_recall(pred, gt):
        pred_bin = pred > 0
        gt_bin = gt > 0
        tp = np.logical_and(pred_bin, gt_bin).sum()
        fp = np.logical_and(pred_bin, ~gt_bin).sum()
        fn = np.logical_and(~pred_bin, gt_bin).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return precision, recall
    
    raw_iou = calc_iou(raw_mask, gt_mask)
    clean_iou = calc_iou(clean_mask, gt_mask)
    
    raw_prec, raw_rec = calc_precision_recall(raw_mask, gt_mask)
    clean_prec, clean_rec = calc_precision_recall(clean_mask, gt_mask)
    
    # Visualization
    fig, axes = plt.subplots(1, 5, figsize=figsize)
    
    # Frame 2 (current frame)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    axes[0].imshow(frame2_rgb)
    axes[0].set_title("Current Frame")
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_mask, cmap='gray')
    gt_pixels = cv2.countNonZero(gt_mask)
    axes[1].set_title(f"Ground Truth\n({gt_pixels:,} pixels)")
    axes[1].axis('off')
    
    # Raw Frame Diff
    axes[2].imshow(raw_mask, cmap='gray')
    raw_pixels = cv2.countNonZero(raw_mask)
    axes[2].set_title(f"Raw Frame Diff\nIoU: {raw_iou:.3f} | P: {raw_prec:.2f} R: {raw_rec:.2f}\n({raw_pixels:,} px)")
    axes[2].axis('off')
    
    # K-Means Cleaned
    axes[3].imshow(clean_mask, cmap='gray')
    clean_pixels = cv2.countNonZero(clean_mask)
    axes[3].set_title(f"K-Means Cleaned\nIoU: {clean_iou:.3f} | P: {clean_prec:.2f} R: {clean_rec:.2f}\n({clean_pixels:,} px)")
    axes[3].axis('off')
    
    # Overlay comparison (Green=TP, Red=FP, Blue=FN)
    H, W = gt_mask.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    
    pred_bin = clean_mask > 0
    gt_bin = gt_mask > 0
    
    # True Positive (Green)
    tp_mask = np.logical_and(pred_bin, gt_bin)
    overlay[tp_mask] = [0, 255, 0]
    
    # False Positive (Red)
    fp_mask = np.logical_and(pred_bin, ~gt_bin)
    overlay[fp_mask] = [255, 0, 0]
    
    # False Negative (Blue)
    fn_mask = np.logical_and(~pred_bin, gt_bin)
    overlay[fn_mask] = [0, 0, 255]
    
    axes[4].imshow(overlay)
    axes[4].set_title("K-Means vs GT\nGreen=TP, Red=FP, Blue=FN")
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Ground Truth pixels: {gt_pixels:,}")
    print(f"\nRaw Frame Diff:")
    print(f"  Pixels: {raw_pixels:,}")
    print(f"  IoU: {raw_iou:.4f}")
    print(f"  Precision: {raw_prec:.4f}, Recall: {raw_rec:.4f}")
    print(f"\nK-Means Cleaned:")
    print(f"  Pixels: {clean_pixels:,}")
    print(f"  IoU: {clean_iou:.4f}")
    print(f"  Precision: {clean_prec:.4f}, Recall: {clean_rec:.4f}")
    print(f"\nImprovement:")
    print(f"  IoU: {clean_iou - raw_iou:+.4f}")
    print(f"  Precision: {clean_prec - raw_prec:+.4f}")
    print(f"  Noise reduction: {100 * (1 - clean_pixels / max(1, raw_pixels)):.1f}%")
    print("=" * 60)
    
    return {
        "raw_iou": raw_iou,
        "clean_iou": clean_iou,
        "raw_precision": raw_prec,
        "raw_recall": raw_rec,
        "clean_precision": clean_prec,
        "clean_recall": clean_rec,
    }


# ---------------------------------------------------------------------------
# Export masks to CVAT-compatible zip
# ---------------------------------------------------------------------------

import os
import zipfile
import shutil
from pathlib import Path


def export_masks_to_zip(
    img_paths,
    zip_path,
    start_idx=0,
    end_idx=None,
    subset_name="default",
    motion_color_rgb=(0, 255, 0),
    background_color_rgb=(0, 0, 0),
    kmeans_params=None,
):
    """
    Runs FrameDiffKMeans on all consecutive frame pairs (t, t+1) and exports
    ONE CVAT "Segmentation mask 1.1" zip containing ALL masks.

    IMPORTANT:
    - CVAT matches masks by filename stem. So image_id MUST equal your CVAT
      image stem.
      Example: if CVAT image is 'ezgif-frame-019.png' -> image_id must be
      'ezgif-frame-019'
    - This function uses the stem of img_paths[t] as image_id automatically.

    Args:
        img_paths: list of image file paths (str or Path)
        zip_path: output zip file path
        start_idx: first frame index (inclusive)
        end_idx: last frame index (inclusive). None = all pairs.
        subset_name: name for the ImageSets file
        motion_color_rgb: RGB colour for motion pixels in saved mask
        background_color_rgb: RGB colour for background pixels
        kmeans_params: optional dict overriding DEFAULT_KMEANS_FD_PARAMS
    """
    params = {**DEFAULT_KMEANS_FD_PARAMS, **(kmeans_params or {})}
    img_paths = [str(p) for p in img_paths]
    if end_idx is None:
        end_idx = len(img_paths) - 2
    end_idx = min(end_idx, len(img_paths) - 2)

    # temp structure
    tmp_root = os.path.splitext(zip_path)[0] + "_tmp"
    seg_class_dir = os.path.join(tmp_root, "SegmentationClass")
    imagesets_dir = os.path.join(tmp_root, "ImageSets", "Segmentation")
    os.makedirs(seg_class_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)

    image_ids = []

    for t in range(start_idx, end_idx + 1):
        frame1 = cv2.imread(img_paths[t])
        frame2 = cv2.imread(img_paths[t + 1])

        fd_kmeans = FrameDiffKMeans(
            frame1=frame1,
            frame2=frame2,
            n_clusters=params.get("n_clusters", 3),
            min_cluster_size=params.get("min_cluster_size", 20),
            min_cluster_density=params.get("min_cluster_density", 0.03),
            use_minibatch=params.get("use_minibatch", True),
            max_pixels_for_kmeans=params.get("max_pixels_for_kmeans", 10000),
            morphology_close_size=params.get("morphology_close_size", 25),
            morphology_open_size=params.get("morphology_open_size", 5),
        )
        mask = fd_kmeans.get_mask()         # cleaned binary mask (0/255)
        mask01 = (mask > 0).astype(np.uint8)  # 0/1

        # Use the actual filename stem so CVAT import matches images
        image_id = Path(img_paths[t]).stem
        image_ids.append(image_id)

        # Save as RGB (visible) mask
        h, w = mask01.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:, :] = np.array(background_color_rgb, dtype=np.uint8)
        out[mask01 == 1] = np.array(motion_color_rgb, dtype=np.uint8)
        cv2.imwrite(os.path.join(seg_class_dir, f"{image_id}.png"), out)

    # labelmap (must match your task labels)
    labelmap_txt = (
        "# label : color (RGB) : 'body' parts : actions\n"
        f"background:{background_color_rgb[0]},{background_color_rgb[1]},{background_color_rgb[2]}::\n"
        f"motion:{motion_color_rgb[0]},{motion_color_rgb[1]},{motion_color_rgb[2]}::\n"
    )
    with open(os.path.join(tmp_root, "labelmap.txt"), "w", encoding="utf-8") as f:
        f.write(labelmap_txt)

    # list file
    with open(os.path.join(imagesets_dir, f"{subset_name}.txt"), "w", encoding="utf-8") as f:
        for image_id in image_ids:
            f.write(f"{image_id}\n")

    # zip
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_root):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, tmp_root)
                z.write(full, rel)

    shutil.rmtree(tmp_root)
    print(f"[OK] Wrote CVAT Segmentation mask 1.1 zip with {len(image_ids)} masks -> {zip_path}")


# ---------------------------------------------------------------------------
# Main – single-pair visualisation + optional zip export
# ---------------------------------------------------------------------------


def main():
    from saliency_methods.frame_diff import load_img_paths

    print(__name__)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_script_dir)
    path = os.path.join(_project_dir, "dataset_20fps")
    img_paths = load_img_paths(path=path, print_paths=True)
    img_paths = [str(p) for p in img_paths]

    idx = 185

    frame1 = cv2.imread(img_paths[idx])
    frame2 = cv2.imread(img_paths[idx + 1])

    fd_kmeans = FrameDiffKMeans(
        frame1=frame1,
        frame2=frame2,
        **DEFAULT_KMEANS_FD_PARAMS,
    )

    merged_bboxes, total_area, mask = fd_kmeans.auto_run(
        plot=False, saliency_measurement="pixel_count"
    )

    # Show visual comparison (raw vs K-means cleaned)
    fd_kmeans.visualize_comparison()

    print("Merged BBoxes:", merged_bboxes)
    print("Total Area:", total_area)

    # --- Export all masks to CVAT zip (uncomment to use) ---
    # zip_name = "frame_diff_kmeans_masks.zip"
    # export_masks_to_zip(
    #     img_paths=img_paths,
    #     zip_path=f"masks/{zip_name}",
    #     start_idx=0,
    #     end_idx=None,   # all pairs
    # )


if __name__ == "__main__":
    main()
