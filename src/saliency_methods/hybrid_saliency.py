"""
Hybrid Motion-Static Saliency Detection

Combines frame differencing (for moving objects) with static saliency detection
(for stationary but salient objects). Also includes temporal persistence to
maintain tracking of objects that stop moving briefly.

This addresses the key limitation of pure frame differencing which fails to
detect stationary objects that are still visually salient.

Approach:
1. Frame Differencing: Detects moving objects via pixel-level changes
2. Spectral Residual Saliency: Fast, lightweight method that detects visually 
   salient objects based on frequency domain analysis (no deep learning needed)
3. Temporal Memory: Maintains saliency for regions that were recently active,
   providing continuity when objects pause briefly
4. Adaptive Fusion: Intelligently combines motion and static saliency masks

Features:
- Works well for both moving AND stationary objects
- Fast enough for real-time applications (~30+ FPS)
- No deep learning required (lightweight)
- Temporal smoothing reduces flickering
- Configurable balance between motion and static saliency
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import time
from collections import deque

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from saliency_methods.frame_diff import merge_direct_neighbors_xywh
from saliency_methods.motion_detection_utils import non_max_suppression


# =============================================================================
# DEFAULT PARAMETERS - Import these in other files for consistency
# =============================================================================
DEFAULT_HYBRID_PARAMS = dict(
    # Constructor parameters
    motion_weight=0.9,        # Weight for motion-based saliency (0-1)
    static_weight=0.1,        # Weight for static saliency (0-1)
    temporal_decay=0.85,      # Temporal memory decay (higher = longer persistence)
    use_temporal_memory=True, # Enable temporal persistence for brief stops
    spectral_scale=6,         # Scale for spectral residual (higher = coarser)
    gaussian_blur=5,          # Gaussian blur kernel size for noise reduction (0=off, higher=smoother)
    # Processing parameters
    threshold=0.3,            # Saliency threshold for binarization (higher = less noise)
    kernel_size=5,            # Morphological kernel size
    open_iterations=2,        # Opening iterations (higher = more small noise removed)
    close_iterations=2,       # Closing iterations
    min_area=80,              # Minimum detection area (higher = filters small noise blobs)
    max_area_ratio=0.4,       # Max area as % of frame (filters large bg)
    nms_threshold=0.3,        # NMS IoU threshold
)


class HybridSaliency:
    """
    Hybrid saliency combining motion detection with static saliency.
    
    Solves the problem of frame differencing missing stationary objects
    by combining:
    - Frame differencing for moving objects
    - Spectral residual saliency for static salient objects
    - Temporal memory for persistence
    """
    
    # Class-level temporal memory (shared across instances for continuity)
    _temporal_buffer = None
    _buffer_size = 10
    
    def __init__(self, frame1, frame2, 
                 motion_weight=0.6,
                 static_weight=0.4,
                 temporal_decay=0.85,
                 use_temporal_memory=True,
                 spectral_scale=6,
                 gaussian_blur=3):
        """
        Initialize hybrid saliency detector.
        
        Args:
            frame1: Previous frame (BGR numpy array)
            frame2: Current frame (BGR numpy array)
            motion_weight: Weight for motion-based saliency (0-1)
            static_weight: Weight for static saliency (0-1) 
            temporal_decay: Decay factor for temporal memory (0-1, higher = longer persistence)
            use_temporal_memory: Whether to use temporal persistence
            spectral_scale: Scale factor for spectral residual (higher = coarser saliency)
            gaussian_blur: Gaussian blur kernel size for noise reduction
        """
        print("HybridSaliency Initiated!")
        
        self.frame1 = frame1
        self.frame2 = frame2
        self.motion_weight = motion_weight
        self.static_weight = static_weight
        self.temporal_decay = temporal_decay
        self.use_temporal_memory = use_temporal_memory
        self.spectral_scale = spectral_scale
        self.gaussian_blur = gaussian_blur
        
        # Convert to grayscale and other color spaces
        self.frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        self.frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        self.frame2_lab = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB)
        
        # Results
        self.motion_saliency = None
        self.static_saliency = None
        self.combined_saliency = None
        self.saliency_map = None
        self.motion_mask = None
        self.nms_bboxes = None
        
        # Initialize temporal buffer if needed
        if self.use_temporal_memory and HybridSaliency._temporal_buffer is None:
            HybridSaliency._temporal_buffer = deque(maxlen=HybridSaliency._buffer_size)
    
    def compute_motion_saliency(self):
        """
        Compute motion-based saliency using frame differencing.
        
        Returns:
            Motion saliency map (H, W) as float32 in [0, 1]
        """
        # Compute absolute difference
        diff = cv2.absdiff(self.frame1_gray, self.frame2_gray)
        
        # Apply Gaussian blur to reduce noise
        if self.gaussian_blur > 0:
            diff = cv2.GaussianBlur(diff, (self.gaussian_blur, self.gaussian_blur), 0)
        
        # Normalize to [0, 1]
        motion_sal = diff.astype(np.float32) / 255.0
        
        # Enhance contrast
        motion_sal = np.clip(motion_sal * 3.0, 0, 1)  # Boost motion signal
        
        self.motion_saliency = motion_sal
        return motion_sal
    
    def compute_spectral_residual_saliency(self):
        """
        Compute static saliency using Spectral Residual method.
        
        This method works in the frequency domain:
        1. Compute log amplitude spectrum
        2. Compute spectral residual (difference from averaged spectrum)
        3. Reconstruct saliency map from residual
        
        Fast and effective for detecting visually distinctive regions.
        
        Returns:
            Static saliency map (H, W) as float32 in [0, 1]
        """
        # Use grayscale for speed, or use L channel from LAB for better results
        img = self.frame2_lab[:, :, 0]  # L channel (luminance)
        
        # Resize for faster processing (spectral residual is scale-invariant)
        scale = self.spectral_scale
        small_h, small_w = img.shape[0] // scale, img.shape[1] // scale
        small_img = cv2.resize(img, (small_w, small_h))
        
        # Compute FFT
        fft = np.fft.fft2(small_img.astype(np.float32))
        
        # Log amplitude spectrum
        amplitude = np.abs(fft)
        log_amplitude = np.log(amplitude + 1e-10)
        
        # Phase spectrum
        phase = np.angle(fft)
        
        # Spectral residual: difference from local average
        # Use a mean filter to get the "average" spectrum
        avg_kernel = np.ones((3, 3), np.float32) / 9
        avg_log_amplitude = cv2.filter2D(log_amplitude.astype(np.float32), -1, avg_kernel)
        spectral_residual = log_amplitude - avg_log_amplitude
        
        # Reconstruct from spectral residual
        saliency_fft = np.exp(spectral_residual) * np.exp(1j * phase)
        saliency_spatial = np.abs(np.fft.ifft2(saliency_fft))
        
        # Square the saliency (as per original paper)
        saliency_spatial = saliency_spatial ** 2
        
        # Gaussian smoothing
        saliency_spatial = cv2.GaussianBlur(saliency_spatial.astype(np.float32), (9, 9), 2.5)
        
        # Resize back to original size
        static_sal = cv2.resize(saliency_spatial, (self.frame2_gray.shape[1], self.frame2_gray.shape[0]))
        
        # Normalize to [0, 1]
        if static_sal.max() > static_sal.min():
            static_sal = (static_sal - static_sal.min()) / (static_sal.max() - static_sal.min())
        else:
            static_sal = np.zeros_like(static_sal)
        
        self.static_saliency = static_sal.astype(np.float32)
        return self.static_saliency
    
    def compute_color_contrast_saliency(self):
        """
        Alternative static saliency based on color contrast.
        
        Detects regions that stand out from their surroundings based on color.
        Useful as a complement to spectral residual.
        
        Returns:
            Color contrast saliency map (H, W) as float32 in [0, 1]
        """
        # Use LAB color space for perceptual color distance
        lab = self.frame2_lab.astype(np.float32)
        
        # Compute mean color of the image
        mean_color = np.mean(lab, axis=(0, 1))
        
        # Compute distance from mean color for each pixel
        diff = lab - mean_color
        color_dist = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Normalize
        if color_dist.max() > color_dist.min():
            color_sal = (color_dist - color_dist.min()) / (color_dist.max() - color_dist.min())
        else:
            color_sal = np.zeros_like(color_dist)
        
        # Smooth
        color_sal = cv2.GaussianBlur(color_sal.astype(np.float32), (5, 5), 1.5)
        
        return color_sal.astype(np.float32)
    
    def apply_temporal_memory(self, current_saliency):
        """
        Apply temporal memory to smooth saliency over time.
        
        Maintains saliency in regions that were recently active, even if
        they're not currently detected. This provides continuity for objects
        that briefly stop moving.
        
        Args:
            current_saliency: Current frame's saliency map
            
        Returns:
            Temporally-smoothed saliency map
        """
        if not self.use_temporal_memory:
            return current_saliency
        
        # Add current saliency to buffer
        HybridSaliency._temporal_buffer.append(current_saliency.copy())
        
        # Compute weighted average with exponential decay
        if len(HybridSaliency._temporal_buffer) == 0:
            return current_saliency
        
        accumulated = np.zeros_like(current_saliency)
        weight_sum = 0
        
        for i, past_sal in enumerate(reversed(list(HybridSaliency._temporal_buffer))):
            weight = self.temporal_decay ** i
            accumulated += weight * past_sal
            weight_sum += weight
        
        if weight_sum > 0:
            accumulated /= weight_sum
        
        # Take max of current and accumulated (preserve strong current detections)
        result = np.maximum(current_saliency, accumulated * 0.7)
        
        return result
    
    def compute_saliency(self):
        """
        Compute combined hybrid saliency map.
        
        Fuses motion saliency and static saliency with temporal smoothing.
        
        Returns:
            Combined saliency map (H, W) as float32 in [0, 1]
        """
        start_time = time.time()
        
        # Compute individual saliency maps
        motion_sal = self.compute_motion_saliency()
        static_sal = self.compute_spectral_residual_saliency()
        
        # Optionally add color contrast
        # color_sal = self.compute_color_contrast_saliency()
        # static_sal = np.maximum(static_sal, color_sal * 0.5)
        
        # Adaptive fusion based on motion level
        motion_level = np.mean(motion_sal)
        
        if motion_level > 0.05:
            # High motion: trust motion more
            effective_motion_weight = min(self.motion_weight * 1.5, 0.9)
            effective_static_weight = 1.0 - effective_motion_weight
        else:
            # Low motion: rely more on static saliency
            effective_motion_weight = self.motion_weight * 0.5
            effective_static_weight = 1.0 - effective_motion_weight
        
        # Weighted combination
        combined = (effective_motion_weight * motion_sal + 
                   effective_static_weight * static_sal)
        
        # Also use OR-like combination for strong signals
        strong_motion = motion_sal > 0.3
        strong_static = static_sal > 0.5
        combined[strong_motion | strong_static] = np.maximum(
            combined[strong_motion | strong_static],
            np.maximum(motion_sal[strong_motion | strong_static], 
                      static_sal[strong_motion | strong_static])
        )
        
        # Apply temporal memory
        combined = self.apply_temporal_memory(combined)
        
        # Normalize final result
        if combined.max() > combined.min():
            combined = (combined - combined.min()) / (combined.max() - combined.min())
        
        self.combined_saliency = combined
        self.saliency_map = combined.astype(np.float32)
        
        inference_time = time.time() - start_time
        print(f"HybridSaliency time: {inference_time:.3f}s | "
              f"motion: [{motion_sal.min():.3f}, {motion_sal.max():.3f}] | "
              f"static: [{static_sal.min():.3f}, {static_sal.max():.3f}] | "
              f"combined: [{combined.min():.3f}, {combined.max():.3f}]")
        
        return self.saliency_map
    
    def get_motion_mask(self, threshold=0.3, kernel_size=5, open_iters=1, close_iters=2):
        """
        Get binary motion mask from combined saliency.
        
        Args:
            threshold: Saliency threshold for binarization
            kernel_size: Morphological kernel size
            open_iters: Opening iterations (noise removal)
            close_iters: Closing iterations (hole filling)
            
        Returns:
            Binary mask (H, W) as uint8
        """
        if self.saliency_map is None:
            self.compute_saliency()
        
        # Threshold
        mask = (self.saliency_map > threshold).astype(np.uint8) * 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
        if close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        
        self.motion_mask = mask
        return mask
    
    def get_filtered_mask(self, threshold=0.3, kernel_size=5, open_iters=1, close_iters=2,
                          min_area=50, max_area_ratio=0.4):
        """
        Get filtered motion mask with area constraints.
        
        Args:
            threshold: Saliency threshold
            kernel_size: Morphological kernel size
            open_iters: Opening iterations
            close_iters: Closing iterations
            min_area: Minimum contour area to keep
            max_area_ratio: Maximum contour area as ratio of frame
            
        Returns:
            Filtered binary mask (H, W) as uint8
        """
        mask = self.get_motion_mask(threshold, kernel_size, open_iters, close_iters)
        
        h, w = mask.shape
        max_area = h * w * max_area_ratio
        
        # Find and filter contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_mask = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        
        return filtered_mask
    
    def get_contour_detections(self, threshold=0.3, kernel_size=5, open_iters=1, close_iters=2,
                               min_area=50, max_area_ratio=0.4, nms_threshold=0.3):
        """
        Get bounding box detections from the saliency mask.
        
        Returns:
            nms_bboxes: numpy array of [x, y, w, h] after NMS
            scores: numpy array of confidence scores
        """
        if self.saliency_map is None:
            self.compute_saliency()
        
        mask = self.get_filtered_mask(threshold, kernel_size, open_iters, close_iters,
                                      min_area, max_area_ratio)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        scores = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append([x, y, w, h])
            
            # Score based on mean saliency in bbox region
            roi = self.saliency_map[y:y+h, x:x+w]
            score = float(np.mean(roi)) if roi.size > 0 else 0.0
            scores.append(score)
        
        if len(bboxes) == 0:
            self.nms_bboxes = np.array([], dtype=int).reshape(0, 4)
            return self.nms_bboxes, np.array([])
        
        # Apply NMS
        bboxes_np = np.array(bboxes)
        scores_np = np.array(scores)
        
        nms_bboxes = non_max_suppression(bboxes_np, scores_np, nms_threshold)
        
        if isinstance(nms_bboxes, np.ndarray):
            self.nms_bboxes = nms_bboxes.astype(int)
        else:
            self.nms_bboxes = np.array([[int(x) for x in box] for box in nms_bboxes])
        
        nms_scores = np.ones(len(self.nms_bboxes))
        
        return self.nms_bboxes, nms_scores
    
    @staticmethod
    def total_box_area(boxes):
        """Calculate total area of bounding boxes."""
        if boxes is None or len(boxes) == 0:
            return 0
        total = 0
        for box in boxes:
            x, y, w, h = box[:4]
            total += w * h
        return total
    
    @staticmethod
    def reset_temporal_memory():
        """Reset the temporal memory buffer."""
        HybridSaliency._temporal_buffer = None
        print("Temporal memory reset")
    
    def auto_run(self, plot=False, saliency_measurement="bbox_area",
                 threshold=0.3, kernel_size=5, open_iterations=1, close_iterations=2,
                 min_area=50, max_area_ratio=0.4, nms_threshold=0.3):
        """
        Run full hybrid saliency detection pipeline.
        
        Compatible with AttentionGrid and saliency_visualizer interfaces.
        
        Args:
            plot: Whether to display results (for debugging)
            saliency_measurement: "bbox_area" or "pixel_count"
            threshold: Saliency threshold
            kernel_size: Morphological kernel size
            open_iterations: Opening iterations (noise removal)
            close_iterations: Closing iterations (hole filling)
            min_area: Minimum detection area
            max_area_ratio: Maximum area as ratio of frame
            nms_threshold: NMS IoU threshold
            
        Returns:
            saliency_score: Total saliency measure
            nms_bboxes: Bounding boxes after NMS
            mask: Binary saliency mask
            saliency_map: Raw saliency map
        """
        # Compute saliency
        saliency_map = self.compute_saliency()
        
        # Get detections
        nms_bboxes, scores = self.get_contour_detections(
            threshold=threshold,
            kernel_size=kernel_size,
            open_iters=open_iterations,
            close_iters=close_iterations,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            nms_threshold=nms_threshold
        )
        
        # Get mask
        mask = self.get_filtered_mask(threshold, kernel_size, open_iterations, close_iterations, min_area, max_area_ratio)
        
        # Compute saliency score (total_area)
        if saliency_measurement == "bbox_area":
            total_area = self.total_box_area(nms_bboxes)
            merged_bboxes = nms_bboxes
        else:  # mask_area, pixel_count, or mask
            total_area = int(np.sum(mask > 0))
            merged_bboxes = nms_bboxes
        
        if plot:
            self._visualize(saliency_map, mask, nms_bboxes)
        
        # Return consistent 3-tuple: (merged_bboxes, total_area, mask)
        return merged_bboxes, total_area, mask
    
    def _visualize(self, saliency_map, mask, bboxes):
        """Visualize results for debugging."""
        # Create visualization
        vis_frame = self.frame2.copy()
        
        # Draw bboxes
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show saliency maps
        motion_vis = (self.motion_saliency * 255).astype(np.uint8) if self.motion_saliency is not None else np.zeros_like(self.frame2_gray)
        static_vis = (self.static_saliency * 255).astype(np.uint8) if self.static_saliency is not None else np.zeros_like(self.frame2_gray)
        combined_vis = (saliency_map * 255).astype(np.uint8)
        
        # Stack for display
        top_row = np.hstack([
            cv2.cvtColor(motion_vis, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(static_vis, cv2.COLOR_GRAY2BGR)
        ])
        bottom_row = np.hstack([
            cv2.cvtColor(combined_vis, cv2.COLOR_GRAY2BGR),
            vis_frame
        ])
        
        # Resize if too large
        display = np.vstack([top_row, bottom_row])
        h, w = display.shape[:2]
        if w > 1200:
            scale = 1200 / w
            display = cv2.resize(display, None, fx=scale, fy=scale)
        
        cv2.imshow("Hybrid Saliency (Motion | Static | Combined | Result)", display)
        cv2.waitKey(1)


# Convenience function for direct use
def compute_hybrid_saliency(frame1, frame2, **kwargs):
    """
    Compute hybrid saliency combining motion and static detection.
    
    Args:
        frame1: Previous frame (BGR)
        frame2: Current frame (BGR)
        **kwargs: Additional arguments for HybridSaliency
        
    Returns:
        saliency_map: Combined saliency map [0, 1]
        motion_mask: Binary mask
        bboxes: Detected bounding boxes
    """
    detector = HybridSaliency(frame1, frame2, **kwargs)
    saliency_score, bboxes, mask, saliency_map = detector.auto_run()
    return saliency_map, mask, bboxes


if __name__ == "__main__":
    import glob
    
    # Test with dataset
    img_paths = sorted(glob.glob("dataset_20fps/*.png"))[:20]
    
    if len(img_paths) < 2:
        print("Need at least 2 images in dataset_20fps/")
        sys.exit(1)
    
    print(f"Testing HybridSaliency with {len(img_paths)} images...")
    
    for i in range(1, len(img_paths)):
        frame1 = cv2.imread(img_paths[i-1])
        frame2 = cv2.imread(img_paths[i])
        
        detector = HybridSaliency(frame1, frame2)
        saliency_score, bboxes, mask, saliency_map = detector.auto_run(plot=True)
        
        print(f"Frame {i}: saliency_score={saliency_score}, {len(bboxes)} boxes")
        
        key = cv2.waitKey(100)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
