"""
Temporal Phase Saliency - Motion Saliency Detection using Temporal Fourier Transform

Implementation based on:
    Chen et al. "Motion saliency detection using a temporal fourier transform"
    Pattern Recognition Letters, 2017

Algorithm Summary:
    1. Build temporal sequence I(x,y,t) from consecutive frames (zero-padded)
    2. Compute FFT along temporal axis: F(x,y,ω) = FFT_t{I(x,y,t)}
    3. Extract phase spectrum: φ(x,y,ω) = angle(F)
    4. Phase-only reconstruction: I'(x,y,t) = IFFT_ω{exp(j·φ)}
    5. Saliency map: S(x,y) = |I'(x,y,t=1)|  (magnitude at second frame)
    6. Spatial Gaussian smoothing to reduce noise
    7. Adaptive threshold: τ = μ + k·σ  (paper uses k=2)
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Union, Optional, Tuple, List
import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from saliency_methods.frame_diff import load_img_paths
from saliency_methods.motion_detection_utils import non_max_suppression, draw_bboxes


# =============================================================================
# DEFAULT PARAMETERS - Import these in other files for consistency
# =============================================================================
DEFAULT_TPS_PARAMS = dict(
    spatial_sigma=3.0,        # Spatial Gaussian smoothing (paper: 3-5)
    threshold_k=2.0,          # Adaptive threshold: μ + k·σ (paper: k=2)
    morph_kernel_size=5,      # Morphological kernel size
    open_iterations=1,        # Opening iterations
    close_iterations=2,       # Closing iterations
    min_area=30,              # Minimum detection area
    nms_threshold=0.3,        # NMS IoU threshold
    saliency_measurement="bbox_area",
)


class TemporalPhaseSaliency:
    """
    Motion saliency using Temporal Fourier Transform phase spectrum.
    
    Based on Chen et al. "Motion saliency detection using a temporal fourier transform"
    """

    def __init__(self, frame1: np.ndarray, frame2: np.ndarray):
        """
        Initialize with two consecutive BGR frames.
        
        Args:
            frame1: Previous frame (BGR, uint8)
            frame2: Current frame (BGR, uint8)
        """
        # Store original frames
        self.frame1_bgr = frame1
        self.frame2_bgr = frame2
        self.frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        self.frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale and normalize to [0, 1] as per paper
        self.img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Pipeline outputs
        self.temporal_sequence = None
        self.phase = None
        self.reconstruction = None
        self.saliency_map = None
        self.motion_mask = None
        
        # Detection outputs
        self.bboxes = None
        self.scores = None
        self.nms_bboxes = None
        self.total_box_area_value = 0

    def build_temporal_sequence(self, T: int = 2) -> np.ndarray:
        """
        Build temporal sequence I(x,y,t) from two frames.
        
        Paper Eq. (4): Zero-pad to length T for FFT.
        Using T=2 is the minimal case (just the two frames).
        Larger T provides better frequency resolution but T=2 works well for motion.
        
        Args:
            T: Temporal sequence length (minimum 2)
            
        Returns:
            Temporal sequence array of shape (H, W, T)
        """
        if T < 2:
            raise ValueError("T must be >= 2")
            
        h, w = self.img1.shape
        seq = np.zeros((h, w, T), dtype=np.float32)
        seq[..., 0] = self.img1
        seq[..., 1] = self.img2
        # Remaining positions stay zero (zero-padding)
        
        self.temporal_sequence = seq
        return seq

    def compute_phase_reconstruction(
        self,
        use_frequency_filter: bool = False,
        freq_sigma: float = 5.0
    ) -> np.ndarray:
        """
        Compute phase-only reconstruction.
        
        Two modes available:
        
        1. PAPER METHOD (use_frequency_filter=False):
           Paper Eq. (5-7):
               F(x,y,ω) = FFT_t{I(x,y,t)}           -- FFT along time
               φ(x,y,ω) = angle(F(x,y,ω))           -- extract phase
               I'(x,y,t) = IFFT_ω{exp(j·φ(x,y,ω))} -- phase-only reconstruction (unit magnitude)
        
        2. FREQUENCY FILTER METHOD (use_frequency_filter=True):
           Your previous implementation that gave better results.
           Applies a Gaussian filter in the frequency domain before IFFT:
               I'(x,y,t) = IFFT_ω{g(ω) · exp(j·φ(x,y,ω))}
           where g(ω) is a 1D Gaussian along the frequency axis.
           This provides additional smoothing/filtering of the phase response.
        
        Args:
            use_frequency_filter: If True, uses your previous frequency-domain Gaussian.
                                  If False, uses paper's pure phase-only method.
            freq_sigma: Sigma for frequency-domain Gaussian (only used if use_frequency_filter=True).
                        Higher values = more frequencies retained, sharper response.
                        Lower values = fewer frequencies, smoother response.
        
        Returns:
            Reconstruction array of shape (H, W, T)
        """
        if self.temporal_sequence is None:
            raise ValueError("Call build_temporal_sequence() first")
        
        # FFT along temporal axis (axis=2)
        F = np.fft.fft(self.temporal_sequence, axis=2)
        
        # Extract phase
        self.phase = np.angle(F)
        
        # Phase-only reconstruction: unit magnitude with original phase
        phase_only = np.exp(1j * self.phase)
        
        if use_frequency_filter:
            # YOUR PREVIOUS METHOD: Apply Gaussian in frequency domain
            # This was in your phase_only_reconstruction_paper() function
            T = phase_only.shape[2]
            g = cv2.getGaussianKernel(ksize=T if T % 2 == 1 else T - 1, sigma=freq_sigma).astype(np.float32)[:, 0]
            if g.shape[0] != T:
                g = np.pad(g, (0, T - g.shape[0]), mode="edge")
            # Multiply phase-only spectrum by frequency Gaussian
            phase_only = phase_only * g[None, None, :]
        
        # Inverse FFT to get reconstruction
        reconstruction = np.fft.ifft(phase_only, axis=2)
        
        # Take real part (imaginary should be ~0 for real input)
        self.reconstruction = np.real(reconstruction).astype(np.float32)
        
        return self.reconstruction

    def compute_saliency_map(
        self,
        spatial_sigma: float = 3.0,
        threshold_k: float = 2.0
    ) -> np.ndarray:
        """
        Compute saliency map from phase reconstruction.
        
        Paper Section 4.1:
            1. Take magnitude at t=1 (second frame): S_raw = |I'(x,y,t=1)|
            2. Apply spatial Gaussian smoothing to reduce noise
            3. Adaptive threshold: τ = μ + k·σ (paper uses k=2)
            
        Args:
            spatial_sigma: Sigma for spatial Gaussian smoothing (paper suggests 3-5)
            threshold_k: Multiplier for adaptive threshold (paper uses k=2)
            
        Returns:
            Saliency map (uint8, 0-255)
        """
        if self.reconstruction is None:
            raise ValueError("Call compute_phase_reconstruction() first")
        
        # Extract magnitude at t=1 (index 1 = second frame)
        # This captures the motion-related response
        saliency_raw = np.abs(self.reconstruction[..., 1])
        
        # Spatial Gaussian smoothing (important for noise reduction)
        if spatial_sigma > 0:
            ksize = int(2 * np.ceil(3 * spatial_sigma) + 1)  # Ensure odd kernel size
            saliency_smooth = cv2.GaussianBlur(saliency_raw, (ksize, ksize), spatial_sigma)
        else:
            saliency_smooth = saliency_raw
        
        # Adaptive thresholding: τ = μ + k·σ
        mu = float(saliency_smooth.mean())
        sigma = float(saliency_smooth.std())
        threshold = mu + threshold_k * sigma
        
        # Apply threshold and normalize to [0, 255]
        saliency_thresholded = np.maximum(saliency_smooth - threshold, 0)
        
        if saliency_thresholded.max() > 0:
            saliency_norm = (saliency_thresholded / saliency_thresholded.max() * 255.0)
        else:
            saliency_norm = np.zeros_like(saliency_thresholded)
        
        self.saliency_map = saliency_norm.astype(np.uint8)
        return self.saliency_map

    def compute_motion_mask(
        self,
        morph_kernel_size: int = 5,
        open_iterations: int = 1,
        close_iterations: int = 2
    ) -> np.ndarray:
        """
        Create binary motion mask from saliency map.
        
        Applies morphological operations to clean up the mask:
            - Opening: removes small noise (isolated pixels)
            - Closing: fills small holes in detected regions
            
        Args:
            morph_kernel_size: Size of morphological kernel (odd number)
            open_iterations: Number of opening iterations
            close_iterations: Number of closing iterations
            
        Returns:
            Binary motion mask (uint8, 0 or 255)
        """
        if self.saliency_map is None:
            raise ValueError("Call compute_saliency_map() first")
        
        # Binarize (saliency map is already thresholded, just need to make it binary)
        _, mask = cv2.threshold(self.saliency_map, 1, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        
        if open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
        if close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        
        self.motion_mask = mask
        return mask

    def detect_motion_regions(
        self,
        min_area: int = 100,
        nms_threshold: float = 0.3
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Extract bounding boxes from motion mask.
        
        Args:
            min_area: Minimum contour area to consider
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Tuple of (bboxes in xyxy format, total area)
        """
        if self.motion_mask is None:
            raise ValueError("Call compute_motion_mask() first")
        
        # Find contours
        contours, _ = cv2.findContours(
            self.motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area >= min_area:
                # Store as [x1, y1, x2, y2, score]
                detections.append([x, y, x + w, y + h, float(area)])
        
        if not detections:
            self.bboxes = None
            self.scores = None
            self.nms_bboxes = []
            self.total_box_area_value = 0
            return None, 0
        
        detections = np.array(detections, dtype=np.float32)
        self.bboxes = detections[:, :4]
        self.scores = detections[:, 4]
        
        # Apply NMS
        self.nms_bboxes = non_max_suppression(self.bboxes, self.scores, threshold=nms_threshold)
        
        # Compute total area
        total_area = sum(
            (box[2] - box[0]) * (box[3] - box[1])
            for box in self.nms_bboxes
        )
        self.total_box_area_value = int(total_area)
        
        return self.nms_bboxes, self.total_box_area_value

    def auto_run(
        self,
        spatial_sigma: float = 3.0,
        threshold_k: float = 2.0,
        morph_kernel_size: int = 5,
        open_iterations: int = 1,
        close_iterations: int = 2,
        min_area: int = 100,
        nms_threshold: float = 0.3,
        saliency_measurement: str = "bbox_area",
        plot: bool = False
    ) -> Tuple[Optional[List], int]:
        """
        Run the complete temporal phase saliency pipeline.
        
        This is the main entry point for motion detection using this method.
        
        PARAMETER EXPLANATIONS:
        =======================
        
        FROM THE PAPER (Chen et al. "Motion saliency detection using a temporal fourier transform"):
        ------------------------------------------------------------------------------------------
        
        spatial_sigma (float, default=3.0):
            - PAPER REFERENCE: Section 4.1, spatial Gaussian smoothing
            - PURPOSE: Controls the amount of spatial smoothing applied to the raw saliency map
            - EFFECT: 
                * Higher values (e.g., 5.0): More smoothing → larger, more connected regions,
                  but may merge separate objects and lose fine detail
                * Lower values (e.g., 1.0): Less smoothing → preserves fine detail,
                  but more susceptible to noise
            - PAPER SUGGESTS: Values in range 3-5 work well
        
        threshold_k (float, default=2.0):
            - PAPER REFERENCE: Equation in Section 4.1, adaptive threshold τ = μ + k·σ
            - PURPOSE: Controls the sensitivity of motion detection
            - EFFECT:
                * Higher values (e.g., 3.0): More strict threshold → only strong motion detected,
                  fewer false positives but may miss subtle motion
                * Lower values (e.g., 1.0): More lenient threshold → detects subtle motion,
                  but more false positives from noise
            - PAPER USES: k=2 (two standard deviations above mean)
        
        NOT FROM THE PAPER (Implementation/Post-processing details I added):
        --------------------------------------------------------------------
        
        morph_kernel_size (int, default=5):
            - NOT IN PAPER - This is standard post-processing
            - PURPOSE: Size of the square kernel used for morphological operations
            - EFFECT:
                * Larger values (e.g., 7, 9): Stronger morphological effect,
                  fills larger gaps but may over-smooth boundaries
                * Smaller values (e.g., 3): Preserves shape detail but less noise removal
        
        open_iterations (int, default=1):
            - NOT IN PAPER - Standard morphological cleanup
            - PURPOSE: Morphological opening = erosion followed by dilation
            - EFFECT:
                * Higher values: Removes more small isolated noise pixels
                * Set to 0 to disable opening
        
        close_iterations (int, default=2):
            - NOT IN PAPER - Standard morphological cleanup
            - PURPOSE: Morphological closing = dilation followed by erosion
            - EFFECT:
                * Higher values: Fills more holes/gaps within detected regions
                * Set to 0 to disable closing
        
        min_area (int, default=100):
            - NOT IN PAPER - Standard detection filtering
            - PURPOSE: Minimum bounding box area (in pixels) to consider as valid detection
            - EFFECT:
                * Higher values: Filters out small detections (good for ignoring noise)
                * Lower values: Keeps smaller detections (good for small objects)
        
        nms_threshold (float, default=0.3):
            - NOT IN PAPER - Standard object detection post-processing
            - PURPOSE: IoU (Intersection over Union) threshold for Non-Maximum Suppression
            - EFFECT:
                * Higher values (e.g., 0.5): More overlapping boxes allowed
                * Lower values (e.g., 0.1): Stricter merging of overlapping detections
        
        saliency_measurement (str, default="bbox_area"):
            - NOT IN PAPER - Implementation choice
            - OPTIONS:
                * "bbox_area": Returns bounding boxes and their total area
                * "pixel_count": Returns None for boxes and count of white pixels in mask
        
        plot (bool, default=False):
            - NOT IN PAPER - Debug/visualization option
            - PURPOSE: If True, displays visualization of the pipeline stages
        
        Returns:
            Tuple of (bboxes or None, total_area)
        """
        # =======================================================================
        # PIPELINE EXECUTION (Steps 1-3 are from the paper, Steps 4-5 are added)
        # =======================================================================
        
        # Step 1: Build temporal sequence (T=2 is sufficient for two frames)
        # PAPER: Section 3, Eq. (4) - zero-padded temporal sequence I(x,y,t)
        self.build_temporal_sequence(T=2)
        
        # Step 2: Compute phase-only reconstruction
        # PAPER: Section 3, Eq. (5-7)
        #   - FFT along temporal axis: F(x,y,ω) = FFT_t{I(x,y,t)}
        #   - Extract phase: φ = angle(F)  
        #   - Reconstruct with unit magnitude: I'(x,y,t) = IFFT{exp(j·φ)}
        self.compute_phase_reconstruction()
        
        # Step 3: Compute saliency map with spatial smoothing and adaptive threshold
        # PAPER: Section 4.1
        #   - Takes magnitude at t=1: S_raw = |I'(x,y,t=1)|
        #   - Applies spatial Gaussian smoothing (controlled by spatial_sigma)
        #   - Adaptive threshold: τ = μ + k·σ (controlled by threshold_k)
        self.compute_saliency_map(
            spatial_sigma=spatial_sigma,
            threshold_k=threshold_k
        )
        
        # Step 4: Create motion mask (NOT IN PAPER - post-processing for cleaner output)
        # Standard morphological operations to clean up the binary mask
        self.compute_motion_mask(
            morph_kernel_size=morph_kernel_size,
            open_iterations=open_iterations,
            close_iterations=close_iterations
        )
        
        if plot:
            self._visualize()
        
        # Step 5: Return based on measurement type (NOT IN PAPER - implementation choice)
        # Store mask for return
        mask = self.motion_mask

        if saliency_measurement == "pixel_count" or saliency_measurement == "mask":
            total_area = int(cv2.countNonZero(self.motion_mask))
            return None, total_area, mask
        else:  # bbox_area
            bboxes, total_area = self.detect_motion_regions(
                min_area=min_area,
                nms_threshold=nms_threshold
            )
            return bboxes, total_area, mask

    def _visualize(self):
        """Display saliency map and motion mask."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original frames
        axes[0, 0].imshow(self.frame1_rgb)
        axes[0, 0].set_title("Frame 1")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(self.frame2_rgb)
        axes[0, 1].set_title("Frame 2")
        axes[0, 1].axis("off")
        
        # Saliency map
        axes[1, 0].imshow(self.saliency_map, cmap="hot")
        axes[1, 0].set_title("Saliency Map")
        axes[1, 0].axis("off")
        
        # Motion mask with bboxes
        mask_vis = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2RGB)
        if self.nms_bboxes:
            for box in self.nms_bboxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[1, 1].imshow(mask_vis)
        axes[1, 1].set_title("Motion Mask + Detections")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        plt.show()

    def get_saliency_map(self) -> Optional[np.ndarray]:
        """Return the computed saliency map."""
        return self.saliency_map

    def get_motion_mask(self) -> Optional[np.ndarray]:
        """Return the computed motion mask."""
        return self.motion_mask

    def export_mask_png(
        self,
        output_path: Union[str, Path],
        motion_color_rgb: Tuple[int, int, int] = (0, 255, 0),
        background_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    ) -> str:
        """
        Export motion mask as colored PNG for CVAT compatibility.
        
        Args:
            output_path: Output file path
            motion_color_rgb: RGB color for motion regions
            background_color_rgb: RGB color for background
            
        Returns:
            Output path string
        """
        if self.motion_mask is None:
            raise ValueError("Run auto_run() first")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        h, w = self.motion_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[:] = background_color_rgb
        colored[self.motion_mask > 0] = motion_color_rgb
        
        cv2.imwrite(str(output_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        return str(output_path)


def export_temporal_phase_masks_cvat_zip(
    frames_dir: Union[str, Path],
    out_zip_path: Union[str, Path],
    spatial_sigma: float = 3.0,
    threshold_k: float = 2.0,
    morph_kernel_size: int = 5,
    open_iterations: int = 1,
    close_iterations: int = 2,
    motion_color_rgb: Tuple[int, int, int] = (0, 255, 0),
    background_color_rgb: Tuple[int, int, int] = (0, 0, 0),
    subset_name: str = "default",
    verbose: bool = True
):
    """
    Export motion masks for a dataset as CVAT-compatible ZIP.
    
    Args:
        frames_dir: Directory containing input frames
        out_zip_path: Output ZIP file path
        spatial_sigma: Spatial smoothing sigma
        threshold_k: Adaptive threshold k value
        morph_kernel_size: Morphological kernel size
        open_iterations: Opening iterations
        close_iterations: Closing iterations
        motion_color_rgb: Motion region color
        background_color_rgb: Background color
        subset_name: Subset name for CVAT
        verbose: Print progress
    """
    frames_dir = Path(frames_dir)
    out_zip_path = Path(out_zip_path)
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    img_paths = load_img_paths(str(frames_dir), print_paths=False)
    img_paths = [str(p) for p in img_paths]
    
    if len(img_paths) < 2:
        raise ValueError(f"Need at least 2 frames, found {len(img_paths)}")
    
    # Create temporary directory structure
    tmp_root = out_zip_path.with_suffix("")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    
    segclass_dir = tmp_root / "SegmentationClass"
    imagesets_dir = tmp_root / "ImageSets" / "Segmentation"
    segclass_dir.mkdir(parents=True)
    imagesets_dir.mkdir(parents=True)
    
    # Write labelmap
    labelmap = (
        "# label : color (RGB) : 'body' parts : actions\n"
        f"background:{background_color_rgb[0]},{background_color_rgb[1]},{background_color_rgb[2]}::\n"
        f"motion:{motion_color_rgb[0]},{motion_color_rgb[1]},{motion_color_rgb[2]}::\n"
    )
    (tmp_root / "labelmap.txt").write_text(labelmap)
    
    ids = []
    
    for i in range(len(img_paths) - 1):
        frame1 = cv2.imread(img_paths[i])
        frame2 = cv2.imread(img_paths[i + 1])
        
        if frame1 is None or frame2 is None:
            continue
        
        image_id = Path(img_paths[i + 1]).stem
        
        tps = TemporalPhaseSaliency(frame1, frame2)
        tps.auto_run(
            spatial_sigma=spatial_sigma,
            threshold_k=threshold_k,
            morph_kernel_size=morph_kernel_size,
            open_iterations=open_iterations,
            close_iterations=close_iterations,
            saliency_measurement="pixel_count"
        )
        
        out_png = segclass_dir / f"{image_id}.png"
        tps.export_mask_png(out_png, motion_color_rgb, background_color_rgb)
        ids.append(image_id)
        
        if verbose and (i % 25 == 0 or i == len(img_paths) - 2):
            print(f"[{i+1:4d}/{len(img_paths)-1}] wrote mask for {image_id}")
    
    # Write subset file
    (imagesets_dir / f"{subset_name}.txt").write_text("\n".join(ids) + "\n")
    
    # Create ZIP
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_root):
            for fn in files:
                full = Path(root) / fn
                rel = full.relative_to(tmp_root)
                z.write(str(full), str(rel))
    
    shutil.rmtree(tmp_root)
    print(f"[OK] Wrote CVAT zip: {out_zip_path} ({len(ids)} masks)")


def main():
    """Test the temporal phase saliency implementation."""
    print("Temporal Phase Saliency Test")
    
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_script_dir)
    path = os.path.join(_project_dir, "dataset_20fps")
    # img_paths = load_img_paths(path, print_paths=True)
    # img_paths = [str(p) for p in img_paths]
    
    # # Pick a frame pair with motion
    # idx = 185
    # frame1 = cv2.imread(img_paths[idx])
    # frame2 = cv2.imread(img_paths[idx + 1])
    
    # # Run the algorithm
    # tps = TemporalPhaseSaliency(frame1, frame2)
    # bboxes, total_area = tps.auto_run(
    #     spatial_sigma=3.0,      # Paper: spatial smoothing
    #     threshold_k=2.0,        # Paper: μ + 2σ threshold
    #     morph_kernel_size=5,    # Clean up mask
    #     open_iterations=1,
    #     close_iterations=2,
    #     min_area=100,
    #     nms_threshold=0.3,
    #     plot=True
    # )
    
    # print(f"Detected {len(bboxes) if bboxes else 0} motion regions")
    # print(f"Total motion area: {total_area} pixels")

    zip_name = "temporal_phase_masks.zip"

    _masks_dir = os.path.join(_project_dir, "masks")
    os.makedirs(_masks_dir, exist_ok=True)
    export_temporal_phase_masks_cvat_zip(
        frames_dir=path,
        out_zip_path=os.path.join(_masks_dir, zip_name),
        spatial_sigma=3.0,
        threshold_k=2.0,
        morph_kernel_size=5,
        open_iterations=1,
        close_iterations=2,
    )


if __name__ == "__main__":
    main()
