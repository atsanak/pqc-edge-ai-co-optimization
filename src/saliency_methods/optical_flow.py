import os
from glob import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import zipfile
import tempfile
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from saliency_methods.motion_detection_utils import *
from saliency_methods.frame_diff import load_img_paths


class OpticalFlow:
    def __init__(self, frame1, frame2, verbose=False):
        self.verbose = verbose
        if verbose:
            print("Optical Flow Initiated!")
        self.frame1 = frame1
        self.frame2 = frame2

        # OpenCV loads BGR by default; keep BGR for processing, RGB for display.
        self.frame1_bgr = self.frame1
        self.frame2_bgr = self.frame2
        self.frame1_rgb = cv2.cvtColor(self.frame1_bgr, cv2.COLOR_BGR2RGB)
        self.frame2_rgb = cv2.cvtColor(self.frame2_bgr, cv2.COLOR_BGR2RGB)

        # convert to grayscale
        self.img1 = cv2.cvtColor(self.frame1_bgr, cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(self.frame2_bgr, cv2.COLOR_BGR2GRAY)


        self.flow = None
        self.rgb_flow = None
        self.mag = None
        self.ang = None
        self.motion_mask = None
        self.detections = None
        self.bboxes = None
        self.scores = None
        self.nms_bboxes = None
        self.total_box_area_value = 0

    def compute_flow(self):
        # convert to grayscale
        gray1 = cv2.cvtColor(self.frame1_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.frame2_bgr, cv2.COLOR_BGR2GRAY)

        # blurr image
        gray1 = cv2.GaussianBlur(gray1, dst=None, ksize=(3,3), sigmaX=5)
        gray2 = cv2.GaussianBlur(gray2, dst=None, ksize=(3,3), sigmaX=5)

        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                            pyr_scale=0.75,
                                            levels=3,
                                            winsize=5,
                                            iterations=3,
                                            poly_n=10,
                                            poly_sigma=1.2,
                                            flags=0)
        self.flow = flow



    def get_flow_viz(self, plot=False):
        """ 
        Obtains BGR image to Visualize the Optical Flow 
        """
        hsv = np.zeros((self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        self.rgb_flow = rgb

        if plot is True:
            # display
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(self.frame1_rgb)
            ax[0].set_title('Frame 1')
            ax[1].imshow(self.frame2_rgb)
            ax[1].set_title('Frame 2')
            ax[2].imshow(np.log(self.mag/self.mag.max()), cmap='hsv_r') # try other cmaps 'hsv_r', 'gist_earth_r', 'rainbow_r', 'twilight_r'
            ax[2].set_title('Log of Dense Optical Flow Magnitude')
            plt.show()

    def compute_mag_ang(self):
        mag, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        self.mag = mag
        self.ang = ang



    def get_motion_mask(self, motion_thresh=1, kernel=np.ones((7,7)), plot=False):
        """ Obtains Detection Mask from Optical Flow Magnitude
            Inputs:
                flow_mag (array) Optical Flow magnitude
                motion_thresh - thresold to determine motion
                kernel - kernal for Morphological Operations
            Outputs:
                motion_mask - Binray Motion Mask
            """
        motion_mask = np.uint8(self.mag > motion_thresh)*255

        motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        self.motion_mask = motion_mask

        if plot is True:
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].imshow(self.motion_mask, cmap='gray')
            ax[0].set_title("Motion Mask")
            ax[1].imshow(self.rgb_flow*50) # scale RGB to see the noise
            ax[1].set_title("Dense Optical Flow");
            plt.show()

    def get_contour_detections_2(self, angle_thresh=2, thresh=400, plot=False):
        """ Obtains initial proposed detections from contours discoverd on the
            mask. Scores are taken as the bbox area, larger is higher.
            Inputs:
                mask - thresholded image mask
                angle_thresh - threshold for flow angle standard deviation
                thresh - threshold for contour size
            Outputs:
                detectons - array of proposed detection bounding boxes and scores 
                            [[x1,y1,x2,y2,s]]
            """
        # get mask contours
        contours, _ = cv2.findContours(self.motion_mask, 
                                    cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_TC89_L1)
        temp_mask = np.zeros_like(self.motion_mask) # used to get flow angle of contours
        angle_thresh = angle_thresh*self.ang.std()
        detections = []
        for cnt in contours:
            # get area of contour
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h

            # get flow angle inside of contour
            cv2.drawContours(temp_mask, [cnt], 0, (255,), -1)
            flow_angle = self.ang[np.nonzero(temp_mask)]

            if (area > thresh) and (flow_angle.std() < angle_thresh): # hyperparameter
                detections.append([x,y,x+w,y+h, area])

        if len(detections) == 0:
            self.detections = np.empty((0, 5), dtype=int)
            self.bboxes = np.empty((0, 4), dtype=int)
            self.scores = np.empty((0,), dtype=float)
            self.mask_rgb = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2RGB)
            if plot is True:
                plt.imshow(self.mask_rgb)
                plt.title("Detected Movers");
                plt.show()
            return

        self.detections = np.array(detections)

        self.mask_rgb = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2RGB)
        # detections = get_contour_detections(mask, thresh=400)

        # separate bboxes and scores
        self.bboxes = self.detections[:, :4]
        self.scores = self.detections[:, -1]

        for box in self.bboxes:
            x1, y1, x2, y2 = box
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(self.mask_rgb, pt1, pt2, (255,0,0), 3)

        if plot is True:
            plt.imshow(self.mask_rgb)
            plt.title("Detected Movers");
            plt.show()

    def perform_nms_suppresion(self, plot=False):
        if self.bboxes is None or len(self.bboxes) == 0:
            self.nms_bboxes = np.empty((0, 4), dtype=int)
            self.frame2 = self.frame2_rgb.copy()
            if plot is True:
                plt.imshow(self.frame2)
                plt.title("Detections on Frame 2 after NMS");
                plt.show()
            return

        nms_bboxes = non_max_suppression(self.bboxes, self.scores, threshold=0.1)
        self.nms_bboxes = nms_bboxes
        if self.verbose:
            print(f"NMS Reduced {len(self.bboxes)} boxes to {len(self.nms_bboxes)} boxes.")

        mask_rgb_detections = cv2.cvtColor(self.motion_mask, cv2.COLOR_GRAY2RGB)
        for det in nms_bboxes:
            x1, y1, x2, y2 = det
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(mask_rgb_detections, pt1, pt2, (255,0,0), 3)

        self.frame2 = self.frame2_rgb.copy()
        draw_bboxes(self.frame2, nms_bboxes)

        if plot is True:
            plt.imshow(mask_rgb_detections)
            plt.title("Non-Max Suppressed Bounding Boxes");
            plt.show()

            plt.imshow(self.frame2);
            plt.title("Detections on Frame 2 after NMS");
            plt.show()


    def compute_total_box_area(self):
        total_area = 0
        for box in self.nms_bboxes:
            x1, y1, x2, y2 = box
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            total_area += w * h
        self.total_box_area_value = total_area

    def auto_run(self, plot=False, saliency_measurement="bbox_area"):
        # Accept legacy alias "mask" → "pixel_count"
        if saliency_measurement == "mask":
            saliency_measurement = "pixel_count"

        self.compute_flow()
        self.compute_mag_ang()
        self.get_flow_viz(plot=plot)
        self.get_motion_mask(plot=plot)

        # Store mask for return
        mask = self.motion_mask

        if saliency_measurement == "bbox_area":
            self.get_contour_detections_2(plot=plot)
            self.perform_nms_suppresion(plot=plot)
            self.compute_total_box_area()
            merged_bboxes, total_area = self.nms_bboxes, self.total_box_area_value
  
        elif saliency_measurement == "pixel_count":
            total_area = cv2.countNonZero(self.motion_mask)
            merged_bboxes = None

        else:
            raise ValueError(f"Unknown saliency_measurement: {saliency_measurement}")
        
        return merged_bboxes, total_area, mask
    




    def build_motion_mask_from_mag(
        self,
        thresh_mode="percentile",
        motion_thresh=1.0,
        percentile=90,
        use_otsu=False,
        kernel_size=7,
        close_iters=2,
        open_iters=1,
        erode_iters=1,
    ):
        """
        Creates self.motion_mask (0/255) from optical flow magnitude in a robust way.

        - thresh_mode="raw": uses self.mag > motion_thresh
        - thresh_mode="percentile": normalizes mag to 0..255 then thresholds by percentile
        - use_otsu=True: uses Otsu threshold on normalized mag (often works well)
        """
        if self.mag is None:
            raise ValueError("Call compute_mag_ang() before building a motion mask.")

        # Normalize magnitude for stable thresholding (critical!)
        mag_norm = cv2.normalize(self.mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if use_otsu:
            # Otsu chooses a threshold automatically
            _, mask = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            if thresh_mode == "raw":
                mask = (self.mag > float(motion_thresh)).astype(np.uint8) * 255
            elif thresh_mode == "percentile":
                thr = np.percentile(mag_norm, float(percentile))
                # Require a minimum absolute motion magnitude to avoid "all-green" on flat frames.
                mask = ((mag_norm >= thr) & (self.mag > float(motion_thresh))).astype(np.uint8) * 255
            else:
                raise ValueError("thresh_mode must be one of: raw, percentile")

        # Morphology cleanup
        k = int(kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)

        if erode_iters > 0:
            mask = cv2.erode(mask, kernel, iterations=int(erode_iters))
        if open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))
        if close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))

        self.motion_mask = mask
        return mask


    @staticmethod
    def export_cvat_segmentation_zip_for_dataset(
        img_paths,
        zip_path,
        use_frame2_name=True,
        use_color_masks=True,
        motion_color_rgb=(0, 255, 0),
        background_color_rgb=(0, 0, 0),
        # mask params (passed to build_motion_mask_from_mag)
        thresh_mode="percentile",
        motion_thresh=1.0,
        percentile=90,
        use_otsu=False,
        kernel_size=7,
        close_iters=1,
        open_iters=1,
        erode_iters=1,
        verbose_every=25,
    ):
        """
        Exports a SINGLE CVAT 'Segmentation mask 1.1' zip containing masks for ALL consecutive pairs.

        Each mask corresponds to motion between frame i and i+1.
        By default, mask name matches frame i+1 (use_frame2_name=True).
        """

        img_paths = [str(p) for p in img_paths]
        zip_path = str(zip_path)

        tmp_root = os.path.splitext(zip_path)[0] + "_tmp"
        seg_class_dir = os.path.join(tmp_root, "SegmentationClass")
        imagesets_dir = os.path.join(tmp_root, "ImageSets", "Segmentation")
        os.makedirs(seg_class_dir, exist_ok=True)
        os.makedirs(imagesets_dir, exist_ok=True)

        # labelmap
        if use_color_masks:
            labelmap_txt = (
                "# label : color (RGB) : 'body' parts : actions\n"
                f"background:{background_color_rgb[0]},{background_color_rgb[1]},{background_color_rgb[2]}::\n"
                f"motion:{motion_color_rgb[0]},{motion_color_rgb[1]},{motion_color_rgb[2]}::\n"
            )
        else:
            # indexed mask: 0 background, 1 motion
            labelmap_txt = (
                "# label : color (RGB) : 'body' parts : actions\n"
                "background:0,0,0::\n"
                "motion:0,255,0::\n"
            )

        with open(os.path.join(tmp_root, "labelmap.txt"), "w", encoding="utf-8") as f:
            f.write(labelmap_txt)

        ids = []

        # loop over all consecutive pairs
        for i in range(len(img_paths) - 1):
            p1 = img_paths[i]
            p2 = img_paths[i + 1]

            frame1 = cv2.imread(p1)
            frame2 = cv2.imread(p2)
            if frame1 is None or frame2 is None:
                raise RuntimeError(f"Failed to read frames: {p1}, {p2}")

            of = OpticalFlow(frame1, frame2)
            of.compute_flow()
            of.compute_mag_ang()

            of.build_motion_mask_from_mag(
                thresh_mode=thresh_mode,
                motion_thresh=motion_thresh,
                percentile=percentile,
                use_otsu=use_otsu,
                kernel_size=kernel_size,
                close_iters=close_iters,
                open_iters=open_iters,
                erode_iters=erode_iters,
            )

            mask01 = (of.motion_mask > 0).astype(np.uint8)  # 0/1

            # choose ID name (frame2 name recommended)
            chosen = p2 if use_frame2_name else p1
            image_id = Path(chosen).stem
            ids.append(image_id)

            out_path = os.path.join(seg_class_dir, f"{image_id}.png")

            if use_color_masks:
                h, w = mask01.shape
                out = np.zeros((h, w, 3), dtype=np.uint8)
                out[:, :] = np.array(background_color_rgb, dtype=np.uint8)
                out[mask01 == 1] = np.array(motion_color_rgb, dtype=np.uint8)
                cv2.imwrite(out_path, out)
            else:
                # indexed (0/1) mask (correct for CVAT; may look dark in some viewers)
                cv2.imwrite(out_path, mask01)

            if verbose_every and (i % int(verbose_every) == 0):
                print(f"[{i:03d}] wrote {image_id}.png")

        # write split file with ALL ids
        with open(os.path.join(imagesets_dir, "default.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")

        # zip it
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(tmp_root):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, tmp_root)
                    z.write(full, rel)

        # cleanup
        import shutil
        shutil.rmtree(tmp_root)

        print(f"[OK] Wrote CVAT segmentation zip with {len(ids)} masks: {zip_path}")

        
def display_frames(frame1, frame2):
    # display the 2 images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.title("Frame 1")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    plt.title("Frame 2")
    plt.axis("off")
    plt.show()

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_script_dir)
    path = os.path.join(_project_dir, "dataset_20fps")
    img_paths = load_img_paths(path=path, print_paths=True)
    # cv2.imread expects a string path, not a Path object
    img_paths = [str(p) for p in img_paths]

    idx = 185

    frame1 = cv2.imread(img_paths[idx])
    frame2 = cv2.imread(img_paths[idx + 1])

    display_frames(frame1, frame2)

    merged_bboxes, total_area = OpticalFlow(frame1, frame2).auto_run(plot=True, saliency_measurement="bbox_area") # set plot=False to disable plotting

    print("Merged BBoxes:", merged_bboxes)
    print("Total Box Area:", total_area)



#     OpticalFlow.export_cvat_segmentation_zip_for_dataset(
#     img_paths=img_paths,
#     zip_path="masks/optical_flow_masks.zip",
#     use_frame2_name=True,
#     use_color_masks=True,     # easier to inspect visually

#     # tune these:
#     percentile=90, # Increase percentile => less noise (fewer pixels kept)
#     kernel_size=5,
#     close_iters=3,
#     verbose_every=25,
# )


if __name__ == "__main__":
    main()
