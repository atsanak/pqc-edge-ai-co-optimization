import os
from glob import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import re
import pprint
import time
from pathlib import Path
import re
import pprint

# for zip exporting
import zipfile
import shutil




def load_img_paths(path, print_paths=False,
                   exts=("jpg","jpeg","png","bmp","tif","tiff","webp"),
                   recursive=True):
    """
    Return a numerically-sorted list of image Paths from `path`.

    Args:
        path (str | Path): root folder
        print_paths (bool): print a short preview of found paths
        exts (tuple[str]): allowed extensions (no dots), case-insensitive
        recursive (bool): search subfolders (rglob) if True, else only the top folder
    """
    root = Path(path)
    exts_lower = {e.lower() for e in exts}

    if recursive:
        files = [p for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower().lstrip(".") in exts_lower]
    else:
        files = [p for p in root.iterdir()
                 if p.is_file() and p.suffix.lower().lstrip(".") in exts_lower]

    def numeric_key(p: Path):
        # sort by last integer in the stem; push non-numbered to the end
        nums = re.findall(r"\d+", p.stem)
        return (int(nums[-1]) if nums else float("inf"), str(p))

    image_paths = sorted(files, key=numeric_key)

    if print_paths:
        preview = [str(p) for p in image_paths[:10]]
        pprint.pprint(preview)
        if len(image_paths) > 10:
            print(f"... and {len(image_paths) - 10} more")
    print(f"Loaded Image Paths! Found {len(image_paths)} images in {root}")
    return image_paths




def merge_direct_neighbors_xywh(
    boxes: np.ndarray,
    min_iou: float = 0.3,
    touch_padding: int = 0
) -> np.ndarray:
    """
    Merge boxes (x, y, w, h) that directly overlap a given anchor box.
    No transitive closure: A merged with B, B merged with C does NOT
    automatically merge A and C together.

    Parameters
    ----------
    boxes : np.ndarray
        Shape (N, 4), (x, y, w, h).
    min_iou : float
        Minimum IoU with the anchor for merging.
        Larger -> more conservative merging.
    touch_padding : int
        Optional padding (pixels) when computing IoU to allow
        'almost touching' boxes to connect.

    Returns
    -------
    np.ndarray
        Merged boxes, shape (M, 4) in (x, y, w, h).
    """
    if boxes is None:
        return boxes
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return boxes

    boxes = boxes.astype(float).copy()

    # Convert to x1, y1, x2, y2
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    if touch_padding != 0:
        x1 -= touch_padding
        y1 -= touch_padding
        x2 += touch_padding
        y2 += touch_padding

    areas = (x2 - x1) * (y2 - y1)
    N = len(boxes)
    used = np.zeros(N, dtype=bool)
    merged_boxes = []

    def iou(i: int, j: int) -> float:
        ix1 = max(x1[i], x1[j])
        iy1 = max(y1[i], y1[j])
        ix2 = min(x2[i], x2[j])
        iy2 = min(y2[i], y2[j])

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0

        union = areas[i] + areas[j] - inter
        if union <= 0:
            return 0.0
        return inter / union

    # Process boxes in some deterministic order (e.g. by area desc)
    order = np.argsort(-areas)

    for anchor_idx in order:
        if used[anchor_idx]:
            continue

        group_indices = [anchor_idx]
        used[anchor_idx] = True

        # Only check overlap w.r.t. the anchor
        for j in order:
            if used[j]:
                continue
            if iou(anchor_idx, j) >= min_iou:
                used[j] = True
                group_indices.append(j)

        # Merge group into one big box
        gx1 = x1[group_indices].min()
        gy1 = y1[group_indices].min()
        gx2 = x2[group_indices].max()
        gy2 = y2[group_indices].max()

        merged_boxes.append([
            gx1,
            gy1,
            gx2 - gx1,   # width
            gy2 - gy1    # height
        ])

    return np.array(merged_boxes, dtype=int)





class FrameDiff:
    def __init__(self, frame1, frame2, verbose=False):
        self.verbose = verbose
        if verbose:
            print("FrameDiff Initiated!")
        self.frame1 = frame1
        self.frame2 = frame2

        self.img1_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        self.img2_rgb = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)

        # convert to grayscale
        self.img1 = cv2.cvtColor(self.img1_rgb, cv2.COLOR_RGB2GRAY)
        self.img2 = cv2.cvtColor(self.img2_rgb, cv2.COLOR_RGB2GRAY)

        self.mask = None


    def visualize_diff(self):
        # compute grayscale image difference
        grayscale_diff = cv2.subtract(self.img2, self.img1)

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(self.img1_rgb)
        ax[0].set_title('Frame 1')
        ax[1].imshow(self.img2_rgb)
        ax[1].set_title('Frame 2')
        ax[2].imshow(grayscale_diff*50) # scale the frame difference to show the noise
        ax[2].set_title('Frame Difference');
        plt.show()
        plt.close()


    def get_mask(self, kernel=None):
        H, W = self.img1.shape

        k = max(3, (min(H, W) // 160) | 1)
        f1 = cv2.GaussianBlur(self.img1, (k, k), 0)
        f2 = cv2.GaussianBlur(self.img2, (k, k), 0)
        diff = cv2.absdiff(f2, f1)

        block = max(11, (min(H, W) // 40) | 1)
        C = 2
        mask = cv2.adaptiveThreshold(
            diff, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block, C
        )

        open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k,  iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)

        mask = mask.astype(np.uint8)
        self.mask = mask
        return mask




    def plot_mask(self, mask):
        plt.imshow(mask, cmap='gray')
        plt.title("Motion Mask")
        plt.show()

    def get_contour_detections(self, mask, thresh=0, plot_bboxes=False):
        H, W = mask.shape[:2]

        # Relative minimum area (~0.05% of image)
        min_area = max(100, int(0.0005 * W * H))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        areas_dbg = []

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            a = w*h
            if a < min_area:
                continue

            # --- fill ratio / solidity tests ---
            # foreground pixels inside the bbox / bbox area
            roi = mask[y:y+h, x:x+w]
            fill_ratio = cv2.countNonZero(roi) / float(a)

            # contour solidity = contour_area / convex_hull_area
            cnt_area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = max(1.0, cv2.contourArea(hull))
            solidity = cnt_area / hull_area

            # simple aspect-ratio guard (optional)
            ar = w / float(h + 1e-6)
            ok_ar = 0.2 <= ar <= 5.0

            # Keep only “dense” enough blobs (tune 0.12–0.25 as needed)
            if fill_ratio >= 0.15 and solidity >= 0.15 and ok_ar:
                detections.append([x, y, x+w, y+h, a])
                areas_dbg.append((a, fill_ratio, solidity))

        detections_array = np.array(detections, dtype=int)
        if detections_array.size == 0:
            if plot_bboxes:
                plt.imshow(mask, cmap="gray"); plt.title("Detected Movers (none)"); plt.show()
            return np.empty((0,4), dtype=int), np.empty((0,), dtype=float)

        bboxes = detections_array[:, :4]
        scores = detections_array[:, -1].astype(float)

        if plot_bboxes:
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            for (x1,y1,x2,y2) in bboxes:
                cv2.rectangle(mask_rgb, (x1,y1), (x2,y2), (255,0,0), 2)
            plt.imshow(mask_rgb); plt.title("Detected Movers"); plt.show()

        return bboxes, scores


    

    def remove_contained_bboxes(self, boxes):
        """ Removes all smaller boxes that are contained within larger boxes.
            Requires bboxes to be soirted by area (score)
            Inputs:
                boxes - array bounding boxes sorted (descending) by area 
                        [[x1,y1,x2,y2]]
            Outputs:
                keep - indexes of bounding boxes that are not entirely contained 
                    in another box
            """
        check_array = np.array([True, True, False, False])
        keep = list(range(0, len(boxes)))
        for i in keep: # range(0, len(bboxes)):
            for j in range(0, len(boxes)):
                # check if box j is completely contained in box i
                if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                    try:
                        keep.remove(j)
                    except ValueError:
                        continue
        return keep


    def non_max_suppression(self, boxes, scores, threshold=1e-1):
        """
        Perform non-max suppression on a set of bounding boxes and corresponding scores.
        Inputs:
            boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
            scores: a list of corresponding scores 
            threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
        Outputs:
            boxes - non-max suppressed boxes
        """
        # Sort the boxes by score in descending order
        boxes = boxes[np.argsort(scores)[::-1]]

        # remove all contained bounding boxes and get ordered index
        order = self.remove_contained_bboxes(boxes)

        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            for j in order:
                # Calculate the IoU between the two boxes
                intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                            max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
                union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                        (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
                iou = intersection / union

                # Remove boxes with IoU greater than the threshold
                if iou > threshold:
                    order.remove(j)

        if self.verbose:
            print(f"NMS reduced {len(boxes)} boxes to {len(keep)} boxes.")
                    
        return boxes[keep]



    def plot_nms_bboxes(self, mask, nms_bboxes):
        mask_rgb_detections = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        for det in nms_bboxes:
            x1, y1, x2, y2 = det
            cv2.rectangle(mask_rgb_detections, (x1,y1), (x2,y2), (255,0,0), 3)


        plt.imshow(mask_rgb_detections)
        plt.title("Non-Max Suppressed Bounding Boxes");

        plt.show()
        plt.close()


    
    def print_nms_box_sizes(self, nms_bboxes):
        """
        Print size (w, h, area) and location (top-left, bottom-right, center)
        for all boxes after NMS. Returns a list of dicts with these fields.
        """
        if nms_bboxes is None or len(nms_bboxes) == 0:
            print("No NMS boxes to report.")
            return []

        info = []
        for i, (x1, y1, x2, y2) in enumerate(nms_bboxes.astype(int)):
            w = int(x2 - x1)
            h = int(y2 - y1)
            area = int(w * h)
            cx = x1 + w // 2
            cy = y1 + h // 2
            rec = {
                "index": i,
                "bbox": (x1, y1, x2, y2),
                "w": w, "h": h, "area": area,
                "top_left": (x1, y1),
                "bottom_right": (x2, y2),
                "center": (cx, cy),
            }
            info.append(rec)
            print(f"[{i}] size: {w}×{h}  area: {area}  "
                  f"top-left: ({x1},{y1})  bottom-right: ({x2},{y2})  center: ({cx},{cy})")
        return info

    def largest_nms_box(self, nms_bboxes):
        """
        Find the largest box by area among NMS boxes.
        Prints its stats and returns a tuple:
        (index, (x1,y1,x2,y2), (w,h), area, (cx,cy))
        """
        if nms_bboxes is None or len(nms_bboxes) == 0:
            print("No NMS boxes; cannot find largest.")
            return None

        nms_bboxes = nms_bboxes.astype(int)
        widths  = nms_bboxes[:, 2] - nms_bboxes[:, 0]
        heights = nms_bboxes[:, 3] - nms_bboxes[:, 1]
        areas   = widths * heights
        idx = int(np.argmax(areas))

        x1, y1, x2, y2 = nms_bboxes[idx]
        w, h = int(widths[idx]), int(heights[idx])
        area = int(areas[idx])
        cx, cy = x1 + w // 2, y1 + h // 2

        print(f"Largest box -> index: {idx}  size: {w}×{h}  area: {area}  "
              f"top-left: ({x1},{y1})  bottom-right: ({x2},{y2})  center: ({cx},{cy})")

        return (idx, (x1, y1, x2, y2), (w, h), area, (cx, cy))
    
    def total_box_area(self, boxes) -> int:
        """
        Return the total pixel area of the given boxes.
        boxes: array-like of shape (N, 4) with [x1, y1, x2, y2].

        Note: This is the simple sum of individual box areas (it does NOT
        subtract overlaps). Call it on NMS boxes if you want minimal overlap.
        """
        if boxes is None or len(boxes) == 0:
            return 0
        b = np.asarray(boxes, dtype=int)
        w = np.clip(b[:, 2] - b[:, 0], a_min=0, a_max=None)
        h = np.clip(b[:, 3] - b[:, 1], a_min=0, a_max=None)
        return int(np.sum(w * h))


    def build_binary_mask(self, mask=None):
        """
        Ensures a clean {0,255} mask stored in self.mask.
        """
        if mask is None:
            if self.mask is None:
                raise ValueError("No mask available. Call get_mask() first.")
            mask = self.mask

        bin_mask = (mask > 0).astype(np.uint8) * 255
        self.mask = bin_mask
        return bin_mask
    

    # saliency_measurement: "bbox_area" or "pixel_count"
    # bbox_area: total area covered by all boxes
    # pixel_count: total number of foreground pixels in the mask (that have value > 0) - 255
    # 0: stands for the colour black in the mask
    # 255: stands for the colour white in the mask
    def auto_run(self, plot=False, saliency_measurement="bbox_area"):

        # Accept legacy alias "mask" → "pixel_count" (unified in both v1 and v2)
        if saliency_measurement == "mask":
            saliency_measurement = "pixel_count"

        mask = self.get_mask()

        if plot is True:
            self.visualize_diff()
            self.plot_mask(mask)

        if saliency_measurement == "pixel_count":
            total_area = cv2.countNonZero(mask)
            merged_bboxes = None

        elif saliency_measurement == "bbox_area":
            bboxes, scores = self.get_contour_detections(mask=mask, plot_bboxes=plot)
            nms_bboxes = self.non_max_suppression(boxes=bboxes, scores=scores, threshold=0.01)

            # merged_bboxes = merge_direct_neighbors_xywh(nms_bboxes, min_iou = 0.4, touch_padding=0)
            merged_bboxes = nms_bboxes

            # Find the total area covered by movement in the current tile
            total_area = self.total_box_area(merged_bboxes)

            if plot is True:
                self.plot_nms_bboxes(mask, merged_bboxes)

        else:
            raise ValueError(f"Unknown saliency_measurement: {saliency_measurement}")

        return merged_bboxes, total_area, mask
    


def export_masks_to_zip(
    img_paths,
    zip_path,
    start_idx=0,
    end_idx=None,
    subset_name="default",
    motion_color_rgb=(0, 255, 0),
    background_color_rgb=(0, 0, 0),
):
    """
    Runs FrameDiff on all consecutive frame pairs (t, t+1) and exports ONE CVAT
    "Segmentation mask 1.1" zip containing ALL masks.

    IMPORTANT:
    - CVAT matches masks by filename stem. So image_id MUST equal your CVAT image stem.
    Example: if CVAT image is 'ezgif-frame-019.png' -> image_id must be 'ezgif-frame-019'
    - This function uses the stem of img_paths[t] as image_id automatically.
    """
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

        fd = FrameDiff(frame1, frame2)
        mask = fd.get_mask()
        mask = fd.build_binary_mask(mask)           # 0/255
        mask01 = (mask > 0).astype(np.uint8)        # 0/1

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

        
    


def main():

    print(__name__)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_script_dir)
    path = os.path.join(_project_dir, "dataset_20fps")
    img_paths = load_img_paths(path=path, print_paths=True)
    # cv2.imread expects string paths, not Path objects.
    img_paths = [str(p) for p in img_paths]

    idx = 185

    frame1 = cv2.imread(img_paths[idx])
    frame2 = cv2.imread(img_paths[idx+1])

    merged_bboxes, total_area = FrameDiff(frame1=frame1, frame2=frame2).auto_run(plot=True, saliency_measurement="bbox_area")

    print("Merged BBoxes:", merged_bboxes)
    print("Total Box Area:", total_area)


    # zip_name = "frame_diff_masks.zip"

    # export_masks_to_zip(
    #     img_paths=img_paths,
    #     zip_path=f"masks/{zip_name}",
    #     start_idx=0,
    #     end_idx=None,   # all pairs
    # )

if __name__ == "__main__":
    main()



