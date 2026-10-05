"""
Phase 2: Classical Candidate Mining (High-Recall Sieve)
Performs:
  1. CLAHE Adaptive Contrast Normalization (clipLimit=2.0, grid=8x8)
  2. Morphological Top-Hat Transform (5x5 Elliptical kernel)
  3. ROI Polygon Masking (isolates nodule, suppresses background tissue)
  4. Local Weber Contrast Peak Detection & Non-Maximum Suppression (NMS, R=4 px)
  5. Native Micro-Patch Extraction (4x4 patches, 16 raw features)
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class CandidateMiner:
    """
    Classical High-Recall Candidate Miner for Ultrasound Microcalcifications.
    """
    def __init__(
        self,
        patch_size: int = 4,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        tophat_kernel_size: Tuple[int, int] = (5, 5),
        nms_radius: int = 4,
        max_candidates: int = 5,
        contrast_epsilon: float = 1e-5
    ):
        self.patch_size = patch_size
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tophat_kernel_size)
        self.nms_radius = nms_radius
        self.max_candidates = max_candidates
        self.contrast_epsilon = contrast_epsilon

    def preprocess_image(self, gray_img: np.ndarray) -> np.ndarray:
        """Applies CLAHE adaptive contrast normalization."""
        return self.clahe.apply(gray_img)

    def extract_tophat(self, enhanced_img: np.ndarray) -> np.ndarray:
        """Applies morphological Top-Hat transform with elliptical structuring element."""
        return cv2.morphologyEx(enhanced_img, cv2.MORPH_TOPHAT, self.tophat_kernel)

    def create_nodule_mask(self, image_shape: Tuple[int, int], polygons: List[np.ndarray]) -> np.ndarray:
        """Creates a binary mask (255 inside nodule, 0 outside) from XML polygon boundaries."""
        mask = np.zeros(image_shape, dtype=np.uint8)
        if polygons:
            cv2.fillPoly(mask, polygons, 255)
        return mask

    def compute_weber_contrast(
        self,
        enhanced_img: np.ndarray,
        coords: List[Tuple[int, int]],
        local_window_radius: int = 5
    ) -> List[float]:
        """
        Computes local Weber contrast score:
        C = (I_peak - I_background) / (I_background + epsilon)
        where I_background is the mean intensity in the local neighborhood excluding the peak.
        """
        h, w = enhanced_img.shape
        scores = []
        for cy, cx in coords:
            y1 = max(0, cy - local_window_radius)
            y2 = min(h, cy + local_window_radius + 1)
            x1 = max(0, cx - local_window_radius)
            x2 = min(w, cx + local_window_radius + 1)

            window = enhanced_img[y1:y2, x1:x2].astype(np.float32)
            i_peak = float(enhanced_img[cy, cx])
            # Local background approximation
            i_bg = float(np.mean(window))
            score = (i_peak - i_bg) / (i_bg + self.contrast_epsilon)
            scores.append(score)
        return scores

    def non_maximum_suppression(
        self,
        coords: List[Tuple[int, int]],
        scores: List[float]
    ) -> List[Tuple[int, int]]:
        """
        Suppresses neighboring peak detections within radius R=4 pixels.
        Returns top candidate coordinates ordered by Weber contrast score.
        """
        if not coords:
            return []

        # Sort indices by score descending
        order = np.argsort(scores)[::-1]
        keep = []

        coords_arr = np.array(coords)
        suppressed = np.zeros(len(coords), dtype=bool)

        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(tuple(coords_arr[idx]))
            if len(keep) >= self.max_candidates:
                break

            # Suppress all peaks within nms_radius
            curr_y, curr_x = coords_arr[idx]
            dist_sq = (coords_arr[:, 0] - curr_y)**2 + (coords_arr[:, 1] - curr_x)**2
            suppressed[dist_sq <= (self.nms_radius ** 2)] = True

        return keep

    def crop_patch(self, img: np.ndarray, center_y: int, center_x: int) -> np.ndarray:
        """
        Extracts a native resolution 4x4 micro-patch around (center_y, center_x).
        Zero padding applied if close to image boundaries.
        """
        h, w = img.shape
        half = self.patch_size // 2
        y1 = center_y - half
        y2 = y1 + self.patch_size
        x1 = center_x - half
        x2 = x1 + self.patch_size

        patch = np.zeros((self.patch_size, self.patch_size), dtype=img.dtype)

        src_y1 = max(0, y1)
        src_y2 = min(h, y2)
        src_x1 = max(0, x1)
        src_x2 = min(w, x2)

        dst_y1 = src_y1 - y1
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = src_x1 - x1
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        return patch

    def mine_candidates(
        self,
        img_gray: np.ndarray,
        polygons: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Runs complete candidate mining pipeline:
        Returns:
            patches: List of 4x4 native pixel numpy arrays (16 values each)
            coordinates: List of (center_y, center_x) candidate positions
        """
        h, w = img_gray.shape

        # Step 1: CLAHE
        enhanced = self.preprocess_image(img_gray)

        # Step 2: Morphological Top-Hat Filter
        tophat = self.extract_tophat(enhanced)

        # Step 3: Nodule ROI Masking
        nodule_mask = self.create_nodule_mask((h, w), polygons)
        masked_tophat = cv2.bitwise_and(tophat, tophat, mask=nodule_mask)

        # Step 4: Candidate Peak Detection inside nodule
        # Dynamic threshold based on masked region stats
        nodule_vals = masked_tophat[nodule_mask > 0]
        if len(nodule_vals) == 0:
            return [], []

        thresh_val = max(15, int(np.mean(nodule_vals) + 1.2 * np.std(nodule_vals)))
        _, binary = cv2.threshold(masked_tophat, thresh_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_coords = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(round(M["m10"] / M["m00"]))
                cy = int(round(M["m01"] / M["m00"]))
            else:
                cx, cy = int(cnt[0][0][0]), int(cnt[0][0][1])

            cx = min(max(0, cx), w - 1)
            cy = min(max(0, cy), h - 1)

            # Ensure detected point is inside the nodule polygon
            if nodule_mask[cy, cx] > 0:
                raw_coords.append((cy, cx))

        if not raw_coords:
            # Fallback to local maximum inside nodule if binary threshold returned 0
            min_v, max_v, min_l, max_l = cv2.minMaxLoc(masked_tophat)
            if max_v > 0:
                raw_coords.append((max_l[1], max_l[0]))

        if not raw_coords:
            return [], []

        # Step 5: Weber Contrast Scoring & NMS
        weber_scores = self.compute_weber_contrast(enhanced, raw_coords)
        selected_coords = self.non_maximum_suppression(raw_coords, weber_scores)

        # Step 6: Native Micro-Patch Extraction (4x4)
        patches = [self.crop_patch(img_gray, cy, cx) for cy, cx in selected_coords]

        return patches, selected_coords


def build_candidate_dataset(records: List[Dict], miner: CandidateMiner) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extracts all candidate patches across a set of patient records.
    Returns:
        X: (N_patches, 16) array of flattened 4x4 raw pixel values
        y: (N_patches,) array of labels (1: microcalcification nodule spot, 0: non-calcification speckle)
        meta: list of metadata dicts tracking originating patient and coordinates
    """
    X_list = []
    y_list = []
    meta_list = []

    for r in records:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        patches, coords = miner.mine_candidates(img, r["polygons"])
        for p, (cy, cx) in zip(patches, coords):
            X_list.append(p.flatten().astype(np.float32))
            y_list.append(r["label"])
            meta_list.append({
                "patient_id": r["patient_id"],
                "image_id": r["image_id"],
                "coord": (cy, cx),
                "label": r["label"]
            })

    if not X_list:
        return np.empty((0, 16), dtype=np.float32), np.empty((0,), dtype=int), []

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int), meta_list


if __name__ == "__main__":
    from dataset import collect_ddti_dataset, get_patient_stratified_split
    import os

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_dir, "data")
    recs = collect_ddti_dataset(data_path)
    train_recs, test_recs = get_patient_stratified_split(recs, test_size=0.2)

    miner = CandidateMiner(patch_size=4, max_candidates=5)

    print("Extracting candidate patches from Train split...")
    X_train, y_train, meta_train = build_candidate_dataset(train_recs, miner)
    print(f"X_train shape: {X_train.shape} (Patches, 16 dims), Class distribution: Microcalc={np.sum(y_train==1)}, Speckle={np.sum(y_train==0)}")

    print("Extracting candidate patches from Test split...")
    X_test, y_test, meta_test = build_candidate_dataset(test_recs, miner)
    print(f"X_test shape:  {X_test.shape} (Patches, 16 dims), Class distribution: Microcalc={np.sum(y_test==1)}, Speckle={np.sum(y_test==0)}")
