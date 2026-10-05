"""
Phase 2: Classical Candidate Mining (High-Recall Sieve)
Performs:
  1. CLAHE Adaptive Contrast Normalization (clipLimit=2.0, grid=8x8)
  2. Morphological Top-Hat Transform (5x5 Elliptical kernel)
  3. Nodule ROI Masking (isolates the TN5000 nodule bbox, suppresses background)
  4. Local Weber Contrast Peak Detection & Non-Maximum Suppression (NMS, R=4 px)
  5. Micro-Patch Extraction (4x4 native pixels) + 5 surround/context features

The ROI comes from the TN5000 VOC <bndbox> (converted to a 4-corner polygon by
src/dataset.py), so everything downstream is unchanged from the polygon-based
version of the pipeline.

Why the context features exist
------------------------------
A 4x4 crop of a bright spot looks almost identical for a calcification and for
acoustic speckle -- in both cases it is "some bright pixels". What actually
separates them is the neighbourhood: a real focus is a compact core sitting in a
darker surround and often casting a soft acoustic shadow downwards, while speckle
is flat and locally noisy. The miner already computed a Weber contrast for
ranking, but threw it away before the classifier ever saw it. These five features
hand that information to the quantum stage instead of discarding it:

  0. Weber contrast        (I_peak - I_bg) / (I_bg + eps)   -- peak vs surround
  1. Local std (7x7)       speckle is noisy, a real focus is smooth
  2. Ring contrast         top-hat energy in the surround minus the core
  3. Shadow ratio          mean intensity below the peak / local mean
  4. Top-hat response      saliency of the peak itself
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from . import config


def _square_region(img: np.ndarray, cy: int, cx: int, radius: int) -> np.ndarray:
    """Values in a (2r+1)^2 window around (cy, cx), clipped at image edges."""
    h, w = img.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    return img[y0:y1, x0:x1]


def _ring_region(
    img: np.ndarray, cy: int, cx: int, inner: int, outer: int
) -> np.ndarray:
    """Pixels with inner < distance <= outer, as a flat array (never empty)."""
    h, w = img.shape
    values = []
    for dy in range(-outer, outer + 1):
        y = cy + dy
        if y < 0 or y >= h:
            continue
        for dx in range(-outer, outer + 1):
            x = cx + dx
            if x < 0 or x >= w:
                continue
            if dy * dy + dx * dx <= inner * inner:
                continue
            values.append(img[y, x])
    return np.asarray(values, dtype=np.float32) if values else np.zeros(1, np.float32)


class CandidateMiner:
    """
    Classical High-Recall Candidate Miner for bright punctate foci inside a
    thyroid nodule ROI (TN5000 bounding box).
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
        """Creates a binary mask (255 inside nodule, 0 outside).

        ``polygons`` are ROI outlines; for TN5000 this is the 4-corner polygon
        of the annotated nodule bounding box.
        """
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

    def context_features(
        self,
        img_gray: np.ndarray,
        tophat: np.ndarray,
        cy: int,
        cx: int,
        weber_score: float,
    ) -> np.ndarray:
        """Five numbers describing the neighbourhood of a candidate peak.

        Returns a 5-vector:
            [weber_contrast, local_std, ring_contrast, shadow_ratio, tophat_peak]

        The first entry is the Weber contrast the miner already computed for NMS
        ranking; keeping it here is the whole point -- previously it was used to
        order candidates and then discarded.
        """
        eps = self.contrast_epsilon
        h, w = img_gray.shape

        # 1. Local standard deviation in a (2r+1)^2 window: speckle is grainy,
        #    a genuine focal echo tends to be smoother than its surround.
        local_std = float(np.std(_square_region(img_gray, cy, cx, config.CONTEXT_WINDOW)))

        # 2. Ring contrast on the top-hat image: surround energy minus core
        #    energy. A compact bright core inside a quiet ring scores high; a
        #    broad diffuse blob scores low.
        core = float(np.mean(_square_region(tophat, cy, cx, config.RING_INNER)))
        ring = float(np.mean(_ring_region(tophat, cy, cx, config.RING_INNER, config.RING_OUTER)))
        ring_contrast = ring - core

        # 3. Posterior shadow proxy: tissue just below a real calcification tends
        #    to attenuate and appear darker than the local mean. Ratio ~1 means no
        #    shadow, <1 means a shadow is present.
        y0 = max(0, cy + config.SHADOW_DEPTH)
        y1 = min(h, y0 + config.SHADOW_HEIGHT)
        x0 = max(0, cx - config.SHADOW_WIDTH)
        x1 = min(w, cx + config.SHADOW_WIDTH + 1)
        if y1 > y0 and x1 > x0:
            below = float(np.mean(img_gray[y0:y1, x0:x1]))
        else:
            below = float(np.mean(_square_region(img_gray, cy, cx, config.CONTEXT_WINDOW)))
        local_mean = float(np.mean(_square_region(img_gray, cy, cx, config.CONTEXT_WINDOW)))
        shadow_ratio = below / (local_mean + eps)

        # 4. Raw saliency of the peak on the top-hat response.
        tophat_peak = float(tophat[cy, cx])

        return np.array(
            [weber_score, local_std, ring_contrast, shadow_ratio, tophat_peak],
            dtype=np.float32,
        )

    def extract_features(
        self,
        img_gray: np.ndarray,
        tophat: np.ndarray,
        cy: int,
        cx: int,
        weber_score: float,
    ) -> np.ndarray:
        """Feature vector for one candidate: 16 raw pixels (+ 5 context values)."""
        patch = self.crop_patch(img_gray, cy, cx).reshape(-1).astype(np.float32)
        if not config.USE_CONTEXT_FEATURES:
            return patch
        context = self.context_features(img_gray, tophat, cy, cx, weber_score)
        return np.concatenate([patch, context])

    def mine_candidates(
        self,
        img_gray: np.ndarray,
        polygons: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Runs complete candidate mining pipeline:
        Returns:
            patches: List of feature vectors (16 raw, or 21 with context)
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

        # Step 6: Micro-Patch Extraction (4x4 raw pixels) + surround context
        selected_weber = self.compute_weber_contrast(enhanced, selected_coords)
        patches = [
            self.extract_features(img_gray, tophat, cy, cx, w)
            for (cy, cx), w in zip(selected_coords, selected_weber)
        ]

        return patches, selected_coords


def build_candidate_dataset(records: List[Dict], miner: CandidateMiner) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extracts all candidate feature vectors across a set of TN5000 image records.
    Returns:
        X: (N_patches, FEATURE_DIM) array -- 16 raw 4x4 pixels, plus 5 surround
           context features when config.USE_CONTEXT_FEATURES is on
        y: (N_patches,) weak labels inherited from the nodule
           (1: malignant nodule, 0: benign nodule)
        meta: list of metadata dicts tracking originating image and coordinates
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
                "case_id": r["case_id"],
                "image_id": r["image_id"],
                "coord": (cy, cx),
                "bbox": r["bbox"],
                "label": r["label"]
            })

    if not X_list:
        return (
            np.empty((0, config.FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=int),
            [],
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int), meta_list


if __name__ == "__main__":
    import os
    from src.dataset import load_dataset

    train_recs, test_recs = load_dataset(verbose=True)
    miner = CandidateMiner(patch_size=4, max_candidates=5)

    for name, recs in (("Train", train_recs), ("Test", test_recs)):
        X, y, meta = build_candidate_dataset(recs, miner)
        print(
            f"{name}: X={X.shape} (patches, {X.shape[1]} features) | "
            f"malignant={int(np.sum(y == 1))} benign={int(np.sum(y == 0))} | "
            f"images mined={len({m['case_id'] for m in meta})}/{len(recs)}"
        )
