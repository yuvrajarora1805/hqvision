"""Classical high-recall candidate mining (OpenCV).

Stage 1 of the hybrid pipeline. Its only job is recall: never discard a real
microcalcification before the quantum verifier sees it. Precision is explicitly
*not* optimised here, because a false negative at this stage is unrecoverable.

Design note on why local-maxima NMS replaces threshold-and-contour detection:
microcalcifications occupy only a few pixels, so their contour area is
comparable to speckle grain area. Sorting contours by area therefore ranks them
essentially at random. Peak *prominence* is scale-free, which is the property
this stage actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Candidate:
    """One mined microcalcification candidate."""

    x: int
    y: int
    weber: float  # local Weber contrast, the ranking score
    tophat: float  # top-hat response at the peak
    patch: np.ndarray  # (PATCH_SIZE, PATCH_SIZE) uint8, native resolution

    @property
    def flat(self) -> np.ndarray:
        return self.patch.reshape(-1)


class CandidateMiner:
    """CLAHE -> morphological top-hat -> ROI mask -> NMS -> top-K patches."""

    def __init__(
        self,
        patch_size: int = config.PATCH_SIZE,
        clahe_clip_limit: float = config.CLAHE_CLIP_LIMIT,
        clahe_tile_grid: tuple[int, int] = config.CLAHE_TILE_GRID,
        kernel_size: int = config.TOPHAT_KERNEL_SIZE,
        nms_radius: int = config.NMS_RADIUS,
        max_candidates: int = config.MAX_CANDIDATES,
        weber_eps: float = config.WEBER_EPS,
    ) -> None:
        self.patch_size = patch_size
        self.nms_radius = nms_radius
        self.max_candidates = max_candidates
        self.weber_eps = weber_eps

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid
        )
        # Elliptical SE matched to sub-millimeter punctate geometry.
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        # Disk of diameter 2R+1 for the NMS suppression mask.
        self._nms_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * nms_radius + 1, 2 * nms_radius + 1)
        )

    # --- Preprocessing -------------------------------------------------------
    def preprocess(self, image_path: Path) -> np.ndarray:
        """Load a scan and normalise acoustic attenuation across depth."""
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self._clahe.apply(gray)

    def tophat(self, enhanced: np.ndarray) -> np.ndarray:
        """Isolate bright punctate structures smaller than the structuring element."""
        return cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, self._kernel)

    def mask_roi(self, tophat: np.ndarray, region) -> np.ndarray:
        """Suppress everything outside the nodule boundary.

        The mask is built per-region, so a frame containing two separate nodules
        does not become one near-image-wide hull.
        """
        mask = np.zeros(tophat.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [region.as_poly()], 255)
        # Dilate by the NMS radius so candidates near the boundary are not clipped.
        mask = cv2.dilate(mask, self._nms_kernel)
        return cv2.bitwise_and(tophat, tophat, mask=mask)

    # --- Peak detection ------------------------------------------------------
    def local_maxima(self, masked_tophat: np.ndarray) -> np.ndarray:
        """Boolean map of pixels equal to their dilated neighbourhood.

        A pixel survives only if it is the strict maximum within radius R, which
        is a direct, scale-free implementation of non-maximum suppression.
        """
        dilated = cv2.dilate(masked_tophat, self._nms_kernel)
        return (masked_tophat == dilated) & (masked_tophat > 0)

    def _background_level(self, enhanced: np.ndarray, x: int, y: int) -> float:
        """Median intensity in an annulus around the peak (local contrast baseline)."""
        h, w = enhanced.shape
        outer = self.nms_radius + config.BACKGROUND_RING_OFFSET
        x0, x1 = max(0, x - outer), min(w, x + outer + 1)
        y0, y1 = max(0, y - outer), min(h, y + outer + 1)
        patch = enhanced[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        inner = self.nms_radius
        ring = np.ones(patch.shape, dtype=bool)
        cy, cx = y - y0, x - x0
        yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
        annulus = (yy - cy) ** 2 + (xx - cx) ** 2
        ring &= ~((annulus <= inner**2) & (annulus > 0))
        values = patch[ring]
        if values.size < 4:  # annulus too small near a border; fall back to median
            values = patch.reshape(-1)
        return float(np.median(values))

    def weber_contrast(self, enhanced: np.ndarray, x: int, y: int) -> float:
        """Local Weber contrast C = (I_peak - I_background) / (I_background + eps)."""
        peak = float(enhanced[y, x])
        background = self._background_level(enhanced, x, y)
        return (peak - background) / (background + self.weber_eps)

    # --- Patch extraction ----------------------------------------------------
    def extract_patch(self, enhanced: np.ndarray, x: int, y: int) -> np.ndarray:
        """Crop a PATCH_SIZE x PATCH_SIZE window at native resolution.

        Patches that overhang the frame edge are edge-padded rather than dropped,
        so inference never silently loses a candidate the miner reported.
        """
        half = self.patch_size // 2
        h, w = enhanced.shape
        x0, x1 = max(0, x - half), min(w, x + half)
        y0, y1 = max(0, y - half), min(h, y + half)
        patch = enhanced[y0:y1, x0:x1]
        pad_y = (max(0, half - y), max(0, (y + half) - (h - 1)))
        pad_x = (max(0, half - x), max(0, (x + half) - (w - 1)))
        if pad_y != (0, 0) or pad_x != (0, 0):
            patch = np.pad(
                patch,
                ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])),
                mode="edge",
            )
        return patch.astype(np.uint8)

    # --- Public API ----------------------------------------------------------
    def mine(self, image_path: Path, region, k: int | None = None) -> list[Candidate]:
        """Return up to ``k`` candidates for one annotated ROI, best first."""
        k = self.max_candidates if k is None else k
        enhanced = self.preprocess(image_path)
        tophat = self.tophat(enhanced)
        masked = self.mask_roi(tophat, region)

        ys, xs = np.nonzero(self.local_maxima(masked))
        if ys.size == 0:
            return []

        scores = np.array(
            [self.weber_contrast(enhanced, int(x), int(y)) for x, y in zip(xs, ys)]
        )
        # Rank by Weber contrast, tie-broken by raw top-hat strength.
        order = np.lexsort((-masked[ys, xs], -scores))[:k]

        candidates = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            candidates.append(
                Candidate(
                    x=x,
                    y=y,
                    weber=float(scores[i]),
                    tophat=float(masked[y, x]),
                    patch=self.extract_patch(enhanced, x, y),
                )
            )
        return candidates

    def mine_batch(self, image_path: Path, regions, k: int | None = None) -> list[Candidate]:
        """Mine every ROI in a frame, de-duplicating peaks shared between ROIs."""
        enhanced = self.preprocess(image_path)
        tophat = self.tophat(enhanced)
        seen: set[tuple[int, int]] = set()
        results: list[Candidate] = []
        for region in regions:
            masked = self.mask_roi(tophat, region)
            ys, xs = np.nonzero(self.local_maxima(masked))
            if ys.size == 0:
                continue
            scores = np.array(
                [self.weber_contrast(enhanced, int(x), int(y)) for x, y in zip(xs, ys)]
            )
            order = np.lexsort((-masked[ys, xs], -scores))
            for i in order:
                x, y = int(xs[i]), int(ys[i])
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                results.append(
                    Candidate(
                        x=x,
                        y=y,
                        weber=float(scores[i]),
                        tophat=float(masked[y, x]),
                        patch=self.extract_patch(enhanced, x, y),
                    )
                )
        k = self.max_candidates if k is None else k
        return sorted(results, key=lambda c: -c.weber)[:k]


def reference_blobs(
    image_path: Path,
    region,
    kernel_size: int = 9,
    min_distance: int = 2,
    max_blobs: int = 12,
) -> list[tuple[int, int]]:
    """Generous reference detector used only as a recall proxy (Experiment C).

    Deliberately over-inclusive: a large structuring element, tight suppression
    and no K cap, so it surfaces punctate echoes the tuned miner may have
    rejected. Recall is measured against *this*, which is a proxy and not
    radiology ground truth -- see README "Experiment C caveat".
    """
    enhanced = CandidateMiner().preprocess(image_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    mask = np.zeros(tophat.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [region.as_poly()], 255)
    masked = cv2.bitwise_and(tophat, tophat, mask=mask)

    nms = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * min_distance + 1, 2 * min_distance + 1)
    )
    peaks = (masked == cv2.dilate(masked, nms)) & (masked > 0)
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return []
    order = np.argsort(-masked[ys, xs])[:max_blobs]
    return [(int(xs[i]), int(ys[i])) for i in order]
