"""Module 4: clinical inference, overlay and nodule-level verdict.

Consumes a TN5000 image plus its nodule bounding box, runs the classical miner
inside that ROI and the quantum verifier on the mined patches, then produces
the two deliverables a capstone has to ship: an annotated clinical image and a
single suspicion verdict for the nodule.

TN5000 carries no TI-RADS / composition / echogenicity annotations (the biopsy
label is the only ground truth), so the DDTI-era TI-RADS re-scoring is replaced
by an explicit nodule verdict: majority vote over the K mined patches plus the
mean signed kernel margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config
from .dataset import bbox_to_polygon
from .mining import CandidateMiner


# --- Nodule-level decision ---------------------------------------------------
@dataclass
class NoduleVerdict:
    """Majority decision over the patches mined from one nodule."""

    n_patches: int
    n_malignant: int
    n_benign: int
    mean_score: float
    label: int  # config.LABEL_MALIGNANT or config.LABEL_BENIGN

    @property
    def confidence(self) -> float:
        """Fraction of patches agreeing with the verdict (0.5 = coin flip)."""
        if self.n_patches == 0:
            return 0.0
        return max(self.n_malignant, self.n_benign) / self.n_patches

    @property
    def name(self) -> str:
        return config.CLASS_NAMES[self.label]

    def describe(self) -> str:
        return (
            f"{self.name.upper()} ({self.n_malignant}/{self.n_patches} malignant patches, "
            f"mean margin {self.mean_score:+.3f})"
        )


def nodule_verdict(labels: list[int], scores: list[float]) -> NoduleVerdict:
    """Aggregate patch predictions into one decision for the nodule.

    Ties (even patch counts split evenly) resolve to the higher mean margin,
    falling back to benign -- the conservative choice when the kernel is
    undecided.
    """
    labels = [int(v) for v in labels]
    n = len(labels)
    n_mal = sum(1 for v in labels if v == config.LABEL_MALIGNANT)
    n_ben = n - n_mal
    mean_score = float(np.mean(scores)) if scores else 0.0

    if n_mal > n_ben:
        label = config.LABEL_MALIGNANT
    elif n_ben > n_mal:
        label = config.LABEL_BENIGN
    else:
        label = config.LABEL_MALIGNANT if mean_score > 0 else config.LABEL_BENIGN
    return NoduleVerdict(
        n_patches=n, n_malignant=n_mal, n_benign=n_ben,
        mean_score=mean_score, label=label,
    )


# --- Inference ---------------------------------------------------------------
@dataclass
class InferenceResult:
    """Quantum-verified classification of every mined candidate in one frame."""

    image_path: Path
    candidates: list[tuple[int, int]]  # (x, y) patch centres
    labels: list[int]
    scores: list[float]
    bbox: tuple[int, int, int, int] | None = None
    patches: list[np.ndarray] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_malignant(self) -> int:
        return int(sum(1 for v in self.labels if v == config.LABEL_MALIGNANT))

    @property
    def n_benign(self) -> int:
        return int(sum(1 for v in self.labels if v == config.LABEL_BENIGN))

    @property
    def verdict(self) -> NoduleVerdict:
        return nodule_verdict(self.labels, self.scores)


class ThyroidCADPipeline:
    """Classical candidate mining followed by quantum-kernel verification."""

    def __init__(self, model, miner: CandidateMiner | None = None) -> None:
        self.model = model
        self.miner = miner or CandidateMiner(
            patch_size=config.PATCH_SIZE, max_candidates=config.MAX_CANDIDATES
        )

    def analyse(
        self, image_path: Path, bbox: tuple[int, int, int, int]
    ) -> InferenceResult:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        polygons = [bbox_to_polygon(bbox)]
        patches, coords = self.miner.mine_candidates(img, polygons)
        if not patches:
            return InferenceResult(
                image_path=image_path, candidates=[], labels=[],
                scores=[], bbox=bbox,
            )

        X = np.array([p.flatten() for p in patches], dtype=np.float64)
        gram = self.model.test_gram(X)
        labels = self.model.predict_from_gram(gram)
        scores = self.model.decision_from_gram(gram)
        return InferenceResult(
            image_path=image_path,
            candidates=[(int(x), int(y)) for y, x in coords],
            labels=[int(v) for v in labels],
            scores=[float(v) for v in scores],
            bbox=bbox,
            patches=patches,
        )


def predict_image(
    model,
    image_path: Path,
    bbox: tuple[int, int, int, int],
    miner: CandidateMiner | None = None,
) -> InferenceResult:
    """Functional wrapper around :class:`ThyroidCADPipeline`."""
    return ThyroidCADPipeline(model=model, miner=miner).analyse(image_path, bbox)


# --- Visualisation -----------------------------------------------------------
def _draw_legend(canvas: np.ndarray, result: InferenceResult) -> None:
    verdict = result.verdict
    lines = [
        "Quantum-Assisted Thyroid CAD",
        f"  candidates mined : {result.n_candidates}",
        f"  malignant (red)  : {result.n_malignant}",
        f"  benign    (green): {result.n_benign}",
        f"  verdict          : {verdict.describe()}",
        f"  confidence       : {verdict.confidence:.0%}",
    ]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    pad, line_h = 10, 18
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_w = max(widths) + 2 * pad
    box_h = line_h * len(lines) + pad

    h, w = canvas.shape[:2]
    x0, y0 = pad, h - box_h - pad
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (200, 200, 200), 1)

    for i, line in enumerate(lines):
        colour = (255, 255, 255) if i == 0 else (210, 210, 210)
        cv2.putText(
            canvas, line, (x0 + pad, y0 + pad + line_h * (i + 1) - 4),
            font, scale, colour, thickness, cv2.LINE_AA,
        )


def render_overlay(
    image_path: Path,
    result: InferenceResult,
    output_path: Path,
    radius: int = 9,
) -> Path:
    """Render the clinical overlay: yellow ROI, red malignant, green benign."""
    canvas = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if result.bbox is not None:
        x0, y0, x1, y1 = result.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), config.COLOR_ROI, 1, cv2.LINE_AA)

    for (x, y), label in zip(result.candidates, result.labels):
        is_malignant = label == config.LABEL_MALIGNANT
        colour = config.COLOR_MALIGNANT if is_malignant else config.COLOR_BENIGN
        # Filled dot plus ring: readable at 4x4 native patch size.
        cv2.circle(canvas, (x, y), radius, colour, 1, cv2.LINE_AA)
        if is_malignant:
            cv2.circle(canvas, (x, y), 1, colour, -1, cv2.LINE_AA)
            cv2.drawMarker(
                canvas, (x, y), colour, cv2.MARKER_CROSS, 4, 1, cv2.LINE_AA
            )

    if result.n_candidates:
        _draw_legend(canvas, result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path
