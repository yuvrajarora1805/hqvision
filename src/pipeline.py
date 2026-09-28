"""Module 4: clinical inference, visual overlay and TI-RADS re-scoring.

Consumes a raw scan plus its ROI annotations, runs the classical miner and the
quantum verifier, and produces the two things a capstone actually has to
deliver: an annotated clinical image and a re-scored suspicion category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config
from .dataset import PatchRecord, Region
from .mining import Candidate, CandidateMiner

# --- ACR TI-RADS component scores -------------------------------------------
# Mapped from the free-text values that appear in the DDTI XML annotations.
_COMPOSITION_POINTS = {
    "cystic": 0,
    "spongiform": 0,
    "predominantly cystic": 1,
    "dense": 2,
    "predominantly solid": 2,
    "solid": 2,
}
_ECHOGENICITY_POINTS = {
    "isoechogenicity": 1,
    "hyperechogenicity": 1,
    "hypoechogenicity": 1,
    "marked hypoechogenicity": 2,
    "anechoic": 0,
}
_MARGIN_POINTS = {
    "well defined": 0,
    "well defined smooth": 0,
    "ill defined": 0,
    "ill- defined": 0,
    "microlobulated": 2,
    "macrolobulated": 2,
    "spiculated": 3,
}

# DDTI stores the 2017 ACR scheme (2, 3, 4a, 4b, 4c, 5); the current standard is
# TR1-TR5. Mapping both lets the re-scored output be validated against the
# radiologist's recorded category.
_XML_TIRADS_TO_TR = {
    "1": "TR1", "2": "TR2", "3": "TR3",
    "4a": "TR4", "4b": "TR4", "4c": "TR5", "5": "TR5",
}


def _points_to_category(points: int) -> str:
    if points <= 1:
        return "TR2"
    if points <= 3:
        return "TR3"
    if points <= 5:
        return "TR4"
    return "TR5"


_RECOMMENDATION = {
    "TR1": "No follow-up.",
    "TR2": "No follow-up.",
    "TR3": "Follow-up ultrasound in 1 year.",
    "TR4": "Ultrasound-guided FNA; consider molecular testing.",
    "TR5": "Ultrasound-guided FNA; high suspicion of malignancy.",
}


@dataclass
class TIRADSScore:
    """A re-scored suspicion category with its provenance."""

    points: int
    category: str
    recommendation: str
    base_points: int = 0
    microcalc_points: int = 0
    xml_category: str | None = None

    def describe(self) -> str:
        parts = [
            f"{self.category} ({self.points} pts: base {self.base_points} "
            f"+ microcalc {self.microcalc_points})"
        ]
        parts.append(f"-> {self.recommendation}")
        if self.xml_category:
            match = "matches" if self.xml_category == self.category else "differs from"
            parts.append(f"| radiologist {self.xml_category} ({match} re-score)")
        return " ".join(parts)


def acr_base_score(case) -> int:
    """ACR TI-RADS points from composition, echogenicity and margins.

    Calcifications are deliberately excluded here: that is exactly the
    contribution this system is being evaluated on, so it must be added
    separately rather than read from the annotation.
    """
    if case is None:
        return 0
    total = 0
    for value, table in (
        (case.composition, _COMPOSITION_POINTS),
        (case.echogenicity, _ECHOGENICITY_POINTS),
        (case.margins, _MARGIN_POINTS),
    ):
        if value:
            total += table.get(str(value).strip().lower(), 0)
    return total


def ti_rads_rescore(
    n_microcalcifications: int,
    case=None,
    base_points: int | None = None,
) -> TIRADSScore:
    """Re-score a nodule given the number of quantum-verified microcalcifications.

    Follows plan.md: one or more verified microcalcifications add points and
    elevate suspicion; if every candidate is classified as speckle, nothing is
    added, which is what prevents an unnecessary biopsy recommendation.
    """
    if base_points is None:
        base_points = acr_base_score(case)
    microcalc_points = (
        config.TI_RADS_MICROCALC_POINTS if n_microcalcifications >= 1 else 0
    )
    total = base_points + microcalc_points
    xml_category = None
    if case is not None and getattr(case, "tirads", None):
        xml_category = _XML_TIRADS_TO_TR.get(str(case.tirads).strip().lower())
    return TIRADSScore(
        points=total,
        category=_points_to_category(total),
        recommendation=_RECOMMENDATION[_points_to_category(total)],
        base_points=base_points,
        microcalc_points=microcalc_points,
        xml_category=xml_category,
    )


# --- Inference ---------------------------------------------------------------
@dataclass
class InferenceResult:
    """Quantum-verified classification of every mined candidate in one frame."""

    image_path: Path
    candidates: list[Candidate]
    labels: list[int]
    scores: list[float]
    regions: list[Region] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_microcalcifications(self) -> int:
        return int(sum(1 for label in self.labels if label == config.LABEL_MICROCALC))

    @property
    def n_speckles(self) -> int:
        return int(sum(1 for label in self.labels if label == config.LABEL_SPECKLE))


class ThyroidCADPipeline:
    """Classical candidate mining followed by quantum-kernel verification."""

    def __init__(self, model, miner: CandidateMiner | None = None) -> None:
        self.model = model
        self.miner = miner or CandidateMiner()

    def analyse(self, image_path: Path, regions: list[Region]) -> InferenceResult:
        candidates = self.miner.mine_batch(image_path, regions)
        if not candidates:
            return InferenceResult(
                image_path=image_path, candidates=[], labels=[], scores=[], regions=regions
            )

        X = np.array([c.flat for c in candidates], dtype=np.float64)
        gram = self.model.test_gram(X)
        labels = self.model.predict_from_gram(gram)
        scores = self.model.decision_from_gram(gram)
        return InferenceResult(
            image_path=image_path,
            candidates=candidates,
            labels=[int(label) for label in labels],
            scores=[float(score) for score in scores],
            regions=regions,
        )


def predict_image(
    model,
    image_path: Path,
    regions: list[Region],
    miner: CandidateMiner | None = None,
) -> InferenceResult:
    """Functional wrapper around :class:`ThyroidCADPipeline`."""
    return ThyroidCADPipeline(model=model, miner=miner).analyse(image_path, regions)


# --- Visualisation -----------------------------------------------------------
def _draw_legend(
    canvas: np.ndarray,
    result: InferenceResult,
    score: TIRADSScore,
) -> None:
    lines = [
        "Quantum-Assisted CAD",
        f"  candidates mined : {result.n_candidates}",
        f"  microcalc (red)  : {result.n_microcalcifications}",
        f"  speckle  (green) : {result.n_speckles}",
        f"  TI-RADS          : {score.describe()}",
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
    base_tirads: str | None = None,
    radius: int = 9,
) -> Path:
    """Render the clinical overlay: yellow ROI, red microcalc, green speckle."""
    canvas = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for region in result.regions:
        cv2.polylines(
            canvas, [region.as_poly()], isClosed=True,
            color=config.COLOR_NODULE, thickness=1,
        )

    for candidate, label in zip(result.candidates, result.labels):
        is_micro = label == config.LABEL_MICROCALC
        colour = config.COLOR_MICROCALC if is_micro else config.COLOR_SPECKLE
        # Filled dot plus ring: readable at 4x4 native patch size.
        cv2.circle(canvas, (candidate.x, candidate.y), radius, colour, 1, cv2.LINE_AA)
        if is_micro:
            cv2.circle(canvas, (candidate.x, candidate.y), 1, colour, -1, cv2.LINE_AA)
            cv2.drawMarker(
                canvas, (candidate.x, candidate.y), colour,
                cv2.MARKER_CROSS, 4, 1, cv2.LINE_AA,
            )

    score = ti_rads_rescore(result.n_microcalcifications)
    _draw_legend(canvas, result, score)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path
