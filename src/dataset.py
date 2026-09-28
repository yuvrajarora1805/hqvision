"""DDTI dataset parsing, integrity auditing and patient-isolated partitioning.

Three defects in the original monolithic script are corrected here:

1. ``<mark><image>`` is a **1-based** index into that patient's image sequence.
   The original script applied every polygon in an XML to every image of the
   same patient, mixing annotations across views.
2. A single ``<svg>`` can hold several **independent** regions (two distinct
   nodules in one frame). The original script unioned them into one polygon,
   inflating the ROI to a near-image-wide hull.
3. Eight XMLs have their embedded SVG JSON **truncated at ~2500 characters** by
   the original dataset release. The original script swallowed them with a bare
   ``except`` and lost the cases entirely.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

# Matches an individual {"x": N, "y": N} point, tolerant of the truncated JSON.
_POINT_RE = re.compile(r'\{\s*"x"\s*:\s*(-?\d+)\s*,\s*"y"\s*:\s*(-?\d+)\s*\}')
# Matches a whole region object so we can recover the "points" array boundaries.
_REGION_RE = re.compile(r'\{\s*"points"\s*:\s*\[(.*?)\]', re.DOTALL)


@dataclass(frozen=True)
class Region:
    """One annotated freehand polygon belonging to a specific image."""

    x: int
    y: int
    image_index: int  # 1-based, as stored in the XML
    patient_id: str
    points: np.ndarray  # (n, 2) int32
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1

    def as_poly(self) -> np.ndarray:
        return self.points.astype(np.int32).reshape(-1, 1, 2)


@dataclass
class Case:
    """A single DDTI patient/case and its usable annotations."""

    case_id: str
    patient_id: str
    label: int | None  # None when excluded as ambiguous
    exclusion_reason: str | None
    tirads: str | None
    composition: str | None = None
    echogenicity: str | None = None
    margins: str | None = None
    regions: list[Region] = field(default_factory=list)
    n_repaired_xml: bool = False
    n_dropped_regions: int = 0

    @property
    def image_indices(self) -> list[int]:
        return sorted({r.image_index for r in self.regions})

    @property
    def image_paths(self) -> list[Path]:
        return [
            config.DATA_DIR / f"{self.case_id}_{i}.jpg" for i in self.image_indices
        ]


@dataclass
class IntegrityReport:
    """Audit trail for the raw DDTI release."""

    total_xml: int = 0
    total_images: int = 0
    xml_without_image: list[str] = field(default_factory=list)
    image_without_xml: list[str] = field(default_factory=list)
    corrupt_xml: list[str] = field(default_factory=list)
    repaired_xml: list[str] = field(default_factory=list)
    out_of_range_index: list[tuple[str, int, list[int]]] = field(default_factory=list)
    multi_region_marks: int = 0
    arrow_regions_skipped: int = 0
    empty_polygons: list[str] = field(default_factory=list)
    calcification_tags: Counter = field(default_factory=Counter)
    label_distribution: Counter = field(default_factory=Counter)
    excluded: Counter = field(default_factory=Counter)
    # Labelled cases that carry no usable freehand polygon (e.g. only an 'arrow'
    # pointer, or an out-of-range image index). Surfaced so losses are auditable.
    unusable_labelled: list[tuple[str, str]] = field(default_factory=list)
    cases: list[Case] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        kept = self.label_distribution[config.LABEL_MICROCALC] + self.label_distribution[config.LABEL_SPECKLE]
        return [
            f"XML files scanned              : {self.total_xml}",
            f"JPG images on disk            : {self.total_images}",
            f"Usable cases (micro + non)    : {kept}",
            f"  microcalcification (label 1) : {self.label_distribution[config.LABEL_MICROCALC]}",
            f"  no calcification  (label 0)  : {self.label_distribution[config.LABEL_SPECKLE]}",
            f"Excluded cases                 : {sum(self.excluded.values())}",
            f"  ambiguous/macro/empty        : {dict(self.excluded)}",
            f"Regions marked as 'arrow'      : {self.arrow_regions_skipped} (skipped)",
            f"Marks holding >1 region        : {self.multi_region_marks}",
            f"Corrupt XML (unparseable)      : {len(self.corrupt_xml)} {self.corrupt_xml}",
            f"Truncated XML repaired by regex: {len(self.repaired_xml)} {self.repaired_xml}",
            f"Out-of-range 1-based image idx : {len(self.out_of_range_index)} {self.out_of_range_index}",
            f"XML with no matching image     : {len(self.xml_without_image)} {self.xml_without_image}",
            f"Image with no matching XML     : {len(self.image_without_xml)} {self.image_without_xml}",
            f"Empty/degenerate polygons      : {len(self.empty_polygons)} {self.empty_polygons}",
            f"Labelled cases w/o polygon     : {len(self.unusable_labelled)} {self.unusable_labelled}",
        ]


# --- SVG parsing -------------------------------------------------------------
def _parse_points_block(block: str) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in _POINT_RE.findall(block)]


def parse_regions(svg_text: str | None) -> tuple[list[list[tuple[int, int]]], bool, int]:
    """Parse an ``<svg>`` payload into point lists.

    Returns ``(regions, repaired, dropped)``. Multiple regions are kept separate
    because they represent distinct nodules in the same frame. A regex pass is
    used to recover polylines from JSON the upstream release truncated.
    """
    if not svg_text or not svg_text.strip():
        return [], False, 0

    try:
        payload = json.loads(svg_text)
        regions = []
        dropped = 0
        for entry in payload:
            points = [
                (int(p["x"]), int(p["y"]))
                for p in entry.get("points", [])
                if p.get("x") is not None and p.get("y") is not None
            ]
            if len(points) < 3:
                dropped += 1
            else:
                regions.append(points)
        return regions, False, dropped
    except json.JSONDecodeError:
        pass  # fall through to the repair path

    # Repair path: the release truncated the JSON string mid-object. Recover
    # every "points" array that was written completely enough to hold a polyline.
    regions: list[list[tuple[int, int]]] = []
    dropped = 0
    for match in _REGION_RE.finditer(svg_text):
        points = _parse_points_block(match.group(1))
        if len(points) < 3:
            dropped += 1
        else:
            regions.append(points)
    return regions, bool(regions), dropped


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# --- Case-level parsing ------------------------------------------------------
def _read_case(
    xml_path: Path,
    n_images: int,
    report: IntegrityReport,
) -> Case:
    """Parse one XML into a Case. Never returns None; unusable cases are recorded."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        report.corrupt_xml.append(xml_path.stem)
        return Case(
            case_id=xml_path.stem,
            patient_id=xml_path.stem.split("_")[0],
            label=None,
            exclusion_reason="unparseable_xml",
            tirads=None,
            regions=[],
        )

    case_id = xml_path.stem
    patient_id = case_id.split("_")[0]

    calc_elem = root.find(".//calcifications")
    raw_tag = (calc_elem.text or "").strip().lower() if calc_elem is not None else ""
    report.calcification_tags[raw_tag or "<empty>"] += 1

    if raw_tag in config.LABEL_MAP:
        label = config.LABEL_MAP[raw_tag]
    else:
        label = None
        if raw_tag.startswith("macro"):
            report.excluded["macrocalcification"] += 1
        else:
            report.excluded["missing_or_empty"] += 1

    def _text(tag: str) -> str | None:
        elem = root.find(f".//{tag}")
        if elem is None:
            return None
        value = (elem.text or "").strip()
        return value or None

    tirads = _text("tirads")

    case = Case(
        case_id=case_id,
        patient_id=patient_id,
        label=label,
        exclusion_reason=None if label is not None else raw_tag or "<empty>",
        tirads=tirads,
        composition=_text("composition"),
        echogenicity=_text("echogenicity"),
        margins=_text("margins"),
    )

    for mark in root.findall(".//mark"):
        image_elem = mark.find("image")
        svg_elem = mark.find("svg")
        if image_elem is None or not (image_elem.text or "").strip().isdigit():
            report.excluded["mark_without_image_index"] += 1
            continue
        image_index = int(image_elem.text.strip())
        if not 1 <= image_index <= n_images:
            report.out_of_range_index.append((case_id, n_images, [image_index]))
            continue

        regions, repaired, dropped = parse_regions(svg_elem.text if svg_elem is not None else None)
        if repaired:
            case.n_repaired_xml = True
            if case_id not in report.repaired_xml:
                report.repaired_xml.append(case_id)
        case.n_dropped_regions += dropped
        report.empty_polygons.extend([f"{case_id}#{image_index}"] * dropped)
        if len(regions) > 1:
            report.multi_region_marks += 1

        for points in regions:
            if len(points) < 3:
                continue
            # 'arrow' marks indicate a pointer, not a nodule boundary.
            if "arrow" in (svg_elem.text or ""):
                report.arrow_regions_skipped += 1
                continue
            case.regions.append(
                Region(
                    x=points[0][0],
                    y=points[0][1],
                    image_index=image_index,
                    patient_id=patient_id,
                    points=np.asarray(points, dtype=np.int32),
                    bbox=_bbox(points),
                )
            )

    if case.regions:
        report.label_distribution[label] += 1
    elif label is not None:
        # Keep the case in the audit trail: it has a label but no usable ROI.
        report.unusable_labelled.append(
            (case_id, f"label={label}, reasons={_reasons(case, report)}")
        )
    return case


def _reasons(case: Case, report: IntegrityReport) -> str:
    parts: list[str] = []
    if case.n_dropped_regions:
        parts.append(f"{case.n_dropped_regions}_degenerate_polygon(s)")
    if report.arrow_regions_skipped:
        parts.append("arrow_pointer_only")
    if any(entry[0] == case.case_id for entry in report.out_of_range_index):
        parts.append("image_index_out_of_range")
    return "+".join(parts) or "no_freehand_polygon"


def _count_images_per_case() -> Counter:
    return Counter(p.stem.rsplit("_", 1)[0] for p in config.DATA_DIR.glob("*.jpg"))


def load_cases(verbose: bool = True) -> IntegrityReport:
    """Parse every DDTI XML, audit the release, and return the usable cases."""
    report = IntegrityReport()
    xml_paths = sorted(config.DATA_DIR.glob("*.xml"))
    image_paths = sorted(config.DATA_DIR.glob("*.jpg"))
    report.total_xml = len(xml_paths)
    report.total_images = len(image_paths)

    images_per_case = _count_images_per_case()
    xml_ids = {p.stem for p in xml_paths}
    report.xml_without_image = sorted(i for i in xml_ids if images_per_case.get(i, 0) == 0)
    report.image_without_xml = sorted(
        {c for c in images_per_case if c not in xml_ids and not c.startswith("_")}
    )

    for xml_path in xml_paths:
        case = _read_case(xml_path, images_per_case.get(xml_path.stem, 0), report)
        if case.regions:
            report.cases.append(case)

    if verbose:
        print("\n=== DDTI Dataset Integrity Report ===")
        for line in report.summary_lines():
            print("  " + line)
        n_regions = sum(len(c.regions) for c in report.cases)
        n_annotated = sum(len(c.image_indices) for c in report.cases)
        print(f"  Usable polygons               : {n_regions}")
        print(f"  Annotated images (1-based idx): {n_annotated}")
    return report


# --- Patch dataset assembly --------------------------------------------------
@dataclass
class PatchRecord:
    """A mined micro-patch with its weak label and provenance."""

    features: np.ndarray  # (16,) uint8, native-resolution patch
    label: int
    case_id: str
    patient_id: str
    image_path: Path
    roi_index: int
    x: int
    y: int
    weber: float


def iter_annotated_images(
    report: IntegrityReport,
) -> list[tuple[Case, int, Path]]:
    """Yield ``(case, image_index, image_path)`` for every annotated image."""
    pairs: list[tuple[Case, int, Path]] = []
    for case in report.cases:
        for image_index in case.image_indices:
            image_path = config.DATA_DIR / f"{case.case_id}_{image_index}.jpg"
            if image_path.exists():
                pairs.append((case, image_index, image_path))
    return pairs


def build_patch_dataset(
    report: IntegrityReport | None = None,
    miner=None,
    verbose: bool = True,
) -> list[PatchRecord]:
    """Run the classical miner over every annotated image.

    Each mined candidate inherits its nodule's label: microcalcification cases
    contribute weak positives, 'non' cases contribute hard speckle negatives.
    """
    from .mining import CandidateMiner  # local import avoids a circular dependency

    if report is None:
        report = load_cases(verbose=verbose)
    miner = miner or CandidateMiner()

    records: list[PatchRecord] = []
    skipped = 0
    for case, image_index, image_path in iter_annotated_images(report):
        if case.label is None:  # macrocalcification / unlabelled -> ambiguous
            skipped += 1
            continue
        regions = [r for r in case.regions if r.image_index == image_index]
        for roi_index, region in enumerate(regions):
            candidates = miner.mine(image_path, region)
            for candidate in candidates:
                records.append(
                    PatchRecord(
                        features=candidate.patch,
                        label=case.label,
                        case_id=case.case_id,
                        patient_id=case.patient_id,
                        image_path=image_path,
                        roi_index=roi_index,
                        x=candidate.x,
                        y=candidate.y,
                        weber=candidate.weber,
                    )
                )

    if verbose:
        counts = Counter(r.label for r in records)
        print(f"\n=== Patch Dataset ===")
        print(f"  total patches : {len(records)}")
        print(f"  label 1 (micro): {counts[config.LABEL_MICROCALC]}")
        print(f"  label 0 (speck): {counts[config.LABEL_SPECKLE]}")
        patients = {r.patient_id for r in records}
        print(f"  patients      : {len(patients)}")
        if skipped:
            print(f"  annotated images skipped (ambiguous label): {skipped}")
    return records


# --- Patient-isolated splitting ---------------------------------------------
@dataclass
class PatientSplit:
    """Patient-isolated train/test partition over patch records."""

    train: list[PatchRecord]
    test: list[PatchRecord]
    train_patients: set[str]
    test_patients: set[str]
    stratification_hits: int = 0

    def verify_isolation(self) -> bool:
        overlap = self.train_patients & self.test_patients
        if overlap:
            raise AssertionError(f"Patient leakage across splits: {sorted(overlap)[:5]}")
        return True


def _stratified_patient_groups(
    patients: list[str],
    patient_label: dict[str, int],
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], int]:
    """Split patient IDs by class so both labels appear on both sides."""
    rng = np.random.default_rng(seed)
    train_set: set[str] = set()
    test_set: set[str] = set()
    hits = 0

    for label in sorted({patient_label[p] for p in patients}):
        group = sorted(p for p in patients if patient_label[p] == label)
        if len(group) < 2:
            continue
        shuffled = list(rng.permutation(group))
        n_test = max(1, int(round(len(group) * test_fraction)))
        n_test = min(n_test, len(group) - 1)
        test_set.update(shuffled[:n_test])
        train_set.update(shuffled[n_test:])

    # Degenerate cohort (single class overall) falls back to a plain shuffle.
    if not train_set or not test_set:
        shuffled = list(rng.permutation(patients))
        cut = max(1, int(round(len(shuffled) * (1 - test_fraction))))
        train_set, test_set = set(shuffled[:cut]), set(shuffled[cut:])
        hits += 1

    return train_set, test_set, hits


def patient_isolated_split(
    records: list[PatchRecord],
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.SEED,
    verbose: bool = True,
) -> PatientSplit:
    """80/20 split at the **patient** level, stratified by nodule label.

    All images of a patient (106_1..106_4) land in the same side of the split,
    so no tissue pattern crosses the boundary.
    """
    if not records:
        raise ValueError("No patch records to split.")

    patient_label = {r.patient_id: r.label for r in records}
    patients = sorted(patient_label)
    train_patients, test_patients, hits = _stratified_patient_groups(
        patients, patient_label, test_fraction, seed
    )

    train = [r for r in records if r.patient_id in train_patients]
    test = [r for r in records if r.patient_id in test_patients]

    split = PatientSplit(
        train=train,
        test=test,
        train_patients=train_patients,
        test_patients=test_patients,
        stratification_hits=hits,
    )
    split.verify_isolation()

    if verbose:
        tr_counts = Counter(r.label for r in train)
        te_counts = Counter(r.label for r in test)
        print(f"\n=== Patient-Isolated Split ({int((1 - test_fraction) * 100)}/"
              f"{int(test_fraction * 100)}) ===")
        print(f"  train : {len(train):4d} patches | {len(train_patients):3d} patients "
              f"| micro={tr_counts[config.LABEL_MICROCALC]} speckle={tr_counts[config.LABEL_SPECKLE]}")
        print(f"  test  : {len(test):4d} patches | {len(test_patients):3d} patients "
              f"| micro={te_counts[config.LABEL_MICROCALC]} speckle={te_counts[config.LABEL_SPECKLE]}")
        print(f"  patient overlap: {len(train_patients & test_patients)}  (must be 0)")
    return split
