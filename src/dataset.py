"""Phase 1: TN5000 ingestion, integrity audit and split assignment.

TN5000 is a PASCAL-VOC dataset:
    <root>/JPEGImages/NNNNNN.jpg     5,000 ultrasound frames
    <root>/Annotations/NNNNNN.xml    one nodule bounding box + biopsy label
    <root>/ImageSets/Main/*.txt      official train / val / test id lists

Each XML carries a single <object> whose <name> is the ground-truth class
(1 = malignant, 0 = benign, confirmed by FNA biopsy) and a <bndbox> giving the
nodule region (xmin, ymin, xmax, ymax).

Pipeline contract: every record keeps the keys ``case_id``, ``image_id``,
``image_path``, ``label`` and ``polygons`` (the ROI as a closed polygon list),
which is what src/mining.py consumes. The VOC rectangle is simply converted to
its 4 corner points, so the miner works unchanged from the DDTI version.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from . import config


# --- Parsing -----------------------------------------------------------------
def bbox_to_polygon(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """(xmin, ymin, xmax, ymax) -> (4, 2) int32 corner polygon for fillPoly."""
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


def parse_voc_annotation(xml_path: Path) -> dict:
    """Parse one TN5000 VOC XML into a flat record.

    Raises ValueError on structurally unusable files so the audit can count them.
    """
    root = ET.parse(xml_path).getroot()

    filename = (root.findtext("filename") or xml_path.stem).strip()
    image_id = Path(filename).stem

    size = root.find("size")
    width = int(float(size.findtext("width"))) if size is not None and size.findtext("width") else 0
    height = int(float(size.findtext("height"))) if size is not None and size.findtext("height") else 0

    objects = root.findall("object")
    if not objects:
        raise ValueError("no <object> element")
    obj = objects[0]

    raw_label = (obj.findtext("name") or "").strip()
    if raw_label not in config.LABEL_MAP:
        raise ValueError(f"unexpected class label {raw_label!r}")
    label = config.LABEL_MAP[raw_label]

    box = obj.find("bndbox")
    if box is None:
        raise ValueError("no <bndbox> element")
    xmin = int(round(float(box.findtext("xmin"))))
    ymin = int(round(float(box.findtext("ymin"))))
    xmax = int(round(float(box.findtext("xmax"))))
    ymax = int(round(float(box.findtext("ymax"))))

    # Clamp to the declared image size; some releases ship 1-based boxes.
    if width and height:
        xmin, xmax = sorted((max(0, min(xmin, width - 1)), max(0, min(xmax, width - 1))))
        ymin, ymax = sorted((max(0, min(ymin, height - 1)), max(0, min(ymax, height - 1))))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"degenerate bbox ({xmin},{ymin},{xmax},{ymax})")

    return {
        "case_id": image_id,
        "image_id": image_id,
        "filename": filename,
        "label": label,
        "class_name": raw_label,
        "bbox": (xmin, ymin, xmax, ymax),
        "width": width,
        "height": height,
        "n_objects": len(objects),
        "xml_path": str(xml_path),
    }


# --- Collection & audit ------------------------------------------------------
def collect_tn5000_records(verbose: bool = True) -> list[dict]:
    """Pair every XML with its JPEG and return usable records (audit + records)."""
    image_dir, annotation_dir = config.IMAGE_DIR, config.ANNOTATION_DIR
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"TN5000 folders missing under {config.DATASET_DIR} "
            "(expected JPEGImages/ and Annotations/)"
        )

    xml_by_stem = {p.stem: p for p in sorted(annotation_dir.glob("*.xml"))}
    jpg_stems = {p.stem for p in sorted(image_dir.glob("*.jpg"))}

    xml_without_image = sorted(set(xml_by_stem) - jpg_stems)
    image_without_xml = sorted(jpg_stems - set(xml_by_stem))

    records: list[dict] = []
    failures: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    n_objects = Counter()

    for stem, xml_path in sorted(xml_by_stem.items()):
        if stem not in jpg_stems:
            continue
        try:
            record = parse_voc_annotation(xml_path)
        except ET.ParseError:
            failures["unparseable_xml"] += 1
            continue
        except ValueError as exc:
            failures[str(exc).split(":")[0]] += 1
            continue

        record["image_path"] = str(image_dir / f"{stem}.jpg")
        record["polygons"] = [bbox_to_polygon(record["bbox"])]
        records.append(record)
        label_counts[record["label"]] += 1
        n_objects[record["n_objects"]] += 1

    if verbose:
        print("\n=== TN5000 Dataset Integrity Report ===")
        print(f"  XML annotations scanned        : {len(xml_by_stem)}")
        print(f"  JPEG images on disk            : {len(jpg_stems)}")
        print(f"  Usable records                 : {len(records)}")
        print(f"    malignant (label 1)          : {label_counts[config.LABEL_MALIGNANT]}")
        print(f"    benign    (label 0)          : {label_counts[config.LABEL_BENIGN]}")
        print(f"  XML without matching image     : {len(xml_without_image)} {xml_without_image[:5]}")
        print(f"  Image without matching XML     : {len(image_without_xml)} {image_without_xml[:5]}")
        if failures:
            print(f"  Rejected annotations           : {dict(failures)}")
        print(f"  Objects per annotation         : {dict(sorted(n_objects.items()))}")
        print(f"  Dataset root                   : {config.DATASET_DIR}")

    return records


# --- Official split ----------------------------------------------------------
def load_split_ids(split_name: str) -> set[str] | None:
    """Read ImageSets/Main/<split>.txt; returns None when unavailable."""
    path = config.IMAGESETS_DIR / f"{split_name}.txt"
    if not path.is_file():
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def assign_splits(
    records: list[dict],
    use_official: bool = config.USE_OFFICIAL_SPLIT,
    test_fraction: float = config.FALLBACK_TEST_FRACTION,
    seed: int = config.SEED,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Split records into (train, test).

    Preferred: the official TN5000 ImageSets (trainval/test), which the dataset
    authors built by keeping one representative image per patient perspective --
    i.e. the leakage-safe partition ships with the data. Fallback: a seeded
    stratified 80/20 image-level split when ImageSets/ is absent.
    """
    if use_official:
        train_ids = load_split_ids(config.TRAIN_SPLIT_NAME)
        test_ids = load_split_ids(config.TEST_SPLIT_NAME)
        if train_ids and test_ids:
            train = [r for r in records if r["case_id"] in train_ids]
            test = [r for r in records if r["case_id"] in test_ids]
            source = f"official {config.TRAIN_SPLIT_NAME}/{config.TEST_SPLIT_NAME}.txt"
        else:
            use_official = False

    if not use_official:
        rng = np.random.default_rng(seed)
        train, test = [], []
        for label in sorted({r["label"] for r in records}):
            group = sorted((r for r in records if r["label"] == label), key=lambda r: r["case_id"])
            order = rng.permutation(len(group))
            n_test = max(1, int(round(len(group) * test_fraction)))
            test_idx = set(order[:n_test].tolist())
            for i, record in enumerate(group):
                (test if i in test_idx else train).append(record)
        source = f"stratified fallback ({int((1 - test_fraction) * 100)}/{int(test_fraction * 100)}, seed {seed})"

    overlap = {r["case_id"] for r in train} & {r["case_id"] for r in test}
    assert not overlap, f"Split leakage: {sorted(overlap)[:5]}"

    if verbose:
        def _counts(recs):
            c = Counter(r["label"] for r in recs)
            return (
                f"{len(recs)} images | malignant={c[config.LABEL_MALIGNANT]} "
                f"benign={c[config.LABEL_BENIGN]}"
            )

        print("\n=== Split ===")
        print(f"  source  : {source}")
        print(f"  train   : {_counts(train)}")
        print(f"  test    : {_counts(test)}")
        print(f"  overlap : {len(overlap)} (must be 0)")

    return train, test


def load_dataset(verbose: bool = True, **split_kwargs) -> tuple[list[dict], list[dict]]:
    """One-call helper: integrity audit + split assignment."""
    records = collect_tn5000_records(verbose=verbose)
    if not records:
        raise RuntimeError("TN5000 yielded no usable records.")
    return assign_splits(records, verbose=verbose, **split_kwargs)


if __name__ == "__main__":
    load_dataset(verbose=True)
