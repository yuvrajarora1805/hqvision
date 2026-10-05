"""
Phase 1: Patient-Isolated Ingestion & Partitioning Engine
Parses DDTI dataset XMLs, extracts patient case numbers, nodule ROIs,
and enforces an 80/20 patient-level stratified split to prevent data leakage.
"""

import os
import glob
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def parse_ddti_xml(xml_path: str) -> Optional[Dict]:
    """
    Parses a single DDTI XML annotation file.
    Returns:
        Dict containing patient case info, calcification label, tirads,
        and nodule boundary polygons per image, or None if invalid/skipped.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return None

    # Case / Patient identifier
    num_elem = root.find("number")
    if num_elem is None or not num_elem.text:
        # Fallback to filename stem
        base = os.path.splitext(os.path.basename(xml_path))[0]
        patient_id = base.strip()
    else:
        patient_id = num_elem.text.strip()

    # Calcifications label
    calc_elem = root.find("calcifications")
    if calc_elem is None or calc_elem.text is None:
        return None

    calc_text = calc_elem.text.strip().lower()
    if calc_text == "microcalcifications":
        label = 1
    elif calc_text == "non":
        label = 0
    else:
        # Skip macro, indeterminate, or blank
        return None

    # TIRADS & other clinical features
    tirads_elem = root.find("tirads")
    tirads = tirads_elem.text.strip() if (tirads_elem is not None and tirads_elem.text) else "unknown"

    # Extract nodule polygon boundaries per image index
    # DDTI format: <mark><image>1</image><svg>[{"points": [{"x":..,"y":..}]}]</svg></mark>
    nodules_by_image = {}
    for mark in root.findall("mark"):
        img_elem = mark.find("image")
        svg_elem = mark.find("svg")
        if img_elem is None or img_elem.text is None:
            continue
        if svg_elem is None or svg_elem.text is None:
            continue

        img_idx = img_elem.text.strip()
        try:
            regions = json.loads(svg_elem.text)
        except Exception:
            continue

        polygons = []
        for region in regions:
            pts = region.get("points", [])
            if len(pts) >= 3:
                poly = [[int(round(p["x"])), int(round(p["y"]))] for p in pts if "x" in p and "y" in p]
                if len(poly) >= 3:
                    polygons.append(np.array(poly, dtype=np.int32))

        if polygons:
            if img_idx not in nodules_by_image:
                nodules_by_image[img_idx] = []
            nodules_by_image[img_idx].extend(polygons)

    if not nodules_by_image:
        return None

    return {
        "patient_id": patient_id,
        "label": label,
        "calcification_str": calc_text,
        "tirads": tirads,
        "nodules": nodules_by_image,
        "xml_path": xml_path
    }


def collect_ddti_dataset(data_dir: str) -> List[Dict]:
    """
    Scans data directory, matches XMLs with available JPEG images,
    and returns a clean structured record for each patient-image sample.
    """
    records = []
    xml_files = sorted(glob.glob(os.path.join(data_dir, "*.xml")))

    for xml_path in xml_files:
        parsed = parse_ddti_xml(xml_path)
        if parsed is None:
            continue

        patient_id = parsed["patient_id"]
        for img_idx, polygons in parsed["nodules"].items():
            # DDTI images typically follow: [patient_id]_[img_idx].jpg
            candidate_img_paths = [
                os.path.join(data_dir, f"{patient_id}_{img_idx}.jpg"),
                os.path.join(data_dir, f"{patient_id}_{img_idx}.JPG"),
                os.path.join(data_dir, f"{patient_id}.jpg")
            ]
            
            matched_img = None
            for p in candidate_img_paths:
                if os.path.exists(p):
                    matched_img = p
                    break

            if matched_img is not None:
                records.append({
                    "patient_id": patient_id,
                    "image_id": f"{patient_id}_{img_idx}",
                    "image_path": matched_img,
                    "xml_path": xml_path,
                    "label": parsed["label"],
                    "tirads": parsed["tirads"],
                    "polygons": polygons
                })

    return records


def get_patient_stratified_split(
    records: List[Dict],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Strict Patient-Isolated Stratified Split:
    Ensures ZERO patient tissue patterns cross between train and test splits,
    while maintaining class balance (label 0 vs label 1).
    """
    patient_map = {}
    for r in records:
        pid = r["patient_id"]
        if pid not in patient_map:
            patient_map[pid] = {
                "label": r["label"],
                "records": []
            }
        patient_map[pid]["records"].append(r)

    patient_ids = np.array(list(patient_map.keys()))
    labels = np.array([patient_map[pid]["label"] for pid in patient_ids])
    groups = patient_ids  # Group by patient ID

    # Use StratifiedGroupKFold to get an 80/20 split (5 folds -> 1 test fold)
    n_splits = int(round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_indices, test_indices = next(sgkf.split(patient_ids, labels, groups=groups))

    train_patients = set(patient_ids[train_indices])
    test_patients = set(patient_ids[test_indices])

    # Assert strict disjointness
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage detected!"

    train_records = [r for r in records if r["patient_id"] in train_patients]
    test_records = [r for r in records if r["patient_id"] in test_patients]

    return train_records, test_records


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_dir, "data")
    print(f"Loading DDTI dataset from: {data_path}")
    recs = collect_ddti_dataset(data_path)
    print(f"Total valid image-nodule records: {len(recs)}")

    if recs:
        train_recs, test_recs = get_patient_stratified_split(recs, test_size=0.2)
        train_p = set(r["patient_id"] for r in train_recs)
        test_p = set(r["patient_id"] for r in test_recs)
        train_labels = [r["label"] for r in train_recs]
        test_labels = [r["label"] for r in test_recs]

        print(f"Train samples: {len(train_recs)} from {len(train_p)} unique patients (Microcalc: {sum(train_labels)}, Non: {len(train_labels)-sum(train_labels)})")
        print(f"Test samples:  {len(test_recs)} from {len(test_p)} unique patients (Microcalc: {sum(test_labels)}, Non: {len(test_labels)-sum(test_labels)})")
        print(f"Patient overlap check: {len(train_p.intersection(test_p))} (Must be 0)")
