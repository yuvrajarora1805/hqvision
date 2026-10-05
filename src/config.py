"""Central configuration for the quantum-assisted thyroid nodule classifier.

Every tunable in the pipeline is declared here so experiment scripts can sweep
values without editing module internals, and so the report can quote a single
authoritative parameter set.

Dataset: TN5000 (PASCAL VOC layout) -- 5,000 B-mode thyroid ultrasound images,
one nodule bounding box per image, biopsy-confirmed label:
    <name>1</name> -> malignant (3,574 images)
    <name>0</name> -> benign    (1,426 images)
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# TN5000 lives outside the repo (5,000 JPEGs are not committed). Resolution
# order: $TN5000_DIR -> data/Main data -> ../TN5000/Main data -> data
_DATASET_CANDIDATES = [
    Path(os.environ["TN5000_DIR"]) if os.environ.get("TN5000_DIR") else None,
    REPO_ROOT / "data" / "Main data",
    REPO_ROOT.parent / "TN5000" / "Main data",
    REPO_ROOT / "data",
]


def _resolve_dataset_dir() -> Path:
    checked = []
    for candidate in _DATASET_CANDIDATES:
        if candidate is None:
            continue
        checked.append(str(candidate))
        if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
            return candidate
    raise FileNotFoundError(
        "TN5000 not found. Checked:\n  - "
        + "\n  - ".join(checked)
        + "\nSet the TN5000_DIR environment variable or copy the dataset to "
        "'data/Main data' (JPEGImages/ + Annotations/ + ImageSets/)."
    )


DATASET_DIR = _resolve_dataset_dir()
IMAGE_DIR = DATASET_DIR / "JPEGImages"
ANNOTATION_DIR = DATASET_DIR / "Annotations"
IMAGESETS_DIR = DATASET_DIR / "ImageSets" / "Main"

OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
ANNOTATED_DIR = OUTPUTS_DIR / "annotated_scans"
MODELS_DIR = OUTPUTS_DIR / "models"
CACHE_DIR = OUTPUTS_DIR / "cache"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Module 1: dataset (TN5000 / VOC) ---------------------------------------
# Ground truth is per *nodule* (biopsy-confirmed), not per pixel or per
# calcification: supervision is weak once the label is inherited by patches.
LABEL_MALIGNANT = 1  # VOC <name>1</name> -- positive / suspicious class
LABEL_BENIGN = 0  # VOC <name>0</name> -- negative class
LABEL_MAP = {"1": LABEL_MALIGNANT, "0": LABEL_BENIGN}
CLASS_NAMES = {LABEL_BENIGN: "benign", LABEL_MALIGNANT: "malignant"}

# Official TN5000 split (ImageSets/Main): train 3500 + val 500 + test 1000.
# The authors kept one representative image per patient perspective, so this
# image-level split is the leakage-safe partition shipped with the dataset.
USE_OFFICIAL_SPLIT = True
TRAIN_SPLIT_NAME = "trainval"  # 4,000 images
TEST_SPLIT_NAME = "test"  # 1,000 images
FALLBACK_TEST_FRACTION = 0.20  # used only when ImageSets/ is missing

# --- Module 2: classical candidate mining -----------------------------------
PATCH_SIZE = 4  # native-resolution micro-patch, 4x4 -> 16 raw features
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
TOPHAT_KERNEL_SIZE = 5  # elliptical SE matched to punctate bright foci
NMS_RADIUS = 4  # pixels; suppress secondary peaks within this radius
MAX_CANDIDATES = 5  # K, retained per nodule ROI
WEBER_EPS = 1e-5

# --- Module 3: quantum engine ------------------------------------------------
N_QUBITS = 4
N_REPS = 2
ENTANGLEMENT = "linear"  # keeps two-qubit gate depth low for NISQ feasibility
PCA_DIMS = 4  # 16 raw pixels -> 4 angle-domain features
ANGLE_MIN = 0.0
ANGLE_MAX = 3.141592653589793  # pi
SVC_CLASS_WEIGHT = "balanced"  # 3574 malignant vs 1426 benign is imbalanced

# Feasibility cap: TN5000 mines ~20k train patches, but a kernel method needs
# an NxN Gram matrix, so training patches are balanced-subsampled to this cap.
MAX_TRAIN_PATCHES = 2000
MAX_TEST_PATCHES = None  # None = keep every mined test patch

# --- Module 3b: baseline circuits --------------------------------------------
# Legacy PennyLane `train_hybrid_model.py` used AngleEmbedding + adjoint
# overlap (a product-cosine kernel), reimplemented in Qiskit for comparison
# against the ZZ feature map.
OVERLAP_N_QUBITS = 4

# --- Module 5: experiments ---------------------------------------------------
SAMPLE_EFFICIENCY_SIZES = (15, 30, 60, 120)
SAMPLE_EFFICIENCY_SEEDS = (0, 1, 2, 3, 4)
NOISE_RATES = (0.001, 0.005, 0.010)
NOISE_SHOTS = 1024

# --- Overlay colours (BGR) ----------------------------------------------------
COLOR_MALIGNANT = (0, 0, 220)  # red: patch classified malignant / suspicious
COLOR_BENIGN = (0, 200, 0)  # green: patch classified benign
COLOR_ROI = (0, 210, 255)  # yellow: nodule bounding box


def ensure_output_dirs() -> None:
    """Create the generated-artefact tree. outputs/ is git-ignored."""
    for directory in (OUTPUTS_DIR, FIGURES_DIR, ANNOTATED_DIR, MODELS_DIR, CACHE_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
