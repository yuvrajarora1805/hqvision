"""Central configuration for the quantum-assisted thyroid microcalcification detector.

Every tunable in the pipeline is declared here so that the experiment scripts can
sweep values without editing module internals, and so the report can quote a
single authoritative parameter set.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
ANNOTATED_DIR = OUTPUTS_DIR / "annotated_scans"
MODELS_DIR = OUTPUTS_DIR / "models"
CACHE_DIR = OUTPUTS_DIR / "cache"

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Module 1: dataset -------------------------------------------------------
# DDTI labels the *nodule*, never the individual calcification, so the supervision
# is weak (see README "Supervision model"). Only these tags are unambiguous.
LABEL_MICROCALC = 1  # positive: real microcalcification
LABEL_SPECKLE = 0  # negative: acoustic speckle artifact
LABEL_MAP = {
    "microcalcifications": LABEL_MICROCALC,
    "microcalcification": LABEL_MICROCALC,
    "non": LABEL_SPECKLE,
}
# 'macrocalcifications' / 'macrocalcification' / missing / empty are ambiguous
# for this binary task and are excluded rather than force-mapped.

TEST_FRACTION = 0.20  # 80/20 patient-level split

# --- Module 2: classical candidate mining ------------------------------------
PATCH_SIZE = 4  # native-resolution micro-patch, 4x4 -> 16 raw features
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
TOPHAT_KERNEL_SIZE = 5  # elliptical SE matched to sub-millimeter geometry
NMS_RADIUS = 4  # pixels; suppress secondary peaks within this radius
MAX_CANDIDATES = 4  # K, retained per ROI
WEBER_EPS = 1e-6

# Local-maximum detection (3x3 neighbourhood) and background ring offset.
LOCAL_MAX_KERNEL = 3
BACKGROUND_RING_OFFSET = 2

# --- Module 3: quantum engine ------------------------------------------------
N_QUBITS = 4
N_REPS = 2
ENTANGLEMENT = "linear"  # keeps CNOT depth low for NISQ feasibility
PCA_DIMS = 4  # 16 raw pixels -> 4 angle-domain features
ANGLE_MIN = 0.0
ANGLE_MAX = 3.141592653589793  # pi
SVC_CLASS_WEIGHT = "balanced"

# --- Module 3b: baseline circuits --------------------------------------------
# PennyLane `train_hybrid_model.py` used AngleEmbedding + adjoint overlap,
# i.e. a raw <psi|phi> (trace) kernel. Reimplemented in Qiskit for comparison
# against the ZZ feature map.
OVERLAP_N_QUBITS = 4

# --- Module 5: experiments ---------------------------------------------------
SAMPLE_EFFICIENCY_SIZES = (15, 30, 60, 120)
SAMPLE_EFFICIENCY_SEEDS = (0, 1, 2, 3, 4)
NOISE_RATES = (0.001, 0.005, 0.010)
NOISE_SHOTS = 1024

# Overlay colours (BGR)
COLOR_MICROCALC = (0, 0, 220)  # red
COLOR_SPECKLE = (0, 200, 0)  # green
COLOR_NODULE = (0, 210, 255)  # yellow

TI_RADS_MICROCALC_POINTS = 2
# plan.md specifies +2 for a verified microcalcification. The 2017 ACR TI-RADS
# actually scores "punctate strong foci" at 3; the lower value is kept to match
# the project specification and is the single place to change it.
ACR_PUNCTATE_STRONG_FOCI_POINTS = 3


def ensure_output_dirs() -> None:
    """Create the generated-artefact tree. outputs/ is git-ignored."""
    for directory in (OUTPUTS_DIR, FIGURES_DIR, ANNOTATED_DIR, MODELS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
