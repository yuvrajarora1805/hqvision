# Quantum-Assisted Thyroid Nodule Malignancy Classifier
## Phase-Wise Implementation Blueprint

**Project:** HQVision — hybrid classical CV + quantum kernel ML pipeline
**Dataset:** TN5000 (PASCAL VOC, 5,000 ultrasound frames, biopsy-confirmed nodule labels)
**Entry point:** `python main.py {integrity | mine | train | predict | circuit}`

---

## Architecture & Data Flow Overview

```
                    Raw ultrasound frame (.jpg) + nodule box (.xml)
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 1: TN5000 Data Sieve & Integrity Audit                               │
   │ • Parse VOC XML: <bndbox> + <name> (1 = malignant, 0 = benign)             │
   │ • Audit: 5000 images / 5000 XMLs / 3574 malignant / 1426 benign            │
   │ • Official ImageSets split: trainval 4000 vs test 1000, overlap = 0         │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │ record = {bbox, label, polygons}
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 2: Classical Candidate Mining (High-Recall Sieve)                    │
   │ • CLAHE Contrast Normalization (clipLimit=2.0, grid=8x8)                    │
   │ • Morphological Top-Hat Transform (5x5 Elliptical Structuring Element)      │
   │ • Nodule ROI Masking (bbox rectangle converted to polygon, background cut)  │
   │ • Weber Contrast Scoring + Non-Maximum Suppression (NMS, R=4 px)            │
   │ • Crop native 4x4 micro-patches (16 raw dimensions), top K=5 per nodule     │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │ 19973 train / 4998 test patches
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 3: Quantum Processing Engine (Qiskit 2.x kernel SVC)                 │
   │ • PCA dimensionality reduction fitted strictly on Train: 16 dims -> 4 dims  │
   │ • Min-Max scaling to angle domain [0, π]                                    │
   │ • 4-Qubit ZZFeatureMap (reps=2, linear entanglement topology)               │
   │ • Symmetric Gram matrix calculation (Statevector inner products, Kii=1.0)   │
   │ • Balanced precomputed-kernel SVC (patches capped at N=2000, seed 42)       │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │ patch label: malignant (1) vs benign (0)
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 4: Clinical Decision Support & Visual Overlay Engine                  │
   │ • Render overlays: red = malignant patch, green = benign patch, yellow ROI  │
   │ • Nodule verdict: majority vote over the K patches + mean kernel margin     │
   │   (TN5000 has no TI-RADS metadata, so the DDTI-era TI-RADS rescoring is     │
   │    removed rather than approximated)                                        │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 5: Experimental Evaluation & Scientific Validation Suite  [PENDING]   │
   │ • Experiment A: Sample-Efficiency Curves (quantum vs RBF-SVM vs RF vs MLP)  │
   │ • Experiment B: Qiskit Aer Depolarizing Noise Model & Shot Budget Analysis  │
   │ • Experiment C: Mining Ablation — K sweep, nodule-level malignant recall    │
   └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase-Wise Breakdown

### Phase 1: Environment Setup & TN5000 Data Ingestion — DONE

**Goal:** an airtight data loading pipeline that eliminates split leakage.

#### 1.1 Dependency installation
`qiskit>=2.x`, `qiskit-aer`, `scikit-learn`, `opencv-python`, `numpy`, `matplotlib`
(in `qip_venv`, Python 3.14; `qiskit-machine-learning` is intentionally absent — no
cp314 wheel — the kernel is built on Qiskit primitives + a precomputed-kernel SVC).

#### 1.2 Dataset integrity & parsing (`src/dataset.py`)
- Ingest TN5000 from `data/Main data` (or `$TN5000_DIR`):
  - `JPEGImages/*.jpg` paired by stem with `Annotations/*.xml`.
  - Parse `<bndbox>` → `(xmin, ymin, xmax, ymax)`, clamped to image size; degenerate
    boxes rejected and counted.
  - Parse `<name>`: `1 → LABEL_MALIGNANT`, `0 → LABEL_BENIGN`; anything else rejected.
  - Convert the box to a 4-corner polygon so the miner's polygon API is unchanged.
- `main.py integrity` prints the full audit (counted, never silently dropped).

#### 1.3 Official split assignment (`src/dataset.py`)
- `ImageSets/Main/trainval.txt` (4000) → train, `test.txt` (1000) → test.
- Leakage-safe by construction: the TN5000 authors kept one representative image per
  patient perspective when building these lists. Overlap is asserted to be 0.
- Fallback (only if `ImageSets/` is missing): seeded stratified 80/20 by label.

---

### Phase 2: Classical Candidate Mining (High-Recall Sieve) — DONE

**Goal:** mine suspected bright punctate spots inside the nodule ROI with ≥ 95% recall.

#### 2.1 CLAHE adaptive contrast normalization (`src/mining.py`)
`clipLimit = 2.0`, `tileGridSize = (8, 8)` — compensates for depth attenuation.

#### 2.2 Morphological top-hat transform (`src/mining.py`)
5×5 elliptical structuring element (`cv2.MORPH_ELLIPSE`) isolates punctate bright
foci against uniform tissue.

#### 2.3 ROI masking (`src/mining.py`)
Binary mask from the bbox polygon; everything outside the nodule (carotid artery,
tracheal reverberations, muscle) is zeroed before peak detection.

#### 2.4 Weber contrast peak detection & NMS (`src/mining.py`)
- Dynamic threshold `max(15, mean + 1.2σ)` over ROI top-hat values, contour centroids.
- Local Weber contrast: C = (I_peak − I_bg) / (I_bg + ε); NMS radius R = 4 px;
  keep top K = 5.

#### 2.5 Native micro-patch extraction (`src/mining.py`)
4×4 crop at native resolution → 16-dim float feature vector. Yields
**19,973 train patches / 4,998 test patches**, and every one of the 5,000 nodules
produced at least one candidate in the verification run.

---

### Phase 3: Quantum Processing Engine (Qiskit kernel SVC) — DONE

**Goal:** classify each patch as malignant (1) or benign (0) with an entangled
quantum feature map; measure nodule-level performance after majority vote.

#### 3.1 Dimensionality reduction & angle mapping (`src/quantum_engine.py`)
PCA 16 → 4 fitted **on train only**; Min-Max into [0, π]. (Fitting the scaler on the
full dataset would leak test statistics — a bug in the original script.)

#### 3.2 Parameterized quantum circuit (`src/quantum_engine.py`)
`ZZFeatureMap`: n = 4 qubits, reps = 2, `entanglement = "linear"`.
Measured circuit stats (`main.py circuit`): depth 19, size 34, 12 cx gates.

#### 3.3 Gram matrix optimization (`src/quantum_engine.py`)
K(a,b) = |⟨ψ(a)|ψ(b)⟩|², symmetrised, unit diagonal. One statevector per sample and
a single matmul — replaces |A|×|B| individual circuit runs.

#### 3.4 Classifier training (`src/quantum_engine.py`, `src/cache.py`)
`SVC(kernel="precomputed", class_weight="balanced")`; training patches
balanced-subsampled to `MAX_TRAIN_PATCHES = 2000` (seed 42) so the O(N²) Gram matrix
stays small; test cross-kernel uses all 4,998 patches. Metrics reported at **both**
patch level (weak labels) and nodule level (majority vote) — the nodule level is the
honest clinical unit because the label is per nodule.

---

### Phase 4: Clinical Decision Support & Visual Overlay — DONE

**Goal:** actionable visual output, not just a number in a terminal.

#### 4.1 Diagnostic scanning pipeline (`src/pipeline.py`)
`ThyroidCADPipeline.analyse(image, bbox)` → mine K candidates → quantum verifier →
labels + signed margins.

#### 4.2 Color-coded overlay (`src/pipeline.py`)
- Red circle + cross marker: patch classified **malignant**.
- Green circle: patch classified **benign**.
- Yellow rectangle: annotated nodule ROI; legend shows counts, verdict, confidence.
- Written by `python main.py predict --image-id NNNNNN` to `outputs/annotated_scans/`.

#### 4.3 Nodule verdict (`src/pipeline.py`)
Majority vote over the K patch decisions, ties broken by mean kernel margin
(fallback benign), reported with a confidence fraction. Compared against the biopsy
label for nodule-level metrics.

---

### Phase 5: Experimental Evaluation & Scientific Validation Suite — PENDING

**Goal:** viva-ready evidence. Planned as before, with TN5000 units:

#### 5.1 Experiment A: sample efficiency
N ∈ {15, 30, 60, 120} training patches, 5 seeds; quantum-kernel SVC vs RBF-SVM vs
Random Forest (50) vs MLP (3 layers), identical splits; curves →
`outputs/figures/sample_efficiency.png`.

#### 5.2 Experiment B: hardware realism & noise
`AerSimulator` depolarizing noise at 0.1 / 0.5 / 1.0% per gate, 1024 shots,
cross-kernel re-evaluated with `noisy_fidelity_gram()` → accuracy decay curve.
(`noisy_fidelity_gram` already fixed and verified: max deviation 0.007 vs the exact
kernel at 8192 shots.)

#### 5.3 Experiment C: mining ablation
Sweep K ∈ {1, 3, 5, 8}; report nodule-level malignant recall (731 malignant test
nodules) and the fraction of nodules producing ≥ 1 candidate.

---

## Verified Run Log (smoke tests, seed 42)

```
python main.py integrity  → 5000/5000 parsed, 3574 malignant / 1426 benign,
                            official split 4000/1000, overlap 0
python main.py mine       → 74 s; train X=(19973,16), test X=(4998,16),
                            all 5000 nodules yielded candidates
python main.py train      → ZZ kernel: patch acc 0.647 / bal 0.628 / AUC 0.654;
                            nodule acc 0.670 / bal 0.647 / AUC 0.700
python main.py train --kernel overlap
                          → patch acc 0.679 / bal 0.610 / AUC 0.657;
                            nodule acc 0.713 / bal 0.632 / AUC 0.689
python main.py predict --image-id 000123 → 5 candidates, verdict + overlay PNG
python main.py circuit    → ZZ: depth 19 / size 34 / 12 cx; overlap: depth 2 / 0 cx
```

These are single-seed **smoke** numbers from an end-to-end wiring check — not
experiment results. Experiments A–C are still to be run, and no claim of quantum
advantage may be made from the table above.

---

## Repository Directory Layout

```
hqvision/
├── main.py                   # CLI: integrity | mine | train | predict | circuit
├── requirements.txt          # environment dependencies
├── train_hybrid_model.py     # DEPRECATED legacy DDTI/PennyLane baseline
│
├── src/
│   ├── config.py             # all tunables: paths, labels, caps, circuit params
│   ├── dataset.py            # VOC parser, integrity audit, official split
│   ├── mining.py             # CLAHE, Top-Hat, Weber contrast, NMS, 4x4 patch crop
│   ├── cache.py              # mined-patch cache + balanced subsampling
│   ├── quantum_engine.py     # ZZFeatureMap, Gram matrices, kernel SVC, Aer noise
│   └── pipeline.py           # inference, nodule verdict, visual overlays
│
├── data/
│   └── Main data/            # TN5000 (git-ignored) — or point $TN5000_DIR at it
│       ├── JPEGImages/  ├── Annotations/  └── ImageSets/Main/
│
├── outputs/                  # git-ignored
│   ├── cache/  ├── models/  ├── figures/  └── annotated_scans/
│
├── experiments/              # Phase 5 scripts (to be created)
│   ├── sample_efficiency.py  # Exp A: few-shot curves vs classical models
│   ├── noise_simulation.py   # Exp B: Aer depolarizing noise & shot analysis
│   └── mining_ablation.py    # Exp C: K sweep, nodule-level malignant recall
│
└── docs/                     # plan.md, IMPLEMENTATION_PLAN.md, STATUS.md,
                              # REVIEW_AND_DEFENSE.md
```

---

## Execution Checklist

- [x] **Phase 1:** Dependencies installed; `src/dataset.py` parses TN5000, audits
      integrity, applies the official split (verified: `main.py integrity`).
- [x] **Phase 2:** Classical candidate mining sieve (`src/mining.py`) with CLAHE,
      Top-Hat, Weber NMS; caches written (`main.py mine`, 74 s full run).
- [x] **Phase 3:** Qiskit ZZFeatureMap, statevector Gram optimizer, balanced kernel
      SVC (`src/quantum_engine.py`); patch + nodule metrics (`main.py train`).
- [x] **Phase 4:** Inference pipeline with red/green overlay and nodule verdict
      (`src/pipeline.py`, `main.py predict`); TI-RADS rescoring removed — TN5000 has
      no TI-RADS metadata.
- [ ] **Phase 5:** Benchmark experiments (sample efficiency, Aer noise, mining
      ablation) and charts in `outputs/figures/`.
- [ ] **Report:** final numbers, comparison tables and figures for the report.
