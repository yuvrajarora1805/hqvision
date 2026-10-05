
---

# Complete Capstone Blueprint: Quantum-Assisted Thyroid Nodule Malignancy Classifier

```
                         [ FULL END-TO-END SYSTEM ]

    Raw Ultrasound Frame (.jpg) + nodule box (.xml)
              │
              ▼
    ┌────────────────────────────────────────┐
    │  MODULE 1: Data Sieve & Split          │  ◄── TN5000 (VOC)
    │  • Parse bbox + biopsy label (1/0)     │
    │  • Official trainval/test split        │
    └──────────────────┬─────────────────────┘
                       │ 4000 train / 1000 test images
                       ▼
    ┌────────────────────────────────────────┐
    │  MODULE 2: Classical Candidate Mining  │  ◄── Classical OpenCV
    │  • CLAHE + Top-Hat saliency + NMS      │
    │  • ROI = nodule bounding box           │
    └──────────────────┬─────────────────────┘
                       │ Extracts up to K candidate patches (4x4)
                       ▼
    ┌────────────────────────────────────────┐
    │  MODULE 3: Quantum Verification Engine │  ◄── Qiskit 2.x
    │  • Train-fitted PCA (16 ──► 4 dims)    │
    │  • ZZFeatureMap quantum state prep     │
    │  • Symmetric kernel evaluation (SVC)   │
    └──────────────────┬─────────────────────┘
                       │ Classifies each patch: malignant (1) vs benign (0)
                       ▼
    ┌────────────────────────────────────────┐
    │  MODULE 4: Clinical Decision & Overlay │  ◄── Output Engine
    │  • Annotates scan (red = malignant)    │
    │  • Majority-vote verdict per nodule    │
    └──────────────────┬─────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────┐
    │  MODULE 5: Research Benchmarking Suite │  ◄── Viva / Paper Proof
    │  • Sample-Efficiency Curves            │
    │  • Qiskit Aer noise hardware modeling  │
    │  • Classical Baseline Comparisons      │
    └────────────────────────────────────────┘
```

**Dataset:** TN5000 (Nature Scientific Data, 2025) — 5,000 thyroid ultrasound
frames, one nodule bounding box per image, biopsy-confirmed label per nodule:
`<name>1</name>` = malignant (3,574), `<name>0</name>` = benign (1,426).
PASCAL-VOC layout: `JPEGImages/`, `Annotations/`, `ImageSets/Main/{train,val,test,trainval}.txt`.

> Why not DDTI: DDTI has ~480 annotated frames with TI-RADS/calcification text tags —
> too few samples to train or evaluate honestly, and no biopsy ground truth.
> TN5000 gives 10x the data with a confirmed label and a leakage-safe published split.

---

## Module 1: Data Pipeline & Split Hygiene

A common failure in medical AI capstones is data leakage between images of the same
patient.

### Implementation Tasks
1. **Dataset Integrity Checker** (`main.py integrity`):
   * Parse all 5,000 VOC XMLs; flag missing images, missing XMLs, corrupt boxes, unexpected labels.
   * Label map: `<name>1</name>` → 1 (malignant), `<name>0</name>` → 0 (benign). Nothing else is accepted.
2. **Official Split** (`ImageSets/Main`):
   * `trainval.txt` (4,000 images) → training, `test.txt` (1,000) → test; overlap asserted to be 0.
   * The TN5000 authors kept one representative image per patient perspective when building this
     split — i.e. the leakage-safe partition ships with the data, so we use it rather than rolling
     our own. Fallback: seeded stratified 80/20 only if `ImageSets/` is absent.

---

## Module 2: Classical Candidate Mining (High-Recall Sieve)

The goal of this stage is to extract candidate spots with **≥ 95% recall**,
ensuring true suspicious foci are almost never discarded upfront.

### Implementation Tasks
1. **Adaptive Contrast Normalization (CLAHE):**
   * Apply CLAHE (`clipLimit=2.0`, grid size 8×8) to normalize acoustic attenuation across depths.
2. **Morphological Extraction:**
   * Top-Hat transform with an elliptical structuring element matching sub-millimeter
     geometry (≈ 5×5 pixels).
3. **Region Masking:**
   * Mask out everything outside the nodule bounding box (neck muscle, carotid artery, trachea).
4. **Non-Maximum Suppression (NMS):**
   * Suppress secondary peaks within a radius of R = 4 pixels.
   * Sort remaining peaks by local Weber contrast score:
     $$C = \frac{I_{\text{peak}} - I_{\text{background}}}{I_{\text{background}} + \epsilon}$$
   * Retain the top K candidate coordinates (K = 5).
5. **Micro-Patch Cropping:**
   * Extract 4×4 pixel patches around each centroid at **original native resolution** (zero
     downsampling) → 16 raw features.

Labels are **weak**: each patch inherits the nodule's biopsy label. A patch inside a
malignant nodule is tagged 1 even if that particular spot is not the lesion itself —
this is stated as a limitation everywhere results appear.

---

## Module 3: Qiskit Quantum Processing Engine

This module replaces heuristic spatial despeckling with an entangled quantum feature map.

### Implementation Tasks
1. **Dimensionality Reduction (Fitted on Train Only):**
   * Fit PCA on `X_train_raw` (16 → 4 components); transform `X_test_raw` with the fitted
     parameters; Min-Max scale into the angle domain [0, π].
2. **Parameterized Circuit Construction:**
   * Qiskit `ZZFeatureMap`: n = 4 qubits, d = 2 repetitions, entanglement `'linear'`
     (keeps CNOT depth low for NISQ feasibility).
3. **Optimized Gram Matrix Calculation:**
   * Compute statevectors with `qiskit.quantum_info.Statevector`; enforce K = Kᵀ and K_ii = 1.
   * One statevector per sample + a single matmul instead of |A|×|B| individual circuit runs.
4. **Classifier Training:**
   * Precomputed-kernel `SVC(class_weight='balanced')` — balancing the 3,574 : 1,426
     malignant/benign imbalance at the patch level.
5. **Feasibility cap:** kernel memory is O(N²), so training patches are balanced-subsampled
   (`MAX_TRAIN_PATCHES = 2000`, seed 42); the test set stays whole (cross-kernel is O(N_test·N_train)).

---

## Module 4: Clinical Inference & Diagnostic Overlay Engine

A project is incomplete if it only outputs an accuracy number in a terminal.

```
       RAW SCAN INPUT                          PROCESSED CLINICAL OVERLAY
┌───────────────────────────┐            ┌───────────────────────────────────┐
│     ~ ~ ~ ~ ~ ~ ~ ~       │            │        ~ ~ ~ ~ ~ ~ ~ ~            │
│   ~ ~ [ NODULE ] ~ ~      │            │     ~ ~ [ NODULE ] ~ ~            │
│       *  .   *            │   ─────►   │         🔴  .   🟢               │
│     *      .              │            │       🔴      .                   │
│    ~ ~ ~ ~ ~ ~ ~ ~ ~      │            │      ~ ~ ~ ~ ~ ~ ~ ~ ~            │
└───────────────────────────┘            └───────────────────────────────────┘
                                          🔴 Red Circle   = patch classified malignant
                                          🟢 Green Circle = patch classified benign
                                          yellow box      = annotated nodule ROI
```

### Implementation Tasks
1. **Full-Image Stitcher** (`src/pipeline.py`, `main.py predict`):
   * Ingest an unseen test image (and its bbox, or read it from the XML).
   * Classical stage flags candidate spots inside the ROI; quantum stage classifies each.
2. **Clinical Visual Overlay:**
   * Render green circles around patches classified as benign (0), red circles +
     cross markers around patches classified as malignant (1); legend with counts.
3. **Nodule-Level Verdict:**
   * Majority vote over the K mined patches (ties broken by mean kernel margin) →
     one decision and a confidence fraction for the whole nodule.
   * TN5000 carries **no TI-RADS / composition / echogenicity metadata**, so the DDTI-era
     TI-RADS re-scoring was removed rather than faked — the verdict is directly comparable
     with the biopsy label.

---

## Module 5: Experimental Evaluation & Scientific Validation Suite

This is the module that proves your thesis and earns top marks during your viva. You must conduct **three mandatory experiments**:

### Experiment A: The Sample-Efficiency Test (Proving the Quantum Advantage)
* **Goal:** Prove that the quantum model outperforms classical models when training data is scarce.
* **Method:** Train all models on sub-sampled training sets: N ∈ {15, 30, 60, 120}.
* **Models Compared:**
  1. Proposed Hybrid Qiskit Model (quantum-kernel SVC)
  2. Classical RBF Support Vector Machine
  3. Classical Random Forest (50 estimators)
  4. Classical 3-layer Multi-Layer Perceptron (MLP)
* **Expected Result:** As N drops below 40, classical networks overfit and their accuracy
  collapses; the quantum kernel may maintain higher accuracy due to its constrained,
  high-inductive-bias hypothesis space. *If it does not, the honest conclusion stands.*

### Experiment B: Hardware Realism & Noise Simulation
* **Goal:** Show that your circuit will survive on real IBM Quantum devices.
* **Method:**
  * Re-run test inferences using `qiskit_aer.AerSimulator` with a **Depolarising Noise Model**
    (error rates: 0.1%, 0.5%, 1.0%) and finite shot sampling (N_shots = 1024).
  * Plot accuracy decay vs. noise rate to demonstrate fault tolerance.

### Experiment C: Candidate Mining Sensitivity (Ablation Study)
* **Goal:** Answer the examiner question: *"Did your classical filter drop real cancers?"*
* **Method:**
  * Sweep K ∈ {1, 3, 5, 8} and the top-hat threshold; retrain and report **nodule-level
    malignant recall** (test: 731 malignant nodules) plus the fraction of nodules yielding
    at least one candidate.
  * Target: ≥ 96% of malignant nodules still produce candidates (mining is recall-first).

---

## Module 6: Repository Layout

```text
hqvision/
│
├── data/
│   └── Main data/              # TN5000 (git-ignored; or set $TN5000_DIR)
│       ├── JPEGImages/         # 5000 .jpg
│       ├── Annotations/        # 5000 .xml (VOC: bbox + label 1/0)
│       └── ImageSets/Main/     # official trainval/test id lists
│
├── src/
│   ├── config.py               # every tunable: paths, labels, caps, circuit params
│   ├── dataset.py              # VOC parser, integrity audit, official split
│   ├── mining.py               # CLAHE, Top-Hat, Weber contrast, NMS, 4x4 crop
│   ├── cache.py                # mined-patch cache + balanced subsampling
│   ├── quantum_engine.py       # ZZFeatureMap, Gram matrices, kernel SVC, noise
│   └── pipeline.py             # inference, nodule verdict, overlay renderer
│
├── outputs/                    # git-ignored
│   ├── cache/                  # mined patch pickles
│   ├── models/                 # trained SVCs + metrics json
│   ├── figures/                # charts (ROC, sample-efficiency, noise)
│   └── annotated_scans/        # red/green overlays
│
├── train_hybrid_model.py       # DEPRECATED legacy DDTI/PennyLane baseline
├── requirements.txt            # locked dependencies
├── main.py                     # CLI: integrity | mine | train | predict | circuit
└── docs/                       # plan, implementation plan, review, status
```

---

## Module 7: Work Distribution & 8-Week Roadmap (2 Students)

| Week | Student A Focus (Computer Vision & Data) | Student B Focus (Quantum Circuits & Qiskit) | Shared Deliverable |
| :--- | :--- | :--- | :--- |
| **W1-2** | TN5000 parser, integrity audit, official split wiring. | Qiskit 2.x environment, `ZZFeatureMap` tests. | Verified 4000/1000 split, zero overlap. |
| **W3-4** | Top-Hat filter, CLAHE, NMS candidate mining on bboxes. | Gram matrix computation, `Statevector` speed test. | Mined patch dataset (4×4), cache layer. |
| **W5-6** | Train classical baselines (RBF-SVM, RF, MLP). | Train quantum-kernel SVC; kernel comparison. | Accuracy / balanced accuracy / AUC tables. |
| **W7** | Full-image visualizer (red/green overlay, verdict). | Aer noise simulations + sample-efficiency runs. | Final charts, overlays, performance tables. |
| **W8** | Draft report (Methodology, Literature Review). | Draft Quantum Circuit Analysis & Defense sections. | Completed thesis report and slide deck. |

---
