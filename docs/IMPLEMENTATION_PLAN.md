# Quantum-Assisted Thyroid Microcalcification Detection System
## Phase-Wise Implementation Blueprint

**Project Root:** `/media/yuvraj/New Volume/xamp/htdocs/hqvision`  
**Target:** Hybrid Classical Computer Vision + Quantum Machine Learning Pipeline (DDTI Ultrasound Dataset)

---

## 🏛 Architecture & Data Flow Overview

```
                      Raw Patient Ultrasound Scan (.jpg + .xml)
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 1: Patient-Isolated Data Sieve & Parsing Engine                      │
   │ • Parse DDTI XML (bounding polygons in <svg> tags, TIRADS, calcifications)  │
   │ • Filter & Map: microcalcifications -> 1, non -> 0                          │
   │ • Group-based 80/20 train/test split on Patient ID (Zero tissue leakage)    │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │ Patient-wise cohorts
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 2: Classical Candidate Mining (High-Recall Sieve)                    │
   │ • CLAHE Contrast Normalization (clipLimit=2.0, grid=8x8)                    │
   │ • Morphological Top-Hat Transform (5x5 Elliptical Structuring Element)      │
   │ • Nodule Polygon Masking (remove carotid artery, muscle wall, trachea)      │
   │ • Weber Contrast Scoring + Non-Maximum Suppression (NMS, R=4 px)            │
   │ • Crop native 4x4 micro-patches (16 raw dimensions)                         │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │ 4x4 candidate patches
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 3: Quantum Processing Engine (Qiskit 1.x / QSVC)                     │
   │ • PCA dimensionality reduction fitted strictly on Train: 16 dims -> 4 dims  │
   │ • Min-Max scaling to angle domain [0, π]                                    │
   │ • 4-Qubit ZZFeatureMap (reps=2, linear entanglement topology)               │
   │ • Symmetric Gram matrix calculation (Statevector inner products, Kii=1.0)   │
   │ • Train Balanced QSVC (Support Vector Classifier with precomputed kernel)   │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │ Classified: Microcalcification vs Speckle
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 4: Clinical Decision Support & Visual Overlay Engine                  │
   │ • Render Diagnostic Overlays: 🔴 Red = Calcification, 🟢 Green = Speckle    │
   │ • Automated TI-RADS Re-Scoring:                                             │
   │     - Verified microcalcifications >= 1 ==> +2 TI-RADS risk points          │
   │     - All candidates confirmed speckle  ==> 0 additional points (safe)      │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ PHASE 5: Experimental Evaluation & Scientific Validation Suite              │
   │ • Experiment A: Sample-Efficiency Curves (QSVC vs RBF-SVM vs RF vs MLP)     │
   │ • Experiment B: Qiskit Aer Depolarizing Noise Model & Shot Budget Analysis │
   │ • Experiment C: Candidate Mining Sensitivity & Ablation (Target >= 95% Rec) │
   └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📅 Phase-Wise Breakdown

### Phase 1: Environment Setup & Patient-Isolated Data Ingestion
**Goal:** Prepare dependencies and create an airtight data loading pipeline that completely eliminates patient data leakage between train and test sets.

#### 1.1 Dependency Installation
- Install required packages in the local virtual environment:
  - `qiskit>=1.0.0`
  - `qiskit-aer`
  - `scikit-learn`
  - `opencv-python`
  - `numpy`, `matplotlib`, `pandas`

#### 1.2 Dataset Integrity & Parsing (`src/dataset.py`)
- Ingest DDTI dataset from `data/`:
  - Locate pairs: `[patient_id].xml` and corresponding `[patient_id]_[image_num].jpg` (e.g., `106.xml` with `106_1.jpg`, `106_2.jpg`, etc.).
  - Parse `<calcifications>`:
    - `microcalcifications` $\to 1$
    - `non` $\to 0$
    - Discard `macro` or indeterminate/empty tags.
  - Parse polygon vertices embedded in `<svg>` tags for each mark and associated image index.
  - Extract baseline clinical metadata: `<tirads>`, `<composition>`, `<echogenicity>`, `<margins>`.

#### 1.3 Patient-Stratified Partitioning (`src/dataset.py`)
- Extract unique patient prefix ID numbers.
- Split patients using an 80/20 stratified split (`GroupKFold` or patient hash partitioning).
- **Verification Guarantee:** Zero image patches belonging to a patient in the training set appear in the testing set.

---

### Phase 2: Classical Candidate Mining (High-Recall Sieve)
**Goal:** Mine suspected sub-millimeter calcification spots within nodule boundaries with $\ge 95\%$ recall.

#### 2.1 CLAHE Adaptive Contrast Normalization (`src/mining.py`)
- Convert ultrasound image to grayscale.
- Apply Contrast Limited Adaptive Histogram Equalization (CLAHE):
  - `clipLimit = 2.0`
  - `tileGridSize = (8, 8)`
  - Compensates for ultrasound depth attenuation and acoustic beam variance.

#### 2.2 Morphological Top-Hat Transform (`src/mining.py`)
- Elliptical structuring element matching physical sub-millimeter punctate microcalcifications:
  - Kernel size: $5 \times 5$ (`cv2.MORPH_ELLIPSE`).
  - Top-Hat operation: isolates high-frequency bright punctate spots while suppressing uniform background tissue.

#### 2.3 Region Masking (`src/mining.py`)
- Construct binary mask from polygon vertices parsed from the XML annotation.
- Zero out all extraneous acoustic artifacts outside the nodule (carotid artery, tracheal reverberations, muscle striations).

#### 2.4 Weber Contrast Peak Detection & NMS (`src/mining.py`)
- Compute local Weber contrast metric for local intensity peaks:
  $$C = \frac{I_{\text{peak}} - I_{\text{background}}}{I_{\text{background}} + \epsilon}$$
- Perform Non-Maximum Suppression (NMS) within radius $R = 4\text{ px}$ to eliminate duplicated neighbor detections.
- Select top $K$ candidate coordinates ($K = 4$ or $5$).

#### 2.5 Native Micro-Patch Extraction (`src/mining.py`)
- Extract native resolution $4 \times 4$ pixel patches around each centroid.
- Flatten patch into a 16-dimensional feature vector per candidate patch.

---

### Phase 3: Quantum Processing Engine (Qiskit 1.x QSVC)
**Goal:** Classify extracted micro-patches as true microcalcification ($1$) or acoustic speckle noise ($0$) using an entangled quantum feature map.

#### 3.1 Dimensionality Reduction & Angle Mapping (`src/quantum_engine.py`)
- Fit PCA on `X_train` native patches ($16 \to 4$ components).
- Apply fitted PCA transformation to `X_test`.
- Min-Max scale values into the angle encoding interval $[0, \pi]$.

#### 3.2 Parameterized Quantum Circuit (`src/quantum_engine.py`)
- Construct Qiskit `ZZFeatureMap`:
  - Number of qubits: $n = 4$
  - Circuit repetitions: $d = 2$
  - Entanglement topology: `'linear'` (keeps CNOT circuit depth low for NISQ feasibility)

#### 3.3 Symmetric Gram Matrix Optimization (`src/quantum_engine.py`)
- Compute statevector representations $|\psi(x)\rangle$ using `qiskit.quantum_info.Statevector`.
- Compute quantum kernel entries:
  $$K_{ij} = |\langle\psi(x_i)|\psi(x_j)\rangle|^2$$
- Symmetrize calculation: compute only upper triangle ($K_{ij} = K_{ji}$) and set diagonals $K_{ii} = 1.0$, cutting quantum execution time by $\approx 50\%$.

#### 3.4 Support Vector Classification (`src/quantum_engine.py`)
- Train `sklearn.svm.SVC(kernel='precomputed', class_weight='balanced')`.
- Balance speckle vs. lesion class distribution.

---

### Phase 4: Clinical Decision Support & Visual Overlay Engine
**Goal:** Provide actionable visual diagnostic overlays and automated risk score calculation.

#### 4.1 Diagnostic Scanning Pipeline (`src/pipeline.py`)
- Ingest full patient scan and nodule coordinates.
- Run candidate mining sieve to locate $K$ ambiguous candidate spots.
- Forward candidates through the trained quantum verification model.

#### 4.2 Color-Coded Overlay Rendering (`src/pipeline.py`)
- 🔴 **Red Circle:** Quantum-Verified Microcalcification ($y = 1$).
- 🟢 **Green Circle:** Filtered Acoustic Speckle Artifact ($y = 0$).
- Save output scan to `outputs/annotated_scans/`.

#### 4.3 Automated TI-RADS Re-Scoring (`src/tirads.py`)
- Base ACR TI-RADS logic:
  - If verified microcalcifications $\ge 1 \implies \mathbf{+2}$ points (elevates nodule risk level, recommending FNA biopsy).
  - If all candidate spots are classified as acoustic speckle $\implies \mathbf{0}$ points (prevents unnecessary biopsy recommendations).

---

### Phase 5: Experimental Evaluation & Scientific Validation Suite
**Goal:** Deliver publication-grade and viva-ready experimental evidence of the hybrid quantum advantage.

#### 5.1 Experiment A: The Sample-Efficiency Test (`experiments/sample_efficiency.py`)
- Train models on restricted sample subsets:
  $$N \in \{15, 30, 60, 120\}$$
- Benchmark models:
  1. Proposed Hybrid Quantum QSVC
  2. Classical RBF Support Vector Machine
  3. Classical Random Forest (50 trees)
  4. Classical Multi-Layer Perceptron (3 layers)
- Generate sample-efficiency learning curves saved to `outputs/figures/sample_efficiency.png`.

#### 5.2 Experiment B: Hardware Realism & Noise Simulation (`experiments/noise_simulation.py`)
- Run inference with `qiskit_aer.AerSimulator` under depolarizing noise models:
  - Noise rates: $0.1\%, 0.5\%, 1.0\%$
  - Finite shot budgets: $N_{\text{shots}} = 1024$
- Plot accuracy decay vs noise rate in `outputs/figures/noise_resilience.png`.

#### 5.3 Experiment C: Candidate Mining Sensitivity & Ablation (`experiments/candidate_ablation.py`)
- Test candidate mining sieve on microcalcification nodules.
- Measure recall of true calcifications in top $K$ candidate selections.
- Ensure sensitivity meets or exceeds the target threshold ($\ge 95\%$).

---

## 📂 Proposed Repository Directory Layout

```
/media/yuvraj/New Volume/xamp/htdocs/hqvision/
├── IMPLEMENTATION_PLAN.md    <-- Phase-by-phase execution guide
├── requirements.txt          <-- Environment dependencies
├── main.py                   <-- Unified CLI entry-point (train / eval / predict)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py            # XML parser, patient grouper, stratified train/test split
│   ├── mining.py             # CLAHE, Top-Hat filter, Weber contrast, NMS, 4x4 patch crop
│   ├── quantum_engine.py     # PCA, ZZFeatureMap, symmetric Gram matrix, QSVC
│   ├── pipeline.py           # End-to-end full image inference & visual overlays
│   └── tirads.py             # TI-RADS risk calculator & clinical reporting
│
├── experiments/
│   ├── sample_efficiency.py  # Exp A: Few-shot curves (N=15..120) vs classical models
│   ├── noise_simulation.py   # Exp B: Aer depolarizing noise & shot analysis
│   └── candidate_ablation.py # Exp C: Classical candidate sieve recall test
│
└── outputs/
    ├── figures/              # Metric charts & comparison graphs
    └── annotated_scans/      # Visual ultrasound overlays (Red / Green circles)
```

---

## 🚀 Execution Checklist

- [ ] **Phase 1:** Update virtual environment dependencies (`pip install qiskit qiskit-aer matplotlib`) and implement `src/dataset.py`.
- [ ] **Phase 2:** Build and unit-test classical candidate mining sieve (`src/mining.py`) with CLAHE, Top-Hat, and NMS.
- [ ] **Phase 3:** Construct Qiskit 1.x ZZFeatureMap, statevector Gram matrix optimizer, and QSVC trainer (`src/quantum_engine.py`).
- [ ] **Phase 4:** Build diagnostic inference pipeline with red/green circles and TI-RADS calculator (`src/pipeline.py`, `src/tirads.py`).
- [ ] **Phase 5:** Run benchmark experiments (Sample-efficiency, Aer noise simulation, candidate ablation) and save charts to `outputs/figures/`.
