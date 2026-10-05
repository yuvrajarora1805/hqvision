
---

# Complete Capstone Blueprint: Quantum-Assisted Thyroid Microcalcification Detection System

```
                         [ FULL END-TO-END SYSTEM ]

    Raw Patient Scan (.jpg) 
              │
              ▼
    ┌────────────────────────────────────────┐
    │  MODULE 1: Macro ROI & Candidate Mining│  ◄── Classical OpenCV
    │  • Nodule Localization (XML or YOLO)  │
    │  • Top-Hat Saliency + NMS Filter       │
    └──────────────────┬─────────────────────┘
                       │ Extracts N Candidate Patches (4x4)
                       ▼
    ┌────────────────────────────────────────┐
    │  MODULE 2: Quantum Verification Engine │  ◄── Qiskit 1.x
    │  • Train-Fitted PCA (16 ──► 4 dims)    │
    │  • ZZFeatureMap Quantum State Prep     │
    │  • Symmetric Kernel Evaluation (QSVC)  │
    └──────────────────┬─────────────────────┘
                       │ Classifies each spot: Real (1) vs Speckle (0)
                       ▼
    ┌────────────────────────────────────────┐
    │  MODULE 3: Clinical Decision & Overlay │  ◄── Output Engine
    │  • Annotates scan (Green=Speckle, Red=Real)
    │  • Recalculates TI-RADS Risk Category  │
    └──────────────────┬─────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────┐
    │  MODULE 4: Research Benchmarking Suite │  ◄── Viva / Paper Proof
    │  • Sample-Efficiency Curves            │
    │  • Qiskit Aer Noise Hardware Modeling  │
    │  • Classical Baseline Comparisons      │
    └────────────────────────────────────────┘
```

---

## Module 1: Data Pipeline & Patient-Isolated Partitioning

A common failure in medical AI capstones is data leakage between images belonging to the same patient.

### Implementation Tasks
1. **Dataset Integrity Checker:**
   * Script to parse all DDTI XMLs and flag missing images, corrupted coordinates, or empty nodule tags.
   * Map labels: `<calcifications>microcalcifications</calcifications>` $\to 1$, `<calcifications>non</calcifications>` $\to 0$. Exclude ambiguous/macro entries.
2. **Patient-Wise Stratified Splitter:**
   * Extract unique patient prefixes (e.g., patient `106` might have images `106_1.jpg`, `106_2.jpg`).
   * Perform an **$80/20$ patient-level split** using `GroupKFold` or hash partitioning so that zero patient tissue patterns cross between train and test splits.

---

## Module 2: Classical Candidate Mining (High-Recall Sieve)

The goal of this stage is to extract candidate spots with **$\ge 95\%$ recall**, ensuring true calcifications are almost never discarded upfront.

### Implementation Tasks
1. **Adaptive Contrast Normalization (CLAHE):**
   * Apply CLAHE (`clipLimit=2.0`, grid size $8 \times 8$) to normalize acoustic attenuation across different depths.
2. **Morphological Extraction:**
   * Apply morphological Top-Hat transformation using an elliptical structuring element matching sub-millimeter geometry ($\approx 5 \times 5$ pixels).
3. **Region Masking:**
   * Mask out everything outside the nodule boundaries (neck muscle, carotid artery, trachea).
4. **Non-Maximum Suppression (NMS):**
   * Suppress secondary peaks within a radius of $R = 4\text{ pixels}$.
   * Sort remaining peaks by local Weber contrast score:
     $$C = \frac{I_{\text{peak}} - I_{\text{background}}}{I_{\text{background}} + \epsilon}$$
   * Retain the top $K$ candidate coordinates ($K=4$ or $5$).
5. **Micro-Patch Cropping:**
   * Extract $4 \times 4$ pixel patches around each centroid at **original native resolution** (zero downsampling).

---

## Module 3: Qiskit Quantum Processing Engine

This module replaces heuristic spatial despeckling with an entangled quantum feature map.

### Implementation Tasks
1. **Dimensionality Reduction (Fitted on Train Only):**
   * Fit PCA on `X_train_raw` ($16 \to 4$ components).
   * Transform `X_test_raw` using the fitted training parameters.
   * Min-Max scale values into the angle domain $[0, \pi]$.
2. **Parameterized Circuit Construction:**
   * Use Qiskit's `ZZFeatureMap`:
     * Number of qubits: $n = 4$
     * Repetitions: $d = 2$
     * Entanglement topology: `'linear'` (keeps CNOT depth low for NISQ feasibility)
3. **Optimized Gram Matrix Calculation:**
   * Compute statevectors using `qiskit.quantum_info.Statevector`.
   * Enforce Gram matrix symmetry ($K_{ij} = K_{ji}$) and set diagonals $K_{ii} = 1.0$ to halve the simulation cost.
4. **Classifier Training:**
   * Train a precomputed Scikit-Learn `SVC` with balanced class weights to compensate for any remaining speckle-vs-lesion imbalance.

---

## Module 4: Clinical Inference & Diagnostic Overlay Engine

A project is incomplete if it only outputs an accuracy number in a terminal. This module processes raw patient images and generates visual clinical deliverables.

```
       RAW SCAN INPUT                          PROCESSED CLINICAL OVERLAY
┌───────────────────────────┐            ┌───────────────────────────────────┐
│     ~ ~ ~ ~ ~ ~ ~ ~       │            │        ~ ~ ~ ~ ~ ~ ~ ~            │
│   ~ ~ [ NODULE ] ~ ~      │            │     ~ ~ [ NODULE ] ~ ~            │
│       *  .   *            │   ─────►   │         🟢  .   🔴               │
│     *      .              │            │       🟢      .                   │
│    ~ ~ ~ ~ ~ ~ ~ ~ ~      │            │      ~ ~ ~ ~ ~ ~ ~ ~ ~            │
└───────────────────────────┘            └───────────────────────────────────┘
                                          🔴 Red Circle   = Quantum-Verified Microcalcification
                                          🟢 Green Circle = Filtered Acoustic Speckle Artifact
```

### Implementation Tasks
1. **Full-Image Stitcher (`inference.py`):**
   * Ingest an unseen test image and its nodule coordinates.
   * Classical stage flags candidate spots.
   * Quantum stage classifies each candidate spot.
2. **Clinical Visual Overlay:**
   * Render green circles around spots classified as **Acoustic Speckle (Class 0)**.
   * Render red circles around spots verified as **True Microcalcification (Class 1)**.
3. **Automated TI-RADS Risk Re-Scoring:**
   * If count of verified microcalcifications $\ge 1 \implies \mathbf{+2\text{ TI-RADS Points}}$ (elevates suspicion level).
   * If all candidate spots are classified as speckle $\implies \mathbf{0\text{ Points}}$ (prevents unnecessary biopsy recommendation).

---

## Module 5: Experimental Evaluation & Scientific Validation Suite

This is the module that proves your thesis and earns top marks during your viva. You must conduct **three mandatory experiments**:

### Experiment A: The Sample-Efficiency Test (Proving the Quantum Advantage)
* **Goal:** Prove that the quantum model outperforms classical models when training data is scarce.
* **Method:** Train all models on sub-sampled training sets: $N \in \{15, 30, 60, 120\}$.
* **Models Compared:**
  1. Proposed Hybrid Qiskit Model (QSVC)
  2. Classical RBF Support Vector Machine
  3. Classical Random Forest (50 estimators)
  4. Classical 3-layer Multi-Layer Perceptron (MLP)
* **Expected Result:** As $N$ drops below 40, classical neural networks overfit and their accuracy collapses; the quantum kernel maintains higher test accuracy due to its constrained, high-inductive-bias hypothesis space.

### Experiment B: Hardware Realism & Noise Simulation
* **Goal:** Show that your circuit will survive on real IBM Quantum devices.
* **Method:**
  * Re-run test inferences using `qiskit_aer.AerSimulator` configured with a simulated **Depolarizing Noise Model** (error rates: $0.1\%, 0.5\%, 1.0\%$) and finite shot sampling ($N_{\text{shots}} = 1024$).
  * Plot the decay of classification accuracy vs. noise rate to demonstrate fault tolerance.

### Experiment C: Candidate Mining Sensitivity (Ablation Study)
* **Goal:** Answer the examiner question: *"Did your classical filter accidentally drop real cancers?"*
* **Method:**
  * Evaluate the candidate miner across 50 known microcalcification nodules.
  * Count how often the true calcification was included in the top $K$ candidate pool.
  * Report the **Candidate Mining Recall** (target: $\ge 96\%$).

---

## Module 6: Recommended Repository Layout

Structure your project repository cleanly:

```text
thyroid-quantum-cad/
│
├── data/                       # DDTI images and XML files
│   ├── 106_1.jpg
│   ├── 106.xml
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Patient-isolated parsing and data loading
│   ├── mining.py               # Classical CLAHE, Top-Hat, and NMS candidate miner
│   ├── quantum_engine.py       # Qiskit ZZFeatureMap and Gram matrix calculation
│   └── pipeline.py             # Full inference pipeline & image overlay renderer
│
├── experiments/
│   ├── benchmark_classical.py  # Comparison against SVM, Random Forest, MLP
│   ├── sample_efficiency.py    # Few-shot accuracy curve generation
│   └── noise_simulation.py     # Qiskit Aer noise and shot-budget analysis
│
├── outputs/
│   ├── figures/                # Saved charts (ROC curves, sample efficiency plots)
│   └── annotated_scans/        # Clinical output images with red/green overlays
│
├── requirements.txt            # Locked dependencies
├── main.py                     # CLI entry-point to run end-to-end training
└── README.md                   # Full documentation with architecture diagram
```

---

## Module 7: Work Distribution & 8-Week Roadmap (2 Students)

| Week | Student A Focus (Computer Vision & Data) | Student B Focus (Quantum Circuits & Qiskit) | Shared Deliverable |
| :--- | :--- | :--- | :--- |
| **W1-2** | Build XML parser, fix patient-level data leakage, implement data loader. | Set up Qiskit 1.x environment, configure `ZZFeatureMap` unit tests. | Clean, isolated dataset split and verified patient cohorts. |
| **W3-4** | Implement Top-Hat filter, CLAHE, and NMS candidate mining. | Implement symmetric Gram matrix computation and test `Statevector` speed. | Verified micro-patch candidate dataset ($4 \times 4$). |
| **W5-6** | Train classical baselines (RBF-SVM, Random Forest, MLP). | Train Qiskit QSVC model; optimize parameter assignments. | Core accuracy and confusion matrix comparisons. |
| **W7** | Build the full-image clinical visualizer (red/green circle overlays). | Run Qiskit Aer noise simulations and sample-efficiency benchmarks. | Final charts, visual overlays, and performance tables. |
| **W8** | Draft final capstone report (Methodology, Literature Review). | Draft Quantum Circuit Analysis, Complexity, and Defense sections. | Completed thesis report and slide deck. |

---