# Status Report — Quantum-Assisted Thyroid Microcalcification Detection

**Date:** 28 September 2026
**Scope:** Phases 0–4 of the plan in `plan.md` (environment, data pipeline, candidate mining, quantum engine, training CLI)
**Headline:** All six modules are built and verified working. However, the classification task
as specified in `plan.md` was found to be **unlearnable on the DDTI dataset**. This document
records the investigation, the evidence, and the decision that must be made before Phase 5.

---

## 1. What was built

| Module | File | Status |
| :--- | :--- | :--- |
| Config | `src/config.py` | Complete |
| Dataset parsing & patient split | `src/dataset.py` | Complete, verified |
| Classical candidate mining | `src/mining.py` | Complete, verified |
| Qiskit quantum engine | `src/quantum_engine.py` | Complete, verified to machine precision |
| Patch dataset cache | `src/cache.py` | Complete |
| Clinical inference & overlay | `src/pipeline.py` | Complete, not yet visually verified |
| CLI entry point | `main.py` | Complete (`integrity`, `train`, `predict`, `circuit`) |
| Experiments (Phase 5) | `experiments/` | **Not started** — blocked, see §6 |

**CLI works end to end:**

```
python main.py integrity     # dataset audit
python main.py train         # mine -> split -> fit -> evaluate
python main.py predict --case 106
python main.py circuit       # gate/depth tables
```

---

## 2. Environment

Interpreters available: system Python 3.14.4 and a pre-built quantum venv at
`../qip_venv`. The venv was used, and was already correct for almost everything.

**Pre-installed in `qip_venv`:** qiskit 2.5.1, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.48.0,
numpy 2.5.1, scipy 1.18.0, matplotlib 3.11.1, seaborn 0.13.2, pandas 3.0.5, pillow 12.3.0.

**Installed by me:** `opencv-python==5.0.0.93`, `scikit-learn==1.9.1`, `joblib==1.6.0`.

### 2.1 Why Qiskit 1.x and PennyLane were both abandoned

`plan.md` specifies Qiskit 1.x; the original `train_hybrid_model.py` used PennyLane.
Neither is viable on Python 3.14:

- **Qiskit 1.x does not support Python 3.14 at all.** The installed qiskit is 2.5.1, whose
  classifiers explicitly list Python 3.14.
- **`qiskit-machine-learning` 0.9.1 publishes no cp314 or abi3 wheel**, so `qiskit.qsvm.QSVC`
  cannot be imported. The quantum kernel is therefore built directly on qiskit primitives with
  a scikit-learn precomputed-kernel SVC — which is what `plan.md` Module 3 already prescribes.
- PennyLane was dropped to keep a single quantum stack. Its kernel was reimplemented in
  Qiskit instead (see §4.3), so the comparison survives at zero dependency cost.

**API note for the report:** the `ZZFeatureMap` *class* is deprecated as of Qiskit 2.1 and
removed in 3.0. The code uses the `zz_feature_map()` *function* and falls back to the class
for older versions.

### 2.2 The performance point that makes the project tractable

Building a Gram matrix naively means |A| x |B| separate circuit simulations — 1.6 M for
this dataset. Because the kernel entry is `K(a_i, b_j) = |<psi(a_i)|psi(b_j)>|^2`, the whole
matrix factorises: simulate |A| + |B| statevectors once, then take **one complex matmul**.

| Approach | Cost |
| :--- | :--- |
| Naive nested loop | ~1.6 M circuit simulations, ~30 min |
| Batched (`statevectors` + one BLAS call) | ~1.6 k simulations + 1 matmul, **< 1 s** |

This is what makes Experiments A and B (sample-efficiency and Aer noise) runnable at all
rather than taking hours per configuration.

---

## 3. Dataset audit (Phase 1)

390 XML files, 480 JPGs, all frames 560x360. Every XML has a matching image and vice versa.

### 3.1 Labels

`<calcifications>` tag values across the corpus:

| Tag | Count | Used |
| :--- | ---: | :--- |
| `microcalcifications` | 121 | label 1 |
| `microcalcification` | 48 | label 1 |
| `non` | 92 | label 0 |
| `macrocalcifications` | 33 | excluded (ambiguous) |
| `macrocalcification` | 5 | excluded (ambiguous) |
| empty | 91 | excluded |

**Final usable set: 257 cases (168 microcalcification, 89 non).**

### 3.2 Three real defects in the original script, now fixed

1. **`<mark><image>` is a 1-based index.** The original applied every polygon in an XML to
   every image of that patient, mixing annotations across views. Cases `114` and `336` index
   image 2 while owning only one image — these are now flagged and skipped, not mismatched.
2. **One `<svg>` can hold 2–5 independent regions** (two separate nodules in one frame —
   179 marks). The original unioned them into a single polygon, inflating the ROI to a
   near-image-wide hull. Masks are now built **per region**.
3. **8 XMLs have their embedded SVG JSON truncated at ~2500 characters** by the original
   dataset release (`127, 165, 166, 176, 197, 203, 205, 54`). The original swallowed these
   with a bare `except` and lost the cases silently. A regex point-extractor now recovers
   them.

Also handled: 6 `arrow` pointer regions (not nodule boundaries) and 8 degenerate polygons.

### 3.3 Every loss is accounted for

No case disappears silently. Four labelled cases yield no usable ROI and are reported with
reasons:

- `336` — image index out of range
- `362`, `378`, `384` — the only mark is an `arrow` pointer, not a freehand nodule polygon

169 labelled microcalcification cases minus `384` = **168**; 92 `non` cases minus `336`,
`362`, `378` = **89**.

### 3.4 Patient-isolated split

All images of a patient (e.g. `106_1..106_4`) land on the same side. Verified overlap: **0**.

| Split | Patches | Patients | micro | speckle |
| :--- | ---: | ---: | ---: | ---: |
| Train (80%) | 1268 | 205 | 864 | 404 |
| Test (20%) | 312 | 52 | 204 | 108 |

Total mined patch set: **1580** (1068 label 1, 512 label 0) across 257 patients, in 6.9 s.

---

## 4. Component verification

### 4.1 Candidate mining

CLAHE (clip 2.0, 8x8) -> elliptical top-hat (5x5) -> per-region mask -> local-maxima NMS
(radius 4) -> rank by Weber contrast -> top K=4 patches at native resolution.

*Why local-maxima NMS replaced threshold-and-contour:* microcalcifications occupy only a few
pixels, so their contour area is comparable to speckle grain area. Sorting contours by area
ranks them essentially at random. Peak prominence is scale-free, which is the property this
stage actually needs.

### 4.2 ROI alignment sanity check

Rendered overlays could not be inspected visually (no image input in this environment), so
alignment was verified numerically:

- 651 / 651 polygons fully in-bounds
- ROI area: min 893, median **9903**, max 61575 px (frame = 201600) — nodule-sized, not image-wide
- `|ROI mean − frame mean|` = 28.0 median; **88%** of ROIs differ by >10 intensity levels

**Conclusion: the masks land on real structures. Parsing is not the cause of §5.**

### 4.3 Quantum circuit verification

| Circuit | Qubits | Params | Depth | Size | 2q gates | Gate mix |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `ZZFeatureMap` (proposed) | 4 | 4 | 19 | 34 | 12 | 8 H, 14 P, 12 CX |
| `AngleEmbedding` (legacy baseline) | 4 | 4 | 2 | 8 | 0 | 4 RY, 4 RZ |

Both Gram matrices verified: unit diagonal, symmetric, PSD.

The legacy PennyLane kernel was reimplemented in Qiskit rather than installing PennyLane.
**A genuine bug was caught and fixed here:** `H` + `P(2x)` is *not* angle-encoding — the
correct form is `RY(x)` then `RZ(x)`. After the fix, the closed-form product-cosine kernel
matches the explicit circuit to **4.441e-16** (machine precision), so the baseline is a
faithful port, not a lookalike.

The PennyLane baseline is scientifically interesting for the report: its zero-weight
entangler is the identity, so the circuit collapses to `prod_i cos^2((x1_i − x2_i)/2)` — a
genuinely different inductive bias from the ZZ map's polynomial structure.

---

## 5. THE CENTRAL FINDING: the task is unlearnable as specified

`python main.py train` completed successfully and reported:

| Metric | Value |
| :--- | --- |
| Accuracy | 0.5064 |
| Balanced accuracy | 0.4788 |
| **ROC AUC** | **0.4568** |
| Confusion matrix (rows = truth, cols = pred) | `[[42, 66], [88, 116]]` |
| PCA variance retained (4 comps) | 0.8705 |

Chance is 0.50. PCA is retaining 87% of variance, so the compression is not destroying the
signal. The features simply do not carry the label.

### 5.1 It is not the quantum kernel

Four independent model families, identical split, identical features:

| Model | Test AUC |
| :--- | ---: |
| Quantum ZZ kernel | **0.457** |
| RBF-SVC | 0.497 |
| RandomForest (50) | 0.489 |
| MLP (16, 8, 4) | 0.504 |

Four different model families cannot all be broken the same way by a weak quantum kernel.
The problem is upstream, in the features.

> **Trap worth noting for the report:** RBF-SVC and RandomForest both "scored" **0.65
> accuracy**. The majority class is **0.654**. A 0.65 accuracy therefore means *zero*
> signal — the model is predicting the majority class. **AUC exposes this; accuracy hides
> it.** Any claim of "65% accuracy" from this run is meaningless.

### 5.2 Root cause: DDTI has no spot-level ground truth

A DDTI XML asserts `<calcifications>microcalcifications</calcifications>`, which means
**"this nodule contains microcalcifications somewhere."** It is a property of the nodule.
There is no annotation anywhere in the file indicating *where* they are or *how many*.

The weak-supervision design (nodule label inherited by each candidate) therefore asserts:

> "These 4 bright specks, which my classical miner selected as the top-4 by contrast,
> are calcifications."

**There is no basis for that claim.** A thyroid nodule contains many bright punctate echoes,
and microcalcifications are a *minority* of them. The top-4 brightest specks in a
microcalcification nodule are, with high probability, ordinary speckle — the same kind of
speckle labelled 0 in benign nodules.

The model is therefore being asked:

> "Here is a 4x4 crop of a bright speck, mined from either a microcalcification nodule or a
> benign nodule. Which was it?"

The information that answers this question lives in the **nodule's overall texture and
echogenicity** — microcalcification nodules in DDTI are predominantly hypoechogenic. But the
feature being extracted is a 4x4 crop that was deliberately cropped to *exclude* the
surrounding context.

**The signal is present in the image; it is being read at the wrong spatial scale.**

### 5.3 The below-chance AUC confirms the mechanism

A single scalar — Weber contrast `C = (I_peak − I_bg)/(I_bg + eps)`, the miner's own ranking
score — scored AUC **0.432**. Reliably inverted, and consistently so:

| Feature | Test AUC |
| :--- | ---: |
| Raw CLAHE intensity at peak | 0.499 |
| Despeckle residual (peak − median-5x5) | 0.464 |
| \|respeckle residual\| | 0.464 |
| Weber contrast (1-D) | 0.432 |

Despeckle survival was tested on the hypothesis that a real calcification is a *coherent*
structure surviving median filtering while speckle is removed. Median Weber contrast was
1.86 (micro) vs 1.79 (speckle) — near-identical.

Widening the candidate pool to K=24 per ROI (to rule out "K=4 misses the rare calcification")
made things **worse**:

| Aggregate over 24 candidates | Test AUC |
| :--- | ---: |
| Max despeckle survival | 0.369 |
| Mean survival | 0.370 |
| p90 survival | 0.462 |
| Count(survival > 6) | 0.481 |

A consistent AUC *below* 0.5 is not noise; it is a mechanism. Microcalcification nodules are
hypoechoic — dark background with strong speckle contrast. A bright speck *relative to its
own dark background* therefore scores a **higher** Weber contrast in a microcalcification
nodule than in a benign one. The feature meant to detect calcifications instead measures
overall speckle prominence, which runs the wrong way.

### 5.4 What was ruled out

- **ROI misalignment** — see §4.2. Verified correct.
- **XML parsing defects** — see §3.2. All three fixed and independently verifiable.
- **Data leakage inflating results** — the opposite; patient overlap is 0.
- **Too few candidates** — tested at K=24, AUC degraded.
- **PCA over-compression** — 87% variance retained.
- **A bug in my own quantum code** — caught (angle-encoding), fixed, re-verified at 1e-16.

**The pipeline is correct. The task is ill-posed.**

---

## 6. The upside: the signal is real at ROI scale

Same images, same miner, same patient-isolated split — only the spatial scale of the feature
changed. 16 hand-crafted ROI texture descriptors (intensity percentiles, top-hat percentiles,
std, mean, bright-pixel fraction), 267 microcalcification / 128 benign ROIs:

| ROI-level model | Test AUC |
| :--- | ---: |
| RandomForest (100) | **0.630** |
| RBF-SVC | 0.558 |
| LogisticRegression | 0.527 |

This is the clean confirmation of the diagnosis. The signal exists in DDTI; it is simply
unreachable at 4x4 patch granularity.

**Honest caveat:** 0.630 is weak, and only 16 hand-crafted features were tested. A richer ROI
descriptor might do better — the ceiling has not been established.

---

## 7. Impact on the plan, and the decision required

### 7.1 What remains valid

The patient-isolated split, the candidate miner and its recall study, the quantum engine and
both circuit analyses, the overlay renderer, and the TI-RADS re-scoring (which now computes
a genuine ACR base score from composition/echogenicity/margins, with the XML's recorded
category available for validation) all stand. `plan.md` Modules 1, 2, 4 and 6 are unaffected.

### 7.2 What is blocked

`plan.md` Module 5 (Experiments A, B, C) is **blocked**, and the blocker is conceptual, not
effort:

- **Experiment A (sample efficiency / "quantum advantage")** would compare four models that
  are all at chance. The plan's stated *expected result* — that the quantum kernel degrades
  more gracefully than classical models below N=40 — is **unfalsifiable**, because there is no
  signal to preserve. Running it would produce a chart of noise.
- **Experiment B (noise robustness)** is unaffected in principle, but measures the
  preservation of a signal that does not exist.
- **Experiment C (candidate mining recall)** remains valid and is already unblocked.

### 7.3 The decision

**What the classifier is asked to predict must change.** Three options:

1. **Reframe to nodule-level as the primary contribution**, keeping the patch-level failure
   as a documented, controlled ablation. The architecture is unchanged: the miner still finds
   candidates for the overlay, the ZZ feature map still classifies, TI-RADS is still
   re-scored — only the granularity of the classified object moves from a 4x4 patch to the
   ROI. Experiment A then becomes a genuinely open question rather than a predetermined
   result. *This is the recommended option, and the most defensible framing: it converts
   "we could not make it work" into a demonstrated, evidence-backed finding.*
2. **Stay faithful to `plan.md`** and report the negative result as the contribution. A
   rigorous, well-instrumented demonstration that the formulation fails on this data — but
   nothing detects anything.
3. **Source spot-level annotations** from another thyroid ultrasound dataset with pixel-level
   calcification masks. Scientifically the correct fix, but needs new data and breaks the
   8-week timeline.

**Status: awaiting a decision before Phase 5.**

---

## 8. Repository state

```
hqvision/
├── plan.md                 # original specification
├── status.md               # this document
├── main.py                 # CLI: integrity | train | predict | circuit
├── requirements.txt        # pinned to the qip_venv versions
├── .gitignore              # outputs/ ignored, as requested
├── src/
│   ├── config.py
│   ├── dataset.py          # XML repair, 1-based index, group split, audit
│   ├── mining.py           # CLAHE, top-hat, per-region mask, NMS, Weber
│   ├── quantum_engine.py   # ZZ map, batched Gram, PCA+SVC, overlap baseline
│   ├── pipeline.py         # inference, overlay, TI-RADS re-scoring
│   └── cache.py
├── outputs/                # git-ignored: figures, overlays, models, caches
└── train_hybrid_model.py   # legacy PennyLane script, kept as reference
```

**Working directory for all commands:** `D:\saksham\college stuff\Munjal\Seventh_Sem\Classroom\QIP\hqvision`
**Interpreter:** `..\qip_venv\Scripts\python.exe`

`train_hybrid_model.py` is superseded and is not imported by the new pipeline. It is retained
because its kernel is reimplemented in `src/quantum_engine.py` as a comparison baseline, and
because the original PennyLane approach is worth citing in the literature review.
