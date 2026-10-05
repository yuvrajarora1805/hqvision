# Quantum-Assisted Thyroid Nodule Classifier — Mid-Term Review

**Status:** DDTI → TN5000 switch done. Phases 1–4 (ingest, mining, quantum kernel,
overlay) are ported and verified end-to-end (`main.py integrity/mine/train/predict/circuit`).
Experiments A–C are designed but **not yet run — the numbers below are smoke tests,
not results**.

---

## 1. Problem and objective

Thyroid ultrasound shows a nodule; the clinical question is **malignant or benign**
(confirmed in TN5000 by FNA biopsy). The image inside the nodule is full of tiny
bright dots and texture patterns; a human reading them is slow and subjective.

**Question:** can a small entangled quantum circuit act as a *verifier* — given a few
classically-mined candidate spots, decide whether each spot looks malignant or
benign — and does it behave differently from classical classifiers when training
data is scarce?

**Pipeline:** classical CV mines candidate spots inside the nodule box → quantum
kernel classifies each 4×4 patch → majority vote over the patches = verdict for the
nodule → overlay (red/green circles + verdict).

**Why quantum:** the input per decision is only 16 pixels compressed to 4 numbers —
small enough to encode honestly on a NISQ device — and the small-data regime is
where a constrained quantum kernel might plausibly help. **Not** for speed.

### Dataset: why we moved from DDTI to TN5000

| | DDTI (old) | TN5000 (now) |
|---|---|---|
| Size | ~480 frames | 5,000 frames |
| Label | text tags (`micro`/`non`) | biopsy-confirmed `1`=malignant (3,574) / `0`=benign (1,426) |
| Split | we rolled our own | official trainval 4,000 / test 1,000, leakage-safe |
| ROI | polygons | one nodule bounding box per image |

DDTI was too small to train or evaluate honestly. Same pipeline idea, new data.

---

## 2. Quantum formulation 🎯

### Representation: patches, not whole images

We do **not** use FRQI/NEQR (whole image in a quantum state). Loading an N×N image
costs O(N²) gates without QRAM, and reading it back costs O(N²/ε²) shots — the
"I/O wall" that erases any speed-up. Instead:

```
image → classical miner → K candidate spots → 4×4 patch (16 px)
      → PCA 16→4 (fitted on train only) → scale to [0,π] → 4-qubit state |ψ(x)⟩
```

One patch = one 4-qubit state. The image itself never enters the quantum side.

### Encoding chain

| Step | Why |
|---|---|
| Crop 4×4 native pixels | the spot is only a few px; upsampling invents data |
| PCA 16 → 4 | one qubit per feature; also decorrelates neighbouring pixels |
| Min–max to [0, π] | rotations repeat every 2π; one non-repeating arc avoids wrapping |
| Angle encoding | cheapest, most precise operation on real hardware |

### The circuit: `ZZFeatureMap` — 4 qubits, reps=2, linear entanglement

Three moves, repeated twice:

1. **H layer** — put qubits in superposition (somewhere to write the data).
2. **Phase layer** — rotations whose angles *are* your four numbers.
3. **Entangling layer (controlled-phase / ZZ)** — couples features pairwise, so the
   state encodes *relationships* ("bright here **and** dark there"), not just four
   independent values. A classical separable encoding does not get this for free.

`reps=2` = data re-uploading: numbers written twice, more interference → more
expressive map, no extra qubits. `entanglement="linear"` (1–2, 2–3, 3–4) matches
real chip connectivity: **depth 19, size 34, 12 two-qubit gates** (run
`python main.py circuit` — quote the tool, not memory).

### The algorithm: quantum kernel SVM

The circuit is **fixed** — nothing in it is trained. Learning is classical:
`SVC(kernel="precomputed")` on the Gram matrix

> K(a,b) = |⟨ψ(a)|ψ(b)⟩|², symmetrised (K = Kᵀ), diagonal forced to 1.

- Swapping the RBF kernel for a quantum one changes exactly one ingredient, so the
  comparison with classical SVMs is clean and fair.
- **Speed trick:** instead of |A|×|B| pairwise circuit runs, simulate each sample
  **once** and get the whole matrix from one matmul (`psi_a @ psi_b.conj().T`, then
  abs²). Quadratic Python work → one BLAS call.

### Second quantum circuit (baseline)

The legacy PennyLane script used `AngleEmbedding` + `StronglyEntanglingLayers` with
**all-zero** weights. Zero weights = identity, so it collapses to encoding the
*difference* of the two inputs and `probs()[0]` becomes a closed-form
**product-cosine kernel**, ∏ᵢ cos²((aᵢ−bᵢ)/2). We keep it as `overlap_gram()` —
a second quantum encoding with a *different* inductive bias: comparing ZZ vs
overlap tells you whether entanglement between features earns its keep.

### Why this connects to the task

The classifier only sees similarity scores. Spots from malignant nodules tend to
carry different local texture/brightness patterns than spots from benign nodules.
The ZZ kernel adds pairwise feature interactions on top of plain distance — image
terms, sensitivity to local texture co-occurrence. **Whether that helps is an
empirical question**, tested by Experiment A. Do not assert it in advance.

---

## 3. Methodology and implementation

- **Stack:** Qiskit 2.x, qiskit-aer, scikit-learn, OpenCV, Python 3.14 (venv
  `qip_venv`). `qiskit-machine-learning` is deliberately absent (no cp314 wheel) —
  the kernel is built on Qiskit primitives + a precomputed-kernel SVC, which is
  what `QSVC` does underneath.
- **Data** (`src/dataset.py`): VOC parser — bbox + `<name>` label; audit counts
  every defect (5,000/5,000 usable in our run, class split 3,574 : 1,426).
  Official `trainval`/`test` split, overlap asserted to be 0.
- **Classical miner** (`src/mining.py`): CLAHE → 5×5 elliptical top-hat → ROI mask
  from the bbox → contour peaks → local Weber contrast → NMS (radius 4 px) → top
  K=5 → crop 4×4. Recall-first: a spot dropped here is unrecoverable; precision is
  the quantum model's job. Full run: 19,973 train / 4,998 test patches, all 5,000
  nodules produced candidates.
- **Classifier** (`src/quantum_engine.py`): PCA + scaler fitted on train only →
  Gram matrix → balanced SVC. Training patches balanced-capped at 2,000 (seed 42)
  because the Gram matrix is O(N²). Metrics at **patch** level (weak labels) and
  **nodule** level (majority vote) — the nodule level is the honest clinical unit.
- **Overlay** (`src/pipeline.py`): red = malignant patch, green = benign patch,
  yellow = ROI box, legend with nodule verdict + confidence.
- **Simulation:** exact statevectors for training (no shot noise);
  `noisy_fidelity_gram()` for Experiment B — transpiled to {rz, sx, x, cx},
  depolarising error per gate, 1024 shots.
- **Missing:** `experiments/` scripts and `outputs/figures` — next phase.

---

## 4. Experimental evaluation (planned)

- **Dataset:** TN5000, official trainval/test split, seed 42; unit = mined 4×4
  patches (weak labels), aggregated to nodules for headline metrics.
- **Metrics:** balanced accuracy (headline — 3,574 : 1,426 imbalance), ROC AUC,
  confusion matrix, per-class F1; nodule-level malignant recall; mining coverage.
- **Baselines:** RBF-SVM (like-for-like), Random Forest (50), MLP (3 layers), and
  the quantum overlap kernel — all on identical splits and features.

| Exp | Question | Method |
|---|---|---|
| A — sample efficiency | Does the quantum kernel hold up with little data? | N ∈ {15,30,60,120}, 5 seeds, accuracy vs N |
| B — noise & shots | Would it survive real hardware? | Aer depolarising 0.1/0.5/1.0%, 1024 shots |
| C — mining ablation | Did the classical sieve drop malignant nodules? | K ∈ {1,3,5,8} → nodule-level malignant recall |

---

## 5. Results and comparison

**No experiment results yet.** What exists is an end-to-end wiring check (seed 42,
full test set, single run — *not* a tuned or claimed result):

| Kernel | patch acc / bal / AUC | nodule acc / bal / AUC |
|---|---|---|
| ZZ feature map | 0.647 / 0.628 / 0.654 | 0.670 / 0.647 / 0.700 |
| Overlap baseline | 0.679 / 0.610 / 0.657 | 0.713 / 0.632 / 0.689 |

Reading these honestly: barely above chance on a weakly-labelled patch task —
exactly why Experiments A–C (and the classical baselines) are the actual story.
When they run:

- Exp A decides the narrative: quantum ahead at N=15–30 → supports the small-data
  hypothesis; classical ahead everywhere → honest conclusion is "no quantum
  advantage demonstrated here", which is still a valid result.
- Always show classical next to quantum, same splits, mean ± spread over 5 seeds
  (never a single best run). Report malignant-class recall — a missed cancer
  matters more than a false alarm.

---

## 6. Quantum advantage and analysis 🎯

**Possible benefit:** a small, rigid hypothesis space suits tiny datasets (a 3-layer
MLP overfits at 15 samples; a fixed 4-qubit map cannot), and the entanglers inject
pairwise feature interactions with zero trainable parameters. That is what
Experiment A measures.

**Where it does not help — say this openly:**

1. **No speed-up.** Statevectors are simulated on a CPU; our quantum stage is
   *slower* than the classical baselines. The claim is a different similarity
   measure, not acceleration.
2. **I/O wall:** whole-image quantum encoding is impractical (O(N²) prepare,
   O(N²/ε²) read out) — that is why we only encode 16-pixel patches.
3. **Kernel cost is quadratic in samples** (N² Gram entries) — why we cap training
   at 2,000 patches; the gap widens as data grows.
4. **Honesty about size:** 4 qubits = at most a 16-dimensional complex feature
   space. No exponential blow-up to boast about — the pitch is *different
   geometry*, not *bigger*.
5. **PCA is classical** and does heavy lifting; any advantage belongs to the hybrid
   pipeline, never to the quantum part alone.

> One-line defence: *"Our quantum stage is a feature map, not an accelerator — a
> possibly well-suited similarity measure for very small medical datasets, at the
> cost of slower training. Whether that trade pays off is exactly what our
> experiments will measure."*

---

## 7. Limitations and failure cases 🎯

**Data / labels**
- **Weak labels (the big one):** TN5000 labels the *nodule* by biopsy, not each
  pixel. Every patch inside a malignant nodule is tagged positive — including
  ordinary tissue between the suspicious foci. Some training labels are wrong by
  construction; this caps achievable accuracy and is why smoke numbers sit near
  chance.
- Class imbalance 3,574 : 1,426 → balanced accuracy is the headline, not accuracy.
- No patient IDs in TN5000 → we cannot re-check leakage ourselves; we rely on the
  authors' published split (one representative image per patient perspective).
- No TI-RADS / composition / echogenicity annotations → the DDTI-era TI-RADS
  re-scoring was **removed**, not approximated. The deliverable is a malignant/
  benign verdict directly comparable to the biopsy label.

**Classical stage**
- Mining recall is a hard ceiling: spot outside the box, ranked below top-K, or
  suppressed within the 4-px NMS radius = lost forever (Experiment C measures this).
- CLAHE/kernel/NMS settings are tuned for this resolution; a different scanner
  would need re-tuning.

**Quantum encoding**
- Min–max is fitted on the train range; test values outside it are **clipped** to
  0 or π — unusual patients get pushed onto the same angle as training extremes.
- Low retained PCA variance means information is lost *before* the circuit sees
  it — always quote retained variance next to accuracy.
- Linear entanglement skips the 1–4 coupling; whether another topology helps is
  untested.
- Phase-only differences are invisible to the kernel (correct quantum mechanics,
  but some info is simply inaccessible).

**Simulation / noise**
- Training uses exact statevectors: no shot noise, no gate error — optimistic by
  construction.
- The noise model is depolarising only: no readout error, no coherent errors, no
  crosstalk, no drift. Treat Experiment B as an optimistic bound.
- *Fixed earlier:* `noisy_fidelity_gram()` used to bind only the second sample and
  measure P(|0000⟩) of a single state — wrong quantity, rows independent of `a`.
  It now builds the proper interference circuit U(a) then U†(b) (verified: max
  error 0.007 vs exact kernel at 8192 shots).
- No real-hardware run yet.

**Statistical / scalability**
- Report across the 5 seeds, never one run.
- Gram memory is O(N²); statevector memory is 2ⁿ — beyond ~30 qubits classical
  simulation is out of reach. Adding qubits (skip PCA, use all 16 pixels) would
  multiply cost.
- One dataset, one set of scanners: no generalisation claims beyond TN5000.

**Next steps:** run A–C → classical baseline table → fix/extend noise model (add
readout error) → robustness check against weak labels → optional small IBM job.

---

## 8. Individual understanding — viva prep

Questions go to **any** member. Core ones everyone answers cold:

1. Problem in one sentence? → Classify thyroid nodules as malignant vs benign from
   ultrasound, using quantum-kernel classification of mined micro-patches.
2. Why this dataset? → TN5000: 5,000 images, biopsy labels, official leakage-safe
   split. DDTI (~480 frames, text tags) was too small to train on.
3. Why quantum, not a CNN? → 16-pixel input is honestly encodable; testing
   behaviour in the small-data regime. Not for speed.
4. What goes into the circuit? → Four angles in [0,π] from a 4×4 patch. Never the
   whole image.
5. Why not FRQI/NEQR? → Image-state loading/reading costs O(N²) gates and
   O(N²/ε²) shots.
6. Walk through the circuit. → H (spread) → phase (write numbers) → ZZ entangle
   (feature pairs), ×2 reps; linear topology for low depth on real chips.
7. Do you train the circuit? → No — fixed feature map; a classical SVM learns on
   the Gram matrix.
8. Define the kernel. → K = |⟨ψ(a)|ψ(b)⟩|², symmetric, unit diagonal.
9. Why that symmetry/diagonal trick? → Correctness for the SVM + fewer quantum
   evaluations.
10. How is the Gram matrix made fast? → One statevector per sample, one matmul.
11. What does entanglement buy? → Pairwise feature interactions, not just four
    separate values.
12. Isn't 4 qubits only a 16-dim feature space? → Yes — we claim different
    geometry, not exponential size.
13. How do you stop leakage? → Use the dataset's official split (built by the
    authors with one image per patient perspective); overlap asserted to be 0.
14. What does the label mean? → Biopsy (FNA) result of the nodule: 1 = malignant,
    0 = benign. It is nodule-level, so patch labels are weak.
15. Why balanced accuracy? → 3,574 malignant vs 1,426 benign is imbalanced.
16. What's unfinished? → Experiments and results; code and evaluation design exist.
17. What was wrong with the noise code and why does it matter? → It measured
    P(|0000⟩) of one state instead of the U(a)·U†(b) interference — wrong
    quantity, rows independent of `a`; now fixed.

| ✅ Say this | ❌ Not this |
|---|---|
| "Hybrid pipeline designed, ported and verified end-to-end" | "It's faster than classical" |
| "Patches encoded, not whole images" | "Images are stored in quantum states" |
| "Biopsy labels at nodule level; patches are weak labels" | "Labels are pixel-accurate" |
| "Official leakage-safe split" | "We proved there is no leakage ourselves" |
| "Exact simulation; noise modelled in Aer" | "We ran on real quantum hardware" |
| "Smoke wiring check reported; experiments pending" | "The quantum model won" — no experiment exists |
