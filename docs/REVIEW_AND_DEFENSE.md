# Quantum-Assisted Thyroid Microcalcification Detection — Mid-Term Review

**Status:** Code is written (`src/`, `main.py`); dataset is in `data/` (390 XMLs, 480 frames).
Experiments (A/B/C) are designed but **not yet run — there are no results to report yet**.

---

## 1. Problem and objective

Thyroid ultrasound scans are full of tiny bright dots. Some are **microcalcifications** (<1 mm,
potentially suspicious for cancer, +TI-RADS points), most are **acoustic speckle** (machine noise).
On screen they look alike: a calcification is ~2–5 px wide, a speckle grain ~1–3 px.

**Question:** can a small entangled quantum circuit act as a *verifier* — given a few classical
candidates, decide real vs speckle — and does it behave differently from classical classifiers when
training data is scarce?

**Pipeline:** classical CV finds candidate spots → quantum kernel classifies each 4×4 patch →
overlay (red/green circles) + TI-RADS re-score.

**Why quantum:** the input per decision is only 16 pixels compressed to 4 numbers — small enough to
encode honestly on a NISQ device — and the small-data regime is where a constrained quantum kernel
might plausibly help. **Not** for speed.

---

## 2. Quantum formulation 🎯

### Representation: patches, not whole images

We do **not** use FRQI/NEQR (whole image in a quantum state). Loading an N×N image costs O(N²)
gates without QRAM, and reading it back costs O(N²/ε²) shots — the "I/O wall" that erases any
speed-up. Instead:

```
image → classical miner → K candidate spots → 4×4 patch (16 px)
      → PCA 16→4 (fitted on train only) → scale to [0,π] → 4-qubit state |ψ(x)⟩
```

One patch = one 4-qubit state. The image itself never enters the quantum side.

### Encoding chain

| Step | Why |
|---|---|
| Crop 4×4 native pixels | The spot is only 2–5 px; upsampling invents data |
| PCA 16 → 4 | One qubit per feature; also decorrelates neighbouring pixels |
| Min–max to [0, π] | Rotations repeat every 2π; one non-repeating arc avoids wrapping |
| Angle encoding | Cheapest, most precise operation on real hardware |

### The circuit: `ZZFeatureMap` — 4 qubits, reps=2, linear entanglement

Three moves, repeated twice:

1. **H layer** — put qubits in superposition (somewhere to write the data).
2. **Phase layer** — rotations whose angles *are* your four numbers.
3. **Entangling layer (controlled-phase / ZZ)** — couples features pairwise, so the state encodes
   *relationships* ("bright here **and** dark there"), not just four independent values. This is the
   part a classical separable encoding cannot do for free.

`reps=2` = data re-uploading: the same numbers are written twice with more chance to interfere →
more expressive map, no extra qubits. `entanglement="linear"` (only 1–2, 2–3, 3–4) matches real
chip connectivity and keeps depth low: **depth 19, size 34, 12 two-qubit gates** for our config
(run `python main.py circuit` to print the table — quote the tool, not memory).

### The algorithm: quantum kernel SVM

The circuit is **fixed** — nothing in it is trained. Learning is classical:
`SVC(kernel="precomputed")` on the Gram matrix

> K(a,b) = |⟨ψ(a)|ψ(b)⟩|², symmetrised (K = Kᵀ), diagonal forced to 1.

- Swapping the RBF kernel for a quantum one changes exactly one ingredient, so the comparison with
  classical SVMs is clean and fair.
- **Speed trick:** instead of |A|×|B| pair-wise circuit runs (~1.6 M for our dataset), simulate each
  sample **once** and get the whole matrix from one matrix product (`psi_a @ psi_b.conj().T`, then
  abs²). Quadratic Python work → one BLAS call.

### Second quantum circuit (baseline)

The legacy PennyLane script used `AngleEmbedding` + `StronglyEntanglingLayers` with **all-zero**
weights. Zero weights = identity, so it collapses to encoding the *difference* of the two inputs,
and `probs()[0]` becomes a closed-form **product-cosine kernel**, ∏ᵢ cos²((aᵢ−bᵢ)/2). We keep it as
`overlap_gram()` (computed in closed form, unit-tested against the explicit circuit) because it is a
second quantum encoding with a *different* inductive bias: comparing ZZ vs overlap tells you
whether entanglement between features actually earns its keep.

### Why this connects to the task

The classifier only sees similarity scores. A microcalcification patch is a bright compact core with
dark surround; speckle is flatter and noisier. The ZZ kernel adds pairwise feature interactions on
top of plain distance — image terms, that is sensitivity to local texture co-occurrence. **Whether
that helps is an empirical question**, tested by Experiment A. Do not assert it in advance.

---

## 3. Methodology and implementation

- **Stack:** Qiskit 2.5.1, qiskit-aer 0.17.2, scikit-learn 1.9.1, OpenCV, Python 3.14.
  `qiskit-machine-learning` is deliberately absent (no cp314 wheel) — the kernel is built on Qiskit
  primitives + a precomputed-kernel SVC, which is exactly what `QSVC` does underneath.
- **Classical miner** (`src/mining.py`): CLAHE → 5×5 elliptical top-hat → ROI mask from the XML
  polygon → local-maximum NMS (radius 4 px) → rank by local Weber contrast → keep top K=4 → crop
  4×4. Recall-first: a spot dropped here is unrecoverable, precision is the quantum model's job.
- **Data** (`src/dataset.py`): `microcalcifications → 1`, `non → 0`, macro/empty excluded;
  **patient-level 80/20 split** (all frames of a patient on one side, overlap asserted to be 0).
  Fixes three parser bugs: 1-based image index applied to all frames, multiple SVG regions unioned
  into one hull, 8 truncated XMLs silently dropped (now regex-repaired).
- **Classifier** (`src/quantum_engine.py`): PCA + scaler fitted on train only → Gram matrix →
  balanced SVC. Reports accuracy, balanced accuracy, ROC AUC, confusion matrix, retained PCA
  variance, circuit stats.
- **Simulation:** exact statevectors for training (no shot noise); `noisy_fidelity_gram()` for
  Experiment B — transpiled to {rz, sx, x, cx}, depolarising error per gate, 1024 shots.
- **Missing:** `experiments/` scripts and all outputs/figures — these are the next phase.

---

## 4. Experimental evaluation (planned)

- **Dataset:** DDTI, patient-level 80/20 split, seed 42; unit = mined 4×4 patches (weak labels).
- **Metrics:** balanced accuracy (headline — classes are imbalanced), ROC AUC, confusion matrix,
  per-class F1; mining recall; TI-RADS agreement with the radiologist.
- **Baselines:** RBF-SVM (like-for-like), Random Forest (50), MLP (3 layers), and the quantum
  overlap kernel — all on identical splits and features.

| Exp | Question | Method |
|---|---|---|
| A — sample efficiency | Does the quantum kernel hold up with little data? | N ∈ {15,30,60,120}, 5 seeds, accuracy vs N |
| B — noise & shots | Would it survive real hardware? | Aer depolarising 0.1/0.5/1.0%, 1024 shots |
| C — mining recall | Did the classical sieve drop a real calcification? | Top-K hit rate on known micro nodules (target ≥95%) |

---

## 5. Results and comparison

**No results yet — no experiments have been run.** When they are:

- Exp A curves decide the story: QSVC ahead at N=15–30 → supports the small-data hypothesis;
  classical ahead everywhere → honest conclusion is "no quantum advantage demonstrated here", which
  is still a valid result.
- Always show classical next to quantum, same splits, mean ± spread over 5 seeds (never a single
  best run). Report recall for class 1 — a missed calcification matters more than a false alarm.

---

## 6. Quantum advantage and analysis 🎯

**Possible benefit:** a small, rigid hypothesis space suits tiny datasets (a 3-layer MLP overfits at
15 samples; a fixed 4-qubit map cannot), and the entanglers inject pairwise feature interactions
with zero trainable parameters. That is what Experiment A measures.

**Where it does not help — say this openly:**

1. **No speed-up.** Statevectors are simulated on a CPU; our quantum stage is *slower* than the
   classical baselines. The claim is a different similarity measure, not acceleration.
2. **I/O wall:** whole-image quantum encoding is impractical (O(N²) prepare, O(N²/ε²) read out) —
   that is why we only encode 16-pixel patches.
3. **Kernel cost is quadratic in samples** (N² Gram entries), and each entry costs a state
   simulation — the gap widens as data grows.
4. **Honesty about size:** 4 qubits = at most a 16-dimensional complex feature space. With 4 qubits
   there is no exponential blow-up to boast about — the pitch is *different geometry*, not *bigger*.
5. **PCA is classical** and does heavy lifting; any advantage belongs to the hybrid pipeline, never
   to the quantum part alone.

> One-line defence: *"Our quantum stage is a feature map, not an accelerator — a possibly
> well-suited similarity measure for very small medical datasets, at the cost of slower training.
> Whether that trade pays off is exactly what our experiments will measure."*

---

## 7. Limitations and failure cases 🎯

**Data / labels**
- **Weak labels:** DDTI labels the *nodule*, not the individual dot, so every candidate inside a
  microcalcification nodule is tagged positive — including speckle grains there. Some training
  labels are wrong by construction; this caps achievable accuracy.
- Experiment C's recall is measured against a *reference detector*, not radiology ground truth.
- Macro / empty / missing calcification tags are excluded — the model is strictly binary.
- Dataset defects (truncated XMLs, arrow pointers, out-of-range indices) are counted in the audit
  report rather than hidden.

**Classical stage**
- Mining recall is a hard ceiling: spot outside the polygon, ranked below top-K, or suppressed by a
  neighbour within the 4-px NMS radius = lost forever.
- CLAHE/kernel/NMS settings are tuned for this resolution; a different scanner would need re-tuning.
- TI-RADS scoring is simplified: we award +2 per plan.md; actual ACR gives 3 for punctate strong
  foci (documented in `config.py`, single place to change).

**Quantum encoding**
- Min–max is fitted on the train range; test values outside it are **clipped** to 0 or π — unusual
  patients get pushed onto the same angle as training extremes.
- Low retained PCA variance means information is lost *before* the circuit sees it — always quote
  retained variance next to accuracy.
- Linear entanglement skips the 1–4 coupling; whether another topology helps is untested.
- Phase-only differences are invisible to the kernel (correct quantum mechanics, but some info is
  simply inaccessible).

**Simulation / noise**
- Training uses exact statevectors: no shot noise, no gate error — optimistic by construction.
- The noise model is depolarising only: no readout error, no coherent errors, no crosstalk, no
  drift. Treat Experiment B as an optimistic bound.
- *Fixed this sprint:* `noisy_fidelity_gram()` used to bind only the second sample and measure
  P(|0000⟩) of a single state — the wrong quantity, and its rows did not depend on `a` at all. It now
  builds the proper interference circuit U(a) then U†(b), which reproduces the exact kernel at 0%
  noise (verified: max error 0.007 at 8192 shots) and decays as expected under noise.
- No real-hardware run yet; qiskit-ibm-runtime is installed but unused.

**Statistical / scalability**
- Correct patient-level splitting leaves few test patients → wide error bars; report across the 5
  seeds.
- Gram memory is O(N²); statevector memory is 2ⁿ — beyond ~30 qubits classical simulation is out of
  reach. Adding qubits (skip PCA, use all 16 pixels) would multiply cost.
- One dataset, one set of scanners: no generalisation claims beyond DDTI.

**Next steps:** run A–C → fix/extend noise model (add readout error) → robustness check against
weak labels → optional small IBM hardware job.

---

## 8. Individual understanding — viva prep

Questions go to **any** member. Core ones everyone answers cold:

1. Problem in one sentence? → Real microcalcification vs speckle in thyroid ultrasound.
2. Why quantum, not a CNN? → 16-pixel input is honestly encodable; testing behaviour in the
   small-data regime. Not for speed.
3. What goes into the circuit? → Four angles in [0,π] from a 4×4 patch. Never the whole image.
4. Why not FRQI/NEQR? → Image-state loading/reading costs O(N²) gates and O(N²/ε²) shots.
5. Walk through the circuit. → H (spread) → phase (write numbers) → ZZ entangle (feature pairs),
   ×2 reps; linear topology for low depth on real chips.
6. Do you train the circuit? → No — fixed feature map; a classical SVM learns on the Gram matrix.
7. Define the kernel. → K = |⟨ψ(a)|ψ(b)⟩|², symmetric, unit diagonal.
8. Why that symmetry/diagonal trick? → Correctness for the SVM + ~50% fewer quantum evaluations.
9. How is the Gram matrix made fast? → One statevector per sample, one matrix product.
10. What does entanglement buy? → Pairwise feature interactions, not just four separate values.
11. Isn't 4 qubits only a 16-dim feature space? → Yes — we claim different geometry, not
    exponential size.
12. How do you stop patient leakage? → Patient-level split; overlap asserted to be zero.
13. Why balanced accuracy? → Class imbalance.
14. What's unfinished? → Experiments and results; code and evaluation design exist.
15. What was wrong with the noise code and why does it matter? → It measured P(|0000⟩) of one state
    instead of the U†(b)U(a) interference — wrong quantity, rows independent of `a`; now fixed.

| ✅ Say this | ❌ Not this |
|---|---|
| "Hybrid pipeline designed and implemented" | "It's faster than classical" |
| "Patches encoded, not whole images" | "Images are stored in quantum states" |
| "Exact simulation; noise modelled in Aer" | "We ran on real quantum hardware" |
| "Protocol designed, results pending" | "The quantum model won" — no numbers exist |
| "Weak labels are a known, quantified limit" | "Labels are pixel-accurate" |
