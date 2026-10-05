# Quantum-Assisted Thyroid Nodule Classifier — Mid-Term Review

**One-line summary:** a normal computer-vision program finds the interesting spots inside a
thyroid nodule, and a small quantum circuit (4 qubits) is used as a "similarity judge" that
decides whether a spot looks like it came from a malignant or a benign nodule.

**Status:** the code is written and runs end to end. The three planned experiments
(sample-efficiency, noise, ablation) have **not** been run yet — the numbers in this document
come from one end-to-end test run, not from final experiments.

---

## 1. What problem are we solving?

A thyroid ultrasound image shows a lump (the *nodule*). The doctor's job is to answer one
question: **cancer or not?** In our dataset (TN5000) that answer is known for every image
because it was confirmed by a needle biopsy:

- **3,574 images** from malignant nodules (cancer)
- **1,426 images** from benign nodules (not cancer)

The difficulty for a computer is that a nodule is full of tiny bright specks. Many of those
specks are just **noise from the ultrasound machine** (speckle) and some are **real
suspicious tissue**. They look extremely similar in a still image. A radiologist judges them
using context — the shape, the texture around the speck, the shadow underneath it.

**Our approach in plain terms:**

1. A classical (normal) computer program looks inside the marked nodule box and picks the **5
   most interesting specks**.
2. Each speck becomes a small feature list (its pixels + a description of its surroundings).
3. A **4-qubit quantum circuit** turns each feature list into a quantum state, and we measure
   **how similar two states are**. That similarity is the input to a normal support vector
   machine (SVM), which produces the final malignant / benign call.
4. The 5 specks in a nodule vote; the majority wins. The doctor gets an annotated image with
   red circles (looks malignant) and green circles (looks benign), plus one verdict line.

So this is a **hybrid**: classical for image work (where computers are excellent and quantum
is hopeless), quantum for one narrow decision step (measuring similarity), classical again for
the final classifier.

### Why we changed dataset (DDTI → TN5000)

| | DDTI (before) | TN5000 (now) |
|---|---|---|
| Size | ~480 images | 5,000 images |
| Label | text tag: "has microcalcifications / doesn't" | biopsy result: cancer / not cancer |
| Split | we made one ourselves | official split given with the data (safe) |

DDTI was too small to train or test on honestly, and its label never told us the location of
any real finding. TN5000 is 10× bigger, its label is a real medical answer, and the dataset
authors provide a split that keeps the same patient from appearing in both train and test.

---

## 2. The quantum part, explained simply

### 2.1 Why use a quantum computer at all?

Being honest: **not for speed.** Our "quantum computer" is a simulator on an ordinary CPU, and
it is *slower* than a plain classifier. The only reason to look at quantum is that a quantum
circuit gives us a **different way to measure similarity** between two data points — and
"similarity" is exactly what an SVM needs. A quantum circuit creates its similarity in a way a
simple distance formula cannot.

**The small-data hope:** when you have very few training examples (say 15–30), flexible models
like neural networks memorise the training data and fail on new data. A quantum feature map is
a *fixed*, rigid description with no trainable parts, so it might not overfit as easily. That
hypothesis is exactly what Experiment A tests. We do not claim it in advance.

### 2.2 We never put the image into the computer

Some quantum papers encode a whole image into quantum memory. We do not, because it needs
roughly as many gates as there are pixels and is very slow — a dead end. Instead:

```
whole image → classical program picks 5 specks → one speck's numbers
            → compressed to 4 numbers → 4-qubit quantum state
```

One speck = one small quantum state. The image itself stays classical.

### 2.3 What the circuit actually does

For each speck we have 4 plain numbers (between 0 and π). The circuit does three things to them,
twice over:

1. **Spread** — each of the 4 qubits is put into a "both 0 and 1 at once" state (Hadamard gate).
   This is where the data can be written.
2. **Write the numbers** — each qubit is rotated by an angle that *is* one of our 4 numbers
   (phase gates). Rotations are the cheapest and most accurate way to put a number into a
   quantum state.
3. **Link the numbers together** — entangling gates (controlled-phase / ZZ) make qubit pairs
   affect each other. This is the important part: it means the final state records not just
   "these 4 numbers" but also "how these numbers relate to each other" (e.g. bright core with a
   darker ring around it). A simple one-number-per-qubit encoding does not get that for free.

Doing this twice (reps = 2) writes the numbers twice with fresh interference in between, which
makes the map more expressive without needing more qubits. The entangling pairs used are only
(1–2), (2–3), (3–4) — a "linear chain" — which matches how real quantum chips are connected and
keeps the circuit shallow. For our settings the circuit is **19 steps deep with 12 two-qubit
gates** (`python main.py circuit` prints this — always run it, don't quote from memory).

**The circuit is never trained.** It is a fixed machine. It is not a neural network; there are
no weights to learn inside it.

### 2.4 So where does the "decision" happen?

Here is the honest part, and it is the sentence to say out loud in the viva:

> The circuit does **not** classify anything and is **not** trained. It only *re-represents*
> each speck as a quantum state. We then measure how similar two states are. Those
> similarity scores go into a normal, classical SVM, and the SVM is what learns and decides.

Mathematically the similarity is one line:

> K(a, b) = |⟨ψ(a)|ψ(b)⟩|²  — "the squared overlap between the two quantum states"

This is the *quantum kernel*. An SVM normally uses a formula like "distance" or "RBF". We
changed exactly one ingredient — the similarity formula — which keeps the comparison against
classical SVMs fair and clean.

**Speed trick:** naively you run the circuit once for every *pair* of specks (millions of runs).
Instead we run the circuit **once per speck** and get every pairwise similarity from one matrix
multiplication. Same answer, vastly cheaper.

### 2.5 The second circuit (a baseline, not the main model)

An older version of our project used a different circuit (PennyLane) that, once you work out
the maths, turns into the simple product ∏ cos²((aᵢ−bᵢ)/2) — a product-cosine kernel. We
kept it as a **second quantum kernel to compare against**. Comparing the two tells us whether
the entangling ("linking") part earns its keep or whether any quantum encoding would do.

### 2.6 Intuition for why this could help *this* problem

The classifier only sees similarity scores. Specks from malignant nodules have different
local texture and surround patterns than specks from benign nodules. The entangling part of our
circuit adds "pairwise feature interaction" on top of plain distance — in image terms, it is
sensitive to patterns that co-occur locally (a bright core *and* a dark ring *and* a shadow
below). **Whether that actually improves accuracy is an experiment, not a fact.** We test it
against classical baselines on the same data with the same splits.

---

## 3. How it is built (implementation)

- **Tools:** Qiskit 2.x, qiskit-aer, scikit-learn, OpenCV, Python 3.14.
  We deliberately do **not** use `qiskit-machine-learning` (no compatible wheel); we build the
  kernel from Qiskit's basic tools plus a normal precomputed-kernel SVM, which is exactly what
  their `QSVC` class does internally.
- **Data (`src/dataset.py`):** reads the PASCAL-style annotation (nodule box + cancer/no-cancer
  label). An integrity report counts every defect instead of hiding it. We use the dataset's
  official train (4,000) / test (1,000) split and assert the overlap is zero.
- **Spot finder (`src/mining.py`):** normal computer-vision steps — brighten contrast evenly
  (CLAHE), keep only small bright blobs (top-hat filter), ignore everything outside the nodule
  box, then rank the specks and keep the 5 best. Its job is *high recall* — never throw away
  something that might matter; deciding what's real is the model's job.
- **Features:** for each speck we take the 4×4 raw pixels **plus 5 numbers describing the
  surroundings** (peak-vs-background contrast, local smoothness, contrast between core and ring,
  a shadow-below measure, and brightness strength). The surroundings matter because a 4×4 crop
  of a real speck and of machine noise look nearly identical otherwise — the context is what
  separates them.
- **Classifier (`src/quantum_engine.py`):** the numbers are standardised, compressed to 4 with
  PCA (fit on training data only), turned into quantum states, compared to all training states,
  and a balanced SVM learns from those comparisons. Because the label is "per nodule", we report
  results both per speck and per nodule (majority vote) — the nodule number is the honest one.
- **Output (`src/pipeline.py`):** an annotated image with red/green circles, the nodule box, and
  a one-line verdict with a confidence percentage.
- **Not yet done:** the three experiment scripts and the charts.

---

## 4. Planned experiments

- **Data:** TN5000 official split, fixed seed. Unit of analysis = mined specks (weak labels),
  reported up to the nodule level.
- **Metrics:** balanced accuracy (headline — the classes are 3,574 vs 1,426, so plain accuracy
  would be misleading), ROC AUC, confusion matrix, per-class F1, and nodule-level cancer recall.
- **Baselines:** classical RBF-SVM, Random Forest, a small MLP, and the second quantum kernel —
  all on the *same* data and splits.

| Experiment | Question it answers | How |
|---|---|---|
| A — sample efficiency | Does the quantum kernel hold up when training data is very small? | Train on 15/30/60/120 examples, 5 seeds each |
| B — noise & shots | Would this still work on real, imperfect hardware? | Add simulated hardware noise (0.1/0.5/1.0%) and limited measurement shots |
| C — ablation | Did the spot finder drop real cancers? And do the surrounding-features matter? | Vary how many specks are kept; retrain with/without the context features; test robustness to wrong speck labels |

---

## 5. Results so far (a wiring check, not final results)

One end-to-end test run (fixed seed, whole test set). These are **not** the experiment results.

| Features | Kernel | per-speck acc / bal / AUC | per-nodule acc / bal / AUC |
|---|---|---|---|
| 16 raw pixels | ZZ feature map | 0.647 / 0.628 / 0.654 | 0.670 / 0.647 / 0.700 |
| 16 raw pixels | Overlap baseline | 0.679 / 0.610 / 0.657 | 0.713 / 0.632 / 0.689 |
| **+ 5 context** | **ZZ feature map** | 0.631 / 0.626 / 0.660 | 0.662 / **0.657** / **0.716** |
| **+ 5 context** | **Overlap baseline** | 0.674 / **0.642** / **0.684** | 0.701 / **0.663** / **0.719** |

Read honestly: the numbers are only a little better than guessing. Two reasons, both understood:
(a) the training label is per nodule, so many individual specks are labelled by their nodule,
not by what they actually are; (b) 4 features are a lot of information to throw at any
classifier. Adding the context features helped both kernels (per-nodule AUC 0.700 → 0.716 for
the main one), which is the right direction but a small gain.

**When the experiments run**, the honest framing is: show quantum and classical side by side,
same data, average over 5 seeds (never a single best run), and let the result decide. If the
quantum kernel wins in the small-data range, that supports our hypothesis. If classical wins
everywhere, "we did not demonstrate a quantum advantage here" is a perfectly valid result and we
will report it that way.

---

## 6. Where quantum helps — and where it does not

**Possible benefit:** a small, rigid feature map may suit very small datasets better than
flexible neural networks, and the entangling gates add feature interactions with nothing to
train. Experiment A measures this.

**Where it does not help — say this openly:**

1. **No speed-up.** Everything is simulated on a normal CPU; our quantum step is *slower* than
   a classical classifier. The claim is a different similarity measure, not faster computing.
2. **We cannot put whole images in.** That would need far too many gates, which is why we only
   encode one speck's numbers.
3. **The similarity table grows fast.** Comparing every speck to every other speck is O(N²) in
   the number of specks. That is why we cap training at 2,000 specks.
4. **Four qubits is small.** Four qubits hold at most 16 numbers. There is no "exponential
   power" to boast about — the pitch is *a different way of measuring similarity*, not *more
   capacity*.
5. **The compression step (PCA) is classical** and does a lot of the work. Any benefit belongs
   to the whole hybrid pipeline, never to the quantum part alone.

> **One-line defence:** *"Our quantum component is a similarity measure, not an accelerator. It
> is a plausible fit for very small medical datasets, at the cost of slower training. Whether
> that trade pays off is exactly what our experiments measure."*

---

## 7. Limitations (be upfront about all of these)

**About the data/labels**
- **The big one — the label is per nodule, not per speck.** Every speck inside a cancer nodule
  is labelled "cancer", even ordinary tissue between the real findings. So some training labels
  are wrong by construction. This puts a ceiling on achievable accuracy and is the main reason
  our current numbers are modest.
- The classes are imbalanced (3,574 vs 1,426), so we report balanced accuracy, not plain accuracy.
- The dataset has no patient ID, so we cannot personally re-check for patient overlap — we rely
  on the authors' official split, which they built to avoid exactly that.
- There is no TI-RADS / composition / echogenicity information in this dataset, so the older
  idea of "re-scoring TI-RADS" was **removed**, not faked. Our output is a cancer / no-cancer
  verdict that can be compared directly with the biopsy label.

**About the spot finder**
- Its recall is a hard ceiling: a speck outside the box, ranked below the top 5, or hidden by a
  near neighbour is lost forever. Experiment C measures this.
- The image-processing settings are tuned for this dataset's resolution; other scanners would
  need re-tuning.

**About the quantum encoding**
- Very high or very low values are clipped to the ends of the angle range, so unusual patients
  get pushed onto the same angle as training extremes.
- If the compression step (PCA) throws away most of the variance, information is lost *before*
  the circuit sees it — always quote how much variance is kept next to any accuracy.
- Only neighbouring qubits are linked (1–2, 2–3, 3–4); whether another layout helps is untested.

**About simulation and noise**
- Training uses exact, noiseless simulation — optimistic by design.
- The noise test only models random gate errors; real chips also have readout errors, coherent
  errors, crosstalk and drift. Treat it as an optimistic bound.
- *Fixed earlier:* our noise function was measuring the wrong thing (it read out a single state
  instead of comparing two). It now builds the proper "compare state a with state b" circuit and
  matches the exact result to within 0.007 at 8,192 shots.
- We have not run on real quantum hardware.

**About scale and statistics**
- Always report across the 5 seeds, never a single run.
- The similarity table is O(N²) and each entry needs a quantum simulation — this gets worse as
  data grows.
- One dataset, one type of scanner: we make no claims beyond TN5000.

**Next steps:** run the three experiments → add classical baselines → extend the noise model →
test robustness to wrong labels → optional small run on real IBM hardware.

---

## 8. Viva preparation

Questions can go to **any** team member. These should be answerable cold:

1. **The problem in one sentence?** → Classify thyroid nodules as cancer or not from ultrasound,
   by classifying small mined specks with a quantum-kernel classifier.
2. **Why this dataset?** → 5,000 images, biopsy-confirmed labels, an official safe split. DDTI
   (~480 images, text tags) was too small to train on.
3. **What actually goes into the quantum computer?** → Four numbers (0 to π) summarising one
   speck. Never the whole image.
4. **What does the quantum circuit do?** → It does not classify and it is not trained. It turns
   one speck's numbers into a quantum state; we then measure how similar two states are.
5. **Where does the decision happen?** → In a normal SVM, using those similarity scores.
6. **Why not put the whole image in?** → It would need roughly one gate per pixel — far too slow.
7. **Walk me through the circuit.** → Spread (Hadamard) → write the numbers (phase rotations) →
   link the numbers together (entangling gates), repeated twice, with a simple linear chain of
   qubits to keep it shallow.
8. **Do you train the circuit?** → No. It is a fixed map. Only the SVM on top is trained.
9. **What is the kernel?** → The squared overlap between two quantum states,
   K = |⟨ψ(a)|ψ(b)⟩|².
10. **How do you make it fast?** → Simulate each speck once, then one matrix multiplication gives
    every pairwise similarity.
11. **What does the entangling part buy?** → It records how the numbers relate to each other, not
    just the numbers themselves.
12. **Isn't 4 qubits tiny?** → Yes. We claim a different way of measuring similarity, not more
    capacity.
13. **How do you stop train/test leakage?** → We use the dataset's own official split, which the
    authors built by keeping one image per patient; we assert the overlap is zero.
14. **What does the label actually mean?** → The needle-biopsy result of the nodule: 1 = cancer,
    0 = not cancer. It is per nodule, so labels for individual specks are "weak" (inherited).
15. **Why report balanced accuracy?** → Because 3,574 vs 1,426 is imbalanced; plain accuracy
    would flatter us.
16. **What is unfinished?** → The three experiments and final results. The code and the plan for
    them exist and have been run end to end.
17. **What was wrong with the noise code, and why did it matter?** → It read out a single state
    instead of comparing two, so its numbers did not depend on the first input at all. Now fixed
    and verified.
18. **What was wrong with the old 4×4 patch idea, and did you fix it?** → A 4×4 crop of a real
    speck and of machine noise look almost the same, because it ignored the surroundings. We
    added 5 context features (contrast vs background, smoothness, core-vs-ring contrast, shadow
    below, brightness strength) and it improved balanced accuracy and AUC for both kernels.

| ✅ Say this | ❌ Not this |
|---|---|
| "A hybrid pipeline: classical for images, quantum for one similarity step" | "It's faster than classical" |
| "The circuit is a fixed map; the SVM does the learning" | "The quantum computer classified the images" |
| "We encode one speck's numbers, not whole images" | "The image is stored in quantum memory" |
| "Labels are biopsy results at nodule level, so speck labels are weak" | "Every speck is labelled perfectly" |
| "We use the dataset's official safe split" | "We personally proved there is no leakage" |
| "Simulated exactly; hardware noise is modelled separately" | "We ran on real quantum hardware" |
| "This is a one-run wiring check; experiments pending" | "The quantum model won" — no experiment has run |
