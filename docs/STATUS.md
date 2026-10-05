# Project Status — HQVision

**Last updated:** after the DDTI → TN5000 dataset switch and full pipeline port.
**Repo state:** Phases 1–4 ported and verified end-to-end. Phase 5 (experiments)
not started. No experimental results exist yet.

---

## What changed in the dataset switch

| | Before (DDTI) | Now (TN5000) |
|---|---|---|
| Images | ~480 annotated frames | 5,000 ultrasound frames |
| Labels | calcification text tags (`micro`/`non`) | biopsy-confirmed per nodule: `1` = malignant (3,574), `0` = benign (1,426) |
| Format | custom XML with `<svg>` polygons | PASCAL VOC `<bndbox>` + `<name>`, plus `ImageSets/Main/*.txt` |
| Split | hand-rolled patient 80/20 | official `trainval` (4,000) / `test` (1,000), overlap asserted 0 |
| ROI | polygon vertices | rectangle (converted to a 4-corner polygon; miner unchanged) |
| Clinical output | TI-RADS re-score | nodule majority-vote verdict (TN5000 has no TI-RADS metadata) |
| Task framing | microcalcification vs speckle | malignant vs benign nodule patches (weak labels) |

Why: DDTI was too small to train or evaluate on honestly (480 frames, no biopsy
ground truth). TN5000 gives 10× the data, a confirmed label per nodule, and a
leakage-safe published split.

---

## Verified working (qip_venv, Python 3.14)

```
python main.py integrity         # 5000/5000 parsed, 0 missing pairs, split overlap 0
python main.py mine              # 32 s: 19,973 train / 4,998 test patches, 21 features (all 5000 nodules hit)
python main.py train             # ZZ kernel SVC, caps at 2000 balanced train patches
python main.py train --kernel overlap
python main.py predict --image-id 000123     # overlay written to outputs/annotated_scans/
python main.py circuit           # ZZ: depth 19 / size 34 / 12 cx; overlap: depth 2 / 0 cx
```

Smoke metrics (seed 42, single run — **not** experiment results):

| Features | Kernel | patch acc / bal / AUC | nodule acc / bal / AUC |
|---|---|---|---|
| 16 raw px | ZZ feature map | 0.647 / 0.628 / 0.654 | 0.670 / 0.647 / 0.700 |
| 16 raw px | Overlap baseline | 0.679 / 0.610 / 0.657 | 0.713 / 0.632 / 0.689 |
| **+5 context** | **ZZ feature map** | 0.631 / 0.626 / 0.660 | 0.662 / **0.657** / **0.716** |
| **+5 context** | **Overlap baseline** | 0.674 / **0.642** / **0.684** | 0.701 / **0.663** / **0.719** |

Adding the five surround/context features improved balanced accuracy and AUC for
**both** kernels (nodule AUC 0.700 → 0.716 for ZZ, 0.689 → 0.719 for overlap). The
gain is modest because the nodule-level label is still the limit, not the features —
which is why Experiment A (small-data regime) and the classical baselines matter.

Also verified earlier: `noisy_fidelity_gram()` builds the proper U(a)·U†(b)
interference circuit (max deviation 0.007 vs exact kernel at 8192 shots).

---

## File map (after the port)

| File | State |
|---|---|
| `src/config.py` | TN5000 paths (`$TN5000_DIR` → `data/Main data` → `../TN5000/Main data`), label map, official split names, patch caps |
| `src/dataset.py` | VOC parser, integrity audit, official/fallback split, bbox → polygon |
| `src/mining.py` | miner unchanged in behaviour; **now emits 21 features** — 16 raw 4×4 pixels + 5 surround-context features (Weber, local std, ring contrast, shadow ratio, top-hat peak); meta is TN5000 (`case_id`) |
| `src/cache.py` | rewritten: per-split mined-patch cache, `balanced_subsample`, `get_split_arrays` |
| `src/pipeline.py` | rewritten: bbox inference, nodule verdict, red/green overlay (no TI-RADS) |
| `src/quantum_engine.py` | StandardScaler added *before* PCA (without z-scoring, PCA discards the 0–3 context features in favour of 0–255 pixels); dataset-agnostic; noise fix intact |
| `main.py` | rewritten: `integrity \| mine \| train \| predict \| circuit` CLI |
| `train_hybrid_model.py` | marked DEPRECATED (legacy DDTI/PennyLane) — do not run |
| `docs/plan.md`, `docs/IMPLEMENTATION_PLAN.md` | rewritten for TN5000 |
| `docs/REVIEW_AND_DEFENSE.md` | redone for TN5000 (mid-term review doc) |

---

## Pending

1. **Phase 5 experiments** (the deliverables that carry the review):
   - Experiment A — sample efficiency: quantum SVC vs RBF-SVM vs RF vs MLP,
     N ∈ {15, 30, 60, 120}, 5 seeds.
   - Experiment B — Aer depolarizing noise (0.1/0.5/1.0%, 1024 shots) accuracy decay.
   - Experiment C — mining ablation: K sweep → nodule-level malignant recall.
   - `experiments/` directory does not exist yet; `outputs/figures/` is empty.
2. Figures/tables for the report; a classical-baseline comparison in `main.py train`
   (currently only the two quantum kernels).
3. Optional: readout-error noise model, small IBM hardware job.

## Known limitations (state these, do not hide them)

- **Weak labels:** every patch inherits its nodule's biopsy label; patches inside a
  malignant nodule that are not themselves lesion are still tagged 1. Caps accuracy.
- **Training cap:** kernel Gram is O(N²), so training uses 2,000 balanced patches
  (of 19,973 mined) — subsampled, seeded, and recorded in config.
- **Single dataset:** no generalisation claim beyond TN5000.
- **Exact-statevector training:** no shot noise in training; noise appears only in
  Experiment B (depolarizing only — optimistic bound).
