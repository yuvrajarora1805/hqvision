"""Dataset caching and shared helpers for the experiment scripts.

Mining 466 annotated DDTI frames takes ~7 s, which is wasted work when five
experiment scripts each rebuild the same patch set. The cache is keyed on the
mining hyper-parameters so changing any of them invalidates it automatically.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from . import config, dataset
from .mining import CandidateMiner


def _cache_key() -> str:
    """Fingerprint every setting that changes the mined patch set."""
    payload = {
        "patch_size": config.PATCH_SIZE,
        "clahe": [config.CLAHE_CLIP_LIMIT, list(config.CLAHE_TILE_GRID)],
        "kernel": config.TOPHAT_KERNEL_SIZE,
        "nms_radius": config.NMS_RADIUS,
        "max_candidates": config.MAX_CANDIDATES,
        "label_map": config.LABEL_MAP,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def get_patch_records(
    use_cache: bool = True,
    verbose: bool = True,
    miner: CandidateMiner | None = None,
) -> list[dataset.PatchRecord]:
    """Return mined patch records, using the on-disk cache when valid."""
    config.ensure_output_dirs()
    cache_path = config.CACHE_DIR / f"patches_{_cache_key()}.pkl"

    if use_cache and cache_path.exists():
        with cache_path.open("rb") as handle:
            records = pickle.load(handle)
        if verbose:
            counts = np.bincount([r.label for r in records], minlength=2)
            print(f"[cache] loaded {len(records)} patches from {cache_path.name}")
            print(f"[cache] label 1 (micro)={counts[1]}  label 0 (speckle)={counts[0]}")
        return records

    report = dataset.load_cases(verbose=verbose)
    records = dataset.build_patch_dataset(report, miner=miner, verbose=verbose)
    if use_cache:
        with cache_path.open("wb") as handle:
            pickle.dump(records, handle)
        if verbose:
            print(f"[cache] wrote {cache_path.name}")
    return records


def records_to_arrays(records: list[dataset.PatchRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten patches to ``(n, 16)`` and collect labels."""
    X = np.array([r.features.reshape(-1) for r in records], dtype=np.float64)
    y = np.array([r.label for r in records], dtype=int)
    return X, y


def get_split(
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[dataset.PatientSplit, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: cached records -> patient-isolated split -> arrays."""
    records = get_patch_records(use_cache=use_cache, verbose=verbose)
    split = dataset.patient_isolated_split(records, verbose=verbose)
    X, y = records_to_arrays(split.train + split.test)
    n_train = len(split.train)
    return (
        split,
        X[:n_train],
        y[:n_train],
        X[n_train:],
        y[n_train:],
    )


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
