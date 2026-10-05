"""Mined-patch caching and array helpers for the TN5000 pipeline.

Mining all 5,000 TN5000 frames takes on the order of a minute, which is
wasted work every time an experiment script rebuilds the same patch set. Each
split is cached as an ``(X, y, meta)`` pickle keyed on the mining
hyper-parameters, the split source and the dataset size, so changing any of
them invalidates the cache automatically.

Caps (``config.MAX_TRAIN_PATCHES``) are applied at *load* time, not cache
time: the cache always holds every mined patch so sweeps can vary the cap
without re-mining.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from . import config, dataset
from .mining import CandidateMiner, build_candidate_dataset


def _cache_key(split: str) -> str:
    """Fingerprint every setting that changes the mined patch set."""
    payload = {
        "split": split,
        "patch_size": config.PATCH_SIZE,
        "clahe": [config.CLAHE_CLIP_LIMIT, list(config.CLAHE_TILE_GRID)],
        "kernel": config.TOPHAT_KERNEL_SIZE,
        "nms_radius": config.NMS_RADIUS,
        "max_candidates": config.MAX_CANDIDATES,
        "label_map": config.LABEL_MAP,
        "official_split": config.USE_OFFICIAL_SPLIT,
        "split_names": [config.TRAIN_SPLIT_NAME, config.TEST_SPLIT_NAME],
        "dataset_root": str(config.DATASET_DIR),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def get_patch_arrays(
    split: str = "train",
    use_cache: bool = True,
    verbose: bool = True,
    miner: CandidateMiner | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return mined patches for one split as ``(X, y, meta)``, cached on disk."""
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    config.ensure_output_dirs()
    cache_path = config.CACHE_DIR / f"mined_{split}_{_cache_key(split)}.pkl"

    if use_cache and cache_path.exists():
        with cache_path.open("rb") as handle:
            X, y, meta = pickle.load(handle)
        if verbose:
            print(
                f"[cache] {split}: {len(y)} patches from {len({m['case_id'] for m in meta})} "
                f"images | malignant={int((y == config.LABEL_MALIGNANT).sum())} "
                f"benign={int((y == config.LABEL_BENIGN).sum())}"
            )
        return X, y, meta

    train_recs, test_recs = dataset.load_dataset(verbose=verbose)
    records = train_recs if split == "train" else test_recs
    if miner is None:
        miner = CandidateMiner(
            patch_size=config.PATCH_SIZE, max_candidates=config.MAX_CANDIDATES
        )

    X, y, meta = build_candidate_dataset(records, miner)

    if use_cache:
        with cache_path.open("wb") as handle:
            pickle.dump((X, y, meta), handle)
        if verbose:
            print(f"[cache] wrote {cache_path.name}")
    return X, y, meta


def write_split(
    split: str, X: np.ndarray, y: np.ndarray, meta: list[dict]
) -> Path:
    """Persist a freshly mined split under the current cache key."""
    config.ensure_output_dirs()
    cache_path = config.CACHE_DIR / f"mined_{split}_{_cache_key(split)}.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump((X, y, meta), handle)
    return cache_path


def balanced_subsample(
    X: np.ndarray,
    y: np.ndarray,
    meta: list[dict],
    cap: int | None,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Cap a patch set while keeping both classes represented evenly.

    ``cap`` is the *total* number of retained patches (per class when both
    classes are present). ``None`` disables capping.
    """
    if cap is None or len(y) <= cap:
        return X, y, meta

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, cap // len(classes))
    keep: list[int] = []
    for label in classes:
        idx = np.flatnonzero(y == label)
        take = min(per_class, len(idx))
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    keep = np.array(sorted(keep))
    return X[keep], y[keep], [meta[i] for i in keep.tolist()]


def get_split_arrays(
    max_train: int | None = config.MAX_TRAIN_PATCHES,
    max_test: int | None = config.MAX_TEST_PATCHES,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """Mined, capped train/test arrays in one call.

    Returns a dict with ``X_train/y_train/meta_train`` and
    ``X_test/y_test/meta_test``. Images with no mined candidate drop out here,
    which is reported so the effective nodule count stays honest.
    """
    X_train, y_train, meta_train = get_patch_arrays("train", use_cache=use_cache, verbose=verbose)
    X_test, y_test, meta_test = get_patch_arrays("test", use_cache=use_cache, verbose=verbose)

    X_train, y_train, meta_train = balanced_subsample(X_train, y_train, meta_train, max_train)
    X_test, y_test, meta_test = balanced_subsample(X_test, y_test, meta_test, max_test)

    if verbose:
        def _summary(X, y, meta, name):
            nodules = len({m["case_id"] for m in meta})
            print(
                f"[arrays] {name}: {len(y)} patches from {nodules} nodules | "
                f"malignant={int((y == config.LABEL_MALIGNANT).sum())} "
                f"benign={int((y == config.LABEL_BENIGN).sum())} | X={X.shape}"
            )

        _summary(X_train, y_train, meta_train, "train")
        _summary(X_test, y_test, meta_test, "test")

    return {
        "X_train": X_train, "y_train": y_train, "meta_train": meta_train,
        "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
