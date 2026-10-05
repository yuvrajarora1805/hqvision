"""
HQVision -- quantum-assisted thyroid nodule classifier (TN5000).

Sub-commands
------------
  integrity   Phase 1: TN5000 audit (labels, boxes, splits)
  mine        Phase 1+2: mine candidate patches, cache + export .npz
  train       Phase 3: quantum-kernel SVC on the mined patches (patch + nodule metrics)
  predict     Phase 4: run one image through miner + quantum verifier, write overlay
  circuit     Phase 5: gate/depth table for the ZZ feature map and the overlap baseline

Dataset resolution order: $TN5000_DIR -> data/Main data -> ../TN5000/Main data.
Pass ``--dataset`` to point at a custom location (sets $TN5000_DIR before any
src import, so config picks it up).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path


# --- Commands ----------------------------------------------------------------
def cmd_integrity(args: argparse.Namespace) -> None:
    from src import dataset

    dataset.load_dataset(verbose=True)


def cmd_mine(args: argparse.Namespace) -> None:
    import numpy as np

    from src import cache, config, dataset
    from src.mining import build_candidate_dataset, CandidateMiner

    config.ensure_output_dirs()
    print("=" * 65)
    print("PHASE 1+2: TN5000 INGESTION & CLASSICAL CANDIDATE MINING")
    print("=" * 65)

    train_recs, test_recs = dataset.load_dataset(verbose=True)

    miner = CandidateMiner(patch_size=args.patch_size, max_candidates=args.max_candidates)
    print(
        f"\nMining top-{args.max_candidates} candidates ({args.patch_size}x{args.patch_size} "
        "patches) per nodule ROI (CLAHE -> Top-Hat -> Weber NMS)..."
    )
    t0 = time.time()
    X_train, y_train, meta_train = build_candidate_dataset(train_recs, miner)
    X_test, y_test, meta_test = build_candidate_dataset(test_recs, miner)
    print(f"mined in {time.time() - t0:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "train_candidates.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "test_candidates.npz", X=X_test, y=y_test)

    for name, X, y, meta, recs in (
        ("train", X_train, y_train, meta_train, train_recs),
        ("test", X_test, y_test, meta_test, test_recs),
    ):
        print(
            f"[{name:5s}] X={X.shape} | malignant={int((y == config.LABEL_MALIGNANT).sum())} "
            f"benign={int((y == config.LABEL_BENIGN).sum())} | "
            f"nodules mined={len({m['case_id'] for m in meta})}/{len(recs)}"
        )

    # Refresh the on-disk cache so `train` reuses exactly these arrays.
    cache.write_split("train", X_train, y_train, meta_train)
    cache.write_split("test", X_test, y_test, meta_test)
    print(f"\nSaved {out_dir/'train_candidates.npz'} and {out_dir/'test_candidates.npz'}")
    print(f"Cached mined patches under {config.CACHE_DIR}")


def _nodule_level_metrics(meta, y_pred, scores):
    """Majority-vote per nodule -> accuracy / balanced accuracy / AUC."""
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    from src.pipeline import nodule_verdict
    from src import config

    by_case: dict[str, list[int]] = {}
    by_case_scores: dict[str, list[float]] = {}
    truth: dict[str, int] = {}
    for m, pred, score in zip(meta, y_pred, scores):
        cid = m["case_id"]
        by_case.setdefault(cid, []).append(int(pred))
        by_case_scores.setdefault(cid, []).append(float(score))
        truth[cid] = int(m["label"])

    case_ids = sorted(truth)
    y_true = np.array([truth[c] for c in case_ids])
    verdicts = [nodule_verdict(by_case[c], by_case_scores[c]) for c in case_ids]
    y_vote = np.array([v.label for v in verdicts])
    y_score = np.array([v.mean_score for v in verdicts])

    metrics = {
        "n_nodules": len(case_ids),
        "accuracy": float(accuracy_score(y_true, y_vote)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_vote)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def cmd_train(args: argparse.Namespace) -> None:
    import numpy as np

    from src import cache, config, quantum_engine as qe

    config.ensure_output_dirs()
    print("=" * 65)
    print(f"PHASE 3: QUANTUM KERNEL VERIFICATION ({args.kernel})")
    print("=" * 65)

    data = cache.get_split_arrays(
        max_train=args.max_train or config.MAX_TRAIN_PATCHES,
        max_test=args.max_test or config.MAX_TEST_PATCHES,
        use_cache=not args.no_cache,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test, meta_test = data["X_test"], data["y_test"], data["meta_test"]
    if len(y_train) == 0 or len(y_test) == 0:
        sys.exit("Empty split -- run `python main.py mine` first.")

    model = qe.QuantumKernelClassifier(kernel=args.kernel)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)

    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, roc_auc_score,
    )

    patch_metrics = {
        "n_patches": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(
            y_test, y_pred,
            target_names=[config.CLASS_NAMES[0], config.CLASS_NAMES[1]],
            output_dict=True, zero_division=0,
        ),
    }
    if len(np.unique(y_test)) > 1:
        patch_metrics["roc_auc"] = float(roc_auc_score(y_test, scores))

    nodule_metrics = _nodule_level_metrics(meta_test, y_pred, scores)

    print(f"\nfit time            : {fit_time:.1f}s")
    print("\n--- patch level (weak labels inherited from the nodule) ---")
    print(f"accuracy            : {patch_metrics['accuracy']:.3f}")
    print(f"balanced accuracy   : {patch_metrics['balanced_accuracy']:.3f}")
    if "roc_auc" in patch_metrics:
        print(f"roc auc             : {patch_metrics['roc_auc']:.3f}")
    print(f"confusion [[TN FP][FN TP]]: {patch_metrics['confusion_matrix']}")

    print("\n--- nodule level (majority vote over mined patches) ---")
    print(f"nodules             : {nodule_metrics['n_nodules']}")
    print(f"accuracy            : {nodule_metrics['accuracy']:.3f}")
    print(f"balanced accuracy   : {nodule_metrics['balanced_accuracy']:.3f}")
    if "roc_auc" in nodule_metrics:
        print(f"roc auc             : {nodule_metrics['roc_auc']:.3f}")

    # --- persist model + metrics -------------------------------------------
    model.circuit = None  # rebuilt lazily; avoids pickling a bound circuit
    model_path = config.MODELS_DIR / f"quantum_svc_{args.kernel}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    summary = {
        "kernel": args.kernel,
        "n_qubits": config.N_QUBITS,
        "n_reps": config.N_REPS,
        "pca_dims": config.PCA_DIMS,
        "max_train_patches": args.max_train or config.MAX_TRAIN_PATCHES,
        "fit_seconds": round(fit_time, 2),
        "patch": patch_metrics,
        "nodule": nodule_metrics,
        "circuit": qe.circuit_summary(qe.build_zz_feature_map()),
    }
    metrics_path = config.MODELS_DIR / f"metrics_{args.kernel}.json"
    cache.save_json(summary, metrics_path)
    print(f"\nSaved model   -> {model_path}")
    print(f"Saved metrics -> {metrics_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    import cv2

    from src import config, dataset, pipeline

    config.ensure_output_dirs()

    image_path = Path(args.image_path) if args.image_path else None
    if image_path is None:
        if not args.image_id:
            sys.exit("provide --image-id NNNNNN or --image-path PATH")
        image_path = config.IMAGE_DIR / f"{args.image_id}.jpg"
        if not image_path.is_file():
            sys.exit(f"image not found: {image_path}")

    if args.bbox:
        bbox = tuple(int(v) for v in args.bbox)
    else:
        xml_path = config.ANNOTATION_DIR / f"{Path(image_path).stem}.xml"
        if not xml_path.is_file():
            sys.exit(f"no annotation for {image_path.name}; pass --bbox xmin ymin xmax ymax")
        bbox = dataset.parse_voc_annotation(xml_path)["bbox"]

    model_path = Path(args.model) if args.model else (
        config.MODELS_DIR / f"quantum_svc_{args.kernel}.pkl"
    )
    if not model_path.is_file():
        sys.exit(f"model not found: {model_path} -- run `python main.py train` first.")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    t0 = time.time()
    result = pipeline.ThyroidCADPipeline(model).analyse(image_path, bbox)
    out_path = (
        Path(args.output) if args.output
        else config.ANNOTATED_DIR / f"{Path(image_path).stem}_overlay.png"
    )
    pipeline.render_overlay(image_path, result, out_path)

    print(f"image        : {image_path.name}")
    print(f"bbox         : {bbox}")
    print(f"candidates   : {result.n_candidates} "
          f"(malignant={result.n_malignant}, benign={result.n_benign})")
    print(f"verdict      : {result.verdict.describe()}")
    print(f"confidence   : {result.verdict.confidence:.0%}")
    print(f"overlay      : {out_path}  ({time.time() - t0:.2f}s)")


def cmd_circuit(args: argparse.Namespace) -> None:
    from src import config, quantum_engine as qe

    print("=" * 65)
    print("CIRCUIT ANALYSIS")
    print("=" * 65)
    rows = [
        ("ZZ feature map (main)", qe.build_zz_feature_map()),
        ("Overlap baseline (legacy)", qe.build_overlap_feature_map()),
    ]
    header = f"{'circuit':28s} {'qubits':>6s} {'params':>7s} {'depth':>6s} {'size':>6s} {'2Q gates':>8s}"
    print(header)
    print("-" * len(header))
    for name, circuit in rows:
        s = qe.circuit_summary(circuit)
        print(
            f"{name:28s} {s['n_qubits']:6d} {s['n_parameters']:7d} "
            f"{s['depth']:6d} {s['size']:6d} {s['two_qubit_gates']:8d}"
        )
    s = qe.circuit_summary(rows[0][1])
    detail = {k: v for k, v in s.items() if k.startswith("gate_")}
    print(f"\nZZ feature map gate counts: {detail}")
    print(f"config: N_QUBITS={config.N_QUBITS} N_REPS={config.N_REPS} "
          f"ENTANGLEMENT={config.ENTANGLEMENT} PCA_DIMS={config.PCA_DIMS}")


# --- CLI ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="HQVision quantum-assisted thyroid CAD (TN5000)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to the TN5000 'Main data' folder (sets $TN5000_DIR)",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("integrity", help="Phase 1: TN5000 audit + split report").set_defaults(
        func=cmd_integrity
    )

    p_mine = sub.add_parser("mine", help="Phase 1+2: mine candidate patches")
    p_mine.add_argument("--output_dir", type=str, default="data/processed")
    p_mine.add_argument("--patch_size", type=int, default=4)
    p_mine.add_argument("--max_candidates", type=int, default=5)
    p_mine.set_defaults(func=cmd_mine)

    p_train = sub.add_parser("train", help="Phase 3: quantum kernel SVC")
    p_train.add_argument("--kernel", choices=["zz_fidelity", "overlap"], default="zz_fidelity")
    p_train.add_argument("--max_train", type=int, default=None,
                         help="balanced cap on training patches (default: config)")
    p_train.add_argument("--max_test", type=int, default=None,
                         help="balanced cap on test patches (default: all)")
    p_train.add_argument("--no-cache", action="store_true", help="ignore mined-patch cache")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Phase 4: annotate one image")
    p_pred.add_argument("--image-id", type=str, default=None, help="e.g. 000123")
    p_pred.add_argument("--image-path", type=str, default=None)
    p_pred.add_argument("--bbox", nargs=4, type=int, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    p_pred.add_argument("--model", type=str, default=None)
    p_pred.add_argument("--kernel", choices=["zz_fidelity", "overlap"], default="zz_fidelity")
    p_pred.add_argument("--output", type=str, default=None)
    p_pred.set_defaults(func=cmd_predict)

    sub.add_parser("circuit", help="Phase 5: circuit gate/depth table").set_defaults(
        func=cmd_circuit
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset:
        # Must happen before any `src` import: config resolves paths on import.
        os.environ["TN5000_DIR"] = args.dataset
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
