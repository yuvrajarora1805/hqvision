"""CLI entry point for the quantum-assisted thyroid microcalcification detector.

Usage
-----
    python main.py integrity                 # audit the raw DDTI release
    python main.py train                     # mine, split, fit, evaluate
    python main.py predict --image data/106_1.jpg
    python main.py predict --case 106         # all annotated frames of a case
    python main.py circuit                   # gate/depth table for the report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src import cache, config, dataset, quantum_engine
from src.mining import CandidateMiner
from src.pipeline import ThyroidCADPipeline, render_overlay, ti_rads_rescore


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _versions() -> None:
    import cv2
    import qiskit
    import qiskit_aer
    import sklearn

    print(f"  python      {sys.version.split()[0]}")
    print(f"  qiskit      {qiskit.__version__}")
    print(f"  qiskit-aer  {qiskit_aer.__version__}")
    print(f"  scikit-learn{sklearn.__version__:>12}")
    print(f"  opencv      {cv2.__version__}")
    print(f"  numpy       {np.__version__}")


# --- Subcommands -------------------------------------------------------------
def cmd_integrity(args: argparse.Namespace) -> int:
    _banner("Module 1 - Dataset Integrity & Patient-Isolated Partitioning")
    _versions()
    report = dataset.load_cases(verbose=True)
    records = dataset.build_patch_dataset(report, verbose=True)
    dataset.patient_isolated_split(records, verbose=True)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    _banner("End-to-End Training: Classical Miner + Quantum Kernel Classifier")
    _versions()
    config.ensure_output_dirs()

    records = cache.get_patch_records(use_cache=not args.no_cache, verbose=True)
    split = dataset.patient_isolated_split(records, verbose=True)

    X, y = cache.records_to_arrays(split.train + split.test)
    n_train = len(split.train)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    kernels = args.kernels.split(",") if args.kernels else ["zz_fidelity"]
    summary: dict[str, dict] = {}

    for kernel in kernels:
        _banner(f"Training QSVC with kernel = {kernel}")
        model = quantum_engine.QuantumKernelClassifier(
            kernel=kernel,
            pca_dims=args.pca_dims,
            reps=args.reps,
            C=args.C,
        ).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        balanced = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

        print(f"\n  Accuracy          : {accuracy:.4f}")
        print(f"  Balanced accuracy : {balanced:.4f}")
        print(f"  ROC AUC           : {auc:.4f}")
        print(f"\n  Confusion matrix (rows = truth [speckle, micro], cols = pred):")
        for row in matrix:
            print("    " + "  ".join(f"{v:5d}" for v in row))
        print("\n  Classification report:")
        print(
            "    "
            + classification_report(
                y_test, y_pred, target_names=["speckle(0)", "microcalc(1)"], zero_division=0
            )
            .replace("\n", "\n    ")
            .rstrip()
        )

        explained = float(np.sum(model.encoding.pca.explained_variance_ratio_))
        print(f"\n  PCA variance retained ({args.pca_dims} comps): {explained:.4f}")

        if model.circuit is not None:
            print("  Circuit:")
            for key, value in quantum_engine.circuit_summary(model.circuit).items():
                print(f"    {key:<16} {value}")

        summary[kernel] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "roc_auc": auc,
            "confusion_matrix": matrix.tolist(),
            "pca_variance_retained": explained,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_patients": sorted(split.train_patients),
            "test_patients": sorted(split.test_patients),
        }

        model_path = config.MODELS_DIR / f"qsvc_{kernel}.joblib"
        joblib.dump(model, model_path)
        print(f"\n  Saved model -> {model_path.relative_to(config.REPO_ROOT)}")
        np.savez_compressed(
            config.CACHE_DIR / f"gram_{kernel}.npz",
            train=model.train_gram,
            test=model.test_gram(X_test),
        )

    cache.save_json(summary, config.OUTPUTS_DIR / "train_summary.json")
    print(f"\n  Summary -> {(config.OUTPUTS_DIR / 'train_summary.json').relative_to(config.REPO_ROOT)}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import joblib

    _banner("Module 4 - Clinical Inference & Diagnostic Overlay")
    config.ensure_output_dirs()

    model_path = config.MODELS_DIR / f"qsvc_{args.kernel}.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}\nRun `python main.py train` first.")
        return 1
    model = joblib.load(model_path)

    report = dataset.load_cases(verbose=False)
    miner = CandidateMiner()

    targets: list[tuple[str, int, Path]] = []
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return 1
        case_id = image_path.stem.rsplit("_", 1)[0]
        index = int(image_path.stem.rsplit("_", 1)[1])
        targets.append((case_id, index, image_path))
    elif args.case:
        case = next((c for c in report.cases if c.case_id == args.case), None)
        if case is None:
            print(f"Case not found in dataset: {args.case}")
            return 1
        for index in case.image_indices:
            path = config.DATA_DIR / f"{case.case_id}_{index}.jpg"
            if path.exists():
                targets.append((case.case_id, index, path))
    else:
        print("Provide --image or --case.")
        return 1

    for case_id, index, image_path in targets:
        case = next((c for c in report.cases if c.case_id == case_id), None)
        regions = [r for r in case.regions if r.image_index == index] if case else []
        if not regions:
            print(f"  {image_path.name}: no ROI annotation, skipping.")
            continue

        pipeline = ThyroidCADPipeline(model=model, miner=miner)
        result = pipeline.analyse(image_path, regions)

        out_path = config.ANNOTATED_DIR / f"{image_path.stem}_annotated.jpg"
        render_overlay(
            image_path,
            result,
            out_path,
            base_tirads=case.tirads if case else None,
        )

        score = ti_rads_rescore(result.n_microcalcifications)
        print(f"\n  {image_path.name}")
        print(f"    candidates mined     : {result.n_candidates}")
        print(f"    verified microcalc   : {result.n_microcalcifications}")
        print(f"    filtered speckle     : {result.n_speckles}")
        print(f"    TI-RADS rescored     : {score.category} ({score.points} pts)")
        print(f"    overlay              : {out_path.relative_to(config.REPO_ROOT)}")
    return 0


def cmd_circuit(args: argparse.Namespace) -> int:
    _banner("Quantum Circuit Analysis")
    for name, circuit in (
        ("ZZFeatureMap (proposed)", quantum_engine.build_zz_feature_map()),
        ("AngleEmbedding (legacy PennyLane baseline)", quantum_engine.build_overlap_feature_map()),
    ):
        summary = quantum_engine.circuit_summary(circuit)
        print(f"\n  {name}")
        for key, value in summary.items():
            print(f"    {key:<16} {value}")
        print("\n    Circuit diagram:")
        try:
            print(circuit.draw("text", fold=100))
        except Exception as exc:  # pragma: no cover - optional pypiton
            print(f"      (text drawer unavailable: {exc})")
    return 0


# --- Parser ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Quantum-assisted thyroid microcalcification detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("integrity", help="audit the DDTI release and print the split")

    train = sub.add_parser("train", help="mine, split, fit and evaluate the QSVC")
    train.add_argument("--kernels", default="zz_fidelity", help="comma-separated kernels")
    train.add_argument("--pca-dims", type=int, default=config.PCA_DIMS)
    train.add_argument("--reps", type=int, default=config.N_REPS)
    train.add_argument("--C", type=float, default=1.0, dest="C")
    train.add_argument("--no-cache", action="store_true", help="re-mine from scratch")

    predict = sub.add_parser("predict", help="annotate a scan with the trained model")
    predict.add_argument("--image", type=str, default=None)
    predict.add_argument("--case", type=str, default=None)
    predict.add_argument("--kernel", default="zz_fidelity")

    sub.add_parser("circuit", help="print gate/depth tables for the report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "integrity": cmd_integrity,
        "train": cmd_train,
        "predict": cmd_predict,
        "circuit": cmd_circuit,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
