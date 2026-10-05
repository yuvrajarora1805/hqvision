"""
Phase 1 & Phase 2 Pipeline Runner
Executes data ingestion, patient-isolated splitting, and candidate patch extraction.
Saves preprocessed candidate datasets ready for Phase 3 (Quantum Verification Engine).
"""

import os
import argparse
import numpy as np
from src.dataset import collect_ddti_dataset, get_patient_stratified_split
from src.mining import CandidateMiner, build_candidate_dataset


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 & Phase 2 Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to DDTI dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to store extracted arrays")
    parser.add_argument("--test_size", type=float, default=0.2, help="Patient test split ratio")
    parser.add_argument("--max_candidates", type=int, default=5, help="Top K candidate patches per nodule")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("PHASE 1: PATIENT-ISOLATED DATA INGESTION & PARTITIONING")
    print("=" * 65)
    records = collect_ddti_dataset(args.data_dir)
    print(f"Total valid image-nodule records discovered: {len(records)}")

    train_recs, test_recs = get_patient_stratified_split(records, test_size=args.test_size)
    train_patients = set(r["patient_id"] for r in train_recs)
    test_patients = set(r["patient_id"] for r in test_recs)

    print(f"Train Cohort: {len(train_recs)} images across {len(train_patients)} unique patients")
    print(f"Test Cohort:  {len(test_recs)} images across {len(test_patients)} unique patients")
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage check failed!"
    print("Leakage Check: PASSED (0% patient overlap between train and test)")

    print("\n" + "=" * 65)
    print("PHASE 2: CLASSICAL CANDIDATE MINING (HIGH-RECALL SIEVE)")
    print("=" * 65)
    miner = CandidateMiner(patch_size=4, max_candidates=args.max_candidates)

    print(f"Mining top {args.max_candidates} candidates (4x4 patches) per image with Top-Hat & Weber NMS...")
    X_train, y_train, meta_train = build_candidate_dataset(train_recs, miner)
    X_test, y_test, meta_test = build_candidate_dataset(test_recs, miner)

    print(f"\n[Dataset Shapes]")
    print(f"X_train: {X_train.shape} (N_patches, 16 features)")
    print(f"y_train: {y_train.shape} (Microcalc: {np.sum(y_train == 1)}, Speckle: {np.sum(y_train == 0)})")
    print(f"X_test:  {X_test.shape}  (N_patches, 16 features)")
    print(f"y_test:  {y_test.shape}  (Microcalc: {np.sum(y_test == 1)}, Speckle: {np.sum(y_test == 0)})")

    # Save arrays for subsequent phases
    train_out = os.path.join(args.output_dir, "train_candidates.npz")
    test_out = os.path.join(args.output_dir, "test_candidates.npz")
    np.savez_compressed(train_out, X=X_train, y=y_train)
    np.savez_compressed(test_out, X=X_test, y=y_test)

    print(f"\nSuccessfully saved candidate data to:")
    print(f" - {train_out}")
    print(f" - {test_out}")
    print("Phase 1 and Phase 2 successfully completed!")


if __name__ == "__main__":
    main()
