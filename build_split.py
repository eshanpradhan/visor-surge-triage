"""Assign a stratified 70/15/15 train/val/test split to the VISOR manifest.

The manifest already holds exactly one row per patient, so splitting rows and
splitting patients are the same operation here. The patient-level assertion
below is kept anyway: if the manifest is ever regenerated with multiple images
per patient, a row-wise split would silently leak the same patient into two
sets, and this is where that would surface.

Stratifying on ``severe`` keeps the 20% prevalence in all three sets. The test
set holds only ~41 positives, so single-run test metrics carry wide confidence
intervals -- treat them as a rough check, not a precise estimate.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

MANIFEST_CSV = "visor_manifest.csv"
OUT_CSV = "visor_manifest_split.csv"
RANDOM_STATE = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def build_split(path: str = MANIFEST_CSV) -> pd.DataFrame:
    manifest = pd.read_csv(path)

    assert manifest["patient_id"].is_unique, "manifest has repeated patient_id"

    train_idx, holdout_idx = train_test_split(
        manifest.index,
        test_size=VAL_FRACTION + TEST_FRACTION,
        stratify=manifest["severe"],
        random_state=RANDOM_STATE,
    )
    # split the holdout evenly into val and test, stratified again
    holdout = manifest.loc[holdout_idx]
    val_idx, test_idx = train_test_split(
        holdout.index,
        test_size=TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION),
        stratify=holdout["severe"],
        random_state=RANDOM_STATE,
    )

    manifest["split"] = "train"
    manifest.loc[val_idx, "split"] = "val"
    manifest.loc[test_idx, "split"] = "test"
    return manifest


def assert_no_patient_leakage(manifest: pd.DataFrame) -> None:
    per_patient_splits = manifest.groupby("patient_id")["split"].nunique()
    offenders = per_patient_splits[per_patient_splits > 1]
    assert offenders.empty, f"patients in multiple splits: {list(offenders.index)[:5]}"

    sets = {s: set(g["patient_id"]) for s, g in manifest.groupby("split")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        assert not overlap, f"{a}/{b} share {len(overlap)} patients"


if __name__ == "__main__":
    manifest = build_split()
    assert_no_patient_leakage(manifest)
    manifest.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(manifest)} rows)")
