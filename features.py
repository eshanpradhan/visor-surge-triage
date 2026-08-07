"""Build the clinical feature frame under the feature_spec.csv leakage contract.

Every function here fails hard rather than warning. A warning about leakage gets
scrolled past; a raised exception stops the run. The checks are deliberately
redundant -- excluded columns are verified absent both by name and by a second
pass over the spec after the frame is built -- because the failure mode being
guarded against is a column slipping through unnoticed, not a column being
obviously wrong.

Typical use::

    from features import build_feature_frame

    train = build_feature_frame(split="train")
    X = train.drop(columns=["patient_id", "severe", "split", "filepath"])
    y = train["severe"]
"""

import pandas as pd

SPEC_CSV = "feature_spec.csv"
CLINICAL_CSV = "deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv"
MANIFEST_SPLIT_CSV = "visor_manifest_split.csv"

VALID_STATUSES = {"SAFE", "LEAK", "LEAK_RISK", "DROP"}
EXCLUDED_STATUSES = {"LEAK", "LEAK_RISK", "DROP"}

# carried through from the manifest, not from the clinical CSV: identifiers, the
# label, the split assignment, and the image path. These are not model inputs and
# are exempt from the spec, which only governs clinical columns.
PASSTHROUGH = ["patient_id", "severe", "split", "filepath"]


class LeakageError(AssertionError):
    """Raised when a non-SAFE column reaches the feature frame."""


def load_spec(path: str = SPEC_CSV) -> pd.DataFrame:
    spec = pd.read_csv(path)

    missing = {"column_name", "category", "status", "reason"} - set(spec.columns)
    assert not missing, f"feature_spec.csv missing required columns: {sorted(missing)}"
    assert spec["column_name"].is_unique, "feature_spec.csv has duplicate column_name entries"

    bad = set(spec["status"]) - VALID_STATUSES
    assert not bad, f"feature_spec.csv has invalid status values: {sorted(bad)}"

    return spec


def safe_columns(spec: pd.DataFrame | None = None, drop_redundant: bool = True) -> list[str]:
    """SAFE column names, optionally minus threshold flags that duplicate a numeric.

    A binned column is dropped only when its ``redundant_with`` source is itself
    in the SAFE set -- otherwise dropping it would discard information rather
    than deduplicate it. Three lab_binned columns (Urine.protein,
    Microscopic_hematuria.above2, Proteinuria.above80) name no source and are
    always kept: no numeric equivalent exists in this file.

    The numeric column is the one kept. A raw value carries strictly more
    information than someone's hand-picked cut of it, and at n=955 training
    patients the redundant dimensions cost more in overfitting risk than the
    thresholds are worth.
    """
    spec = load_spec() if spec is None else spec
    safe = spec[spec["status"] == "SAFE"]

    if not drop_redundant:
        return safe["column_name"].tolist()

    safe_names = set(safe["column_name"])
    source = safe["redundant_with"].fillna("")
    keep = safe[~(source != "") | ~source.isin(safe_names)]
    return keep["column_name"].tolist()


def excluded_columns(spec: pd.DataFrame | None = None) -> dict[str, str]:
    """Map every non-SAFE column to its status, for error messages."""
    spec = load_spec() if spec is None else spec
    excluded = spec[spec["status"].isin(EXCLUDED_STATUSES)]
    return dict(zip(excluded["column_name"], excluded["status"]))


def assert_no_leakage(frame: pd.DataFrame, spec: pd.DataFrame | None = None) -> None:
    """Raise LeakageError if any LEAK/LEAK_RISK/DROP column is present.

    Also raises if a clinical column appears that the spec does not classify at
    all -- an unclassified column is one nobody audited, which is the same risk
    with less evidence.
    """
    spec = load_spec() if spec is None else spec
    excluded = excluded_columns(spec)

    offenders = {c: excluded[c] for c in frame.columns if c in excluded}
    if offenders:
        detail = ", ".join(f"{c} ({s})" for c, s in sorted(offenders.items()))
        raise LeakageError(f"{len(offenders)} excluded column(s) reached the feature frame: {detail}")

    known = set(spec["column_name"]) | set(PASSTHROUGH)
    unclassified = [c for c in frame.columns if c not in known]
    if unclassified:
        raise LeakageError(
            f"{len(unclassified)} column(s) absent from feature_spec.csv: {sorted(unclassified)}. "
            "Classify them in build_feature_spec.py before using this frame."
        )


def build_feature_frame(
    split: str | None = None,
    drop_redundant: bool = True,
    manifest_path: str = MANIFEST_SPLIT_CSV,
    clinical_path: str = CLINICAL_CSV,
    spec_path: str = SPEC_CSV,
) -> pd.DataFrame:
    """Join the manifest to the clinical CSV, keeping only SAFE clinical columns.

    Parameters
    ----------
    split
        Restrict to one of 'train' / 'val' / 'test'. None returns all rows.
    drop_redundant
        Drop threshold flags whose source numeric column is also present. See
        :func:`safe_columns`.
    """
    spec = load_spec(spec_path)
    manifest = pd.read_csv(manifest_path)
    clinical = pd.read_csv(clinical_path, low_memory=False)

    if split is not None:
        assert split in set(manifest["split"]), f"unknown split: {split!r}"
        manifest = manifest[manifest["split"] == split]

    keep = safe_columns(spec, drop_redundant=drop_redundant)
    present = [c for c in keep if c in clinical.columns]
    assert len(present) == len(keep), (
        f"{len(keep) - len(present)} SAFE column(s) not found in {clinical_path}: "
        f"{sorted(set(keep) - set(present))}"
    )

    # to_patient_id is DROP-status, so it is used for the join and immediately removed
    clinical_safe = clinical[["to_patient_id"] + present]

    frame = manifest[PASSTHROUGH].merge(
        clinical_safe, how="left", left_on="patient_id", right_on="to_patient_id"
    )
    frame = frame.drop(columns=["to_patient_id"])

    assert len(frame) == len(manifest), "join changed row count -- duplicate patient_id somewhere"
    assert frame["patient_id"].is_unique, "duplicate patient_id in feature frame"

    assert_no_leakage(frame, spec)
    return frame


if __name__ == "__main__":
    spec = load_spec()
    print("feature_spec.csv status counts:")
    print(spec["status"].value_counts().to_string())
    print()

    full = len(safe_columns(spec, drop_redundant=False))
    reduced = len(safe_columns(spec, drop_redundant=True))
    print(f"SAFE columns: {full} raw -> {reduced} after dropping {full - reduced} redundant bins")
    print()

    for split in [None, "train", "val", "test"]:
        frame = build_feature_frame(split=split)
        name = split or "all"
        n_features = frame.shape[1] - len(PASSTHROUGH)
        print(
            f"{name:6s} rows={len(frame):5d}  features={n_features:4d}  "
            f"severe={frame['severe'].mean() * 100:.1f}%"
        )
