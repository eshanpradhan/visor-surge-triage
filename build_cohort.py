"""Join the VISOR labels to the COVID-19-NY-SBU chest X-ray archive.

Produces visor_cohort.csv: one row per labeled patient, carrying the severe
label, whether imaging exists, the patient's imaging directory, and per-patient
image counts.

metadata.csv is the join source for patient IDs -- it is the archive's own
index, and its patient_id field matches the on-disk directory names exactly.
The directory listing is used only to confirm that agreement.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "COVID-19-NY-SBU"
LABELS_CSV = ROOT / "visor_labels.csv"
METADATA_CSV = ROOT / "metadata.csv"
OUT_CSV = ROOT / "visor_cohort.csv"


def build_cohort() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV)
    meta = pd.read_csv(METADATA_CSV, low_memory=False)

    # sanity: metadata patient_ids and on-disk directory names must agree
    dirs = {p.name for p in IMAGE_ROOT.iterdir() if p.is_dir()}
    meta_ids = set(meta["patient_id"])
    assert dirs == meta_ids, (
        f"metadata/disk mismatch: {len(meta_ids - dirs)} in metadata only, "
        f"{len(dirs - meta_ids)} on disk only"
    )

    # per-patient image inventory. 'enhanced' marks contrast-processed copies of
    # the same acquisition, so count raw and enhanced separately.
    per_patient = meta.groupby("patient_id").agg(
        n_images=("id", "size"),
        n_series=("series", "nunique"),
        n_enhanced=("enhanced", "sum"),
    )
    per_patient["n_raw"] = per_patient["n_images"] - per_patient["n_enhanced"]

    cohort = labels.merge(
        per_patient, how="left", left_on="to_patient_id", right_index=True
    )
    cohort["has_imaging"] = cohort["n_images"].notna()
    cohort["image_dir"] = cohort.apply(
        lambda r: f"COVID-19-NY-SBU/{r['to_patient_id']}" if r["has_imaging"] else "",
        axis=1,
    )
    for c in ["n_images", "n_series", "n_enhanced", "n_raw"]:
        cohort[c] = cohort[c].fillna(0).astype(int)

    return cohort.rename(columns={"to_patient_id": "patient_id"})[
        [
            "patient_id",
            "severe",
            "is_icu",
            "was_ventilated",
            "deceased",
            "has_imaging",
            "image_dir",
            "n_images",
            "n_series",
            "n_raw",
            "n_enhanced",
        ]
    ]


if __name__ == "__main__":
    cohort = build_cohort()
    cohort.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.name}  ({len(cohort)} rows)")
