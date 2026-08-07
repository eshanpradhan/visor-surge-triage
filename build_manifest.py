"""Build the trainable manifest: one chest X-ray per patient, the earliest one.

Why one image per patient
-------------------------
Severe patients carry ~12x more images than non-severe ones (mean 37.3 vs 3.2),
because sicker patients get serial portable films. Any model trained on all
images can score well by learning image count rather than radiographic finding.
Taking a single earliest study per patient removes that shortcut and keeps the
class balance at the patient level.

Selection rules
---------------
1. Raw arm only (``enhanced == False``). The enhanced rows are contrast-processed
   copies of the same acquisitions; keeping both would put near-identical
   anatomy on either side of a train/val split. All 1365 imaged patients have at
   least one raw image, so this drops no patients.
2. Sort within patient by the date parsed out of the series string. Years are
   date-shifted for de-identification (they land in 1900/1901), so the ordering
   is only meaningful *within* a patient -- absolute dates are not comparable
   across patients and are not used for anything else.
3. Ties within the earliest date are broken by sorting on series then filename
   and taking the first. This tiebreak is ARBITRARY, not clinically chosen: when
   a patient has several films from their first study, nothing in the metadata
   says which is the better or more diagnostic view, so a deterministic string
   sort stands in for a choice we cannot make from this data.

Two of the 6545 series strings are malformed (they embed the patient prefix and
a trailing slash), so the date is located by search rather than a full-string
match.
"""

import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "COVID-19-NY-SBU"
COHORT_CSV = ROOT / "visor_cohort.csv"
METADATA_CSV = ROOT / "metadata.csv"
OUT_CSV = ROOT / "visor_manifest.csv"

DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


def parse_series_date(series: str) -> str:
    """Return the series date as a sortable YYYY-MM-DD string.

    Dates are de-identified by year shifting, so this is only used to order
    studies within a single patient.
    """
    match = DATE_RE.search(series)
    if not match:
        raise ValueError(f"no date found in series: {series!r}")
    month, day, year = match.groups()
    return f"{year}-{month}-{day}"


def build_manifest() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_CSV)
    meta = pd.read_csv(METADATA_CSV, low_memory=False)

    raw = meta[~meta["enhanced"]].copy()
    raw["study_date"] = raw["series"].map(parse_series_date)

    # earliest study per patient; arbitrary but deterministic tiebreak
    raw = raw.sort_values(["patient_id", "study_date", "series", "filename"])
    first = raw.groupby("patient_id", as_index=False).first()

    manifest = (
        cohort[cohort["has_imaging"]]
        .merge(first, left_on="patient_id", right_on="patient_id", how="inner")
        .rename(columns={"id": "image_id", "filename": "rel_path"})
    )
    manifest["filepath"] = "COVID-19-NY-SBU/" + manifest["rel_path"]

    return manifest[
        [
            "patient_id",
            "severe",
            "is_icu",
            "was_ventilated",
            "deceased",
            "filepath",
            "image_id",
            "study_date",
            "series",
            "n_images",
            "n_series",
        ]
    ]


if __name__ == "__main__":
    manifest = build_manifest()

    missing = [p for p in manifest["filepath"] if not (ROOT / p).is_file()]
    assert not missing, f"{len(missing)} manifest paths do not exist, e.g. {missing[:3]}"

    manifest.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.name}  ({len(manifest)} rows)")
