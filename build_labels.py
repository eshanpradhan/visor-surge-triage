"""Build the VISOR outcome labels from the SBU clinical CSV.

Primary training label
----------------------
severe = is_icu OR was_ventilated

This is an escalation/urgency label: it marks patients whose course required
critical-care resources. Death is deliberately excluded. Of the 183 deceased
patients, 69 were never admitted to the ICU and never ventilated, which is
consistent with comfort-care / DNR goals-of-care decisions rather than a
missed escalation. Folding those rows into the positive class would ask the
model to predict a care-planning decision, not clinical deterioration.

Secondary metric
----------------
deceased = (last.status == 'deceased')

Reported alongside the primary label for context. Not used for training.
"""

import pandas as pd

CLINICAL_CSV = "deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv"
OUT_CSV = "visor_labels.csv"


def build_labels(path: str = CLINICAL_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    is_icu = df["is_icu"].astype(bool)
    was_ventilated = df["was_ventilated"].eq("Yes")

    labels = pd.DataFrame(
        {
            "to_patient_id": df["to_patient_id"],
            "is_icu": is_icu,
            "was_ventilated": was_ventilated,
            "severe": is_icu | was_ventilated,
            # secondary / reporting only -- never a training target
            "deceased": df["last.status"].eq("deceased"),
        }
    )
    return labels


if __name__ == "__main__":
    labels = build_labels()

    print("severe = is_icu | was_ventilated")
    print(labels["severe"].value_counts(dropna=False).to_string())
    print(f"prevalence: {labels['severe'].mean() * 100:.1f}%  (n={len(labels)})")
    print()

    print("component overlap")
    print(
        pd.crosstab(
            labels["is_icu"],
            labels["was_ventilated"],
            rownames=["is_icu"],
            colnames=["was_ventilated"],
        ).to_string()
    )
    print()

    print("secondary metric: deceased, stratified by severe")
    print(
        pd.crosstab(
            labels["severe"], labels["deceased"], rownames=["severe"], colnames=["deceased"]
        ).to_string()
    )
    print()

    labels.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(labels)} rows)")
