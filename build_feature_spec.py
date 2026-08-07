"""Generate feature_spec.csv: a per-column leakage audit of the SBU clinical CSV.

The spec is generated rather than hand-typed so that every one of the 131 source
columns is guaranteed to be classified -- an unlisted column is a column nobody
decided about, which is exactly how a leak survives review. The script asserts
full coverage before writing.

Status vocabulary
-----------------
SAFE       Known at or near presentation. Usable as a model feature.
LEAK       Known only after the outcome, or computed from the ICU/vent/death
           event itself.
LEAK_RISK  Timing is ambiguous or the value reflects a clinician's reaction to
           severity. Excluded by default; resolve against the data dictionary
           before promoting to SAFE.
DROP       Identifier, constant, exact duplicate, or too sparse to model.

The ``redundant_with`` column names the numeric column a threshold flag was cut
from, empty for everything else. It exists so that redundancy handling reads a
structured field rather than parsing the free-text reason, and so the binned
column and its source are always resolved as an exact pair.
"""

import pandas as pd

CLINICAL_CSV = "deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv"
OUT_CSV = "feature_spec.csv"

# binned column -> the numeric column it was thresholded from
BIN_SOURCES = {
    "BMI.over30": "39156-5_Body mass index (BMI) [Ratio]",
    "BMI.over35": "39156-5_Body mass index (BMI) [Ratio]",
    "temperature.over38": "8331-1_Oral temperature",
    "pulseOx.under90": "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry",
    "Respiration.over24": "9279-1_Respiratory rate",
    "HeartRate.over100": "76282-3_Heart rate.beat-to-beat by EKG",
    "Lymphocytes.under1k": "731-0_Lymphocytes [#/volume] in Blood by Automated count",
    "Aspartate.over40": "1920-8_Aspartate aminotransferase",
    "Alanine.over60": "1744-2_Alanine aminotransferase",
    "A1C": "4548-4_Hemoglobin A1c/Hemoglobin.total in Blood",
    "Sodium": "2951-2_Sodium [Moles/volume] in Serum or Plasma",
    "Potassium": "2823-3_Potassium [Moles/volume] in Serum or Plasma",
    "Chloride": "2075-0_Chloride [Moles/volume] in Serum or Plasma",
    "Bicarbonate": "1963-8_Bicarbonate [Moles/volume] in Serum or Plasma",
    "Blood_Urea_Nitrogen": "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma",
    "Creatinine": "2160-0_Creatinine [Mass/volume] in Serum or Plasma",
    "eGFR": "62238-1_Glomerular filtration rate",
    "blood_pH": "33254-4_pH of Arterial blood",
    "Troponin.above0.01": "6598-7_Troponin T.cardiac",
    "D_dimer": "48058-2_Fibrin D-dimer DDU",
    "ESR.above30": "30341-2_Erythrocyte sedimentation rate",
    "SBP": "8480-6_Systolic blood pressure",
    "MAP": "76536-2_Mean blood pressure by Noninvasive",
    "procalcitonin": "75241-0_Procalcitonin",
    "ferritin.above1k": "2276-4_Ferritin [Mass/volume] in Serum or Plasma",
}

EXPLICIT = {
    # --- identifiers / admin ---
    "to_patient_id": ("admin", "DROP", "Patient identifier. Join key only, never a feature."),
    "covid19_statuses": ("admin", "DROP", "Constant: 'positive' for all 1384 rows. Zero variance."),
    "visit_start_datetime": (
        "admin",
        "SAFE",
        "Admission date, known at presentation. Date-shifted for de-identification; "
        "spans Dec-1900 to Sep-1901. Carries a calendar-wave confound because COVID "
        "treatment protocols changed over 2020 -- use with caution or bin coarsely.",
    ),
    "visit_concept_name": (
        "admin",
        "LEAK_RISK",
        "Inpatient 1025 / ER 357 / Outpatient 2. Encodes the admission decision, which "
        "a clinician makes in reaction to observed severity.",
    ),
    # --- demographics ---
    "age.splits": ("demographics", "SAFE", "Age band: [18,59], (59,74], (74,90]. Known at presentation."),
    "gender_concept_name": ("demographics", "SAFE", "MALE 760 / FEMALE 592 / null 32. Known at presentation."),
    # --- outcome and label-derived ---
    "last.status": ("outcome", "LEAK", "Discharge disposition (discharged/deceased). Secondary reporting metric only."),
    "is_icu": ("outcome", "LEAK", "Component of the severe label. Training on it is circular."),
    "was_ventilated": ("outcome", "LEAK", "Component of the severe label. Training on it is circular."),
    "invasive_vent_days": (
        "outcome",
        "LEAK",
        "Computed from the ventilation event. NaN corresponds exactly to was_ventilated=='No'.",
    ),
    "length_of_stay": ("outcome", "LEAK", "Whole-stay quantity, unknown until discharge."),
    # --- in-hospital complications and treatments ---
    "Acute.Hepatic.Injury..during.hospitalization.": (
        "in_hospital",
        "LEAK",
        "Column name states the observation window is the hospitalization.",
    ),
    "Acute.Kidney.Injury..during.hospitalization.": (
        "in_hospital",
        "LEAK",
        "Column name states the observation window is the hospitalization.",
    ),
    "kidney_replacement_therapy": (
        "in_hospital",
        "LEAK",
        "In-hospital dialysis (70 Yes). A treatment given in response to deterioration.",
    ),
    "therapeutic.exnox.Boolean": (
        "in_hospital",
        "LEAK",
        "Therapeutic enoxaparin started during the stay in response to clinical course.",
    ),
    "therapeutic.heparin.Boolean": (
        "in_hospital",
        "LEAK",
        "Therapeutic heparin started during the stay in response to clinical course.",
    ),
    "Other.anticoagulation.therapy": (
        "in_hospital",
        "LEAK",
        "In-hospital anticoagulation agent (8 levels, 'not documented' for 1155).",
    ),
    # --- baseline comorbidities and home medications ---
    "kidney_transplant": (
        "comorbidity",
        "DROP",
        "98.4% null, only 22 Yes. Too sparse to estimate anything from.",
    ),
    "antibiotics_use_v": (
        "comorbidity",
        "DROP",
        "Ambiguous timing, unresolved without data dictionary -- excluded conservatively. "
        "The _v suffix suggests baseline/home use, but if this records in-hospital "
        "administration it is a response to severity. Promote to SAFE only after the TCIA "
        "documentation confirms it is a pre-admission value.",
    ),
    "nsaid_use_v": (
        "comorbidity",
        "DROP",
        "Ambiguous timing, unresolved without data dictionary -- excluded conservatively. "
        "Same concern as antibiotics_use_v.",
    ),
    "hf_ef_v": ("comorbidity", "SAFE", "Heart failure phenotype: No / HFpEF / HFrEF. Baseline history."),
    "smoking_status_v": ("comorbidity", "SAFE", "Never 764 / Former 250 / Current 36. Baseline history."),
    # --- symptoms at presentation ---
    "days_prior_sx": ("symptom", "SAFE", "Days of symptoms before presentation. Reported at admission."),
    "dyspnea_admission_v": ("symptom", "SAFE", "Dyspnea recorded at admission."),
    # --- labs with no numeric counterpart ---
    "Urine.protein": ("lab_binned", "SAFE", "Normal/Abnormal, 60.3% null. No numeric counterpart in this file."),
    "Microscopic_hematuria.above2": (
        "lab_binned",
        "SAFE",
        "Threshold flag with no numeric counterpart in this file.",
    ),
    "Proteinuria.above80": (
        "lab_binned",
        "SAFE",
        "Threshold flag with no numeric counterpart in this file. 84.5% null.",
    ),
    # --- post-imputation near-constant: >75% null before imputation ---
    # After median/mode filling these are dominated by a single value, so any model
    # weight on them reflects the imputation choice rather than physiology.
    "Proteinuria.above80": (
        "lab_binned",
        "DROP",
        "Post-imputation near-constant, >75% missing pre-impute (84.5%).",
    ),
    "33254-4_pH of Arterial blood adjusted to patient's actual temperature": (
        "lab_numeric",
        "DROP",
        "Post-imputation near-constant, >75% missing pre-impute (83.5%). Arterial blood gas "
        "is ordered selectively, so the missingness is also informative in the wrong direction.",
    ),
    "13457-7_Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation": (
        "lab_numeric",
        "DROP",
        "Post-imputation near-constant, >75% missing pre-impute (78.2%).",
    ),
    "13458-5_Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation": (
        "lab_numeric",
        "DROP",
        "Post-imputation near-constant, >75% missing pre-impute (78.2%).",
    ),
    "2085-9_Cholesterol in HDL [Mass/volume] in Serum or Plasma": (
        "lab_numeric",
        "DROP",
        "Post-imputation near-constant, >75% missing pre-impute (78.0%).",
    ),
    "2571-8_Triglyceride [Mass/volume] in Serum or Plasma": (
        "lab_numeric",
        "DROP",
        "Post-imputation near-constant, >75% missing pre-impute (77.2%).",
    ),
    # The blood_pH threshold flags must follow their source to DROP. Left SAFE, they
    # would stop being redundant (their source is no longer SAFE) and would silently
    # re-enter the feature set carrying the same 83.5%-missing signal.
    "blood_pH.above7.45": (
        "lab_binned",
        "DROP",
        "Threshold flag of a DROP column (33254-4 arterial pH, >75% missing pre-impute). "
        "Dropped with its source rather than allowed back in as a non-redundant column.",
    ),
    "blood_pH.between7.35and7.45": (
        "lab_binned",
        "DROP",
        "Threshold flag of a DROP column (33254-4 arterial pH, >75% missing pre-impute). "
        "Dropped with its source rather than allowed back in as a non-redundant column.",
    ),
    "blood_pH.below7.35": (
        "lab_binned",
        "DROP",
        "Threshold flag of a DROP column (33254-4 arterial pH, >75% missing pre-impute). "
        "Dropped with its source rather than allowed back in as a non-redundant column.",
    ),
    # --- exact duplicate ---
    "2951-2_Sodium [Moles/volume] in Serum or Plasma.1": (
        "lab_numeric",
        "DROP",
        "Byte-identical duplicate of '2951-2_Sodium [Moles/volume] in Serum or Plasma' "
        "(verified with Series.equals). The .1 suffix is a pandas rename of a repeated header.",
    ),
}

COMORBIDITIES = {
    "htn_v": "hypertension",
    "dm_v": "diabetes mellitus",
    "cad_v": "coronary artery disease",
    "ckd_v": "chronic kidney disease",
    "malignancies_v": "malignancy",
    "copd_v": "COPD",
    "other_lung_disease_v": "other chronic lung disease",
    "acei_v": "home ACE inhibitor",
    "arb_v": "home ARB",
}

SYMPTOMS = {
    "cough_v": "cough",
    "nausea_v": "nausea",
    "vomiting_v": "vomiting",
    "diarrhea_v": "diarrhea",
    "abdominal_pain_v": "abdominal pain",
    "fever_v": "fever",
}


def bin_source_prefix(column: str) -> str | None:
    """Return the numeric-source prefix for a binned column, or None."""
    if column in EXPLICIT:
        return None
    for prefix, source in BIN_SOURCES.items():
        if column.startswith(prefix):
            return source
    return None


def resolve_source(prefix: str, columns: list[str], statuses: dict[str, str]) -> str:
    """Resolve a source prefix to exactly one non-DROP column name.

    The DROP filter matters: the Sodium prefix matches both the real column and
    its byte-identical '.1' duplicate, and the bins were cut from the original.
    """
    candidates = [c for c in columns if c.startswith(prefix) and statuses[c] != "DROP"]
    assert len(candidates) == 1, f"prefix {prefix!r} resolved to {len(candidates)} columns: {candidates}"
    return candidates[0]


def classify(column: str) -> tuple:
    if column in EXPLICIT:
        return EXPLICIT[column]

    if column in COMORBIDITIES:
        return ("comorbidity", "SAFE", f"Baseline {COMORBIDITIES[column]} history, known before admission.")

    if column in SYMPTOMS:
        return ("symptom", "SAFE", f"{SYMPTOMS[column].capitalize()} reported at presentation.")

    # binned vitals/labs: timing-safe, but each duplicates a numeric column
    for prefix, source in BIN_SOURCES.items():
        if column.startswith(prefix):
            return (
                "lab_binned",
                "SAFE",
                f"Threshold flag derived from '{source}'. Timing-safe but redundant with that "
                "numeric column -- pick one representation, do not model both.",
            )

    # everything remaining in this file is a LOINC-coded numeric vital or lab
    return (
        "lab_numeric",
        "SAFE",
        "LOINC-coded vital or laboratory value at presentation.",
    )


def build_spec(path: str = CLINICAL_CSV) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()

    records = []
    for column in columns:
        category, status, reason = classify(column)
        records.append(
            {"column_name": column, "category": category, "status": status, "reason": reason}
        )
    spec = pd.DataFrame(records)

    # second pass: resolve each binned column's source to an exact column name
    statuses = dict(zip(spec["column_name"], spec["status"]))
    spec["redundant_with"] = [
        resolve_source(prefix, columns, statuses) if (prefix := bin_source_prefix(c)) else ""
        for c in spec["column_name"]
    ]

    # coverage guarantees: every source column classified exactly once, no stray entries
    assert len(spec) == len(columns), "spec length does not match source column count"
    assert spec["column_name"].is_unique, "duplicate column_name in spec"
    assert set(spec["column_name"]) == set(columns), "spec/source column mismatch"
    assert spec["status"].isin({"SAFE", "LEAK", "LEAK_RISK", "DROP"}).all(), "invalid status value"
    assert spec[["category", "reason"]].notna().all().all(), "missing category or reason"

    # every named source must itself exist and be SAFE, or dropping the bin would
    # discard information rather than deduplicate it
    named = spec.loc[spec["redundant_with"] != "", "redundant_with"]
    assert named.isin(columns).all(), "redundant_with names a column absent from the source CSV"
    assert named.map(statuses).eq("SAFE").all(), "redundant_with names a non-SAFE column"

    return spec


if __name__ == "__main__":
    spec = build_spec()
    spec.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(spec)} columns)")
    print()
    print(spec["status"].value_counts().to_string())
    print()
    print(pd.crosstab(spec["category"], spec["status"]).to_string())
