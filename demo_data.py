"""Synthetic demo patients for the public deployment. No real patient data.

Why this exists
---------------
app.py normally scores real patients from the held-out test split, which lives
in gitignored files under the data-use terms. A deployed copy of the dashboard
has neither those files nor permission to show them, so demo mode substitutes:

* three public NIH ChestX-ray14 sample images (`demo_assets/`), which are
  unrelated to the SBU cohort and carry no COVID severity label, and
* clinical values invented by hand to span a plausible clinical range.

Nothing here corresponds to a real person. The vitals and labs below were
written to illustrate a mild, an intermediate and a severe-looking presentation;
they are not drawn from, sampled from, or fitted to any patient record.

What the resulting scores mean
------------------------------
Very little. The model was trained on SBU radiographs to predict escalation in
that cohort; an NIH image paired with invented labs is out of distribution on
both inputs. The numbers demonstrate that the pipeline runs end to end and that
the interface behaves. They are not evidence about the model's accuracy, and the
app says so in a banner that cannot be dismissed.

Unspecified variables are filled from the committed train-split imputation
statistics, the same values the real pipeline would use for a missing entry.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DEMO_ASSETS = ROOT / "demo_assets"

# Column names as they appear in the SBU schema, so the saved encoder can consume
# them unchanged. Values are invented.
DEMO_PATIENTS = [
    {
        "demo_id": "DEMO-A",
        "summary": "68-year-old, hypertensive, mild presentation",
        "image": "00000017_001.png",
        "narrative": "Room-air saturation preserved, unremarkable inflammatory markers.",
        "values": {
            "age.splits": "(59,74]",
            "gender_concept_name": "FEMALE",
            "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": 97.0,
            "9279-1_Respiratory rate": 18.0,
            "76282-3_Heart rate.beat-to-beat by EKG": 78.0,
            "8331-1_Oral temperature": 36.9,
            "8480-6_Systolic blood pressure": 134.0,
            "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": 1.8,
            "731-0_Lymphocytes [#/volume] in Blood by Automated count": 1.6,
            "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay": 320.0,
            "2524-7_Lactate [Moles/volume] in Serum or Plasma": 1.1,
            "htn_v": "Yes",
            "dm_v": "No",
            "copd_v": "No",
            "dyspnea_admission_v": "No",
            "days_prior_sx": 3.0,
        },
    },
    {
        "demo_id": "DEMO-B",
        "summary": "55-year-old, diabetic, intermediate presentation",
        "image": "00000032_001.png",
        "narrative": "Mild hypoxia and tachypnoea with a raised inflammatory profile.",
        "values": {
            "age.splits": "[18,59]",
            "gender_concept_name": "MALE",
            "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": 93.0,
            "9279-1_Respiratory rate": 24.0,
            "76282-3_Heart rate.beat-to-beat by EKG": 102.0,
            "8331-1_Oral temperature": 38.2,
            "8480-6_Systolic blood pressure": 128.0,
            "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": 14.5,
            "731-0_Lymphocytes [#/volume] in Blood by Automated count": 0.8,
            "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay": 1450.0,
            "2524-7_Lactate [Moles/volume] in Serum or Plasma": 2.1,
            "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay": 0.6,
            "htn_v": "Yes",
            "dm_v": "Yes",
            "copd_v": "No",
            "dyspnea_admission_v": "Yes",
            "days_prior_sx": 7.0,
        },
    },
    {
        "demo_id": "DEMO-C",
        "summary": "80-year-old, COPD, severe-looking presentation",
        "image": "00000013_005.png",
        "narrative": "Marked hypoxia, tachypnoea, lymphopenia and a high D-dimer.",
        "values": {
            "age.splits": "(74,90]",
            "gender_concept_name": "MALE",
            "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": 86.0,
            "9279-1_Respiratory rate": 32.0,
            "76282-3_Heart rate.beat-to-beat by EKG": 118.0,
            "8331-1_Oral temperature": 38.8,
            "8480-6_Systolic blood pressure": 104.0,
            "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": 28.0,
            "731-0_Lymphocytes [#/volume] in Blood by Automated count": 0.4,
            "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay": 4200.0,
            "2524-7_Lactate [Moles/volume] in Serum or Plasma": 3.4,
            "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay": 1.9,
            "2276-4_Ferritin [Mass/volume] in Serum or Plasma": 1850.0,
            "htn_v": "Yes",
            "dm_v": "Yes",
            "copd_v": "Yes",
            "dyspnea_admission_v": "Yes",
            "days_prior_sx": 9.0,
        },
    },
]

# Clinically plausible sweep ranges, chosen to span what an admitting clinician
# might actually see rather than the full numeric range of the training data.
# A feature without an entry here is not offered as a slider.
SWEEP_RANGES = {
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": (80, 100, "%", "SpO₂"),
    "9279-1_Respiratory rate": (12, 40, "/min", "Respiratory rate"),
    "76282-3_Heart rate.beat-to-beat by EKG": (50, 140, "bpm", "Heart rate"),
    "8331-1_Oral temperature": (35.5, 40.0, "°C", "Temperature"),
    "8480-6_Systolic blood pressure": (80, 180, "mmHg", "Systolic BP"),
    "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": (0.1, 40, "mg/dL", "CRP"),
    "731-0_Lymphocytes [#/volume] in Blood by Automated count": (0.1, 3.0, "K/µL", "Lymphocytes"),
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay":
        (200, 6000, "ng/mL", "D-dimer"),
    "2524-7_Lactate [Moles/volume] in Serum or Plasma": (0.5, 6.0, "mmol/L", "Lactate"),
    "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay":
        (0.02, 5.0, "ng/mL", "Procalcitonin"),
    "2276-4_Ferritin [Mass/volume] in Serum or Plasma": (50, 3000, "ng/mL", "Ferritin"),
    "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma":
        (10, 400, "U/L", "AST"),
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma": (0.4, 4.0, "mg/dL", "Creatinine"),
    "39156-5_Body mass index (BMI) [Ratio]": (18, 45, "kg/m²", "BMI"),
    "days_prior_sx": (0, 21, "days", "Days of symptoms"),
}

# How many slider features and how many positions each. The static export
# precomputes a 7^5 joint grid from these; the live app uses the same ranges but
# calls the model directly, so the step count only controls slider granularity.
SWEEP_FEATURES = 5
SWEEP_STEPS = 7

DISCLAIMER = (
    "**Demonstration mode — synthetic data.** These are not real patients. The "
    "radiographs are public NIH ChestX-ray14 samples unrelated to the training cohort, "
    "and the clinical values are invented. Scores show that the pipeline runs; they are "
    "not clinically meaningful and are not evidence about model accuracy. Reported "
    "performance comes from the held-out SBU test split, not from anything shown here."
)


def load_saved_encoder():
    """Rebuild the fitted ClinicalEncoder from models/encoder_state.json.

    Demo mode has no access to the training data, so the encoder cannot be refit;
    it is restored from the statistics saved at training time.
    """
    from clinical_encoding import ClinicalEncoder

    with open(MODELS / "encoder_state.json") as fh:
        state = json.load(fh)

    encoder = ClinicalEncoder()
    encoder.numeric_columns = state["numeric_columns"]
    encoder.log1p_columns = state["log1p_columns"]
    encoder.ordinal_columns = state["ordinal_columns"]
    encoder.onehot_levels = state["onehot_levels"]
    encoder.date_columns = state["date_columns"]
    encoder.month_buckets = state["month_buckets"]
    encoder.feature_names = state["feature_names"]
    encoder.means = np.asarray(state["means"], dtype=float)
    encoder.stds = np.asarray(state["stds"], dtype=float)
    return encoder


def build_demo_frame() -> pd.DataFrame:
    """One row per demo patient, every encoder input present.

    Values the demo does not specify are filled from the committed train-split
    medians and modes -- the same defaults the real pipeline applies to a missing
    entry, so an unspecified variable behaves as it would in production.
    """
    encoder = load_saved_encoder()
    with open(MODELS / "impute_stats_final.json") as fh:
        stats = json.load(fh)
    defaults = {**stats["numeric_medians"], **stats["categorical_modes"]}

    required = (
        list(encoder.numeric_columns)
        + list(encoder.ordinal_columns)
        + list(encoder.onehot_levels)
    )

    rows = []
    for patient in DEMO_PATIENTS:
        unknown = set(patient["values"]) - set(required)
        assert not unknown, f"{patient['demo_id']} sets unknown columns: {sorted(unknown)}"

        row = {column: defaults.get(column) for column in required}
        missing = [c for c, v in row.items() if v is None]
        assert not missing, f"no default available for: {missing[:3]}"

        row.update(patient["values"])
        row["patient_id"] = patient["demo_id"]
        row["severe"] = False  # no ground truth exists for these images
        row["filepath"] = str(DEMO_ASSETS / patient["image"])
        rows.append(row)

    return pd.DataFrame(rows), encoder


if __name__ == "__main__":
    frame, encoder = build_demo_frame()
    matrix = encoder.transform(frame)
    print(f"demo patients: {len(frame)}   encoded: {matrix.shape}")
    print(f"finite: {np.isfinite(matrix).all()}")
    for _, row in frame.iterrows():
        print(f"  {row['patient_id']}: {Path(row['filepath']).name}  "
              f"SpO2={row['59408-5_Oxygen saturation in Arterial blood by Pulse oximetry']}")
