"""Calendar-free ablations: fusion vs clinical-only MLP, matched protocol.

The first fusion run put `visit_start_datetime` in the clinical branch, so its
0.826 was comparable to the naive clinical 0.843, not to the calendar-ablated
0.804. These two runs fix that and separate the two things that were confounded:

  A. fusion, no calendar          -> comparable to 0.804 directly
  B. clinical-only MLP, no calendar -> same architecture as A's clinical branch,
     image branch removed entirely, so A-minus-B isolates the modality-fusion
     contribution from the MLP-vs-LightGBM architecture difference

Both keep modality dropout settings consistent with the settled ablation
(p=0.2 for fusion; it is inert and forced to 0.0 when there is no image branch).
The calendar probe runs on held-out representations from each run.
"""

import numpy as np
import pandas as pd

from dataset import ClinicalEncoder
from features import build_feature_frame
from fusion_model import (
    MODALITY_DROPOUT_P,
    calendar_probe,
    pick_device,
    run_cv,
)
from impute import apply_impute_stats, fit_impute_stats

CALENDAR_COLUMN = "visit_start_datetime"


def report(name: str, table: pd.DataFrame, fused, frame, buckets_source) -> dict:
    print()
    print(f"=== PER-FOLD: {name} ===")
    print(table.to_string(index=False))
    summary = {
        "model": name,
        "cv_auroc": table.hold_auroc.mean(),
        "cv_auroc_sd": table.hold_auroc.std(ddof=0),
        "cv_auprc": table.hold_auprc.mean(),
        "cv_auprc_sd": table.hold_auprc.std(ddof=0),
        "mean_gap": table.gap.mean(),
    }
    print(
        f"  CV AUROC {summary['cv_auroc']:.3f} +/- {summary['cv_auroc_sd']:.3f}   "
        f"CV AUPRC {summary['cv_auprc']:.3f} +/- {summary['cv_auprc_sd']:.3f}   "
        f"mean gap {summary['mean_gap']:+.3f}"
    )

    buckets = buckets_source(frame)
    mean_auroc, sd = calendar_probe(fused, buckets)
    summary["probe_auroc"] = mean_auroc
    summary["probe_sd"] = sd
    print(f"  calendar probe on representation: AUROC {mean_auroc:.3f} +/- {sd:.3f}")
    return summary


def main() -> None:
    pd.set_option("display.width", 220)
    device = pick_device()

    pool = build_feature_frame(split=None)
    pool = pool[pool["split"].isin(["train", "val"])].reset_index(drop=True)
    assert pool["patient_id"].is_unique, "duplicate patient in CV pool"
    assert CALENDAR_COLUMN in pool.columns, f"{CALENDAR_COLUMN} not in feature frame"
    cache: dict = {}

    # bucket labels come from the undropped pool, so the probe still has ground
    # truth for a column the models never saw
    bucket_encoder = ClinicalEncoder().fit(
        apply_impute_stats(pool, fit_impute_stats(pool, require_train_split=False))
    )

    def buckets_for(frame: pd.DataFrame) -> np.ndarray:
        return bucket_encoder._bucket_months(
            frame[CALENDAR_COLUMN], CALENDAR_COLUMN
        ).to_numpy()

    print(f"=== CALENDAR-FREE ABLATIONS (device={device}) ===")
    print(f"  dropping '{CALENDAR_COLUMN}' before imputation and encoding")
    print("  benchmarks: clinical-only LightGBM, calendar-ablated 0.804 | image-only 0.756")
    print()

    print(f"--- A. FUSION, no calendar (modality dropout p={MODALITY_DROPOUT_P}) ---")
    fusion_table, fusion_repr, fusion_frame = run_cv(
        pool, cache, device, MODALITY_DROPOUT_P,
        probe=True, drop_columns=(CALENDAR_COLUMN,), use_image=True,
    )

    print()
    print("--- B. CLINICAL-ONLY MLP, no calendar (image branch removed) ---")
    clinical_table, clinical_repr, clinical_frame = run_cv(
        pool, cache, device, 0.0,
        probe=True, drop_columns=(CALENDAR_COLUMN,), use_image=False,
    )

    summaries = [
        report("A. fusion, no calendar", fusion_table, fusion_repr, fusion_frame, buckets_for),
        report("B. clinical-only MLP, no calendar", clinical_table, clinical_repr,
               clinical_frame, buckets_for),
    ]

    print()
    print("=== FULL BENCHMARK TABLE ===")
    print(f"  {'model':38s} {'CV AUROC':>16s}  {'CV AUPRC':>16s}  {'gap':>7s}  {'probe':>7s}")
    print(f"  {'clinical LightGBM, naive':38s} {'0.843':>16s}  {'0.635':>16s}  {'-':>7s}  {'-':>7s}")
    print(f"  {'clinical LightGBM, calendar-ablated':38s} {'0.804':>16s}  {'-':>16s}  {'-':>7s}  {'-':>7s}")
    print(f"  {'image-only, layer4 (3 seeds)':38s} {'0.756 +/- 0.003':>16s}  {'0.457':>16s}  {'-':>7s}  {'0.636':>7s}")
    print(f"  {'fusion, WITH calendar':38s} {'0.826 +/- 0.011':>16s}  {'0.603':>16s}  {'+0.103':>7s}  {'0.828':>7s}")
    for summary in summaries:
        print(
            f"  {summary['model']:38s} "
            f"{summary['cv_auroc']:.3f} +/- {summary['cv_auroc_sd']:.3f}  "
            f"{summary['cv_auprc']:.3f} +/- {summary['cv_auprc_sd']:.3f}  "
            f"{summary['mean_gap']:+7.3f}  {summary['probe_auroc']:7.3f}"
        )

    fusion_auroc = summaries[0]["cv_auroc"]
    clinical_auroc = summaries[1]["cv_auroc"]
    print()
    print("=== THE TWO COMPARISONS THAT MATTER ===")
    print(f"  fusion - clinical MLP  (isolates modality fusion):    {fusion_auroc - clinical_auroc:+.3f}")
    print(f"  fusion - clinical LGBM (0.804, cross-architecture):   {fusion_auroc - 0.804:+.3f}")
    print(f"  clinical MLP - clinical LGBM (architecture only):     {clinical_auroc - 0.804:+.3f}")

    fusion_table.to_csv("ablation_fusion_no_calendar.csv", index=False)
    clinical_table.to_csv("ablation_clinical_mlp_no_calendar.csv", index=False)
    print()
    print("  wrote ablation_fusion_no_calendar.csv, ablation_clinical_mlp_no_calendar.csv")


if __name__ == "__main__":
    main()
