"""3-seed CV for the two calendar-free ablations, with a paired comparison.

The single-seed run put fusion at +0.019 AUROC over its matched clinical-only
MLP, which was inside fold-level variance and so could not be distinguished from
fold luck. This repeats both arms at seeds 42, 7 and 2024 and pairs them *by
seed*: within a seed both arms see identical fold assignments, so the difference
isolates the model rather than the partition.

Seeds drive fold assignment, the inner early-stopping split, weight init and
augmentation order together, matching the image-only multi-seed protocol.
"""

import numpy as np
import pandas as pd

from features import build_feature_frame
from fusion_model import MODALITY_DROPOUT_P, pick_device, run_cv

CALENDAR_COLUMN = "visit_start_datetime"
SEEDS = (42, 7, 2024)


def main() -> None:
    pd.set_option("display.width", 220)
    device = pick_device()

    pool = build_feature_frame(split=None)
    pool = pool[pool["split"].isin(["train", "val"])].reset_index(drop=True)
    assert pool["patient_id"].is_unique, "duplicate patient in CV pool"
    cache: dict = {}

    print(f"=== 3-SEED CALENDAR-FREE ABLATIONS (device={device}, seeds={list(SEEDS)}) ===")
    print()

    records = []
    for seed in SEEDS:
        for name, use_image, dropout in [
            ("fusion", True, MODALITY_DROPOUT_P),
            ("clinical_mlp", False, 0.0),
        ]:
            table, _, _ = run_cv(
                pool, cache, device, dropout,
                seed=seed, probe=False,
                drop_columns=(CALENDAR_COLUMN,), use_image=use_image,
            )
            records.append(
                {
                    "seed": seed,
                    "model": name,
                    "cv_auroc": table.hold_auroc.mean(),
                    "cv_auprc": table.hold_auprc.mean(),
                    "fold_sd": table.hold_auroc.std(ddof=0),
                    "folds": table.hold_auroc.tolist(),
                }
            )
            print(f"  seed {seed} {name:13s} CV AUROC {records[-1]['cv_auroc']:.3f}  "
                  f"AUPRC {records[-1]['cv_auprc']:.3f}  folds {records[-1]['folds']}",
                  flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv("multiseed_ablations.csv", index=False)

    print()
    print("=== ACROSS-SEED SUMMARY ===")
    for name in ["fusion", "clinical_mlp"]:
        rows = frame[frame.model == name]
        print(
            f"  {name:13s} AUROC {rows.cv_auroc.mean():.3f} +/- {rows.cv_auroc.std(ddof=1):.3f}   "
            f"AUPRC {rows.cv_auprc.mean():.3f} +/- {rows.cv_auprc.std(ddof=1):.3f}   "
            f"mean within-seed fold sd {rows.fold_sd.mean():.3f}"
        )

    print()
    print("=== PAIRED BY SEED (identical folds within each seed) ===")
    wide = frame.pivot(index="seed", columns="model", values="cv_auroc")
    wide["delta_auroc"] = wide["fusion"] - wide["clinical_mlp"]
    wide_pr = frame.pivot(index="seed", columns="model", values="cv_auprc")
    wide["delta_auprc"] = wide_pr["fusion"] - wide_pr["clinical_mlp"]
    print(wide.round(4).to_string())

    deltas = wide["delta_auroc"]
    print()
    print(
        f"  fusion - clinical MLP:  AUROC {deltas.mean():+.4f} +/- {deltas.std(ddof=1):.4f}  "
        f"(seeds favouring fusion: {int((deltas > 0).sum())}/{len(deltas)})"
    )
    print(
        f"                          AUPRC {wide['delta_auprc'].mean():+.4f} +/- "
        f"{wide['delta_auprc'].std(ddof=1):.4f}"
    )
    if len(deltas) > 1 and deltas.std(ddof=1) > 0:
        from scipy.stats import ttest_rel

        stat, pvalue = ttest_rel(wide["fusion"], wide["clinical_mlp"])
        print(f"  paired t-test over {len(deltas)} seeds: t={stat:.2f}, p={pvalue:.3f}")
        print("  (n=3 pairs -- reported for completeness, not as a strong significance claim)")
    print()
    print("  wrote multiseed_ablations.csv")


if __name__ == "__main__":
    main()
