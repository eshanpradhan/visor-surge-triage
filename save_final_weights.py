"""Train the final models once and save their parameters to models/.

Purpose: let anyone who clones the repo run inference without retraining or
redownloading the 2.1 GB image archive. Nothing patient-level is written here --
only fitted parameters and aggregate preprocessing statistics.

Which models get saved, and why these ones
------------------------------------------
Everything is fit on the **train split alone**, not train+val. That is a
deliberate trade of a little accuracy for a valid calibrator: the isotonic
calibrator has to be fitted on predictions the model made out-of-sample, and val
is the only split available for that. Saving the train+val model from the test
table would mean shipping a calibrator fitted against a different model's
outputs, which is the failure mode calibrate.py exists to avoid.

Consequence to document wherever these weights are used: the saved fusion model
sees 955 patients rather than 1160, so its discrimination is marginally below the
test-table figure. It is the only configuration that emits trustworthy
*probabilities* rather than ranking scores.

The ResNet-50 backbone below layer4 is frozen at ImageNet initialization and is
NOT saved -- torchvision downloads those weights on demand. Only the parameters
that actually changed during training are written, which keeps the checkpoint
well under GitHub's file size limits.
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

import fusion_model as FM
from features import build_feature_frame
from impute import apply_impute_stats, fit_impute_stats

SEED = 42
CALENDAR_COLUMN = "visit_start_datetime"
MODELS_DIR = "models"

# parameters that are trained; everything else stays at ImageNet initialization
TRAINABLE_PREFIXES = ("backbone.layer4.", "image_projection.", "clinical_branch.", "head.")


def encoder_state(encoder) -> dict:
    """Serialize the fitted ClinicalEncoder so inference can reproduce the transform."""
    return {
        "numeric_columns": encoder.numeric_columns,
        "log1p_columns": encoder.log1p_columns,
        "ordinal_columns": encoder.ordinal_columns,
        "onehot_levels": encoder.onehot_levels,
        "date_columns": encoder.date_columns,
        "month_buckets": encoder.month_buckets,
        "feature_names": encoder.feature_names,
        "means": encoder.means.tolist(),
        "stds": encoder.stds.tolist(),
    }


def main() -> None:
    from pathlib import Path

    out = Path(MODELS_DIR)
    out.mkdir(exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = FM.pick_device()

    pool = build_feature_frame(split=None)
    train = pool[pool["split"] == "train"].reset_index(drop=True)
    val = pool[pool["split"] == "val"].reset_index(drop=True)
    assert not set(train["patient_id"]) & set(val["patient_id"]), "train/val overlap"

    print(f"=== SAVING FINAL WEIGHTS (device={device}, seed={SEED}) ===")
    print(f"  fit on train n={len(train)}, val n={len(val)} reserved for calibration")
    print()

    # ---- fusion, no calendar -------------------------------------------------
    prepared, encoder = FM.prepare_fold(train, {"val": val}, (CALENDAR_COLUMN,))
    fit_imputed, fit_clinical = prepared["__fit__"]
    val_imputed, val_clinical = prepared["val"]

    model, info = FM.train_fold(
        fit_imputed, fit_clinical, {}, device, FM.MODALITY_DROPOUT_P, SEED, True
    )
    print(f"  fusion trained, best epoch {info['best_epoch']}")

    # stored in fp16: halves the file from 60.5 MB to 30.3 MB, which matters for
    # clone and deploy footprint. Verified to leave demo-case probabilities and
    # tiers identical, and val/test AUPRC and Brier unchanged to 3 dp; val AUROC
    # moves 0.796 -> 0.795 through a single tie flip, with no patient's predicted
    # probability shifting by more than 1e-4. Loaders cast back to fp32.
    trainable = {
        key: value.cpu().half()
        for key, value in model.state_dict().items()
        if key.startswith(TRAINABLE_PREFIXES)
    }
    frozen_count = len(model.state_dict()) - len(trainable)
    torch.save(trainable, out / "fusion_no_calendar.pt")
    print(f"  saved {len(trainable)} trained tensors "
          f"({frozen_count} frozen ImageNet tensors omitted)")

    # ---- isotonic calibrator, fitted on held-out val -------------------------
    loader = FM.make_loader(val_imputed, val_clinical, False, {}, False, True)
    val_logits, y_val = FM.predict_scores(model, loader, device)
    val_logits = np.asarray(val_logits, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    isotonic = IsotonicRegression(out_of_bounds="clip").fit(val_logits, y_val)
    with open(out / "calibrator_isotonic.json", "w") as fh:
        json.dump(
            {
                "method": "isotonic",
                "fitted_on": "val split, out-of-sample for this model",
                "x_thresholds": isotonic.X_thresholds_.tolist(),
                "y_thresholds": isotonic.y_thresholds_.tolist(),
                "input": "raw model logit",
            },
            fh,
            indent=2,
        )
    print("  saved isotonic calibrator (fitted on held-out val logits)")
    del model

    # ---- clinical LightGBM, both variants ------------------------------------
    # imported here, after all torch work is finished, because LightGBM and torch
    # cannot share a process safely on macOS -- see clinical_encoding.py
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-u", "-c", CLINICAL_SUBPROCESS], capture_output=True, text=True
    )
    print(result.stdout.strip())
    assert result.returncode == 0, f"clinical model subprocess failed:\n{result.stderr}"

    # ---- preprocessing artifacts --------------------------------------------
    stats = fit_impute_stats(train.drop(columns=[CALENDAR_COLUMN]), require_train_split=True)
    with open(out / "impute_stats_final.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    with open(out / "encoder_state.json", "w") as fh:
        json.dump(encoder_state(encoder), fh, indent=2)

    with open(out / "MODEL_CARD.json", "w") as fh:
        json.dump(
            {
                "fusion_no_calendar": {
                    "file": "fusion_no_calendar.pt",
                    "contents": "trained tensors only; ResNet-50 below layer4 is frozen "
                                "at ImageNet init and loaded from torchvision",
                    "fit_on": "train split (n=955)",
                    "best_epoch": int(info["best_epoch"]),
                    "modality_dropout": FM.MODALITY_DROPOUT_P,
                    "excluded_feature": CALENDAR_COLUMN,
                    "seed": SEED,
                    "dtype": "float16 (cast to float32 on load)",
                    "fp16_note": "Stored half-precision to halve clone/deploy footprint. "
                                 "Demo-case probabilities and tiers identical to fp32; "
                                 "val/test AUPRC and Brier unchanged to 3 dp; val AUROC "
                                 "0.796 -> 0.795 via one tie flip. Max per-patient "
                                 "probability change 9.2e-05.",
                },
                "calibrator": {
                    "file": "calibrator_isotonic.json",
                    "applies_to": "fusion_no_calendar.pt only",
                    "note": "fitted on val predictions from this exact model; do not "
                            "apply to a model trained on different rows",
                },
                "clinical_lightgbm": {
                    "files": ["clinical_lightgbm_naive.txt",
                              "clinical_lightgbm_calendar_ablated.txt"],
                    "fit_on": "train split (n=955)",
                },
                "preprocessing": {
                    "impute_stats_final.json": "train medians/modes, calendar column dropped",
                    "encoder_state.json": "fitted ClinicalEncoder: vocabularies, log1p "
                                          "selection, standardization",
                },
                "contains_patient_data": False,
            },
            fh,
            indent=2,
        )
    print("  saved preprocessing artifacts and MODEL_CARD.json")
    print()

    sizes = sorted(
        ((p.stat().st_size / 1e6, p.name) for p in out.iterdir()), reverse=True
    )
    print("=== models/ contents ===")
    for size, name in sizes:
        print(f"  {size:8.2f} MB  {name}")
    print(f"  {sum(s for s, _ in sizes):8.2f} MB  TOTAL")


CLINICAL_SUBPROCESS = """
import pandas as pd
import baseline as B
from features import build_feature_frame

pool = build_feature_frame(split=None)
train = pool[pool["split"] == "train"].reset_index(drop=True)

for name, drop in [("naive", False), ("calendar_ablated", True)]:
    frame = train.drop(columns=["visit_start_datetime"]) if drop else train
    matrices, _ = B.prepare(frame, {}, strict=True)
    x, y = matrices["__fit__"]
    booster = B.fit_models(x, y)["lightgbm"]
    booster.save_model(f"models/clinical_lightgbm_{name}.txt")
    print(f"  saved clinical_lightgbm_{name}.txt")
"""


if __name__ == "__main__":
    main()
