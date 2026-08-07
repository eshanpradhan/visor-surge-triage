"""Torch-free LightGBM scoring service, invoked as a subprocess by app.py.

LightGBM cannot run in a process that has imported torch: on macOS the two
OpenMP runtimes collide and any call into the booster segfaults, including a
single-row predict. The Streamlit app needs torch for the fusion model and
Grad-CAM, so the clinical comparison model is scored here instead and the result
is passed back as JSON on stdout.

Usage:
    python3 clinical_service.py 107 16 126 176 19

Emits, for each requested test-split row index, the LightGBM probability and its
exact TreeSHAP contributions. Nothing is written to disk -- the output contains
per-patient values and is held in the caller's memory only.
"""

import json
import sys

import baseline as B
from features import build_feature_frame

CALENDAR_COLUMN = "visit_start_datetime"
BOOSTER_PATH = "models/clinical_lightgbm_calendar_ablated.txt"


def main() -> None:
    assert "torch" not in sys.modules, "torch is loaded; LightGBM will segfault"
    import lightgbm as lgb

    indices = [int(a) for a in sys.argv[1:]]
    assert indices, "no row indices given"

    pool = build_feature_frame(split=None)
    train = pool[pool["split"] == "train"].reset_index(drop=True)
    test = pool[pool["split"] == "test"].reset_index(drop=True)

    # identical preprocessing to fusion_model.prepare_fold: fit on train, drop
    # the calendar column, apply to test
    matrices, encoder = B.prepare(
        train.drop(columns=[CALENDAR_COLUMN]),
        {"test": test.drop(columns=[CALENDAR_COLUMN])},
        strict=True,
    )
    x_test, _y_test = matrices["test"]

    booster = lgb.Booster(model_file=BOOSTER_PATH)
    assert booster.num_feature() == x_test.shape[1], (
        f"booster expects {booster.num_feature()} features, matrix has {x_test.shape[1]}"
    )

    rows = x_test[indices]
    probabilities = booster.predict(rows)
    contributions = booster.predict(rows, pred_contrib=True)

    json.dump(
        {
            "feature_names": encoder.feature_names,
            "patients": {
                str(index): {
                    "probability": float(probabilities[position]),
                    # final column is the model's base value, not a feature
                    "contributions": contributions[position][:-1].tolist(),
                }
                for position, index in enumerate(indices)
            },
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
