"""Test-set stage 1: clinical LightGBM models. Torch is never imported here.

Split out from the torch models because LightGBM and PyTorch each bundle their
own OpenMP runtime. In one process on macOS the combination fails two ways: with
torch imported first a multithreaded lgb.train segfaults immediately, and with
LightGBM imported first it survives one call and then deadlocks on the next at
0% CPU. Separate processes remove the interaction instead of tuning around it,
and keep num_threads identical to the cross-validation runs so the test numbers
stay comparable to the CV numbers.

Writes raw predictions to disk; final_test_report.py computes the metrics.
"""

import numpy as np
import pandas as pd

import baseline as B
from features import build_feature_frame

OUT_NPZ = "test_preds_clinical.npz"
CALENDAR_COLUMN = "visit_start_datetime"


def evaluate(fit_frame, test_frame, drop_calendar: bool):
    if drop_calendar:
        fit_frame = fit_frame.drop(columns=[CALENDAR_COLUMN])
        test_frame = test_frame.drop(columns=[CALENDAR_COLUMN])
    matrices, _encoder = B.prepare(fit_frame, {"test": test_frame}, strict=False)
    x_fit, y_fit = matrices["__fit__"]
    x_test, y_test = matrices["test"]
    booster = B.fit_models(x_fit, y_fit)["lightgbm"]
    return B.predict(booster, x_test), y_test


def main() -> None:
    import sys

    assert "torch" not in sys.modules, "torch was imported; the OpenMP conflict is back"

    pool = build_feature_frame(split=None)
    fit_frame = pool[pool["split"].isin(["train", "val"])].reset_index(drop=True)
    test_frame = pool[pool["split"] == "test"].reset_index(drop=True)

    assert len(test_frame) == 205, f"unexpected test size {len(test_frame)}"
    assert not set(fit_frame["patient_id"]) & set(test_frame["patient_id"]), "train/test overlap"

    print("=== TEST STAGE 1: clinical LightGBM (torch-free process) ===", flush=True)
    print(f"  fit n={len(fit_frame)}  test n={len(test_frame)}  "
          f"positives={int(test_frame['severe'].sum())}", flush=True)

    payload = {}
    for name, drop in [
        ("clinical LightGBM, naive", False),
        ("clinical LightGBM, calendar-ablated", True),
    ]:
        scores, targets = evaluate(fit_frame, test_frame, drop)
        payload[f"pred::{name}"] = scores
        payload["y_true"] = targets
        print(f"  done: {name}", flush=True)

    np.savez(OUT_NPZ, **payload)
    print(f"  wrote {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
