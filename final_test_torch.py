"""Test-set stage 2: the torch models. LightGBM is never imported here.

Companion to final_test_clinical.py; see that module for why the two stages run
as separate processes.

Each model is refit on train+val with settings frozen before the test split was
read, then scored once. Early stopping uses an inner slice carved out of
train+val, never test. Writes raw prediction logits for final_test_report.py.
"""

import numpy as np
import torch

import finetune as FT
import fusion_model as FM
from features import build_feature_frame

OUT_NPZ = "test_preds_torch.npz"
SEED = 42
CALENDAR_COLUMN = "visit_start_datetime"


def image_only(fit_frame, test_frame, cache, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, info = FT.train_one(fit_frame, cache, device, seed=SEED)
    loader = torch.utils.data.DataLoader(
        FT.ImageOnlyDataset(test_frame, train=False, cache=cache),
        batch_size=FT.BATCH_SIZE, shuffle=False, num_workers=0,
    )
    scores, targets = FT.predict_scores(model, loader, device)
    del model
    return scores, targets, info


def fusion(fit_frame, test_frame, cache, device, drop_calendar: bool):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    drop = (CALENDAR_COLUMN,) if drop_calendar else ()
    prepared, _encoder = FM.prepare_fold(fit_frame, {"test": test_frame}, drop)
    fit_imputed, fit_clinical = prepared["__fit__"]
    test_imputed, test_clinical = prepared["test"]

    model, info = FM.train_fold(
        fit_imputed, fit_clinical, cache, device, FM.MODALITY_DROPOUT_P, SEED, True
    )
    loader = FM.make_loader(test_imputed, test_clinical, False, cache, False, True)
    scores, targets = FM.predict_scores(model, loader, device)
    del model
    return scores, targets, info


def main() -> None:
    import sys

    assert "lightgbm" not in sys.modules, "lightgbm was imported; the OpenMP conflict is back"

    device = FM.pick_device()
    pool = build_feature_frame(split=None)
    fit_frame = pool[pool["split"].isin(["train", "val"])].reset_index(drop=True)
    test_frame = pool[pool["split"] == "test"].reset_index(drop=True)

    assert len(test_frame) == 205, f"unexpected test size {len(test_frame)}"
    assert not set(fit_frame["patient_id"]) & set(test_frame["patient_id"]), "train/test overlap"

    print(f"=== TEST STAGE 2: torch models (device={device}, seed={SEED}) ===", flush=True)
    cache: dict = {}
    payload = {}

    scores, targets, info = image_only(fit_frame, test_frame, cache, device)
    payload["pred::image-only, layer4 fine-tuned"] = scores
    payload["y_true"] = targets
    print(f"  done: image-only (epoch {info['best_epoch']})", flush=True)

    scores, targets, info = fusion(fit_frame, test_frame, cache, device, drop_calendar=False)
    payload["pred::fusion, with calendar"] = scores
    print(f"  done: fusion with calendar (epoch {info['best_epoch']})", flush=True)

    scores, targets, info = fusion(fit_frame, test_frame, cache, device, drop_calendar=True)
    payload["pred::fusion, no calendar"] = scores
    print(f"  done: fusion no calendar (epoch {info['best_epoch']})", flush=True)

    np.savez(OUT_NPZ, **payload)
    print(f"  wrote {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
