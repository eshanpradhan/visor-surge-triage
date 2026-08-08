"""Check whether storing the fusion checkpoint in fp16 changes anything measurable.

HISTORICAL: the committed checkpoint is now fp16, so re-running this compares
fp16 against itself and reports no difference. It is kept as the record of the
comparison that justified the switch, and of the one metric that did move.

The checkpoint is 60.5 MB in fp32, above GitHub's 50 MB warning threshold. Half
precision would halve it, but only if the rounding is invisible in the outputs.

Weights are stored as fp16 and cast back to fp32 at load time, so this tests
storage precision alone -- the forward pass runs in fp32 either way. Metrics are
compared on val and test; the decision rule is that AUROC, AUPRC and Brier must
agree to three decimal places on both.

Re-running inference on test here is a numerical equivalence check between two
serializations of one frozen model, not an evaluation and not model selection.
Nothing is chosen on the basis of a test metric.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import fusion_model as FM
from features import build_feature_frame

SEED = 42
CALENDAR_COLUMN = "visit_start_datetime"
FP32_PATH = "models/fusion_no_calendar.pt"
FP16_PATH = "models/fusion_no_calendar_fp16.pt"
TOLERANCE_DECIMALS = 3


def brier(y_true, y_prob) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def build_model(state: dict, n_clinical: int, device):
    model = FM.FusionModel(n_clinical, FM.MODALITY_DROPOUT_P, use_image=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    # the frozen ImageNet tensors below layer4 are intentionally absent
    assert all(not k.startswith(FM_TRAINABLE) for k in missing for FM_TRAINABLE in
               ("backbone.layer4.", "image_projection.", "clinical_branch.", "head.")), \
        "a trained tensor is missing from the checkpoint"
    return model.eval().to(device)


def predict(model, frame, clinical, device) -> np.ndarray:
    loader = FM.make_loader(frame, clinical, False, CACHE, False, True)
    logits, targets = FM.predict_scores(model, loader, device)
    return np.asarray(logits, dtype=float), np.asarray(targets, dtype=int)


CACHE: dict = {}


def main() -> None:
    pd.set_option("display.width", 200)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = FM.pick_device()

    pool = build_feature_frame(split=None)
    train = pool[pool["split"] == "train"].reset_index(drop=True)
    val = pool[pool["split"] == "val"].reset_index(drop=True)
    test = pool[pool["split"] == "test"].reset_index(drop=True)

    prepared, _encoder = FM.prepare_fold(train, {"val": val, "test": test}, (CALENDAR_COLUMN,))
    n_clinical = prepared["__fit__"][1].shape[1]

    fp32_state = torch.load(FP32_PATH, map_location="cpu")
    fp16_state = {k: v.half() for k, v in fp32_state.items()}
    torch.save(fp16_state, FP16_PATH)

    # load fp16 back from disk and restore to fp32 for the forward pass
    reloaded = {k: v.float() for k, v in torch.load(FP16_PATH, map_location="cpu").items()}

    max_weight_delta = max(
        float((fp32_state[k].float() - reloaded[k]).abs().max()) for k in fp32_state
    )
    print("=== fp16 STORAGE CHECK ===")
    print(f"  tensors: {len(fp32_state)}   max |weight delta| after round-trip: "
          f"{max_weight_delta:.3e}")
    print()

    rows = []
    prediction_deltas = {}
    for precision, state in [("fp32", fp32_state), ("fp16", reloaded)]:
        model = build_model(state, n_clinical, device)
        for split_name in ["val", "test"]:
            frame, clinical = prepared[split_name]
            logits, targets = predict(model, frame, clinical, device)
            probability = sigmoid(logits)
            rows.append(
                {
                    "precision": precision,
                    "split": split_name,
                    "auroc": round(roc_auc_score(targets, probability), TOLERANCE_DECIMALS),
                    "auprc": round(average_precision_score(targets, probability),
                                   TOLERANCE_DECIMALS),
                    "brier": round(brier(targets, probability), TOLERANCE_DECIMALS),
                }
            )
            prediction_deltas.setdefault(split_name, {})[precision] = probability
        del model

    table = pd.DataFrame(rows)
    print("=== METRICS, rounded to 3 dp ===")
    print(table.to_string(index=False))
    print()

    print("=== PER-PATIENT PREDICTION DIFFERENCES ===")
    for split_name, probs in prediction_deltas.items():
        delta = np.abs(probs["fp32"] - probs["fp16"])
        print(f"  {split_name:5s} n={len(delta):4d}  max |delta p| = {delta.max():.3e}   "
              f"mean = {delta.mean():.3e}")
    print()

    fp32_rows = table[table.precision == "fp32"].drop(columns="precision").reset_index(drop=True)
    fp16_rows = table[table.precision == "fp16"].drop(columns="precision").reset_index(drop=True)
    identical = fp32_rows.equals(fp16_rows)

    print("=== DECISION ===")
    if identical:
        print(f"  MATCH: all metrics agree to {TOLERANCE_DECIMALS} dp on val and test.")
        print("  -> fp16 is safe; replacing the committed checkpoint.")
    else:
        print(f"  MISMATCH at {TOLERANCE_DECIMALS} dp:")
        print(fp32_rows.compare(fp16_rows).to_string())
        print("  -> keeping fp32.")
    return identical


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
