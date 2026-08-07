"""Post-hoc probability calibration for the fusion model.

Why this trains a fresh model instead of reusing the final one
--------------------------------------------------------------
The fusion model in the final test table was fit on train+val, so its
predictions on val are in-sample. A calibrator fit on those would be learning to
correct predictions the model had already memorized, and would then be applied
to outputs it had never seen out-of-sample -- which defeats the purpose.

So this trains one fusion model on the train split alone (early-stopping on an
inner slice of train, exactly as in cross-validation), leaving val genuinely
held out. That single frozen model produces the val predictions the calibrator
is fitted on and the test predictions it is applied to. Because it sees 955 rows
rather than 1160, its raw discrimination differs slightly from the table model;
the quantity being measured here is the calibration change, which transfers.

No test data influences anything fitted here. The choice between Platt scaling
and isotonic regression is made on val Brier score alone.

Platt vs isotonic at this sample size
-------------------------------------
Val has 205 rows and 41 positives. Isotonic is non-parametric and can fit any
monotone mapping, which at this n tends to overfit into step functions that
generalize poorly; Platt fits two parameters to a sigmoid. Both are reported and
the winner is picked on val, but the prior favours Platt.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import fusion_model as FM
from features import build_feature_frame

SEED = 42
CALENDAR_COLUMN = "visit_start_datetime"
OUT_PNG = "calibration_fusion_before_after.png"
OUT_CSV = "calibration_fusion_results.csv"


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def calibration_table(y_true, y_prob, n_bins: int = 5) -> pd.DataFrame:
    order = np.argsort(y_prob)
    rows = []
    for index, chunk in enumerate(np.array_split(order, n_bins), start=1):
        rows.append(
            {
                "bin": index,
                "n": len(chunk),
                "mean_pred": round(float(y_prob[chunk].mean()), 3),
                "observed": round(float(y_true[chunk].mean()), 3),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = (table["mean_pred"] - table["observed"]).round(3)
    return table


def train_fusion_on_train_only(device, cache):
    """Fit on the train split; val stays held out so it can calibrate."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pool = build_feature_frame(split=None)
    train = pool[pool["split"] == "train"].reset_index(drop=True)
    val = pool[pool["split"] == "val"].reset_index(drop=True)
    test = pool[pool["split"] == "test"].reset_index(drop=True)

    for name, frame in [("val", val), ("test", test)]:
        assert not set(train["patient_id"]) & set(frame["patient_id"]), f"train/{name} overlap"
    assert not set(val["patient_id"]) & set(test["patient_id"]), "val/test overlap"

    prepared, _encoder = FM.prepare_fold(
        train, {"val": val, "test": test}, (CALENDAR_COLUMN,)
    )
    fit_imputed, fit_clinical = prepared["__fit__"]

    model, info = FM.train_fold(
        fit_imputed, fit_clinical, cache, device, FM.MODALITY_DROPOUT_P, SEED, True
    )

    outputs = {}
    for name in ["val", "test"]:
        imputed, clinical = prepared[name]
        loader = FM.make_loader(imputed, clinical, False, cache, False, True)
        logits, targets = FM.predict_scores(model, loader, device)
        outputs[name] = (np.asarray(logits, dtype=float), np.asarray(targets, dtype=int))

    del model
    return outputs, info


def main() -> None:
    pd.set_option("display.width", 220)
    device = FM.pick_device()
    cache: dict = {}

    print("=== POST-HOC CALIBRATION: fusion, no calendar ===")
    print("  model refit on train only (n=955) so val is held out for calibration")
    outputs, info = train_fusion_on_train_only(device, cache)
    print(f"  trained, best epoch {info['best_epoch']}")
    print()

    val_logits, y_val = outputs["val"]
    test_logits, y_test = outputs["test"]

    raw_val = sigmoid(val_logits)
    raw_test = sigmoid(test_logits)

    # --- fit both calibrators on val only ---
    platt = LogisticRegression(C=1e10, solver="lbfgs").fit(
        val_logits.reshape(-1, 1), y_val
    )
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(val_logits, y_val)

    candidates = {
        "platt": (
            platt.predict_proba(val_logits.reshape(-1, 1))[:, 1],
            platt.predict_proba(test_logits.reshape(-1, 1))[:, 1],
        ),
        "isotonic": (
            isotonic.predict(val_logits),
            isotonic.predict(test_logits),
        ),
    }

    print("=== CALIBRATOR SELECTION (on val only -- test plays no part) ===")
    print(f"  {'method':10s} {'val Brier':>10s} {'val AUROC':>10s}")
    print(f"  {'uncalibrated':10s} {brier(y_val, raw_val):10.4f} {roc_auc_score(y_val, raw_val):10.3f}")
    for name, (val_prob, _) in candidates.items():
        print(f"  {name:10s} {brier(y_val, val_prob):10.4f} {roc_auc_score(y_val, val_prob):10.3f}")

    chosen = min(candidates, key=lambda k: brier(y_val, candidates[k][0]))
    print(f"  -> chosen by val Brier: {chosen}")
    print()

    cal_test = candidates[chosen][1]

    print("=== TEST: BEFORE vs AFTER CALIBRATION ===")
    rows = []
    for label, probability in [("uncalibrated", raw_test), (f"{chosen}-calibrated", cal_test)]:
        rows.append(
            {
                "model": f"fusion, no calendar ({label})",
                "test_auroc": round(roc_auc_score(y_test, probability), 3),
                "test_auprc": round(average_precision_score(y_test, probability), 3),
                "test_brier": round(brier(y_test, probability), 4),
            }
        )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print()
    print(f"  Brier improvement: {rows[0]['test_brier'] - rows[1]['test_brier']:+.4f}")
    print("  Ranking metrics move only slightly, and only because of ties. Platt is")
    print("  strictly increasing and leaves AUROC/AUPRC exactly unchanged. Isotonic is a")
    print("  step function: it maps distinct scores onto identical values, and those ties")
    print("  are scored differently by AUROC (half credit) and AUPRC. Neither calibrator")
    print("  reorders any pair of patients, so the discrimination claim is unaffected.")
    print()

    print("=== TEST CALIBRATION TABLES (equal-count bins) ===")
    for label, probability in [("uncalibrated", raw_test), (f"{chosen}-calibrated", cal_test)]:
        print(f"  {label}  (Brier {brier(y_test, probability):.4f})")
        print(calibration_table(y_test, probability).to_string(index=False))
        print()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for axis, (label, probability) in zip(
        axes, [("uncalibrated", raw_test), (f"{chosen}-calibrated", cal_test)]
    ):
        table = calibration_table(y_test, probability)
        axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
        axis.plot(table["mean_pred"], table["observed"], "o-", color="#2b6cb0")
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("observed frequency")
        axis.set_title(
            f"fusion, no calendar -- {label}\n"
            f"test n=205 | AUROC {roc_auc_score(y_test, probability):.3f} | "
            f"Brier {brier(y_test, probability):.3f}",
            fontsize=9,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT_PNG, dpi=140)

    summary.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_PNG} and {OUT_CSV}")


if __name__ == "__main__":
    main()
