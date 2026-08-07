"""Test-set stage 3: merge stage-1 and stage-2 predictions into the final table.

Pure reporting -- no model is fit here. Neither torch nor LightGBM is imported,
so this stage cannot hit the OpenMP conflict that forced the split.

At 41 test positives an AUROC carries roughly a +/-0.06-0.08 confidence interval.
These numbers confirm the cross-validated ranking rather than sharpen it; where
test and CV disagree, the CV estimate is better supported.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CLINICAL_NPZ = "test_preds_clinical.npz"
TORCH_NPZ = "test_preds_torch.npz"
ORDER = [
    "clinical LightGBM, naive",
    "clinical LightGBM, calendar-ablated",
    "image-only, layer4 fine-tuned",
    "fusion, with calendar",
    "fusion, no calendar",
]
CALIBRATION_MODELS = [
    "clinical LightGBM, calendar-ablated",
    "image-only, layer4 fine-tuned",
    "fusion, no calendar",
]
CV_REFERENCE = {
    "clinical LightGBM, naive": 0.843,
    "clinical LightGBM, calendar-ablated": 0.804,
    "image-only, layer4 fine-tuned": 0.756,
    "fusion, with calendar": 0.826,
    "fusion, no calendar": 0.793,
}


def to_probability(scores: np.ndarray) -> np.ndarray:
    """Torch stages emit logits; LightGBM already emits probabilities."""
    if scores.min() < 0 or scores.max() > 1:
        return 1.0 / (1.0 + np.exp(-scores))
    return scores


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


def plot_calibration(results: dict, path: str = "calibration_test.png") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in CALIBRATION_MODELS if n in results]
    figure, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.7))
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        result = results[name]
        table = result["calibration"]
        axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
        axis.plot(table["mean_pred"], table["observed"], "o-", color="#2b6cb0")
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("observed frequency")
        axis.set_title(
            f"{name}\ntest n=205 | AUROC {result['auroc']:.3f} | Brier {result['brier']:.3f}",
            fontsize=9,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    return path


def main() -> None:
    pd.set_option("display.width", 220)

    clinical = np.load(CLINICAL_NPZ, allow_pickle=True)
    torch_preds = np.load(TORCH_NPZ, allow_pickle=True)

    y_true = clinical["y_true"].astype(int)
    assert np.array_equal(y_true, torch_preds["y_true"].astype(int)), (
        "the two stages scored different test rows -- check split ordering"
    )
    assert len(y_true) == 205, f"unexpected test size {len(y_true)}"

    predictions = {}
    for blob in (clinical, torch_preds):
        for key in blob.files:
            if key.startswith("pred::"):
                predictions[key[len("pred::"):]] = blob[key]

    missing = [name for name in ORDER if name not in predictions]
    assert not missing, f"missing predictions for: {missing}"

    print("=== FINAL TEST-SET RESULTS (single pass, no tuning after this) ===")
    print(f"  test n={len(y_true)}  positives={int(y_true.sum())}  "
          f"prevalence={y_true.mean() * 100:.1f}%")
    print()

    results = {}
    rows = []
    for name in ORDER:
        probability = to_probability(np.asarray(predictions[name], dtype=float))
        auroc = roc_auc_score(y_true, probability)
        results[name] = {
            "auroc": auroc,
            "auprc": average_precision_score(y_true, probability),
            "brier": float(np.mean((probability - y_true) ** 2)),
            "calibration": calibration_table(y_true, probability),
        }
        rows.append(
            {
                "model": name,
                "test_auroc": round(auroc, 3),
                "test_auprc": round(results[name]["auprc"], 3),
                "test_brier": round(results[name]["brier"], 3),
                "cv_auroc": CV_REFERENCE[name],
                "test_minus_cv": round(auroc - CV_REFERENCE[name], 3),
            }
        )

    summary = pd.DataFrame(rows)
    print("=== FINAL TEST TABLE ===")
    print(summary.to_string(index=False))
    print()

    print("=== CALIBRATION ON TEST (equal-count bins) ===")
    for name in CALIBRATION_MODELS:
        print(f"  {name}  (Brier {results[name]['brier']:.3f})")
        print(results[name]["calibration"].to_string(index=False))
        print()

    summary.to_csv("final_test_results.csv", index=False)
    print("wrote", plot_calibration(results), "and final_test_results.csv")


if __name__ == "__main__":
    main()
