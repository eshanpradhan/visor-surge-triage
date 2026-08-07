"""Clinical-only baselines: L2 logistic regression and LightGBM.

Establishes the floor the image model has to beat. The test split is not touched
anywhere in this file.

Fold-wise refitting
-------------------
Imputation medians/modes and the encoder's standardization, log1p selection, and
category vocabularies are all fitted statistics. In the holdout evaluation they
come from train and are applied to val. In cross-validation they are refitted
inside every fold, on that fold's training rows only. Fitting them once over all
of train+val and then cross-validating would leak each fold's held-out rows into
its own preprocessing, which is the quiet version of the leak this pipeline has
been guarding against everywhere else.

Splitting is stratified on the label and, because the manifest holds exactly one
row per patient, each patient lands in exactly one fold. That is asserted rather
than assumed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

# clinical_encoding, not dataset: keeps this module importable without torch, so
# LightGBM never shares a process with torch's OpenMP runtime (see clinical_encoding)
from clinical_encoding import ClinicalEncoder
from features import build_feature_frame
from impute import apply_impute_stats, fit_impute_stats

RANDOM_STATE = 42
N_FOLDS = 5
N_CALIBRATION_BINS = 5

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "is_unbalance": True,
    "verbose": -1,
    "seed": RANDOM_STATE,
    "num_threads": 4,
}
LGB_ROUNDS = 300


def prepare(fit_frame: pd.DataFrame, apply_frames: dict[str, pd.DataFrame], strict: bool = True):
    """Fit imputation + encoding on fit_frame, apply to each frame in apply_frames."""
    stats = fit_impute_stats(fit_frame, require_train_split=strict)
    fitted = apply_impute_stats(fit_frame, stats)
    encoder = ClinicalEncoder().fit(fitted)

    matrices = {"__fit__": (encoder.transform(fitted), fitted["severe"].to_numpy(dtype=int))}
    for name, frame in apply_frames.items():
        imputed = apply_impute_stats(frame, stats)
        matrices[name] = (encoder.transform(imputed), imputed["severe"].to_numpy(dtype=int))
    return matrices, encoder


def fit_models(x_train: np.ndarray, y_train: np.ndarray):
    logistic = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    ).fit(x_train, y_train)

    booster = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(x_train, label=y_train),
        num_boost_round=LGB_ROUNDS,
    )
    return {"logistic": logistic, "lightgbm": booster}


def predict(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, lgb.Booster):
        return model.predict(x)
    return model.predict_proba(x)[:, 1]


def score(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS):
    """Equal-count bins, which behave better than equal-width at n=205."""
    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)
    rows = []
    for index, chunk in enumerate(bins, start=1):
        rows.append(
            {
                "bin": index,
                "n": len(chunk),
                "mean_pred": float(y_prob[chunk].mean()),
                "observed": float(y_true[chunk].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = (table["mean_pred"] - table["observed"]).round(3)
    return table


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def run_holdout():
    train = build_feature_frame(split="train")
    val = build_feature_frame(split="val")

    matrices, encoder = prepare(train, {"val": val})
    x_train, y_train = matrices["__fit__"]
    x_val, y_val = matrices["val"]

    models = fit_models(x_train, y_train)
    results = {}
    for name, model in models.items():
        y_prob = predict(model, x_val)
        results[name] = {
            **score(y_val, y_prob),
            "brier": brier(y_val, y_prob),
            "calibration": calibration_table(y_val, y_prob),
            "y_prob": y_prob,
        }
    return results, models, encoder, y_val


def run_cv():
    """5-fold stratified CV over train+val, refitting preprocessing per fold."""
    combined = build_feature_frame(split=None)
    combined = combined[combined["split"].isin(["train", "val"])].reset_index(drop=True)
    assert combined["patient_id"].is_unique, "combined frame has repeated patients"

    labels = combined["severe"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    per_fold = {"logistic": [], "lightgbm": []}
    seen_holdout: set[str] = set()

    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(combined, labels), start=1):
        fit_frame = combined.iloc[fit_idx]
        holdout_frame = combined.iloc[holdout_idx]

        overlap = set(fit_frame["patient_id"]) & set(holdout_frame["patient_id"])
        assert not overlap, f"fold {fold}: {len(overlap)} patients in both halves"
        assert not (seen_holdout & set(holdout_frame["patient_id"])), (
            f"fold {fold}: a patient appears in two holdout folds"
        )
        seen_holdout |= set(holdout_frame["patient_id"])

        matrices, _ = prepare(fit_frame, {"holdout": holdout_frame}, strict=False)
        x_fit, y_fit = matrices["__fit__"]
        x_hold, y_hold = matrices["holdout"]

        for name, model in fit_models(x_fit, y_fit).items():
            per_fold[name].append(score(y_hold, predict(model, x_hold)))

    assert len(seen_holdout) == len(combined), "CV folds did not cover every patient exactly once"
    return per_fold


def importances(models: dict, encoder: ClinicalEncoder, top: int = 15) -> pd.DataFrame:
    names = np.array(encoder.feature_names)

    coefs = models["logistic"].coef_[0]
    lr = pd.DataFrame({"feature": names, "lr_coef": coefs, "lr_abs": np.abs(coefs)})
    lr["lr_rank"] = lr["lr_abs"].rank(ascending=False).astype(int)

    gain = models["lightgbm"].feature_importance(importance_type="gain")
    gb = pd.DataFrame({"feature": names, "lgb_gain": gain})
    gb["lgb_pct"] = (gb["lgb_gain"] / gb["lgb_gain"].sum() * 100).round(2)
    gb["lgb_rank"] = gb["lgb_gain"].rank(ascending=False).astype(int)

    return lr.merge(gb, on="feature")


def plot_calibration(results: dict, path: str = "calibration_val.png") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, (name, result) in zip(axes, results.items()):
        table = result["calibration"]
        axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
        axis.plot(table["mean_pred"], table["observed"], "o-", color="#2b6cb0", label=name)
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("observed frequency")
        axis.set_title(
            f"{name}  (val n=205)\nAUROC {result['auroc']:.3f}  Brier {result['brier']:.3f}"
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(loc="upper left", fontsize=8)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    return path


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    results, models, encoder, y_val = run_holdout()

    print("=== HOLDOUT: fit on train (n=955), evaluated on val (n=205) ===")
    for name, result in results.items():
        print(
            f"  {name:9s} AUROC={result['auroc']:.3f}  AUPRC={result['auprc']:.3f}  "
            f"Brier={result['brier']:.3f}"
        )
    print(f"  {'baseline':9s} AUROC=0.500  AUPRC={y_val.mean():.3f} (prevalence)")
    print()

    print("=== CALIBRATION (val, equal-count bins) ===")
    for name, result in results.items():
        print(f"  {name}")
        print(result["calibration"].to_string(index=False))
    print()

    print(f"=== {N_FOLDS}-FOLD CV on train+val (n=1160), preprocessing refit per fold ===")
    cv = run_cv()
    for name, folds in cv.items():
        aurocs = np.array([f["auroc"] for f in folds])
        auprcs = np.array([f["auprc"] for f in folds])
        print(
            f"  {name:9s} AUROC {aurocs.mean():.3f} +/- {aurocs.std():.3f}   "
            f"AUPRC {auprcs.mean():.3f} +/- {auprcs.std():.3f}   "
            f"folds {np.round(aurocs, 3).tolist()}"
        )
    print()

    table = importances(models, encoder)
    print("=== TOP 15 BY LOGISTIC |COEF| ===")
    print(
        table.sort_values("lr_abs", ascending=False)
        .head(15)[["feature", "lr_coef", "lr_rank", "lgb_pct", "lgb_rank"]]
        .to_string(index=False)
    )
    print()
    print("=== TOP 15 BY LIGHTGBM GAIN ===")
    print(
        table.sort_values("lgb_gain", ascending=False)
        .head(15)[["feature", "lgb_pct", "lgb_rank", "lr_coef", "lr_rank"]]
        .to_string(index=False)
    )
    print()

    print("=== CALENDAR FEATURE RANKS ===")
    calendar = table[table["feature"].str.startswith("visit_start_datetime")]
    print(calendar[["feature", "lr_coef", "lr_rank", "lgb_pct", "lgb_rank"]].to_string(index=False))
    print(f"  (of {len(table)} encoded features)")
    print()

    print("wrote", plot_calibration(results))
