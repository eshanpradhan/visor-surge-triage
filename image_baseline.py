"""Image-only baseline: frozen ImageNet ResNet-50 embeddings + a simple head.

Establishes what the chest X-ray alone contributes, to be read against the
calendar-ablated clinical floor of 0.804 AUROC -- not against the 0.843 figure,
which is inflated by an admission-month feature that cannot exist prospectively.

Embeddings are extracted with the deterministic transform (no augmentation):
augmentation exists to regularize a model being trained, and here the backbone
is frozen, so jitter would only add noise to a fixed feature vector.

Extracting embeddings for all three splits at once is not leakage. The backbone
is ImageNet-pretrained and frozen, so nothing about our data is fitted during
extraction, and no label is read. Everything that *is* fitted -- scaler, head --
is fitted per split or per fold. The test split's metrics are not computed
anywhere in this file.

A note on normalization: images are normalized with the train-split statistics
from norm_stats.json rather than ImageNet's, following the rest of the pipeline.
These X-rays are already contrast-normalized upstream (mean 0.496, std 0.240)
and sit some distance from ImageNet's natural-image statistics, so the frozen
features are somewhat off-distribution either way.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import models

from dataset import ClinicalEncoder, VisorDataset, load_norm_stats
from impute import build_imputed_frames

RANDOM_STATE = 42
N_FOLDS = 5
EMBED_BATCH = 64

# ImageNet per-channel statistics, applied after the L -> 3-channel repeat. The
# repeat makes all three channels identical, so this shifts and scales each copy
# slightly differently -- which is what the pretrained filters were trained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CACHE_NPZ = {
    "dataset": "resnet50_embeddings.npz",
    "imagenet": "resnet50_embeddings_imagenet.npz",
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_backbone(device: torch.device) -> torch.nn.Module:
    """ImageNet ResNet-50 with the classification layer removed, frozen."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone.eval().to(device)


def extract_embeddings(normalization: str = "dataset") -> tuple[dict, dict, dict]:
    """Return per-split (embeddings, labels, patient_ids), caching to disk.

    ``normalization`` selects 'dataset' (train-split statistics from
    norm_stats.json) or 'imagenet' (the statistics the backbone was trained on).
    """
    assert normalization in CACHE_NPZ, f"unknown normalization: {normalization!r}"
    cache = CACHE_NPZ[normalization]
    frames, _ = build_imputed_frames()

    if __import__("os").path.exists(cache):
        blob = np.load(cache, allow_pickle=True)
        embeddings = {s: blob[f"x_{s}"] for s in ["train", "val", "test"]}
        labels = {s: blob[f"y_{s}"] for s in ["train", "val", "test"]}
        patients = {s: blob[f"p_{s}"] for s in ["train", "val", "test"]}
        return embeddings, labels, patients

    device = pick_device()
    backbone = build_backbone(device)
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = load_norm_stats()
    encoder = ClinicalEncoder().fit(frames["train"])

    embeddings, labels, patients = {}, {}, {}
    for split in ["train", "val", "test"]:
        # train=False everywhere: deterministic transform for feature extraction
        dataset = VisorDataset(frames[split], encoder, train=False, mean=mean, std=std)
        loader = DataLoader(dataset, batch_size=EMBED_BATCH, shuffle=False, num_workers=0)

        chunks = []
        with torch.no_grad():
            for images, _clinical, _label in loader:
                chunks.append(backbone(images.to(device)).cpu().numpy())

        embeddings[split] = np.vstack(chunks).astype(np.float32)
        labels[split] = frames[split]["severe"].to_numpy(dtype=int)
        patients[split] = frames[split]["patient_id"].to_numpy()

        assert embeddings[split].shape == (len(frames[split]), 2048), (
            f"{split}: unexpected embedding shape {embeddings[split].shape}"
        )
        print(f"  extracted {split}: {embeddings[split].shape} on {device}")

    np.savez_compressed(
        cache,
        **{f"x_{s}": embeddings[s] for s in embeddings},
        **{f"y_{s}": labels[s] for s in labels},
        **{f"p_{s}": patients[s] for s in patients},
    )
    return embeddings, labels, patients


def build_heads() -> dict:
    return {
        "logistic": LogisticRegression(
            penalty="l2",
            C=0.01,  # 2048 dims against 955 rows; strong shrinkage
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        # 2-layer MLP: 2048 -> 256 -> 1. MLPClassifier has no class_weight, which
        # is tolerable here because AUROC and AUPRC are both threshold-free.
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1.0,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    }


def fit_and_score(x_fit, y_fit, x_eval, y_eval) -> dict:
    scaler = StandardScaler().fit(x_fit)
    x_fit_s, x_eval_s = scaler.transform(x_fit), scaler.transform(x_eval)

    results = {}
    for name, head in build_heads().items():
        head.fit(x_fit_s, y_fit)
        probability = head.predict_proba(x_eval_s)[:, 1]
        results[name] = {
            "auroc": roc_auc_score(y_eval, probability),
            "auprc": average_precision_score(y_eval, probability),
            "prob": probability,
        }
    return results


def run_cv(x: np.ndarray, y: np.ndarray, patients: np.ndarray) -> dict:
    """5-fold stratified CV; scaler and head refit per fold, backbone is frozen."""
    assert len(set(patients)) == len(patients), "duplicate patient in CV pool"

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    per_fold = {"logistic": [], "mlp": []}
    seen: set = set()

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(x, y), start=1):
        overlap = set(patients[fit_idx]) & set(patients[hold_idx])
        assert not overlap, f"fold {fold}: {len(overlap)} patients on both sides"
        assert not (seen & set(patients[hold_idx])), f"fold {fold}: repeated holdout patient"
        seen |= set(patients[hold_idx])

        scored = fit_and_score(x[fit_idx], y[fit_idx], x[hold_idx], y[hold_idx])
        for name, result in scored.items():
            per_fold[name].append((result["auroc"], result["auprc"]))

    assert len(seen) == len(y), "CV did not cover every patient exactly once"
    return per_fold


def calendar_check(x, y, patients, frames) -> None:
    """Do the embeddings carry the admission-month signal that the clinical data did?"""
    encoder = ClinicalEncoder().fit(frames["train"])
    pool = pd.concat([frames["train"], frames["val"]]).reset_index(drop=True)
    bucket = encoder._bucket_months(pool["visit_start_datetime"], "visit_start_datetime")
    bucket = bucket.to_numpy()
    assert len(bucket) == len(y), "calendar bucket length mismatch"

    print("=== CALENDAR INDEPENDENCE OF THE IMAGE BRANCH ===")

    # 1. can a linear probe recover admission month from the embeddings alone?
    is_december = (bucket == "1900-12").astype(int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probe = []
    for fit_idx, hold_idx in splitter.split(x, is_december):
        scaler = StandardScaler().fit(x[fit_idx])
        head = LogisticRegression(
            penalty="l2", C=0.01, class_weight="balanced", max_iter=5000,
            random_state=RANDOM_STATE,
        ).fit(scaler.transform(x[fit_idx]), is_december[fit_idx])
        probe.append(
            roc_auc_score(is_december[hold_idx], head.predict_proba(scaler.transform(x[hold_idx]))[:, 1])
        )
    print(
        f"  linear probe, embeddings -> admission month: AUROC "
        f"{np.mean(probe):.3f} +/- {np.std(probe):.3f}   (0.500 = images carry no date signal)"
    )

    # 2. severity AUROC computed within each calendar bucket separately
    for label in sorted(set(bucket)):
        mask = bucket == label
        if y[mask].sum() < 10 or (~y[mask].astype(bool)).sum() < 10:
            print(f"  {label}: too few of one class to score (n={mask.sum()})")
            continue
        folds = run_cv(x[mask], y[mask], patients[mask])
        aurocs = np.array([a for a, _ in folds["logistic"]])
        print(
            f"  within {label}: n={mask.sum():4d}  prevalence={y[mask].mean() * 100:4.1f}%  "
            f"logistic AUROC {aurocs.mean():.3f} +/- {aurocs.std():.3f}"
        )


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    print("=== EXTRACTING FROZEN ResNet-50 EMBEDDINGS (deterministic transform) ===")
    embeddings, labels, patients = extract_embeddings()
    frames, _ = build_imputed_frames()
    print()

    print("=== HOLDOUT: fit on train (n=955), evaluated on val (n=205) ===")
    holdout = fit_and_score(
        embeddings["train"], labels["train"], embeddings["val"], labels["val"]
    )
    for name, result in holdout.items():
        print(f"  {name:9s} AUROC={result['auroc']:.3f}  AUPRC={result['auprc']:.3f}")
    print(f"  {'chance':9s} AUROC=0.500  AUPRC={labels['val'].mean():.3f} (prevalence)")
    print()

    x_pool = np.vstack([embeddings["train"], embeddings["val"]])
    y_pool = np.concatenate([labels["train"], labels["val"]])
    p_pool = np.concatenate([patients["train"], patients["val"]])

    print(f"=== {N_FOLDS}-FOLD CV on train+val (n={len(y_pool)}), head refit per fold ===")
    cv = run_cv(x_pool, y_pool, p_pool)
    for name, folds in cv.items():
        aurocs = np.array([a for a, _ in folds])
        auprcs = np.array([b for _, b in folds])
        print(
            f"  {name:9s} AUROC {aurocs.mean():.3f} +/- {aurocs.std():.3f}   "
            f"AUPRC {auprcs.mean():.3f} +/- {auprcs.std():.3f}   "
            f"folds {np.round(aurocs, 3).tolist()}"
        )
    print()

    calendar_check(x_pool, y_pool, p_pool, frames)
