"""Test 2: fine-tune ResNet-50 layer4 on the severity task.

Frozen ImageNet features plateaued at 0.690 CV AUROC and input rescaling did not
move it (test 1: +0.005). This unfreezes the last residual block to see whether
the domain gap closes when the top-level features can adapt.

BatchNorm is held in eval mode everywhere, including inside layer4
--------------------------------------------------------------------
The whole network stays in ``.eval()`` for the entire run. Gradients still flow
to layer4's weights -- ``requires_grad`` and train/eval mode are independent --
but no BatchNorm layer updates its running statistics, and none normalizes by
batch statistics. With 955 training images at batch 32, batch-estimated BN
statistics are noisy enough to destabilize training on their own, and letting
layer4's running stats drift while the frozen blocks below keep ImageNet's
creates an inconsistent normalization stack. Freezing all BN is the standard
small-sample recipe and makes the forward pass deterministic given the input.

Early stopping does not touch the fold holdout
----------------------------------------------
Each CV fold carves an inner validation slice out of its own training rows and
early-stops on that. Stopping on the fold's holdout would select the epoch that
best fits the data the fold is scored on, which inflates exactly the number the
CV exists to estimate.

Learning rates: 1e-4 for layer4 as specified, 1e-3 for the randomly-initialized
head, which starts from noise and needs to move faster than a pretrained block.
"""

import copy

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from dataset import ClinicalEncoder, build_image_transform
from image_baseline import IMAGENET_MEAN, IMAGENET_STD, N_FOLDS, RANDOM_STATE, pick_device
from impute import build_imputed_frames

BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 8
LR_LAYER4 = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
INNER_VAL_FRACTION = 0.15


class ImageOnlyDataset(Dataset):
    """Images plus labels, with source images cached in memory across epochs."""

    def __init__(self, frame: pd.DataFrame, train: bool, cache: dict) -> None:
        self.frame = frame.reset_index(drop=True)
        self.paths = self.frame["filepath"].tolist()
        self.labels = self.frame["severe"].to_numpy(dtype=np.float32)
        self.transform = build_image_transform(train, IMAGENET_MEAN, IMAGENET_STD)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        path = self.paths[index]
        if path not in self.cache:
            with Image.open(path) as handle:
                self.cache[path] = handle.convert("L").copy()
        return self.transform(self.cache[path]), self.labels[index]


class FineTuneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        for parameter in backbone.layer4.parameters():
            parameter.requires_grad = True

        self.backbone = backbone
        self.head = nn.Linear(2048, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)).squeeze(1)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def train(self, mode: bool = True) -> "FineTuneModel":
        """Always eval mode: keeps every BatchNorm frozen. See module docstring."""
        return super().train(False)


def assert_batchnorm_frozen(model: nn.Module) -> None:
    live = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
    ]
    assert not live, f"{len(live)} BatchNorm layer(s) in training mode, e.g. {live[:3]}"


def predict_scores(model: FineTuneModel, loader: DataLoader, device) -> tuple:
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            scores.append(model(images.to(device)).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(scores), np.concatenate(targets)


def embed_all(model: FineTuneModel, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for images, _labels in loader:
            chunks.append(model.embed(images.to(device)).cpu().numpy())
    return np.vstack(chunks)


def train_one(fit_frame: pd.DataFrame, cache: dict, device, verbose: bool = False,
              seed: int = RANDOM_STATE):
    """Train on fit_frame, early-stopping on an inner split carved from it."""
    inner_train, inner_val = train_test_split(
        fit_frame,
        test_size=INNER_VAL_FRACTION,
        stratify=fit_frame["severe"],
        random_state=seed,
    )
    overlap = set(inner_train["patient_id"]) & set(inner_val["patient_id"])
    assert not overlap, "inner split shares patients"

    train_loader = DataLoader(
        ImageOnlyDataset(inner_train, train=True, cache=cache),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    # deterministic transform for scoring, even on training rows
    inner_train_eval = DataLoader(
        ImageOnlyDataset(inner_train, train=False, cache=cache),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    inner_val_loader = DataLoader(
        ImageOnlyDataset(inner_val, train=False, cache=cache),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = FineTuneModel().to(device)
    model.train()
    assert_batchnorm_frozen(model)

    positives = inner_train["severe"].sum()
    pos_weight = torch.tensor(
        [(len(inner_train) - positives) / positives], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.layer4.parameters(), "lr": LR_LAYER4},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    best_auroc, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

        scores, targets = predict_scores(model, inner_val_loader, device)
        auroc = roc_auc_score(targets, scores)
        if verbose:
            print(f"    epoch {epoch:2d}  inner-val AUROC {auroc:.3f}")

        if auroc > best_auroc:
            best_auroc, best_epoch, stale = auroc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    train_scores, train_targets = predict_scores(model, inner_train_eval, device)
    return model, {
        "best_epoch": best_epoch,
        "inner_val_auroc": best_auroc,
        "train_auroc": roc_auc_score(train_targets, train_scores),
    }


def calendar_probe(embeddings: np.ndarray, buckets: np.ndarray) -> tuple:
    """Can a linear probe recover admission month from these features?"""
    target = (buckets == "1900-12").astype(int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fit_idx, hold_idx in splitter.split(embeddings, target):
        scaler = StandardScaler().fit(embeddings[fit_idx])
        probe = LogisticRegression(
            penalty="l2", C=0.01, class_weight="balanced", max_iter=5000,
            random_state=RANDOM_STATE,
        ).fit(scaler.transform(embeddings[fit_idx]), target[fit_idx])
        scores.append(
            roc_auc_score(
                target[hold_idx], probe.predict_proba(scaler.transform(embeddings[hold_idx]))[:, 1]
            )
        )
    return float(np.mean(scores)), float(np.std(scores))


def run_cv_for_seed(seed: int, device, cache: dict, frames: dict, probe: bool = True) -> dict:
    """Full 5-fold CV at one seed. The seed drives fold assignment, the inner
    early-stopping split, weight init and augmentation order together, so the
    spread across seeds is the spread you would actually see on a re-run."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    pool = pd.concat([frames["train"], frames["val"]]).reset_index(drop=True)
    labels = pool["severe"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    rows, seen = [], set()
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(pool, labels), start=1):
        fit_frame, hold_frame = pool.iloc[fit_idx], pool.iloc[hold_idx]
        assert not set(fit_frame["patient_id"]) & set(hold_frame["patient_id"]), "fold overlap"
        assert not seen & set(hold_frame["patient_id"]), "repeated holdout patient"
        seen |= set(hold_frame["patient_id"])

        model, info = train_one(fit_frame, cache, device, seed=seed)
        hold_loader = DataLoader(
            ImageOnlyDataset(hold_frame, train=False, cache=cache),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        scores, targets = predict_scores(model, hold_loader, device)
        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "epoch": info["best_epoch"],
                "train_auroc": round(info["train_auroc"], 3),
                "hold_auroc": round(roc_auc_score(targets, scores), 3),
                "hold_auprc": round(average_precision_score(targets, scores), 3),
            }
        )
        del model

    assert len(seen) == len(pool), "CV did not cover every patient exactly once"
    return pd.DataFrame(rows)


def main_multiseed(seeds=(42, 7, 2024)) -> None:
    pd.set_option("display.width", 220)
    device = pick_device()
    frames, _ = build_imputed_frames()
    cache: dict = {}

    print(f"=== MULTI-SEED layer4 FINE-TUNING CV (seeds={list(seeds)}, device={device}) ===")
    tables = []
    for seed in seeds:
        table = run_cv_for_seed(seed, device, cache, frames)
        mean_auroc = table.hold_auroc.mean()
        print(f"  seed {seed}: CV AUROC {mean_auroc:.3f}  epochs {table.epoch.tolist()}  "
              f"folds {table.hold_auroc.tolist()}")
        tables.append(table)

    combined = pd.concat(tables)
    per_seed = combined.groupby("seed")[["hold_auroc", "hold_auprc"]].mean()
    print()
    print("=== PER-SEED CV MEANS ===")
    print(per_seed.round(3).to_string())
    print()
    print(f"  ACROSS-SEED  AUROC {per_seed.hold_auroc.mean():.3f} +/- "
          f"{per_seed.hold_auroc.std(ddof=1):.3f}   "
          f"AUPRC {per_seed.hold_auprc.mean():.3f} +/- {per_seed.hold_auprc.std(ddof=1):.3f}")
    print(f"  within-seed fold sd (mean): "
          f"{combined.groupby('seed').hold_auroc.std(ddof=0).mean():.3f}")
    print(f"  stopping epochs across all runs: {sorted(combined.epoch.tolist())}")
    combined.to_csv("finetune_multiseed.csv", index=False)
    print("  wrote finetune_multiseed.csv")


def main() -> None:
    pd.set_option("display.width", 220)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = pick_device()
    frames, _ = build_imputed_frames()
    pool = pd.concat([frames["train"], frames["val"]]).reset_index(drop=True)
    assert pool["patient_id"].is_unique, "duplicate patient in CV pool"
    labels = pool["severe"].to_numpy(dtype=int)
    cache: dict = {}

    print(f"=== FINE-TUNING ResNet-50 layer4 (device={device}, ImageNet normalization) ===")
    print(
        f"  lr layer4={LR_LAYER4}  lr head={LR_HEAD}  batch={BATCH_SIZE}  "
        f"max_epochs={MAX_EPOCHS}  patience={PATIENCE}  all BatchNorm frozen"
    )
    print()

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, seen = [], set()

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(pool, labels), start=1):
        fit_frame, hold_frame = pool.iloc[fit_idx], pool.iloc[hold_idx]
        assert not set(fit_frame["patient_id"]) & set(hold_frame["patient_id"]), "fold overlap"
        assert not seen & set(hold_frame["patient_id"]), "repeated holdout patient"
        seen |= set(hold_frame["patient_id"])

        model, info = train_one(fit_frame, cache, device)
        hold_loader = DataLoader(
            ImageOnlyDataset(hold_frame, train=False, cache=cache),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        scores, targets = predict_scores(model, hold_loader, device)

        rows.append(
            {
                "fold": fold,
                "epoch": info["best_epoch"],
                "train_auroc": round(info["train_auroc"], 3),
                "inner_val_auroc": round(info["inner_val_auroc"], 3),
                "hold_auroc": round(roc_auc_score(targets, scores), 3),
                "hold_auprc": round(average_precision_score(targets, scores), 3),
            }
        )
        rows[-1]["gap"] = round(rows[-1]["train_auroc"] - rows[-1]["hold_auroc"], 3)
        print(f"  fold {fold}: {rows[-1]}")
        del model

    assert len(seen) == len(pool), "CV did not cover every patient exactly once"
    table = pd.DataFrame(rows)
    print()
    print("=== PER-FOLD (train AUROC uses the deterministic transform) ===")
    print(table.to_string(index=False))
    print()
    print(
        f"  CV AUROC {table.hold_auroc.mean():.3f} +/- {table.hold_auroc.std(ddof=0):.3f}   "
        f"CV AUPRC {table.hold_auprc.mean():.3f} +/- {table.hold_auprc.std(ddof=0):.3f}"
    )
    print(
        f"  mean train AUROC {table.train_auroc.mean():.3f}   "
        f"mean train-holdout gap {table.gap.mean():+.3f}"
    )
    print()

    # one model on the named train split, early-stopped internally, for the probe
    print("=== CALENDAR PROBE ON FINE-TUNED FEATURES ===")
    model, info = train_one(frames["train"], cache, device)
    pool_loader = DataLoader(
        ImageOnlyDataset(pool, train=False, cache=cache),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    embeddings = embed_all(model, pool_loader, device)

    encoder = ClinicalEncoder().fit(frames["train"])
    buckets = encoder._bucket_months(pool["visit_start_datetime"], "visit_start_datetime").to_numpy()
    mean_auroc, sd = calendar_probe(embeddings, buckets)
    print(f"  fine-tuned embeddings -> admission month: AUROC {mean_auroc:.3f} +/- {sd:.3f}")
    print("  frozen embeddings (reference):            AUROC 0.624 +/- 0.026")


if __name__ == "__main__":
    import sys

    if "--multiseed" in sys.argv:
        main_multiseed()
    else:
        main()
