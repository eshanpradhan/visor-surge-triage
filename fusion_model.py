"""Fusion model: fine-tuned image branch + clinical MLP, joint head.

Benchmarks
----------
0.804  clinical-only, calendar-ablated  <- the honest floor
0.753  image-only, layer4 fine-tuned
0.843  clinical-only, naive             <- inflated by an admission-month feature
                                           that cannot exist prospectively; listed
                                           only for contrast, never as the target

Why the image branch is retrained per fold
------------------------------------------
The standalone image branch was validated by training layer4 from ImageNet
initialization inside each fold, on that fold's training rows only. Loading a
checkpoint fitted on a different partition would let the fusion model's image
branch see rows that its standalone counterpart never did, and the comparison
between the two numbers would stop meaning anything. So the image branch starts
from ImageNet weights in every fold here too. It costs compute and buys a
comparison that holds.

Preprocessing statistics -- imputation medians/modes and the clinical encoder's
standardization, log1p selection and category vocabularies -- are refit inside
each fold on that fold's training rows, matching the clinical baseline protocol.

Modality dropout zeroes the projected image representation with p=0.2 per sample
during training, so the joint head cannot become wholly dependent on imaging. It
is a robustness property rather than an accuracy one, and the ``--ablation`` run
measures what it costs.
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
from features import build_feature_frame
from image_baseline import IMAGENET_MEAN, IMAGENET_STD, N_FOLDS, RANDOM_STATE, pick_device
from impute import apply_impute_stats, fit_impute_stats

BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 8
LR_LAYER4 = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
INNER_VAL_FRACTION = 0.15

IMAGE_PROJECTION_DIM = 64
CLINICAL_HIDDEN = (64, 32)
MODALITY_DROPOUT_P = 0.2


class FusionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, clinical: np.ndarray, train: bool, cache: dict,
                 use_image: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.paths = self.frame["filepath"].tolist()
        self.labels = self.frame["severe"].to_numpy(dtype=np.float32)
        self.clinical = torch.from_numpy(clinical.astype(np.float32))
        self.transform = build_image_transform(train, IMAGENET_MEAN, IMAGENET_STD)
        self.cache = cache
        self.use_image = use_image
        # placeholder returned in clinical-only mode so the collate signature is
        # unchanged; nothing decodes an image that the model will not look at
        self.empty_image = torch.zeros(1)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        if not self.use_image:
            return self.empty_image, self.clinical[index], self.labels[index]
        path = self.paths[index]
        if path not in self.cache:
            with Image.open(path) as handle:
                self.cache[path] = handle.convert("L").copy()
        return self.transform(self.cache[path]), self.clinical[index], self.labels[index]


class FusionModel(nn.Module):
    def __init__(self, n_clinical: int, modality_dropout: float = MODALITY_DROPOUT_P,
                 use_image: bool = True):
        super().__init__()
        self.use_image = use_image

        if use_image:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Identity()
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            for parameter in backbone.layer4.parameters():
                parameter.requires_grad = True
            self.backbone = backbone

            # project 2048 down to 64 so the image side does not swamp the 32-dim
            # clinical representation purely by width at the concatenation
            self.image_projection = nn.Sequential(
                nn.Linear(2048, IMAGE_PROJECTION_DIM), nn.ReLU(), nn.Dropout(0.3)
            )

        layers, in_features = [], n_clinical
        for width in CLINICAL_HIDDEN:
            layers += [nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(0.3)]
            in_features = width
        self.clinical_branch = nn.Sequential(*layers)

        head_in = CLINICAL_HIDDEN[-1] + (IMAGE_PROJECTION_DIM if use_image else 0)
        self.head = nn.Linear(head_in, 1)
        self.modality_dropout = modality_dropout if use_image else 0.0
        self._training_mode = False

    def train(self, mode: bool = True) -> "FusionModel":
        """Keep every BatchNorm frozen; track intent for modality dropout."""
        self._training_mode = bool(mode)
        return super().train(False)

    def represent(self, images: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        if not self.use_image:
            return self.clinical_branch(clinical)

        image_repr = self.image_projection(self.backbone(images))
        if self._training_mode and self.modality_dropout > 0:
            # zero the whole image vector for a random subset of the batch. No
            # 1/(1-p) rescaling: the point is that the head must produce a sane
            # score from clinical data alone, not that the expectation matches.
            keep = (torch.rand(image_repr.size(0), 1, device=image_repr.device)
                    >= self.modality_dropout).float()
            image_repr = image_repr * keep
        return torch.cat([image_repr, self.clinical_branch(clinical)], dim=1)

    def forward(self, images: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return self.head(self.represent(images, clinical)).squeeze(1)


def assert_batchnorm_frozen(model: nn.Module) -> None:
    live = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
    ]
    assert not live, f"{len(live)} BatchNorm layer(s) in training mode, e.g. {live[:3]}"


def prepare_fold(fit_frame: pd.DataFrame, other_frames: dict, drop_columns: tuple = ()):
    """Refit imputation + clinical encoding on this fold's training rows only.

    ``drop_columns`` are removed before imputation and encoding, so a dropped
    column contributes nothing -- not a median, not a category vocabulary, not a
    standardization statistic. The caller keeps the undropped frames for anything
    that still needs them, such as the calendar probe's bucket labels.
    """
    if drop_columns:
        fit_frame = fit_frame.drop(columns=list(drop_columns))
        other_frames = {k: v.drop(columns=list(drop_columns)) for k, v in other_frames.items()}

    stats = fit_impute_stats(fit_frame, require_train_split=False)
    fitted = apply_impute_stats(fit_frame, stats)
    encoder = ClinicalEncoder().fit(fitted)

    out = {"__fit__": (fitted, encoder.transform(fitted))}
    for name, frame in other_frames.items():
        imputed = apply_impute_stats(frame, stats)
        out[name] = (imputed, encoder.transform(imputed))
    return out, encoder


def predict_scores(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for images, clinical, labels in loader:
            scores.append(model(images.to(device), clinical.to(device)).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(scores), np.concatenate(targets)


def represent_all(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for images, clinical, _labels in loader:
            chunks.append(model.represent(images.to(device), clinical.to(device)).cpu().numpy())
    return np.vstack(chunks)


def make_loader(frame, clinical, train, cache, shuffle, use_image=True):
    return DataLoader(
        FusionDataset(frame, clinical, train, cache, use_image),
        batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
    )


def train_fold(fit_frame, fit_clinical, cache, device, modality_dropout, seed, use_image=True):
    """Train on fit_frame, early-stopping on an inner slice carved from it."""
    indices = np.arange(len(fit_frame))
    inner_train_idx, inner_val_idx = train_test_split(
        indices,
        test_size=INNER_VAL_FRACTION,
        stratify=fit_frame["severe"].to_numpy(),
        random_state=seed,
    )
    inner_train = fit_frame.iloc[inner_train_idx]
    inner_val = fit_frame.iloc[inner_val_idx]
    assert not set(inner_train["patient_id"]) & set(inner_val["patient_id"]), "inner overlap"

    train_loader = make_loader(inner_train, fit_clinical[inner_train_idx], True, cache, True, use_image)
    train_eval = make_loader(inner_train, fit_clinical[inner_train_idx], False, cache, False, use_image)
    val_loader = make_loader(inner_val, fit_clinical[inner_val_idx], False, cache, False, use_image)

    model = FusionModel(fit_clinical.shape[1], modality_dropout, use_image).to(device)
    model.train()
    assert_batchnorm_frozen(model)

    positives = inner_train["severe"].sum()
    pos_weight = torch.tensor(
        [(len(inner_train) - positives) / positives], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    groups = [
        {"params": model.clinical_branch.parameters(), "lr": LR_HEAD},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ]
    if use_image:
        groups += [
            {"params": model.backbone.layer4.parameters(), "lr": LR_LAYER4},
            {"params": model.image_projection.parameters(), "lr": LR_HEAD},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

    best_auroc, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for images, clinical, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device), clinical.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

        scores, targets = predict_scores(model, val_loader, device)
        auroc = roc_auc_score(targets, scores)
        if auroc > best_auroc:
            best_auroc, best_epoch, stale = auroc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    train_scores, train_targets = predict_scores(model, train_eval, device)
    return model, {
        "best_epoch": best_epoch,
        "inner_val_auroc": best_auroc,
        "train_auroc": roc_auc_score(train_targets, train_scores),
    }


def calendar_probe(representations: np.ndarray, buckets: np.ndarray, seed: int = RANDOM_STATE):
    target = (buckets == "1900-12").astype(int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scores = []
    for fit_idx, hold_idx in splitter.split(representations, target):
        scaler = StandardScaler().fit(representations[fit_idx])
        probe = LogisticRegression(
            penalty="l2", C=0.01, class_weight="balanced", max_iter=5000, random_state=seed
        ).fit(scaler.transform(representations[fit_idx]), target[fit_idx])
        scores.append(
            roc_auc_score(
                target[hold_idx],
                probe.predict_proba(scaler.transform(representations[hold_idx]))[:, 1],
            )
        )
    return float(np.mean(scores)), float(np.std(scores))


def run_cv(pool, cache, device, modality_dropout, seed=RANDOM_STATE, probe=False,
           drop_columns=(), use_image=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    labels = pool["severe"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    rows, seen = [], set()
    probe_reprs, probe_order = [], []

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(pool, labels), start=1):
        fit_frame, hold_frame = pool.iloc[fit_idx], pool.iloc[hold_idx]
        assert not set(fit_frame["patient_id"]) & set(hold_frame["patient_id"]), "fold overlap"
        assert not seen & set(hold_frame["patient_id"]), "repeated holdout patient"
        seen |= set(hold_frame["patient_id"])

        prepared, _encoder = prepare_fold(fit_frame, {"hold": hold_frame}, drop_columns)
        fit_imputed, fit_clinical = prepared["__fit__"]
        hold_imputed, hold_clinical = prepared["hold"]

        model, info = train_fold(
            fit_imputed, fit_clinical, cache, device, modality_dropout, seed, use_image
        )
        hold_loader = make_loader(hold_imputed, hold_clinical, False, cache, False, use_image)
        scores, targets = predict_scores(model, hold_loader, device)

        row = {
            "fold": fold,
            "epoch": info["best_epoch"],
            "train_auroc": round(info["train_auroc"], 3),
            "inner_val_auroc": round(info["inner_val_auroc"], 3),
            "hold_auroc": round(roc_auc_score(targets, scores), 3),
            "hold_auprc": round(average_precision_score(targets, scores), 3),
        }
        row["gap"] = round(row["train_auroc"] - row["hold_auroc"], 3)
        rows.append(row)
        print(f"  fold {fold}: {row}", flush=True)

        if probe:
            # fused representations for held-out rows only, so the probe never
            # reads a representation of a row the model was trained on
            probe_reprs.append(represent_all(model, hold_loader, device))
            probe_order.append(hold_frame)
        del model

    assert len(seen) == len(pool), "CV did not cover every patient exactly once"
    table = pd.DataFrame(rows)
    if probe:
        return table, np.vstack(probe_reprs), pd.concat(probe_order)
    return table, None, None


def summarize(name: str, table: pd.DataFrame) -> None:
    print(f"  {name:28s} CV AUROC {table.hold_auroc.mean():.3f} +/- "
          f"{table.hold_auroc.std(ddof=0):.3f}   "
          f"AUPRC {table.hold_auprc.mean():.3f} +/- {table.hold_auprc.std(ddof=0):.3f}   "
          f"mean gap {table.gap.mean():+.3f}")


def main() -> None:
    pd.set_option("display.width", 220)
    device = pick_device()

    pool = build_feature_frame(split=None)
    pool = pool[pool["split"].isin(["train", "val"])].reset_index(drop=True)
    assert pool["patient_id"].is_unique, "duplicate patient in CV pool"
    cache: dict = {}

    print(f"=== FUSION MODEL (device={device}) ===")
    print(f"  image: ResNet-50 layer4 fine-tuned per fold -> {IMAGE_PROJECTION_DIM}-d")
    print(f"  clinical: encoded -> {' -> '.join(map(str, CLINICAL_HIDDEN))}")
    print(f"  benchmarks: clinical-only (calendar-ablated) 0.804 | image-only 0.753")
    print()

    print(f"--- WITH modality dropout (p={MODALITY_DROPOUT_P}) ---")
    with_table, fused, fused_frame = run_cv(
        pool, cache, device, MODALITY_DROPOUT_P, probe=True
    )
    print()

    print("--- WITHOUT modality dropout (p=0.0) ---")
    without_table, _, _ = run_cv(pool, cache, device, 0.0)
    print()

    print("=== PER-FOLD, WITH modality dropout ===")
    print(with_table.to_string(index=False))
    print()
    print("=== PER-FOLD, WITHOUT modality dropout ===")
    print(without_table.to_string(index=False))
    print()

    print("=== ABLATION SUMMARY ===")
    summarize(f"fusion + modality dropout", with_table)
    summarize("fusion, no modality dropout", without_table)
    delta = without_table.hold_auroc.mean() - with_table.hold_auroc.mean()
    print(f"  cost of modality dropout: {-delta:+.3f} AUROC")
    print()

    print("=== BENCHMARKS ===")
    print(f"  clinical-only, calendar-ablated   0.804")
    print(f"  image-only, layer4 fine-tuned     0.753")
    print(f"  fusion (+dropout)                 {with_table.hold_auroc.mean():.3f}")
    print(f"  fusion (no dropout)               {without_table.hold_auroc.mean():.3f}")
    print(f"  [clinical-only, naive             0.843  <- inflated, not a target]")
    print()

    encoder = ClinicalEncoder().fit(
        apply_impute_stats(pool, fit_impute_stats(pool, require_train_split=False))
    )
    buckets = encoder._bucket_months(
        fused_frame["visit_start_datetime"], "visit_start_datetime"
    ).to_numpy()
    mean_auroc, sd = calendar_probe(fused, buckets)
    print("=== CALENDAR PROBE ON FUSED REPRESENTATION ===")
    print(f"  fused representation -> admission month: AUROC {mean_auroc:.3f} +/- {sd:.3f}")
    print("  image-only, fine-tuned (reference):      AUROC 0.636 +/- 0.058")
    print("  image-only, frozen (reference):          AUROC 0.624 +/- 0.026")
    print("  note: the clinical branch contains the admission-month one-hot by design,")
    print("        so a high value here is expected and is not evidence about the image side.")

    with_table.to_csv("fusion_cv_with_dropout.csv", index=False)
    without_table.to_csv("fusion_cv_no_dropout.csv", index=False)


if __name__ == "__main__":
    main()
