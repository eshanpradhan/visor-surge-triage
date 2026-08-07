"""PyTorch Dataset and DataLoaders for the VISOR chest X-ray severity task.

Each item is ``(image, clinical, label)``:

* ``image``    float32 [3, 224, 224], normalized with train-split statistics
* ``clinical`` float32 [n_features], categoricals encoded, nulls already filled
* ``label``    float32 scalar, ``severe`` as 0.0 / 1.0

Design notes
------------
*Aspect-preserving pad, then resize.* Native heights run 327-616 against a fixed
width of 512, so a direct resize to a square would stretch the ~22% of images
outside the 425/426 mode differently from the rest. Padding to a square with
black first keeps thoracic proportions constant across the cohort.

*Encoders and normalization are fit on train only.* Category vocabularies, the
date origin, and the feature standardization all come from the train split, for
the same reason the pixel and imputation statistics do.

*Augmentation is train-only.* Rotation and brightness/contrast jitter are applied
before padding so that rotation-exposed corners land on the black border rather
than inside the lung fields. Val and test go through the deterministic path.

Known limitation, worth restating where results are reported: roughly 20% of the
comorbidity flags were null and were filled with the train mode (the majority
class) rather than an explicit "Unknown" level, because an "Unknown" level is a
missingness indicator and missingness in this dataset leaks severity. This may
understate true comorbidity prevalence. See impute.py.
"""

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from features import PASSTHROUGH
from impute import build_imputed_frames

NORM_STATS_JSON = "norm_stats.json"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Encoding logic lives in clinical_encoding, which imports no torch, so that
# LightGBM-based evaluation can run in a torch-free process. Re-exported here so
# existing `from dataset import ClinicalEncoder` imports keep working.
from clinical_encoding import (  # noqa: E402
    ClinicalEncoder,
    LOG1P_MAX_Z_THRESHOLD,
    MIN_MONTH_BUCKET_N,
    ORDINAL_LEVELS,
)


def load_norm_stats(path: str = NORM_STATS_JSON) -> tuple[float, float]:
    with open(path) as fh:
        stats = json.load(fh)
    assert stats["split"] == "train", "norm_stats.json was not fit on the train split"
    return float(stats["mean"]), float(stats["std"])


class PadToSquare:
    """Pad the shorter side with black so the image is square before resizing."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        if width == height:
            return image
        canvas = Image.new(image.mode, (side, side), color=0)
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


def build_image_transform(train: bool, mean, std) -> transforms.Compose:
    """Deterministic pipeline, plus light augmentation when ``train``.

    ``mean``/``std`` accept either a scalar (broadcast to all three channels, the
    dataset-statistics case) or a 3-sequence (the ImageNet-statistics case).
    """
    mean = [float(mean)] * 3 if np.isscalar(mean) else [float(v) for v in mean]
    std = [float(std)] * 3 if np.isscalar(std) else [float(v) for v in std]
    assert len(mean) == len(std) == 3, "mean/std must resolve to three channels"

    steps = []
    if train:
        # light augmentation: rotation first so exposed corners fall on the pad border
        steps.append(transforms.RandomRotation(degrees=7, fill=0))
        steps.append(transforms.ColorJitter(brightness=0.10, contrast=0.10))
    steps += [
        PadToSquare(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # [1, H, W] in 0-1
        transforms.Lambda(lambda t: t.expand(3, -1, -1)),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


class VisorDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        encoder: ClinicalEncoder,
        train: bool,
        mean: float,
        std: float,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.clinical = torch.from_numpy(encoder.transform(self.frame))
        self.labels = torch.tensor(self.frame["severe"].to_numpy(dtype=np.float32))
        self.paths = self.frame["filepath"].tolist()
        self.transform = build_image_transform(train=train, mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as handle:
            image = handle.convert("L")
        return self.transform(image), self.clinical[index], self.labels[index]


def build_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 0) -> dict[str, DataLoader]:
    frames, _ = build_imputed_frames()
    mean, std = load_norm_stats()

    encoder = ClinicalEncoder().fit(frames["train"])

    loaders = {}
    for split in ["train", "val", "test"]:
        is_train = split == "train"
        dataset = VisorDataset(frames[split], encoder, is_train, mean, std)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders, encoder


if __name__ == "__main__":
    torch.manual_seed(42)
    loaders, encoder = build_dataloaders()

    print(f"clinical features after encoding: {len(encoder.feature_names)}")
    print(
        f"  numeric {len(encoder.numeric_columns)} (log1p {len(encoder.log1p_columns)})   "
        f"date {len(encoder.date_columns)} -> buckets {encoder.month_buckets}   "
        f"ordinal {len(encoder.ordinal_columns)}   one-hot groups {len(encoder.onehot_levels)}"
    )
    print()

    for split, loader in loaders.items():
        images, clinical, labels = next(iter(loader))
        print(
            f"{split:6s} batches={len(loader):3d}  "
            f"image={tuple(images.shape)} {images.dtype}  "
            f"clinical={tuple(clinical.shape)}  labels={tuple(labels.shape)}"
        )
        print(
            f"        image range [{images.min():.2f}, {images.max():.2f}]  "
            f"clinical range [{clinical.min():.2f}, {clinical.max():.2f}]  "
            f"positives in batch={int(labels.sum())}/{len(labels)}"
        )
