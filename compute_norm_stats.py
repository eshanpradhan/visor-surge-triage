"""Compute pixel normalization statistics from the TRAIN split only.

Train-only is the point of this script. Val and test images are never opened
here: folding them in would let held-out intensity distributions influence the
inputs the model sees at training time, which inflates held-out metrics in a way
that is hard to notice and impossible to undo after the fact.

Two weightings are computed:

* ``per_image`` (saved) -- every image contributes equally, which matches
  training, where each image is resized to a fixed resolution and so
  contributes the same pixel count regardless of its native height.
* ``pixel_pooled`` (reported only) -- every raw pixel contributes equally.
  Native heights range 327-616, so tall images would carry more weight than
  short ones. Kept as a sanity check against the saved value.

Statistics are accumulated as sums of x and x^2 in float64 to avoid the
precision loss of a running-average over ~200M pixels.
"""

import json
import numpy as np
import pandas as pd
from PIL import Image

SPLIT_CSV = "visor_manifest_split.csv"
OUT_JSON = "norm_stats.json"


def compute_stats(path: str = SPLIT_CSV) -> dict:
    manifest = pd.read_csv(path)
    train = manifest[manifest["split"] == "train"]

    assert len(train) > 0, "no train rows found"
    assert set(train["split"]) == {"train"}, "non-train rows leaked into selection"

    per_image_means = []
    per_image_sqmeans = []
    pixel_sum = 0.0
    pixel_sqsum = 0.0
    pixel_count = 0

    for filepath in train["filepath"]:
        arr = np.asarray(Image.open(filepath), dtype=np.float64) / 255.0
        per_image_means.append(arr.mean())
        per_image_sqmeans.append((arr**2).mean())
        pixel_sum += arr.sum()
        pixel_sqsum += (arr**2).sum()
        pixel_count += arr.size

    per_image_mean = float(np.mean(per_image_means))
    per_image_std = float(np.sqrt(np.mean(per_image_sqmeans) - per_image_mean**2))

    pooled_mean = pixel_sum / pixel_count
    pooled_std = float(np.sqrt(pixel_sqsum / pixel_count - pooled_mean**2))

    return {
        "mean": round(per_image_mean, 6),
        "std": round(per_image_std, 6),
        "weighting": "per_image",
        "split": "train",
        "n_images": int(len(train)),
        "n_pixels": int(pixel_count),
        "scale": "0-1 (uint8 / 255)",
        "channels": 1,
        "pixel_pooled_mean": round(pooled_mean, 6),
        "pixel_pooled_std": round(pooled_std, 6),
        "source_manifest": SPLIT_CSV,
    }


if __name__ == "__main__":
    stats = compute_stats()
    with open(OUT_JSON, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
