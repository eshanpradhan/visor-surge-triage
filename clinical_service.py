"""Torch-free LightGBM scoring service, invoked as a subprocess by app.py.

LightGBM cannot run in a process that has imported torch: on macOS the two
OpenMP runtimes collide and any call into the booster segfaults, including a
single-row predict. The Streamlit app needs torch for the fusion model and
Grad-CAM, so the clinical comparison model is scored here instead.

The caller sends an already-encoded feature matrix on stdin and receives
probabilities and exact TreeSHAP contributions on stdout. Taking the matrix
rather than re-deriving it keeps this script independent of the training data,
so it works identically for real patients and for demo mode, which has no access
to the cohort files.

Protocol:
    stdin   {"features": [[...74 floats...], ...]}
    stdout  {"probabilities": [...], "contributions": [[...], ...]}

Nothing is written to disk.
"""

import json
import sys

BOOSTER_PATH = "models/clinical_lightgbm_calendar_ablated.txt"


def main() -> None:
    assert "torch" not in sys.modules, "torch is loaded; LightGBM will segfault"
    import lightgbm as lgb
    import numpy as np

    payload = json.load(sys.stdin)
    features = np.asarray(payload["features"], dtype=float)
    assert features.ndim == 2, f"expected a 2-D feature matrix, got shape {features.shape}"

    booster = lgb.Booster(model_file=BOOSTER_PATH)
    assert booster.num_feature() == features.shape[1], (
        f"booster expects {booster.num_feature()} features, received {features.shape[1]}"
    )

    probabilities = booster.predict(features)
    contributions = booster.predict(features, pred_contrib=True)

    json.dump(
        {
            "probabilities": [float(p) for p in probabilities],
            # final column is the model's base value, not a feature
            "contributions": [row[:-1].tolist() for row in contributions],
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
