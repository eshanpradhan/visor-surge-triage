"""VISOR — local Streamlit demo. Run with: streamlit run app.py

Loads the committed artifacts in models/ and scores real patients from the test
split. Inference is local: no telemetry, and no patient data leaves the process.

One exception to "no network", on first boot only. The checkpoint stores just the
68 tensors that were trained; the ResNet-50 layers below layer4 are frozen at
ImageNet initialization and are fetched by torchvision. On a machine with a warm
torch cache that is a disk read, but a fresh deployment downloads ~98 MB from
download.pytorch.org before the first prediction. Nothing about a patient is sent
anywhere -- it is a one-time model-weight fetch -- but it does mean the first load
is slow, so it gets its own progress state rather than hiding inside a generic
spinner.

Two modes
---------
Real patients are read at runtime from visor_manifest_split.csv and the image
archive, which are gitignored per data-use terms and are deliberately NOT baked
into this file -- hardcoding patient rows here would put them in git history.
When those files are absent (a fresh clone, or a deployment) the app falls back
to demo_data.py: synthetic cases with public NIH images, behind a banner.

What each explanation explains
------------------------------
Grad-CAM and the "Clinical drivers (fusion model)" chart both come from one
backward pass through the fusion model, so together they attribute the score
actually on screen across both of its inputs. The clinical attribution is
gradient x input -- a local linearization, not an exact decomposition.

"Feature importance (clinical-only reference model)" is exact TreeSHAP from the
LightGBM comparison model. It explains that model, not the fusion decision, and
is collapsed and labelled accordingly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CALENDAR_COLUMN = "visit_start_datetime"

# calibrated-probability tiers. Cohort base rate is 20%, so amber starts below it
TIER_AMBER = 0.10
TIER_RED = 0.35

FOOTER = (
    "Fusion: 0.793 ± 0.009 CV AUROC (0.834 test) vs. Clinical-only: 0.804 CV AUROC "
    "— see README for full methodology"
)

st.set_page_config(page_title="VISOR — surge triage demo", layout="wide")


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------

def backbone_weights_cached() -> bool:
    """Is the ImageNet ResNet-50 checkpoint already on disk?

    torchvision downloads it on demand, so this is only used to decide whether to
    warn the user about a slow first load -- not to gate anything.
    """
    import torch
    from torchvision.models import ResNet50_Weights

    url = ResNet50_Weights.IMAGENET1K_V1.url
    destination = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    return destination.exists()


@st.cache_resource(show_spinner=False)
def ensure_backbone_weights() -> bool:
    """Fetch the ImageNet weights up front so the wait is visible and explained.

    Without this the download happens silently inside model construction, and a
    cold deployment looks hung for a minute with a spinner that says it is
    loading models. Returns True if a download was performed.
    """
    import torch
    from torchvision.models import ResNet50_Weights

    if backbone_weights_cached():
        return False

    with st.status(
        "First run: downloading ImageNet ResNet-50 weights (~98 MB). "
        "This happens once, then it is cached.",
        expanded=False,
    ):
        torch.hub.load_state_dict_from_url(
            ResNet50_Weights.IMAGENET1K_V1.url, progress=False
        )
    return True


def real_cohort_available() -> bool:
    """Demo mode is the fallback whenever the gitignored cohort files are absent.

    A deployed copy of this app has the committed artifacts but not the patient
    data, so it lands in demo mode automatically rather than erroring. Setting
    VISOR_DEMO_MODE=1 forces demo mode even on a machine that has the data,
    which is how to rehearse the public build locally.
    """
    import os

    if os.environ.get("VISOR_DEMO_MODE") == "1":
        return False
    return (ROOT / "visor_manifest_split.csv").exists() and (ROOT / "COVID-19-NY-SBU").is_dir()


@st.cache_resource(show_spinner="Loading models and cohort…")
def load_everything():
    import torch

    import fusion_model as FM

    device = torch.device("cpu")  # one image at a time; CPU keeps this deterministic
    demo_mode = not real_cohort_available()

    if demo_mode:
        # encoder restored from committed statistics; no training data required
        from demo_data import build_demo_frame

        test_imputed, encoder = build_demo_frame()
        test_clinical = encoder.transform(test_imputed)
    else:
        from features import build_feature_frame

        pool = build_feature_frame(split=None)
        train = pool[pool["split"] == "train"].reset_index(drop=True)
        test = pool[pool["split"] == "test"].reset_index(drop=True)

        # same preprocessing path as training: fit on train, apply to test
        prepared, encoder = FM.prepare_fold(train, {"test": test}, (CALENDAR_COLUMN,))
        test_imputed, test_clinical = prepared["test"]

    n_clinical = test_clinical.shape[1]

    model = FM.FusionModel(n_clinical, FM.MODALITY_DROPOUT_P, use_image=True)
    # the checkpoint is stored in fp16 to halve its size; the forward pass runs in
    # fp32, so cast on load rather than relying on load_state_dict to convert
    state = {
        key: value.float()
        for key, value in torch.load(MODELS / "fusion_no_calendar.pt", map_location="cpu").items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:3]}"
    model.eval().to(device)

    with open(MODELS / "calibrator_isotonic.json") as fh:
        calibrator = json.load(fh)

    return {
        "device": device,
        "FM": FM,
        "model": model,
        "encoder": encoder,
        "test_frame": test_imputed.reset_index(drop=True),
        "test_clinical": test_clinical,
        "calibrator": calibrator,
        "feature_names": encoder.feature_names,
        "demo_mode": demo_mode,
    }


@st.cache_resource(show_spinner="Scoring the clinical comparison model…")
def load_clinical_scores(feature_key: str, features: tuple) -> dict:
    """Score the clinical LightGBM model in a separate, torch-free process.

    LightGBM cannot be called at all from a process that has imported torch: the
    two OpenMP runtimes collide and even a single-row predict segfaults with no
    traceback. Streamlit needs torch in-process for the fusion model and
    Grad-CAM, so the booster runs in a subprocess. The already-encoded matrix is
    sent on stdin, which keeps the subprocess independent of the training data
    and so works in demo mode too.

    ``feature_key`` exists only to give the cache a cheap hashable identity.
    """
    import subprocess
    import sys

    payload = json.dumps({"features": [list(row) for row in features]})
    result = subprocess.run(
        [sys.executable, str(ROOT / "clinical_service.py")],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"clinical_service.py failed (exit {result.returncode}):\n{result.stderr[-800:]}"
        )
    return json.loads(result.stdout)


def apply_calibrator(logit: float, calibrator: dict) -> float:
    """Isotonic regression is piecewise-linear between its fitted thresholds.

    The output is clipped away from 0 and 1. Isotonic saturates in its end bins
    whenever every calibration point beyond a threshold shared one outcome, which
    at 205 calibration rows happens readily and reflects the size of that set
    rather than certainty about the patient. Displaying "100% risk" from a model
    fitted on 205 examples would overstate what it can support.
    """
    x = np.asarray(calibrator["x_thresholds"], dtype=float)
    y = np.asarray(calibrator["y_thresholds"], dtype=float)
    probability = float(np.interp(logit, x, y, left=y[0], right=y[-1]))
    return float(np.clip(probability, 0.01, 0.99))


def tier_for(probability: float) -> tuple[str, str]:
    if probability >= TIER_RED:
        return "HIGH", "#c0392b"
    if probability >= TIER_AMBER:
        return "MODERATE", "#d68910"
    return "LOW", "#1e8449"


# --------------------------------------------------------------------------
# inference + explanations
# --------------------------------------------------------------------------

def fusion_score_and_cam(artifacts, row_index: int):
    """Return (logit, grad-CAM HxW in 0-1, clinical gradient x input attributions).

    Both explanations come from the same backward pass, and both describe the
    fusion model that produced the displayed score.
    """
    import torch

    FM = artifacts["FM"]
    model = artifacts["model"]
    device = artifacts["device"]

    frame = artifacts["test_frame"].iloc[[row_index]]
    clinical = artifacts["test_clinical"][[row_index]]
    loader = FM.make_loader(frame, clinical, False, {}, False, True)
    images, clinical_tensor, _label = next(iter(loader))
    images = images.to(device)
    clinical_tensor = clinical_tensor.to(device).requires_grad_(True)

    activations = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        activations["value"] = output

    handle = model.backbone.layer4.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logit = model(images, clinical_tensor)
            model.zero_grad(set_to_none=True)
            logit.backward()

        maps = activations["value"]
        gradients = maps.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * maps).sum(dim=1))[0]
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]

        # gradient x input on the standardized clinical vector: a first-order
        # attribution of this logit to each clinical feature. Cheaper and weaker
        # than SHAP -- it is a local linearization, not an exact decomposition --
        # but it does describe the fusion model rather than a stand-in.
        clinical_attribution = (
            clinical_tensor.grad[0] * clinical_tensor.detach()[0]
        ).cpu().numpy()
    finally:
        handle.remove()

    return float(logit.detach().cpu()), cam.detach().cpu().numpy(), clinical_attribution


def overlay_cam(image_path: str, cam: np.ndarray):
    """Blend the Grad-CAM heatmap over the padded, resized radiograph."""
    import matplotlib.cm as cm
    from PIL import Image

    from dataset import PadToSquare

    with Image.open(image_path) as handle:
        base = handle.convert("L")
    base = PadToSquare()(base).resize((224, 224))
    base_rgb = np.stack([np.asarray(base, dtype=float) / 255.0] * 3, axis=-1)

    heat = cm.get_cmap("jet")(cam)[:, :, :3]
    blended = 0.6 * base_rgb + 0.4 * heat
    return base, (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def top_contributions(names, values, top_n: int = 8) -> pd.DataFrame:
    frame = pd.DataFrame({"feature": names, "contribution": values})
    frame["magnitude"] = frame["contribution"].abs()
    return frame.nlargest(top_n, "magnitude").sort_values("contribution")


def shorten(name: str, limit: int = 42) -> str:
    """LOINC column names are unreadable at full length in a chart."""
    label = name.split("_", 1)[-1] if name[:1].isdigit() else name
    label = label.replace("__log1p", " (log)").replace("__ordinal", "")
    for noise in [" [Mass/volume]", " [#/volume]", " [Moles/volume]", " in Serum or Plasma",
                  " in Blood by Automated count", " [Ratio]", " by Immunoassay"]:
        label = label.replace(noise, "")
    return label if len(label) <= limit else label[: limit - 1] + "…"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def pick_demo_patients(_frame: pd.DataFrame, demo_mode: bool,
                       n_severe: int = 3, n_mild: int = 2):
    """Which rows to offer as one-click cases.

    In demo mode every synthetic patient is shown. Otherwise a fixed,
    reproducible mix of real outcomes is drawn from the test split.
    """
    if demo_mode:
        return list(range(len(_frame)))

    severe = _frame.index[_frame["severe"]].tolist()
    mild = _frame.index[~_frame["severe"]].tolist()
    rng = np.random.default_rng(42)
    chosen = list(rng.choice(severe, n_severe, replace=False)) + \
             list(rng.choice(mild, n_mild, replace=False))
    return [int(i) for i in chosen]


def main() -> None:
    st.title("VISOR — surge triage demo")
    st.caption(
        "Predicting critical-care escalation (ICU or mechanical ventilation) from an "
        "admission chest radiograph plus clinical data. Research demo — not a clinical tool."
    )

    downloaded = ensure_backbone_weights()

    try:
        artifacts = load_everything()
    except FileNotFoundError as error:
        st.error(
            f"Missing a required file: `{error}`\n\n"
            "The committed artifacts in `models/` and `demo_assets/` are needed even in "
            "demo mode. For real patients, regenerate the cohort files by running the "
            "pipeline in the order documented in README.md."
        )
        return

    if downloaded:
        st.caption(
            "ImageNet backbone downloaded and cached on this instance — later loads are fast."
        )

    demo_mode = artifacts["demo_mode"]
    if demo_mode:
        from demo_data import DEMO_PATIENTS, DISCLAIMER

        st.warning(DISCLAIMER)
        narratives = {p["demo_id"]: p for p in DEMO_PATIENTS}

    frame = artifacts["test_frame"]
    demo_indices = pick_demo_patients(frame, demo_mode)

    with st.sidebar:
        st.header("Demo patients" if demo_mode else "Patients")
        st.caption(
            "Synthetic cases — not real people."
            if demo_mode
            else "Held-out test split. Ground truth shown after scoring."
        )
        if "selected" not in st.session_state:
            st.session_state.selected = demo_indices[0]
        for index in demo_indices:
            row = frame.loc[index]
            if demo_mode:
                label = f"{row['patient_id']} · {narratives[row['patient_id']]['summary']}"
            else:
                label = f"{row['patient_id']} · {row['age.splits']} · {row['gender_concept_name']}"
            if st.button(label, key=f"patient_{index}", use_container_width=True):
                st.session_state.selected = index
        st.divider()
        st.caption(
            f"Tiers on calibrated probability: LOW < {TIER_AMBER:.0%} · "
            f"MODERATE {TIER_AMBER:.0%}–{TIER_RED:.0%} · HIGH ≥ {TIER_RED:.0%}"
        )

    index = st.session_state.selected
    row = frame.loc[index]

    rows = tuple(tuple(float(v) for v in artifacts["test_clinical"][i]) for i in demo_indices)
    clinical_scores = load_clinical_scores("demo" if demo_mode else "test", rows)
    position = demo_indices.index(index)

    with st.spinner("Scoring…"):
        logit, cam, clinical_attribution = fusion_score_and_cam(artifacts, index)
        probability = apply_calibrator(logit, artifacts["calibrator"])
        base_image, blended = overlay_cam(row["filepath"], cam)

    clinical_probability = clinical_scores["probabilities"][position]
    fusion_top = top_contributions(artifacts["feature_names"], clinical_attribution)
    reference_top = top_contributions(
        artifacts["feature_names"], clinical_scores["contributions"][position]
    )

    tier, colour = tier_for(probability)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Chest radiograph")
        image_col, cam_col = st.columns(2)
        image_col.image(base_image, caption="Admission film", use_container_width=True)
        cam_col.image(blended, caption="Grad-CAM (fusion model)", use_container_width=True)
        st.caption(
            "Grad-CAM is computed on the fusion model being scored, so it reflects "
            "this prediction."
        )

    with right:
        st.subheader("Predicted risk")
        st.markdown(
            f"<div style='padding:1.1rem;border-radius:8px;background:{colour};color:white'>"
            f"<div style='font-size:2.6rem;font-weight:700;line-height:1'>{probability:.0%}</div>"
            f"<div style='font-size:1rem;letter-spacing:.08em'>{tier} RISK</div>"
            f"<div style='font-size:.8rem;opacity:.85;margin-top:.4rem'>"
            f"calibrated probability of ICU admission or ventilation</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**Clinical-only model (LightGBM): {clinical_probability:.0%}**  \n"
            f"<span style='color:#666;font-size:.85rem'>Uncalibrated. The gap against the "
            f"fusion score is what the radiograph contributes for this patient.</span>",
            unsafe_allow_html=True,
        )

        if demo_mode:
            st.info(
                f"**{narratives[row['patient_id']]['narrative']}**  \n"
                "No ground truth exists — this case is synthetic and the radiograph "
                "carries no COVID severity label."
            )
        else:
            actual = "ICU or ventilated" if row["severe"] else "neither"
            st.info(f"Ground truth for this held-out patient: **{actual}**")

        st.markdown("**Clinical drivers (fusion model)**")
        chart = fusion_top.copy()
        chart["feature"] = chart["feature"].map(shorten)
        st.bar_chart(
            chart.set_index("feature")["contribution"],
            horizontal=True,
            color="#2b6cb0",
            height=290,
        )
        st.caption(
            "Gradient × input on the fusion model's clinical branch — a first-order "
            "attribution of *this* score, not an exact decomposition. Together with the "
            "Grad-CAM overlay it covers both of the fusion model's inputs."
        )

    with st.expander("Feature importance (clinical-only reference model)"):
        st.markdown(
            "Exact TreeSHAP from the **LightGBM clinical-only model** shown above for "
            "comparison. This explains that reference model's prediction — **not** the "
            "fusion model's decision. Shown because TreeSHAP is exact where the "
            "gradient attribution above is approximate, which makes it a useful "
            "cross-check on which clinical variables matter for this patient."
        )
        reference_chart = reference_top.copy()
        reference_chart["feature"] = reference_chart["feature"].map(shorten)
        st.bar_chart(
            reference_chart.set_index("feature")["contribution"],
            horizontal=True,
            color="#8a8a8a",
            height=290,
        )
        st.caption("Log-odds contributions, clinical-only LightGBM (calendar-ablated).")

    st.divider()
    st.caption(FOOTER)
    st.caption(
        "Inference runs locally — no patient data leaves this machine. The frozen "
        "ImageNet backbone is fetched from torchvision once on first run and cached "
        "thereafter. Model fit on the train split (n=955); val was reserved to fit "
        "the calibrator."
    )


if __name__ == "__main__":
    main()
