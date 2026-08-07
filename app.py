"""VISOR — local Streamlit demo. Run with: streamlit run app.py

Loads the committed artifacts in models/ and scores real patients from the test
split. Everything runs on this machine: no network calls, no telemetry, no
patient data leaving the process. The ResNet-50 layers below layer4 come from
torchvision's local cache rather than a download.

Requires the local data files
-----------------------------
Demo patients are read at runtime from visor_manifest_split.csv and the image
archive. Those are gitignored per data-use terms and are deliberately NOT baked
into this file -- hardcoding patient rows here would put them in git history.
Anyone cloning the repo needs to regenerate them by running the pipeline in the
order documented in README.md.

What the two explanations actually explain
------------------------------------------
Grad-CAM is computed on the fusion model itself, so it shows what the displayed
risk score is looking at.

The SHAP bar chart is exact TreeSHAP from the clinical-only LightGBM booster,
not from the fusion model's clinical branch. It explains the comparison model
shown alongside, and is labelled that way in the UI. Attributing the fusion
model's clinical pathway would need a sampling-based explainer and is not what
this chart is.
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

@st.cache_resource(show_spinner="Loading models and cohort…")
def load_everything():
    import torch

    import fusion_model as FM
    from features import build_feature_frame

    device = torch.device("cpu")  # one image at a time; CPU keeps this deterministic

    pool = build_feature_frame(split=None)
    train = pool[pool["split"] == "train"].reset_index(drop=True)
    test = pool[pool["split"] == "test"].reset_index(drop=True)

    # same preprocessing path as training: fit on train, apply to test
    prepared, encoder = FM.prepare_fold(train, {"test": test}, (CALENDAR_COLUMN,))
    test_imputed, test_clinical = prepared["test"]
    n_clinical = test_clinical.shape[1]

    model = FM.FusionModel(n_clinical, FM.MODALITY_DROPOUT_P, use_image=True)
    state = torch.load(MODELS / "fusion_no_calendar.pt", map_location="cpu")
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
        "test_frame": test_imputed,
        "test_clinical": test_clinical,
        "calibrator": calibrator,
        "feature_names": encoder.feature_names,
    }


@st.cache_resource(show_spinner="Scoring the clinical comparison model…")
def load_clinical_scores(indices: tuple[int, ...]) -> dict:
    """Score the clinical LightGBM model in a separate, torch-free process.

    LightGBM cannot be called at all from a process that has imported torch: the
    two OpenMP runtimes collide and even a single-row predict segfaults with no
    traceback. Streamlit needs torch in-process for the fusion model and
    Grad-CAM, so the booster runs in a subprocess and returns JSON. One call
    covers every demo patient, so this happens once per session.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "clinical_service.py"), *map(str, indices)],
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
def pick_demo_patients(_frame: pd.DataFrame, n_severe: int = 3, n_mild: int = 2):
    """A fixed, reproducible mix of outcomes from the test split."""
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

    try:
        artifacts = load_everything()
    except FileNotFoundError as error:
        st.error(
            f"Missing a local data file: `{error}`\n\n"
            "Demo patients are read from the gitignored cohort files. Regenerate them by "
            "running the pipeline in the order documented in README.md."
        )
        return

    frame = artifacts["test_frame"]
    demo_indices = pick_demo_patients(frame)

    with st.sidebar:
        st.header("Demo patients")
        st.caption("Held-out test split. Ground truth shown after scoring.")
        if "selected" not in st.session_state:
            st.session_state.selected = demo_indices[0]
        for index in demo_indices:
            row = frame.loc[index]
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

    clinical_scores = load_clinical_scores(tuple(demo_indices))

    with st.spinner("Scoring…"):
        logit, cam, clinical_attribution = fusion_score_and_cam(artifacts, index)
        probability = apply_calibrator(logit, artifacts["calibrator"])
        base_image, blended = overlay_cam(row["filepath"], cam)

    entry = clinical_scores["patients"][str(index)]
    clinical_probability = entry["probability"]
    fusion_top = top_contributions(artifacts["feature_names"], clinical_attribution)
    reference_top = top_contributions(clinical_scores["feature_names"], entry["contributions"])

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
        "Runs entirely locally — no network calls, no data leaves this machine. "
        "Model fit on the train split (n=955); val was reserved to fit the calibrator."
    )


if __name__ == "__main__":
    main()
