"""Render demo.html: a static, dependency-free snapshot of the dashboard output.

Runs the real inference path over the synthetic demo cases and bakes the results
into a single self-contained HTML file. No Python, no server, no network at view
time -- images are embedded as base64 data URIs, charts are inline SVG, and the
what-if grids are inlined JSON, so the file works from a file:// URL anywhere.

Synthetic cases only, deliberately
----------------------------------
The live dashboard can score real held-out SBU patients, but their radiographs
and outcomes are gitignored under the data-use terms. Baking those into a
committed HTML file would publish precisely what the rest of this repo takes care
not to. demo_data.py exists for this: public NIH images, invented clinical
values, no real person.

The what-if sliders
-------------------
For each case, the five highest-attribution numeric features get a slider with
seven positions, and the FULL JOINT GRID over them is precomputed: 7^5 = 16,807
combinations per case, every one a real forward pass through the trained fusion
model followed by the isotonic calibrator. The page looks up the exact cell, so
any combination a viewer can reach is genuine model output, not an estimate.

An earlier version stored one-at-a-time sweeps and summed the deviations. That
was measured against the true joint output and was wrong by up to 24.8
percentage points, because the model is not additive across features. A grid
costs a larger file and coarser steps and is exact, which is the right trade for
something presented as a model prediction.

Sweeping is exact but cheap. A patient's image never changes, so the projected
image vector is computed once and only the clinical branch and joint head are
re-run per grid point. The two branches are independent until concatenation and
modality dropout is inactive in eval, so this is identical to a full forward
pass -- asserted against one below, not assumed.
"""

import base64
import html
import io
import json
import os
from pathlib import Path

import numpy as np

os.environ["VISOR_DEMO_MODE"] = "1"

ROOT = Path(__file__).resolve().parent
OUT_HTML = ROOT / "demo.html"
TOP_N = 8
SWEEP_FEATURES = 5
SWEEP_STEPS = 7

# Which case is selected on page load. DEMO-A's raw score sits deep inside the
# bottom bin of the isotonic calibrator, so single-slider moves rarely cross a bin
# edge and the badge appears frozen at the 1% floor -- a poor first impression even
# though nothing is wrong. DEMO-B responds to every combination. Button order is
# unchanged and all three stay one click away.
DEFAULT_CASE_ID = "DEMO-B"

SNAPSHOT_NOTE = (
    "This is a static snapshot of the live dashboard's output for demonstration "
    "purposes — see the GitHub repo for the full interactive version and source code."
)
REPO_URL = "https://github.com/eshanpradhan/visor-surge-triage"

# Clinically plausible sweep ranges, chosen to span what an admitting clinician
# might actually see rather than the full numeric range of the training data.
# A feature without an entry here is not offered as a slider.
SWEEP_RANGES = {
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": (80, 100, "%", "SpO₂"),
    "9279-1_Respiratory rate": (12, 40, "/min", "Respiratory rate"),
    "76282-3_Heart rate.beat-to-beat by EKG": (50, 140, "bpm", "Heart rate"),
    "8331-1_Oral temperature": (35.5, 40.0, "°C", "Temperature"),
    "8480-6_Systolic blood pressure": (80, 180, "mmHg", "Systolic BP"),
    "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": (0.1, 40, "mg/dL", "CRP"),
    "731-0_Lymphocytes [#/volume] in Blood by Automated count": (0.1, 3.0, "K/µL", "Lymphocytes"),
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay":
        (200, 6000, "ng/mL", "D-dimer"),
    "2524-7_Lactate [Moles/volume] in Serum or Plasma": (0.5, 6.0, "mmol/L", "Lactate"),
    "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay":
        (0.02, 5.0, "ng/mL", "Procalcitonin"),
    "2276-4_Ferritin [Mass/volume] in Serum or Plasma": (50, 3000, "ng/mL", "Ferritin"),
    "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma":
        (10, 400, "U/L", "AST"),
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma": (0.4, 4.0, "mg/dL", "Creatinine"),
    "39156-5_Body mass index (BMI) [Ratio]": (18, 45, "kg/m²", "BMI"),
    "days_prior_sx": (0, 21, "days", "Days of symptoms"),
}


def png_data_uri(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def bar_chart_svg(rows, colour: str, width: int = 430, row_height: int = 30) -> str:
    """Horizontal diverging bar chart as inline SVG. No JS, no chart library."""
    if not rows:
        return ""
    height = row_height * len(rows) + 16
    label_width = 210
    plot_width = width - label_width - 12
    centre = label_width + plot_width / 2
    largest = max(abs(value) for _, value in rows) or 1.0

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">']
    parts.append(
        f'<line x1="{centre:.1f}" y1="6" x2="{centre:.1f}" y2="{height - 10}" '
        f'stroke="currentColor" stroke-opacity=".25"/>'
    )
    for index, (label, value) in enumerate(rows):
        y = 8 + index * row_height
        length = abs(value) / largest * (plot_width / 2 - 6)
        x = centre if value >= 0 else centre - length
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" height="{row_height - 12}" '
            f'rx="2" fill="{colour}" fill-opacity="{0.9 if value >= 0 else 0.55}"/>'
        )
        parts.append(
            f'<text x="{label_width - 6}" y="{y + row_height - 16}" text-anchor="end" '
            f'font-size="11" fill="currentColor" fill-opacity=".78">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def joint_grid(context, row, columns):
    """Exact risk for every combination of the slider features.

    Builds all 7^5 rows at once, encodes them in a single call and runs one
    batched forward pass, so the whole grid costs about as much as a few hundred
    individual predictions.
    """
    import itertools

    import pandas as pd
    import torch

    import app

    encoder, model, image_repr, calibrator = context

    # Each axis is snapped to include this case's actual value, replacing whichever
    # grid point is nearest. Without that, the sliders' home position would be a
    # neighbouring grid point and resetting them would not reproduce the score shown
    # on load -- the page would contradict itself.
    axes = []
    for column in columns:
        axis = np.linspace(*SWEEP_RANGES[column][:2], SWEEP_STEPS)
        actual = float(row[column])
        if axis[0] <= actual <= axis[-1]:
            axis[int(np.abs(axis - actual).argmin())] = actual
        axes.append(axis)

    combinations = list(itertools.product(*axes))

    grid_frame = pd.DataFrame([row] * len(combinations)).reset_index(drop=True)
    for position, column in enumerate(columns):
        grid_frame[column] = [combo[position] for combo in combinations]

    encoded = encoder.transform(grid_frame).astype(np.float32)
    with torch.no_grad():
        clinical = model.clinical_branch(torch.from_numpy(encoded))
        joint = torch.cat([image_repr.expand(len(encoded), -1), clinical], dim=1)
        logits = model.head(joint).squeeze(1).numpy()

    risks = [round(app.apply_calibrator(float(v), calibrator), 4) for v in logits]
    # The isotonic calibrator is a step function fitted on 205 validation points, so
    # the calibrated probability is quantised: it holds flat across a range of logits
    # and then jumps. The underlying score does move continuously, so it is carried
    # alongside -- otherwise most sliders look broken when they are in fact working.
    raw = [round(float(1.0 / (1.0 + np.exp(-v))), 4) for v in logits]
    return axes, risks, raw


def build_cases():
    import torch
    from PIL import Image

    import app
    from demo_data import DEMO_PATIENTS

    artifacts = app.load_everything()
    assert artifacts["demo_mode"], "expected demo mode; refusing to snapshot real patients"

    model = artifacts["model"]
    encoder = artifacts["encoder"]
    frame = artifacts["test_frame"]
    indices = app.pick_demo_patients(frame, True)
    narratives = {p["demo_id"]: p for p in DEMO_PATIENTS}

    rows = tuple(tuple(float(v) for v in artifacts["test_clinical"][i]) for i in indices)
    clinical_scores = app.load_clinical_scores("demo", rows)

    cases = []
    for position, index in enumerate(indices):
        row = frame.loc[index]
        logit, cam, attribution = app.fusion_score_and_cam(artifacts, index)
        probability = app.apply_calibrator(logit, artifacts["calibrator"])
        tier, colour = app.tier_for(probability)
        base_image, blended = app.overlay_cam(row["filepath"], cam)

        # cache this case's projected image vector: it is constant across the sweep
        loader = artifacts["FM"].make_loader(
            frame.iloc[[index]], artifacts["test_clinical"][[index]], False, {}, False, True
        )
        images, clinical_tensor, _ = next(iter(loader))
        with torch.no_grad():
            image_repr = model.image_projection(model.backbone(images))
            fast_logit = float(
                model.head(
                    torch.cat([image_repr, model.clinical_branch(clinical_tensor)], dim=1)
                ).squeeze(1)
            )
        assert abs(fast_logit - logit) < 1e-4, (
            f"cached-image shortcut diverges from the full forward pass "
            f"({fast_logit} vs {logit}); the sweep would not reflect the real model"
        )

        fusion_top = app.top_contributions(artifacts["feature_names"], attribution, TOP_N)
        reference_top = app.top_contributions(
            artifacts["feature_names"], clinical_scores["contributions"][position], TOP_N
        )

        # sliders: highest-attribution numeric features that have a defined range
        context = (encoder, model, image_repr, artifacts["calibrator"])
        ranked = sorted(
            zip(artifacts["feature_names"], np.abs(attribution)), key=lambda p: -p[1]
        )
        columns, seen = [], set()
        for name, _ in ranked:
            column = name.replace("__log1p", "")
            if column in seen or column not in SWEEP_RANGES:
                continue
            seen.add(column)
            columns.append(column)
            if len(columns) == SWEEP_FEATURES:
                break

        axes, risks, raw = joint_grid(context, row, columns)
        sliders = [
            {
                "label": SWEEP_RANGES[column][3],
                "unit": SWEEP_RANGES[column][2],
                "actual": round(float(row[column]), 3),
                "values": [round(float(v), 3) for v in axis],
            }
            for column, axis in zip(columns, axes)
        ]

        meta = narratives[row["patient_id"]]
        cases.append(
            {
                "id": row["patient_id"],
                "summary": meta["summary"],
                "narrative": meta["narrative"],
                "probability": probability,
                "tier": tier,
                "colour": colour,
                "clinical_probability": clinical_scores["probabilities"][position],
                "xray": png_data_uri(base_image),
                "cam": png_data_uri(Image.fromarray(blended)),
                "sliders": sliders,
                "grid": risks,
                "raw_grid": raw,
                "raw_baseline": round(float(1.0 / (1.0 + np.exp(-logit))), 4),
                "fusion_rows": [
                    (app.shorten(f, 34), float(v))
                    for f, v in zip(fusion_top["feature"], fusion_top["contribution"])
                ],
                "reference_rows": [
                    (app.shorten(f, 34), float(v))
                    for f, v in zip(reference_top["feature"], reference_top["contribution"])
                ],
            }
        )
        print(f"  {cases[-1]['id']}: {probability:.1%} {tier}, "
              f"{len(sliders)} sliders x {SWEEP_STEPS} steps = {len(risks)} exact "
              f"grid points", flush=True)
    return cases, app.FOOTER, app.TIER_AMBER, app.TIER_RED


JS_TEMPLATE = """
const CASES = __PAYLOAD__;
const AMBER = __AMBER__, RED = __RED__;

function tierFor(p) {
  if (p >= RED) return ["HIGH", "#c0392b"];
  if (p >= AMBER) return ["MODERATE", "#d68910"];
  return ["LOW", "#1e8449"];
}

// Exact lookup into the precomputed joint grid. The index is row-major over the
// five slider axes, matching itertools.product order in the exporter. No
// interpolation and no combining of separate sweeps: an earlier additive version
// was measured wrong by up to 25 percentage points, because the model is not
// additive across features.
function gridRisk(c, positions) {
  let index = 0;
  for (const p of positions) index = index * c.steps + p;
  return c.grid[index];
}

function positionsFor(ci) {
  return CASES[ci].sliders.map((s, j) =>
    parseInt(document.getElementById(`in-${ci}-${j}`).value, 10));
}

function recompute(ci) {
  const c = CASES[ci];
  const pos = positionsFor(ci);
  const risk = gridRisk(c, pos);
  let ri = 0; for (const p of pos) ri = ri * c.steps + p;
  const raw = c.rawGrid[ri];
  const [tier, colour] = tierFor(risk);
  const rawDelta = raw - c.rawBaseline;
  const arrow = Math.abs(rawDelta) < 0.0005 ? "" :
    (rawDelta > 0 ? ` ▲ +${(rawDelta * 100).toFixed(1)}` : ` ▼ ${(rawDelta * 100).toFixed(1)}`);
  document.getElementById(`raw-${ci}`).textContent =
    `Uncalibrated model score: ${(raw * 100).toFixed(1)}%${arrow} `
    + `(at this case's values: ${(c.rawBaseline * 100).toFixed(1)}%)`;
  document.getElementById(`pct-${ci}`).textContent = Math.round(risk * 100) + "%";
  document.getElementById(`tier-${ci}`).textContent = tier + " RISK";
  document.getElementById(`risk-${ci}`).style.background = colour;

  const delta = risk - c.baseline;
  const note = document.getElementById(`delta-${ci}`);
  if (Math.abs(delta) < 0.005) { note.hidden = true; }
  else {
    note.hidden = false;
    note.textContent = `${delta > 0 ? "+" : ""}${Math.round(delta * 100)} points vs this `
      + `case's actual values (${Math.round(c.baseline * 100)}%).`;
  }
}

// The trace for slider j is the raw model score across its seven positions with
// every other slider held where the viewer has left it. It is therefore conditional
// on the current state and is redrawn whenever anything moves. All values come from
// the precomputed grid -- nothing is inferred or smoothed.
function drawSpark(ci, j) {
  const c = CASES[ci], pos = positionsFor(ci), n = c.steps;
  const series = [];
  for (let k = 0; k < n; k++) {
    const p = pos.slice(); p[k === k ? j : j] = k;
    let i = 0; for (const q of p) i = i * n + q;
    series.push(c.rawGrid[i]);
  }
  const lo = Math.min(...series), hi = Math.max(...series);
  const span = (hi - lo) || 1;
  const x = k => 4 + k * (192 / (n - 1));
  // flat traces sit mid-height rather than pinning to an edge
  const y = v => (hi - lo < 1e-6) ? 13 : 22 - ((v - lo) / span) * 18;

  const points = series.map((v, k) => `${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const here = pos[j];
  const rising = series[n - 1] >= series[0];
  const colour = rising ? "#c0392b" : "#1e8449";
  const svg = document.getElementById(`spark-${ci}-${j}`);
  svg.innerHTML =
    `<polyline points="${points}" fill="none" stroke="${colour}" stroke-width="1.6"
       stroke-opacity=".85" vector-effect="non-scaling-stroke"/>` +
    `<circle cx="${x(here).toFixed(1)}" cy="${y(series[here]).toFixed(1)}" r="3"
       fill="${colour}"/>`;

  const note = document.getElementById(`note-${ci}-${j}`);
  note.textContent = (hi - lo < 1e-6)
    ? `model score flat at ${(lo * 100).toFixed(1)}% across this range`
    : `model score ${(series[0] * 100).toFixed(1)}% → ${(series[n - 1] * 100).toFixed(1)}% `
      + `across this range (now ${(series[here] * 100).toFixed(1)}%)`;
  note.style.color = colour;
}

function onSlide(ci, j, pos) {
  const s = CASES[ci].sliders[j];
  const value = s.values[parseInt(pos, 10)];
  document.getElementById(`out-${ci}-${j}`).textContent =
    value.toFixed(Math.abs(value) < 10 ? 1 : 0) + " " + s.unit;
  recompute(ci);
  CASES[ci].sliders.forEach((_, k) => drawSpark(ci, k));
}

function resetCase(ci) {
  CASES[ci].sliders.forEach((s, j) => {
    document.getElementById(`in-${ci}-${j}`).value = s.baseIndex;
    onSlide(ci, j, s.baseIndex);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  CASES.forEach((c, ci) => c.sliders.forEach((_, j) => drawSpark(ci, j)));
});

document.addEventListener("wheel", e => {
  if (e.target instanceof HTMLInputElement && e.target.type === "range") e.target.blur();
}, {passive: true});

function showCase(n) {
  document.querySelectorAll('.case').forEach((el, i) => el.hidden = (i !== n));
  document.querySelectorAll('.case-btn').forEach((el, i) => el.classList.toggle('active', i === n));
}
"""

def render(cases, footer, amber, red) -> str:
    from demo_data import DISCLAIMER

    disclaimer = DISCLAIMER.replace("**", "")

    default = next((i for i, c in enumerate(cases) if c["id"] == DEFAULT_CASE_ID), 0)
    buttons = "".join(
        f'<button class="case-btn{" active" if i == default else ""}" '
        f'onclick="showCase({i})"><b>{html.escape(c["id"])}</b>'
        f'<span>{html.escape(c["summary"])}</span></button>'
        for i, c in enumerate(cases)
    )

    payload = json.dumps(
        [
            {
                "id": c["id"],
                "baseline": round(c["probability"], 4),
                "sliders": [
                    {k: v for k, v in s.items() if k != "actual"} | {"actual": s["actual"]}
                    for s in c["sliders"]
                ],
                "grid": c["grid"],
                "rawGrid": c["raw_grid"],
                "rawBaseline": c["raw_baseline"],
                "steps": SWEEP_STEPS,
            }
            for c in cases
        ],
        separators=(",", ":"),
    )

    panels = []
    for i, c in enumerate(cases):
        slider_rows = "".join(
            f"""<div class="slider-row">
  <label>{html.escape(s['label'])}
    <output id="out-{i}-{j}">{s['actual']:g} {html.escape(s['unit'])}</output></label>
  <input type="range" id="in-{i}-{j}" min="0" max="{SWEEP_STEPS - 1}" step="1"
         value="{s['baseIndex']}" oninput="onSlide({i},{j},this.value)">
  <svg class="spark" id="spark-{i}-{j}" viewBox="0 0 200 26" preserveAspectRatio="none"></svg>
  <p class="spark-note" id="note-{i}-{j}"></p>
</div>"""
            for j, s in enumerate(c["sliders"])
        )

        panels.append(f"""
<section class="case" id="case-{i}" {'' if i == default else 'hidden'}>
  <div class="grid">
    <div class="col">
      <h3>Chest radiograph</h3>
      <div class="imgs">
        <figure><img src="{c['xray']}" alt="Chest radiograph"><figcaption>Admission film</figcaption></figure>
        <figure><img src="{c['cam']}" alt="Grad-CAM overlay"><figcaption>Grad-CAM (fusion model)</figcaption></figure>
      </div>
      <p class="muted">Grad-CAM is computed on the fusion model being scored, so it
      reflects this prediction. The image is fixed — moving a clinical slider does not
      change it.</p>

      <h4>Explore how each factor affects this case</h4>
      <p class="muted">Precomputed from the trained model — not a live prediction tool.
      Every reachable combination of these five sliders was evaluated ahead of time
      ({SWEEP_STEPS}<sup>5</sup> = {SWEEP_STEPS ** 5:,} forward passes per case), so any
      score shown is exact model output rather than an estimate. Sliders snap to the
      precomputed values.</p>
      <p class="muted">The calibrated percentage moves in visible steps, and some
      sliders will not move it at all. That is the isotonic calibrator: fitted on 205
      validation points, it maps a range of model scores onto each of nine output
      levels. The trace under each slider shows the model score across that slider's
      whole range with your current position marked, so the direction of effect is
      visible even when the headline percentage has not crossed a step.</p>
      {slider_rows}
      <button class="reset" onclick="resetCase({i})">Reset to this case's values</button>
    </div>
    <div class="col">
      <h3>Predicted risk</h3>
      <div class="risk" id="risk-{i}" style="background:{c['colour']}">
        <div class="pct" id="pct-{i}">{c['probability']:.0%}</div>
        <div class="tier" id="tier-{i}">{c['tier']} RISK</div>
        <div class="sub">calibrated probability of ICU admission or ventilation</div>
      </div>
      <p class="baseline-note" id="delta-{i}" hidden></p>
      <p class="baseline-note" id="raw-{i}">Uncalibrated model score:
      {c['raw_baseline'] * 100:.1f}% (at this case's values:
      {c['raw_baseline'] * 100:.1f}%)</p>
      <p class="compare"><b>Clinical-only model (LightGBM):
      {c['clinical_probability']:.1%}</b><br>
      <span class="muted">Uncalibrated, and fixed at this case's actual values — the
      sliders drive the fusion model only.</span></p>
      <div class="note">{html.escape(c['narrative'])}<br>
      <span class="muted">No ground truth exists — this case is synthetic and the
      radiograph carries no COVID severity label.</span></div>
      <h4>Clinical drivers (fusion model)</h4>
      {bar_chart_svg(c['fusion_rows'], '#2b6cb0')}
      <p class="muted">Gradient × input on the fusion model's clinical branch — a
      first-order attribution of <i>this</i> score at its actual values, not an exact
      decomposition, and not recomputed as you move the sliders.</p>
      <details>
        <summary>Feature importance (clinical-only reference model)</summary>
        <p class="muted">Exact TreeSHAP from the LightGBM clinical-only model shown
        above. Explains that reference model's prediction — <b>not</b> the fusion
        model's decision.</p>
        {bar_chart_svg(c['reference_rows'], '#8a8a8a')}
      </details>
    </div>
  </div>
</section>""")

    js = (JS_TEMPLATE
          .replace("__PAYLOAD__", payload)
          .replace("__AMBER__", repr(amber))
          .replace("__RED__", repr(red)))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VISOR — surge triage demo (static snapshot)</title>
<style>
:root {{ color-scheme: light dark; --fg:#1a1a1a; --bg:#fff; --muted:#666; --line:#e2e2e2; --card:#fafafa; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg:#e8e8e8; --bg:#141414; --muted:#9a9a9a; --line:#2e2e2e; --card:#1c1c1c; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:1.6rem clamp(1rem,4vw,3rem); background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
h1 {{ font-size:1.55rem; margin:0 0 .3rem; }}
h3 {{ font-size:1rem; margin:0 0 .7rem; }}
h4 {{ font-size:.92rem; margin:1.5rem 0 .4rem; }}
.muted {{ color:var(--muted); font-size:.82rem; }}
.banner {{ border-left:4px solid #d68910; background:color-mix(in srgb,#d68910 9%,transparent);
  padding:.8rem 1rem; border-radius:5px; margin:1rem 0; font-size:.86rem; }}
.snapshot {{ border-left:4px solid #2b6cb0; background:color-mix(in srgb,#2b6cb0 9%,transparent);
  padding:.8rem 1rem; border-radius:5px; margin:1rem 0; font-size:.86rem; }}
.cases {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:1.3rem 0; }}
.case-btn {{ flex:1 1 200px; text-align:left; padding:.6rem .8rem; border:1px solid var(--line);
  background:var(--card); color:inherit; border-radius:6px; cursor:pointer; font:inherit; }}
.case-btn.active {{ border-color:#2b6cb0; box-shadow:inset 0 0 0 1px #2b6cb0; }}
.case-btn b {{ display:block; font-size:.9rem; }}
.case-btn span {{ display:block; color:var(--muted); font-size:.78rem; margin-top:.15rem; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:2rem; }}
@media (max-width:860px) {{ .grid {{ grid-template-columns:1fr; }} }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:.7rem; }}
figure {{ margin:0; }} img {{ width:100%; border-radius:5px; display:block; }}
figcaption {{ color:var(--muted); font-size:.76rem; margin-top:.3rem; text-align:center; }}
.risk {{ padding:1.1rem; border-radius:8px; color:#fff; transition:background .18s; }}
.pct {{ font-size:2.6rem; font-weight:700; line-height:1; }}
.tier {{ font-size:1rem; letter-spacing:.08em; }}
.sub {{ font-size:.78rem; opacity:.85; margin-top:.4rem; }}
.baseline-note {{ font-size:.82rem; color:var(--muted); margin:.5rem 0 0; }}
.compare {{ margin:1rem 0 .2rem; font-size:.92rem; }}
.note {{ border-left:3px solid var(--line); padding:.6rem .8rem; background:var(--card);
  border-radius:5px; font-size:.86rem; margin:.9rem 0; }}
.slider-row {{ margin:.55rem 0; }}
.slider-row label {{ display:flex; justify-content:space-between; font-size:.83rem;
  color:var(--muted); margin-bottom:.15rem; }}
.slider-row output {{ font-variant-numeric:tabular-nums; color:var(--fg); font-weight:600; }}
.slider-row input {{ width:100%; accent-color:#2b6cb0; }}
.spark {{ width:100%; height:26px; display:block; margin-top:-2px; overflow:visible; }}
.spark-note {{ font-size:.72rem; opacity:.85; margin:.1rem 0 .3rem; }}
.reset {{ margin-top:.6rem; padding:.35rem .7rem; font:inherit; font-size:.8rem;
  border:1px solid var(--line); background:var(--card); color:inherit; border-radius:5px;
  cursor:pointer; }}
details {{ margin-top:1.2rem; border-top:1px solid var(--line); padding-top:.7rem; }}
summary {{ cursor:pointer; font-size:.9rem; font-weight:600; }}
footer {{ margin-top:2.2rem; border-top:1px solid var(--line); padding-top:1rem;
  color:var(--muted); font-size:.8rem; }}
a {{ color:#2b6cb0; }}
</style></head><body>

<h1>VISOR — surge triage demo</h1>
<p class="muted">Predicting critical-care escalation (ICU or mechanical ventilation)
from an admission chest radiograph plus clinical data. Research demo — not a clinical tool.</p>

<div class="snapshot"><b>Static snapshot.</b> {html.escape(SNAPSHOT_NOTE)}
<a href="{REPO_URL}">{REPO_URL}</a></div>

<div class="banner"><b>Demonstration mode — synthetic data.</b>
{html.escape(disclaimer.split('.', 1)[1].strip())}</div>

<div class="cases">{buttons}</div>
{''.join(panels)}

<footer>
  <p>{html.escape(footer)}</p>
  <p>Tiers on calibrated probability: LOW &lt; {amber:.0%} · MODERATE {amber:.0%}–{red:.0%}
  · HIGH ≥ {red:.0%}. Model fit on the train split (n=955); val was reserved to fit the
  isotonic calibrator. Every number here was produced by the live pipeline and frozen
  into this page.</p>
</footer>

<script>
{js}
</script>
</body></html>"""


def nearest_step(slider) -> int:
    """Grid index of this case's actual value.

    joint_grid snaps one point on each axis to the actual value, so this is an
    exact hit rather than a rounding, except where the value lies outside the
    clinically plausible range, in which case it clamps to the nearest end.
    """
    values = np.asarray(slider["values"], dtype=float)
    return int(np.abs(values - float(slider["actual"])).argmin())


def main() -> None:
    print("=== building static snapshot (synthetic demo cases only) ===")
    cases, footer, amber, red = build_cases()
    for case in cases:
        for slider in case["sliders"]:
            slider["baseIndex"] = nearest_step(slider)

    OUT_HTML.write_text(render(cases, footer, amber, red), encoding="utf-8")
    size = OUT_HTML.stat().st_size / 1e6
    print(f"\nwrote {OUT_HTML.name}  ({size:.2f} MB, {len(cases)} cases, self-contained)")


if __name__ == "__main__":
    main()
