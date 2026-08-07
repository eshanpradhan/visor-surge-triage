"""Fill nulls in the clinical feature frame using TRAIN-split statistics only.

Same leak-prevention pattern as norm_stats.json: fit on train, apply everywhere.
Val and test rows never contribute to a fill value.

No missingness indicators
-------------------------
Nulls are filled silently and no ``*_isnull`` style column is ever created. This
is not stylistic. Missingness in this dataset is strongly informative in exactly
the wrong way: among the ~208 patients missing a basic metabolic panel, zero are
severe (Fisher OR 0.00, p ~ 2.5e-22), because the panel was ordered on every
patient sick enough to be escalated. A missingness indicator would hand the
model a near-perfect shortcut that reflects clinician test-ordering behaviour,
not radiographic or physiologic signal. :func:`assert_no_missingness_indicators`
enforces this, and the column set is asserted identical before and after
imputation so nothing can be appended.

For the same reason, categorical nulls are filled with the train mode rather
than an explicit "Unknown" level -- an "Unknown" category is a missingness
indicator wearing a different hat.

What this costs
---------------
Median and mode imputation both shrink variance and pull imputed rows toward the
population centre. For the ~15% of patients missing a chem panel this is a real
distortion, and it is preferred here only because the alternative leaks. If a
model later depends heavily on a lab with high missingness (blood pH is 83.7%
null, lipids ~78%), treat that dependence as suspect and consider dropping the
column outright instead.
"""

import json
import re
import pandas as pd

from features import PASSTHROUGH, build_feature_frame

OUT_JSON = "impute_stats.json"

# names that would signal "this value was missing" to a downstream model
INDICATOR_PATTERN = re.compile(
    r"(_isnull|_isna|_is_null|_missing|_was_missing|_nan|_notnull|_present)$", re.IGNORECASE
)


def assert_no_missingness_indicators(frame: pd.DataFrame) -> None:
    offenders = [c for c in frame.columns if INDICATOR_PATTERN.search(str(c))]
    assert not offenders, (
        f"missingness indicator column(s) present: {offenders}. Missingness in this dataset "
        "encodes test-ordering behaviour and must never reach the model."
    )


def split_column_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition feature columns into numeric and categorical, excluding passthrough."""
    features = [c for c in frame.columns if c not in PASSTHROUGH]
    numeric = frame[features].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def fit_impute_stats(train: pd.DataFrame, require_train_split: bool = True) -> dict:
    """Compute fill values from the train split.

    Numeric columns get the median (robust to the heavy right tails in ferritin,
    D-dimer and NT-proBNP). Categorical columns get the mode.

    ``require_train_split`` may be disabled only by cross-validation, where the
    fitting rows are a CV fold drawn from train+val rather than the named train
    split. The caller is then responsible for ensuring the rows are that fold's
    training portion -- the point of the assertion is to make bypassing it a
    deliberate act rather than an accident.
    """
    if require_train_split:
        assert set(train["split"]) == {"train"}, "fit_impute_stats received non-train rows"

    numeric, categorical = split_column_types(train)

    medians = {}
    for column in numeric:
        values = train[column].dropna()
        assert not values.empty, f"column {column!r} is entirely null in train; cannot impute"
        medians[column] = float(values.median())

    modes = {}
    for column in categorical:
        values = train[column].dropna()
        assert not values.empty, f"column {column!r} is entirely null in train; cannot impute"
        modes[column] = str(values.mode().iloc[0])

    return {
        "split": "train",
        "n_train": int(len(train)),
        "numeric_medians": medians,
        "categorical_modes": modes,
        "n_numeric": len(medians),
        "n_categorical": len(modes),
        "adds_missingness_indicators": False,
    }


def apply_impute_stats(frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Fill nulls in-place-by-copy using previously fitted train statistics."""
    before = list(frame.columns)
    filled = frame.copy()

    fill_values = {**stats["numeric_medians"], **stats["categorical_modes"]}
    for column, value in fill_values.items():
        assert column in filled.columns, f"impute_stats.json names unknown column {column!r}"
        filled[column] = filled[column].fillna(value)

    assert list(filled.columns) == before, "imputation changed the column set"
    assert len(filled) == len(frame), "imputation changed the row count"
    assert_no_missingness_indicators(filled)
    return filled


def build_imputed_frames(spec_path: str | None = None) -> dict[str, pd.DataFrame]:
    """Build train/val/test frames with train-fitted imputation applied to each."""
    full = build_feature_frame(split=None)
    assert_no_missingness_indicators(full)

    train = full[full["split"] == "train"]
    stats = fit_impute_stats(train)

    return {split: apply_impute_stats(full[full["split"] == split], stats) for split in
            ["train", "val", "test"]}, stats


if __name__ == "__main__":
    frames, stats = build_imputed_frames()

    with open(OUT_JSON, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"wrote {OUT_JSON}")
    print(f"  fitted on split={stats['split']}, n={stats['n_train']}")
    print(f"  numeric medians: {stats['n_numeric']}   categorical modes: {stats['n_categorical']}")
    print()

    for name, frame in frames.items():
        features = [c for c in frame.columns if c not in PASSTHROUGH]
        n_null = int(frame[features].isnull().sum().sum())
        print(
            f"{name:6s} rows={len(frame):5d}  features={len(features):3d}  "
            f"remaining NaNs={n_null}  severe={frame['severe'].mean() * 100:.1f}%"
        )
