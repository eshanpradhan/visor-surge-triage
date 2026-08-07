"""Clinical feature encoding, deliberately free of any torch import.

This lives apart from dataset.py because LightGBM and PyTorch each bundle their
own OpenMP runtime, and on macOS loading both into one process breaks: a
multithreaded lgb.train either segfaults outright or, if LightGBM is imported
first, deadlocks on a later call with the process pinned at 0% CPU. Keeping the
encoder importable without torch lets LightGBM-based evaluation run in a process
that never loads torch at all, which removes the conflict rather than working
around it.

dataset.py re-exports everything here, so existing imports keep working.
"""

import numpy as np
import pandas as pd

from features import PASSTHROUGH

# ordinal columns carry a real order; everything else categorical is one-hot encoded
ORDINAL_LEVELS = {
    "age.splits": ["[18,59]", "(59,74]", "(74,90]"],
    "smoking_status_v": ["Never", "Former", "Current"],
}

# numeric features whose train max |z| exceeds this are log1p-transformed before
# standardization. Several biomarkers here are lognormal over 3-4 orders of
# magnitude (NT-proBNP spans 5 to 267,600), and a 25-sigma feature dominates
# early gradients in any linear head.
LOG1P_MAX_Z_THRESHOLD = 8.0

# admission-month buckets with fewer than this many train patients are merged
# into the nearest frequent bucket rather than becoming near-empty one-hot columns
MIN_MONTH_BUCKET_N = 20




class ClinicalEncoder:
    """Encode the imputed clinical frame to a float matrix, fit on train only."""

    def __init__(self) -> None:
        self.numeric_columns: list[str] = []
        self.log1p_columns: list[str] = []
        self.ordinal_columns: list[str] = []
        self.onehot_levels: dict[str, list[str]] = {}
        self.date_columns: list[str] = []
        self.month_buckets: dict[str, list[str]] = {}
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.feature_names: list[str] = []

    def _feature_columns(self, frame: pd.DataFrame) -> list[str]:
        return [c for c in frame.columns if c not in PASSTHROUGH]

    @staticmethod
    def _months(values: pd.Series) -> pd.Series:
        return pd.to_datetime(values).dt.to_period("M").astype(str)

    def _bucket_months(self, values: pd.Series, column: str) -> pd.Series:
        """Map each admission month onto a frequent-month bucket."""
        frequent = self.month_buckets[column]
        months = self._months(values)
        periods = {m: pd.Period(m) for m in set(months) | set(frequent)}
        nearest = {
            m: min(frequent, key=lambda f: abs((periods[m] - periods[f]).n)) for m in set(months)
        }
        return months.map(nearest)

    def fit(self, train: pd.DataFrame) -> "ClinicalEncoder":
        columns = self._feature_columns(train)

        for column in columns:
            if column == "visit_start_datetime":
                # Severe prevalence swings 41.6% (1900-12) to 11.9% (1901-01),
                # chi2=146.5 p<1e-30 -- a real calendar-wave effect, not noise. Kept
                # as a coarse month bucket rather than a raw day count, which was the
                # highest-leverage numeric in the matrix at 23.5 sigma.
                self.date_columns.append(column)
                counts = self._months(train[column]).value_counts()
                frequent = sorted(counts[counts >= MIN_MONTH_BUCKET_N].index)
                assert frequent, f"no admission month reaches n>={MIN_MONTH_BUCKET_N} in train"
                self.month_buckets[column] = frequent
            elif pd.api.types.is_numeric_dtype(train[column]):
                self.numeric_columns.append(column)
            elif column in ORDINAL_LEVELS:
                self.ordinal_columns.append(column)
            else:
                self.onehot_levels[column] = sorted(train[column].astype(str).unique())

        for column in self.ordinal_columns:
            unseen = set(train[column].astype(str)) - set(ORDINAL_LEVELS[column])
            assert not unseen, f"{column} has levels outside ORDINAL_LEVELS: {sorted(unseen)}"

        # select log1p columns from raw train values, before any standardization
        for column in self.numeric_columns:
            values = train[column].to_numpy(dtype=np.float64)
            spread = values.std()
            if spread < 1e-8:
                continue
            max_z = np.abs((values - values.mean()) / spread).max()
            # Known imperfect case: Troponin T ranges 0.01-1.8, and log1p(x) is very
            # nearly x for x << 1, so it compresses only 20.2 -> 16.9 sigma. Accepted
            # rather than given a custom scale factor: it matters only for the linear
            # baseline, since tree models are scale-invariant.
            if max_z > LOG1P_MAX_Z_THRESHOLD:
                assert values.min() >= 0, (
                    f"{column!r} has negative values ({values.min()}); log1p would produce NaN"
                )
                self.log1p_columns.append(column)

        matrix = self._encode_raw(train)
        self.means = matrix.mean(axis=0)
        self.stds = matrix.std(axis=0)
        # a zero-variance column would divide by zero; leave it at its centred value
        self.stds[self.stds < 1e-8] = 1.0
        return self

    def _encode_raw(self, frame: pd.DataFrame) -> np.ndarray:
        blocks, names = [], []

        log1p_set = set(self.log1p_columns)
        for column in self.numeric_columns:
            values = frame[column].to_numpy(dtype=np.float64)
            if column in log1p_set:
                assert values.min() >= 0, f"{column!r} has negative values at transform time"
                values = np.log1p(values)
                names.append(f"{column}__log1p")
            else:
                names.append(column)
            blocks.append(values.reshape(-1, 1))

        for column in self.date_columns:
            bucketed = self._bucket_months(frame[column], column)
            for level in self.month_buckets[column]:
                blocks.append((bucketed == level).to_numpy(dtype=np.float64).reshape(-1, 1))
                names.append(f"{column}__{level}")

        for column in self.ordinal_columns:
            levels = ORDINAL_LEVELS[column]
            codes = frame[column].astype(str).map({v: i for i, v in enumerate(levels)})
            assert codes.notna().all(), f"unseen level in ordinal column {column!r}"
            blocks.append(codes.to_numpy(dtype=np.float64).reshape(-1, 1))
            names.append(f"{column}__ordinal")

        for column, levels in self.onehot_levels.items():
            values = frame[column].astype(str)
            unseen = set(values) - set(levels)
            assert not unseen, f"unseen level(s) in {column!r} at transform time: {sorted(unseen)}"
            for level in levels:
                blocks.append((values == level).to_numpy(dtype=np.float64).reshape(-1, 1))
                names.append(f"{column}__{level}")

        self.feature_names = names
        matrix = np.hstack(blocks)
        assert np.isfinite(matrix).all(), "non-finite value in encoded clinical matrix"
        return matrix

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.means is not None, "ClinicalEncoder.transform called before fit"
        matrix = self._encode_raw(frame)
        standardized = (matrix - self.means) / self.stds
        assert np.isfinite(standardized).all(), "non-finite value after standardization"
        return standardized.astype(np.float32)

