"""
Diagnosis-label normalisation and integer encoding.
"""

from typing import Iterable

import numpy as np

from .. import config as C


def labels_to_int(raw_labels: Iterable[str]) -> np.ndarray:
    """Convert raw diagnosis strings to {0=control, 1=pd} via config maps."""
    out = []
    for s in raw_labels:
        canonical = C.LABEL_NORMALISE.get(str(s).strip().lower())
        if canonical is None:
            raise ValueError(f"Unrecognised label: {s!r}")
        out.append(C.LABEL_TO_INT[canonical])
    return np.asarray(out, dtype=np.int64)
