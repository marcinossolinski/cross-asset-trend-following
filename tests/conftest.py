import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """Eight years of daily returns for one asset per asset class plus a second FX pair."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2000-01-03", "2007-12-31")
    vols = {"US_EQ": 0.16, "AUD": 0.10, "CAD": 0.07, "WTI": 0.35}
    data = {k: rng.normal(0.0002, v / np.sqrt(256), len(idx)) for k, v in vols.items()}
    # add some persistent drift so that trend signals are not all noise
    data["US_EQ"] += np.sin(np.arange(len(idx)) / 120) * 0.0008
    return pd.DataFrame(data, index=idx)
