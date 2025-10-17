\
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any

def ks_tests(real: pd.DataFrame, synth: pd.DataFrame) -> Dict[str, float]:
    """Kolmogorov–Smirnov p-values for numeric columns present in both dataframes."""
    pvals = {}
    common = [c for c in real.columns if c in synth.columns]
    for col in common:
        if pd.api.types.is_numeric_dtype(real[col]):
            r = pd.to_numeric(real[col], errors="coerce").dropna().values
            s = pd.to_numeric(synth[col], errors="coerce").dropna().values
            if len(r) > 10 and len(s) > 10:
                k, p = stats.ks_2samp(r, s, alternative="two-sided", mode="asymp")
                pvals[col] = float(p)
    return pvals

def correlation_delta(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Return mean absolute difference between numeric correlation matrices."""
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c]) and c in synth.columns]
    if len(num_cols) < 2:
        return float("nan")
    r_corr = real[num_cols].corr(numeric_only=True).fillna(0.0).values
    s_corr = synth[num_cols].corr(numeric_only=True).fillna(0.0).values
    return float(np.mean(np.abs(r_corr - s_corr)))

def naive_reid_risk(real: pd.DataFrame, synth: pd.DataFrame, threshold: float = 0.15) -> float:
    """
    Naive nearest-neighbor re-identification proxy:
    - Scale numeric columns jointly (shared columns only).
    - For each real row, find nearest neighbor in synthetic.
    - If the distance < threshold, count as "potential match".
    Returns fraction of real rows with a "too-close" synthetic neighbor.
    """
    shared = [c for c in real.columns if c in synth.columns and pd.api.types.is_numeric_dtype(real[c])]
    if len(shared) == 0:
        return float("nan")

    r = real[shared].dropna().values
    s = synth[shared].dropna().values
    if len(r) == 0 or len(s) == 0:
        return float("nan")

    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r)
    s_scaled = scaler.transform(s)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(s_scaled)
    distances, _ = nbrs.kneighbors(r_scaled)
    risk = float(np.mean((distances.flatten() < threshold).astype(float)))
    return risk

def summarize_privacy_fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> Dict[str, Any]:
    ks = ks_tests(real, synth)
    corr = correlation_delta(real, synth)
    risk = naive_reid_risk(real, synth)
    return {
        "ks_pvalues": ks,                     # higher p ~ more similar distributions
        "corr_mean_abs_delta": corr,          # lower is better
        "naive_reid_risk_fraction": risk      # lower is better
    }
