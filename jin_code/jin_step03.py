"""
Jin Ke — step03 (Fig 1 impression distance). Verbatim functions extracted (via ast) from his repo clone step03_impression-updates.py for faithful replication.
Only module-level path/`main()`/file-IO scaffolding was omitted so this imports cleanly; function bodies unchanged.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
try:
    from joblib import Parallel, delayed
except Exception: pass

flist = {
    1: ['sub-1001','sub-1005','sub-1008','sub-1011','sub-1014','sub-1017',
        'sub-1020','sub-1023','sub-1026','sub-1029','sub-1033','sub-1039'],
    2: ['sub-2006','sub-2009','sub-2012','sub-2015','sub-2018','sub-2021',
        'sub-2024','sub-2027','sub-2034','sub-2038','sub-2040'],  # sub-2030 excluded
    3: ['sub-3004','sub-3007','sub-3013','sub-3016','sub-3019','sub-3022',
        'sub-3025','sub-3031','sub-3037','sub-3041'],             # sub-3010, sub-3028 excluded
}


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors (handles NaN vectors → NaN)."""
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return np.nan
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def nan_spearmanr(tc1, tc2):
    """Spearman r after dropping positions where either array is NaN."""
    tc1, tc2 = np.asarray(tc1, dtype=float), np.asarray(tc2, dtype=float)
    nan_idx = np.union1d(np.where(np.isnan(tc1))[0], np.where(np.isnan(tc2))[0])
    tc1 = np.delete(tc1, nan_idx)
    tc2 = np.delete(tc2, nan_idx)
    return spearmanr(tc1, tc2)

def embed(texts):
    return use_model(texts).numpy()

