"""
helpers.py — shared IS-RSA helpers. r2z/z2r/nanspear/rearrange_new/bootstrap_p are Jin Ke's step04
functions (VERIFIED numerically identical: nanspear==nanspearmanr, r2z==conv_r2z, z2r==conv_z2r).
His step04 originals are not cleanly importable (they read module-level globals n_iterations/event_idxs),
so they live self-contained here; the verbatim originals are in jin_code/jin_step04.py. Attribution: Jin Ke.
_pair_mask / scalar_similarity / boot_onetail are mine.

Notebooks do `from helpers import *`. `__all__` is set so the underscore-prefixed
names (_JIN_FLIST, _pair_mask, _norm) import too. `event_idxs` is loaded once from
config.EVENT_PATH, so `rearrange_new` is self-contained.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from config import EVENT_PATH

__all__ = ["NROI", "N_BOOT", "_JIN_FLIST", "r2z", "z2r", "nanspear", "bootstrap_p",
           "boot_onetail", "scalar_similarity", "_pair_mask", "_norm",
           "rearrange_new", "event_idxs"]

NROI = 116
N_BOOT = 10000
_JIN_FLIST = {
    1: ["sub-1001","sub-1005","sub-1008","sub-1011","sub-1014","sub-1017","sub-1020","sub-1023","sub-1026","sub-1029","sub-1033","sub-1039"],
    2: ["sub-2006","sub-2009","sub-2012","sub-2015","sub-2018","sub-2021","sub-2024","sub-2027","sub-2034","sub-2038","sub-2040"],
    3: ["sub-3004","sub-3007","sub-3013","sub-3016","sub-3019","sub-3022","sub-3025","sub-3031","sub-3037","sub-3041"],
}

def r2z(r):   # = Jin step04 conv_r2z (verified identical)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

def z2r(z):   # = Jin step04 conv_z2r (verified identical)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def nanspear(a, b):   # = Jin step04 nanspearmanr (verified identical)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.union1d(np.where(np.isnan(a))[0], np.where(np.isnan(b))[0])
    return spearmanr(np.delete(a, idx), np.delete(b, idx))[0]

def bootstrap_p(data, n=N_BOOT, seed=42, two_sided=True):
    """Jin's step04 bootstrap. two_sided=True = posted step04 (q<0.01 usage);
    two_sided=False = one-sided p (matches his published Figure 2 at q<0.05)."""
    np.random.seed(seed); d = np.asarray(data, float)          # legacy RNG == Jin's step04
    bm = np.array([np.nanmean(np.random.choice(d, size=len(d), replace=True)) for _ in range(n)])
    one = min(np.nanmean(bm <= 0), np.nanmean(bm >= 0))
    return min((2 if two_sided else 1) * one, 1.0)

def boot_onetail(data, n=N_BOOT, seed=42):   # MINE -- one-sided extension of Jin's bootstrap (figure-match threshold)
    return bootstrap_p(data, n=n, seed=seed, two_sided=False)

def scalar_similarity(vals, metric="annak"):   # MINE -- AnnaK scalar RSA (not in Jin's step04)
    v = np.asarray(vals, float); n = len(v); out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(-abs(v[i] - v[j]) if metric == "annak" else (v[i] + v[j]) / 2.0)
    return np.array(out)

def _pair_mask(g, overlap_ids):   # MINE -- 29-overlap pair mask (not in Jin's step04)
    his = _JIN_FLIST[g]; keep = set(overlap_ids); m = []
    for i in range(len(his)):
        for j in range(i + 1, len(his)):
            m.append(his[i] in keep and his[j] in keep)
    return np.array(m)

def _norm(x):
    return "".join(ch for ch in str(x) if ch.isdigit())

event_idxs = pd.read_csv(EVENT_PATH)

def rearrange_new(gid, run_id, brain):   # = Jin step04 rearrange_new (event_idxs loaded from config here vs his module scope)
    cur = event_idxs[event_idxs["run"] == run_id][f"g{gid}.char"].tolist()
    b = brain[np.argsort(cur)]
    if 5 in cur:
        return np.array([np.nanmean(b[[i, 4], :], axis=0) if i in [1, 3] else b[i] for i in range(4)])
    dup = [x for x in set(cur) if cur.count(x) > 1]
    if not dup:
        return b
    return np.array([np.nanmean(b[[i, i - 1], :], axis=0) if dup[0] == i else b[i] for i in range(1, 5)])
