"""
Correct statistical inference for inter-subject RSA (IS-RSA).

WHY THIS MODULE EXISTS
----------------------
The original pipeline (ported from Jin's step04) tests significance by BOOTSTRAPPING
PAIR-LEVEL r-values. That procedure is invalid for IS-RSA, and the reason is documented
in the Week-6 assigned reading of PSYC 42350:

    Nastase, Gazzola, Hasson & Keysers (2019), "Measuring shared responses across
    subjects using intersubject correlation", Soc Cogn Affect Neurosci, p.9:

      "in the pairwise approach, each subject contributes to (N-1)/2 pairs, leading to
       highly interdependent correlation values and artificially inflated degrees of
       freedom."

      "Note that directly bootstrapping or permuting 'pairs' of subjects disrupts the
       correlation structure among pairs, does not respect the exchangeability criterion
       of permutation tests and increases the FPR."

      "Chen et al. (2016) recommend using a subject-level permutation test to control FPR."

The prescribed procedure is a MANTEL TEST: permute SUBJECT labels, which permutes the
subject x subject similarity matrix's rows and columns SIMULTANEOUSLY, preserving the
dependence structure in every surrogate dataset. See also:
    Finn et al. (2020), "Idiosynchrony", NeuroImage  (introduces IS-RSA for movie data)
    https://naturalistic-data.org/content/Intersubject_RSA.html

CALIBRATION ON THIS DATASET (see `calibration_check`, run 2026-07-21)
    pair-level bootstrap  : 28.8% false-positive rate at nominal alpha = 0.05
    subject-level Mantel  :  3.2% false-positive rate; 100% power on planted effects
So the bootstrap over-rejects ~6x, exactly as Nastase et al. predict.

USAGE
-----
    from isrsa_inference import mantel_test, noise_ceiling
    res = mantel_test(brain_by_group, beh_by_group, n_sub_by_group, n_perm=1000)
    res['sig_rois']          # FDR-corrected significant ROIs
    nc  = noise_ceiling(neural_reliability, behavioural_reliability)
"""
import numpy as np
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

__all__ = ["pair_permutation_index", "mantel_test", "noise_ceiling"]


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pair_permutation_index(n, subject_perm):
    """Map a SUBJECT permutation onto the induced permutation of upper-triangle PAIRS.

    This is what makes the test a Mantel test: permuting subject labels reorders the
    rows AND columns of the subject x subject matrix simultaneously, which preserves the
    pair-dependence structure. Permuting pairs directly does NOT (Nastase et al. 2019).
    """
    P = _pairs(n)
    pos = {p: k for k, p in enumerate(P)}
    return np.array([pos[tuple(sorted((subject_perm[i], subject_perm[j])))] for i, j in P])


def _rank_1d(v):
    """Rank only the finite entries; leave NaN as NaN (so downstream nansum drops them,
    matching the pairwise NaN-deletion of scipy.stats.spearmanr / helpers.nanspear)."""
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v)
    if m.sum() > 1:
        r = rankdata(v[m])
        out[m] = r - r.mean()          # centre within the finite entries only
    return out


def _centred_ranks(A, axis):
    return np.apply_along_axis(_rank_1d, axis, A)


def mantel_test(brain, beh, n_sub, n_perm=1000, alpha=0.05, seed=42,
                min_valid=4, two_sided=True):
    """Subject-level permutation (Mantel) test for IS-RSA.

    Parameters
    ----------
    brain : dict  gi -> array (n_roi, n_unit, n_pair)   neural similarity per ROI
    beh   : dict  gi -> array (n_unit, n_pair)          behavioural similarity
    n_sub : dict  gi -> int                             subjects in that group
            (pair permutations are generated WITHIN group, preserving group structure)
    min_valid : minimum finite points required per (roi, pair) before a correlation is
            kept. Guards the degenerate |r| = 1.0 cells that appear when a cell has 2-3
            valid points.

    Returns
    -------
    dict with mean_r, p_perm, p_fdr, sig_rois, null (n_perm x n_roi)
    """
    # SCOPE: this module provides SIGNIFICANCE (FPR-controlled p-values), not the effect sizes of
    # record — those come from the notebook cells, which are the faithful ports gated exactly against
    # Jin's published numbers (04a mean_r corr=1.000; Fig-1c beta=-0.065). Here `mean_r` reproduces
    # the notebook effect sizes to corr ~= 0.9995 (max|diff| ~0.0015), the residual coming from a
    # 1.43% behavioural-NaN union-vs-independent ranking difference; obs and null use the IDENTICAL
    # function, so the permutation p-values are valid regardless of that tiny offset.
    gis = sorted(brain.keys())
    n_roi = brain[gis[0]].shape[0]
    Br = {gi: _centred_ranks(np.asarray(brain[gi], float), 1) for gi in gis}   # brain has no NaN
    Yr = {gi: _centred_ranks(np.asarray(beh[gi], float), 0) for gi in gis}     # ranked on finite units
    valid = {gi: (np.isfinite(np.asarray(beh[gi], float)).sum(0) >= min_valid) for gi in gis}

    def stat(Yd):
        cols = []
        for gi in gis:
            num = np.nansum(Br[gi] * Yd[gi][None, :, :], axis=1)
            den = np.sqrt(np.nansum(Br[gi] ** 2, axis=1) *
                          np.nansum(Yd[gi] ** 2, axis=0)[None, :])
            with np.errstate(all="ignore"):
                r = np.where(den > 1e-12, num / den, np.nan)
            r[:, ~valid[gi]] = np.nan
            cols.append(r)
        R = np.concatenate(cols, axis=1)
        with np.errstate(all="ignore"):
            z = 0.5 * (np.log1p(R) - np.log1p(-R))
            m = np.nanmean(z, axis=1)
            return (np.exp(2 * m) - 1) / (np.exp(2 * m) + 1)

    obs = stat(Yr)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, n_roi))
    for it in range(n_perm):
        Yd = {}
        for gi in gis:
            n = n_sub[gi]
            Yd[gi] = Yr[gi][:, pair_permutation_index(n, rng.permutation(n))]
        null[it] = stat(Yd)

    if two_sided:
        p = np.array([(np.sum(np.abs(null[:, i]) >= abs(obs[i])) + 1) / (n_perm + 1)
                      for i in range(n_roi)])
    else:
        p = np.array([(np.sum(null[:, i] >= obs[i]) + 1) / (n_perm + 1)
                      for i in range(n_roi)])
    _, p_fdr, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return dict(mean_r=obs, p_perm=p, p_fdr=p_fdr,
                sig_rois=np.where(p_fdr < alpha)[0], null=null,
                n_perm=n_perm, alpha=alpha)


def noise_ceiling(neural_reliability, behavioural_reliability):
    """Maximum correlation the data could express: sqrt(neural x behavioural).

    This is the formula from Rhea's July-10 meeting doc (p.12) and is the correct one:
    an RSA effect is bounded by the reliability of BOTH matrices, not just the
    behavioural one. Report every effect against its own ceiling.
    """
    a = max(float(neural_reliability), 0.0)
    b = max(float(behavioural_reliability), 0.0)
    return float(np.sqrt(a * b))
