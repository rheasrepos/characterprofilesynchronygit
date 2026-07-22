"""
Jin Ke — step04b (Fisher's combined test across the before/after IS-RSA analyses).
Logic extracted verbatim from his repo `step04b_IS-RSA_combinep.py` for faithful replication.

His procedure, in order:
  1. Direction filter — only combine ROIs where sign(r_after) == sign(r_before).
     Opposite-direction ROIs are EXCLUDED and assigned p = 1.
  2. Fisher's combined test on the surviving ROIs:  chi2(df=4) = -2 * (ln p_after + ln p_before)
  3. FDR (Benjamini-Hochberg) at **alpha = 0.05**, significance taken from `rejected`.

NOTE: our earlier notebook cells omitted step 1 and used alpha=0.01. That difference is
material — see the docstring of `fisher_combine` below.
"""
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests

EPS = 1e-300


def fisher_combine(mean_r_after, mean_r_before, p_after, p_before,
                   alpha=0.05, n_roi=116):
    """Jin's step04b combined test.

    Parameters
    ----------
    mean_r_after, mean_r_before : (n_roi,) IS-RSA effect sizes for each analysis.
    p_after, p_before           : (n_roi,) bootstrap p-values for each analysis.
    alpha                       : FDR level. Jin uses 0.05 (NOT the 0.01 used elsewhere in step04).

    Returns
    -------
    dict with keys:
      same_direction : (n_roi,) bool — ROIs whose r-values agree in sign (the only ones combined)
      fisher_stat    : (n_roi,) chi2 statistic (0 for excluded ROIs)
      p_combined     : (n_roi,) combined p (1.0 for excluded ROIs)
      p_fdr          : (n_roi,) FDR-corrected p
      sig_rois       : indices where BH rejected the null at `alpha`

    Why the direction filter matters: without it, an ROI with a strong POSITIVE effect
    before-movie and a strong NEGATIVE effect after-movie can produce a small combined p,
    and be declared "significant" despite the two analyses disagreeing. Jin excludes these.
    """
    r_after = np.asarray(mean_r_after, float)
    r_before = np.asarray(mean_r_before, float)
    pa = np.clip(np.asarray(p_after, float), EPS, 1.0)
    pb = np.clip(np.asarray(p_before, float), EPS, 1.0)

    # 1. direction filter (his `same_direction`)
    same_direction = np.sign(r_after) == np.sign(r_before)

    # 2. Fisher, same-direction ROIs only; others keep p = 1
    fisher_stat = np.zeros(n_roi)
    p_combined = np.ones(n_roi)
    fisher_stat[same_direction] = -2.0 * (np.log(pa[same_direction]) +
                                          np.log(pb[same_direction]))
    p_combined[same_direction] = scipy.stats.chi2.sf(fisher_stat[same_direction], df=4)

    # 3. FDR-BH at alpha (his alpha=0.05); significance from `rejected`
    rejected, p_fdr, _, _ = multipletests(p_combined, alpha=alpha, method='fdr_bh')

    return {
        "same_direction": same_direction,
        "fisher_stat": fisher_stat,
        "p_combined": p_combined,
        "p_fdr": p_fdr,
        "sig_rois": np.where(rejected)[0],
        "alpha": alpha,
    }
