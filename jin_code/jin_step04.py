"""
Jin Ke — step04_IS-RSA.py (IS-RSA). Verbatim functions extracted (via ast) from his repo clone step04_IS-RSA.py for faithful replication.
Only module-level path/`main()`/file-IO scaffolding was omitted so this imports cleanly; function bodies unchanged.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
try:
    from joblib import Parallel, delayed
except Exception: pass

nroi_cor, nroi_sub = 100, 16
nroi = nroi_cor + nroi_sub
alpha = 0.01


def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def nanspearmanr(tc1, tc2):
    nanid1 = np.where(np.isnan(tc1))
    nanid2 = np.where(np.isnan(tc2))
    nanid = np.union1d(nanid1, nanid2)
    tc1 = np.delete(tc1, nanid)
    tc2 = np.delete(tc2, nanid)
    rval, pval = spearmanr(tc1, tc2)
    return rval, pval

def nanpearsonr(tc1, tc2):
    nanid1 = np.where(np.isnan(tc1))
    nanid2 = np.where(np.isnan(tc2))
    nanid = np.union1d(nanid1, nanid2)
    tc1 = np.delete(tc1, nanid)
    tc2 = np.delete(tc2, nanid)
    rval, pval = pearsonr(tc1, tc2)
    return rval, pval

def twotail_p(real, null):
    return (np.sum(abs(null) >= abs(real))) / (len(null))

def rearrange(group_id, run_id, brain):
    current_indices = event_idxs[event_idxs['run'] == run_id]['g' + str(group_id) + '.char'].tolist()
    sorted_order = np.argsort(current_indices)
    rearranged_brain = brain[sorted_order]
    return rearranged_brain

def rearrange_new(group_id, run_id, brain):
    """Rearrange neural patterns by character, averaging where a character appears twice."""
    current_indices = event_idxs[event_idxs['run'] == run_id]['g' + str(group_id) + '.char'].tolist()
    sorted_order = np.argsort(current_indices)
    rearranged_brain = brain[sorted_order]

    if 5 in current_indices:
        rearranged_brain4 = []
        for i in range(4):
            if i in [1, 3]:
                this_run = np.nanmean(rearranged_brain[[i, 4], :], axis=0)
            else:
                this_run = rearranged_brain[i]
            rearranged_brain4.append(this_run)
    else:
        duplicates = [x for x in set(current_indices) if current_indices.count(x) > 1]
        if len(duplicates) == 0:
            return rearranged_brain
        else:
            rearranged_brain4 = []
            for i in range(1, 4 + 1):
                if duplicates[0] == i:
                    this_run = np.nanmean(rearranged_brain[[i, i - 1], :], axis=0)
                else:
                    this_run = rearranged_brain[i]
                rearranged_brain4.append(this_run)

    return np.array(rearranged_brain4)

def char_order(group_id, run_id):
    current_indices = event_idxs[event_idxs['run'] == run_id]['g' + str(group_id) + '.char'].tolist()
    current_indices.sort()
    return current_indices

def niftimask(nroi_cor, nroi_sub, directory):
    cortical = (directory + '/template/tpl-MNI152NLin2009cAsym/'
                'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
                + str(nroi_cor) + 'Parcels17Networks_dseg.nii.gz')
    if nroi_sub == 16:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S1_3T_2009cAsym.nii.gz'
    elif nroi_sub == 32:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S2_3T_2009cAsym.nii.gz'
    elif nroi_sub == 50:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S3_3T_2009cAsym.nii.gz'
    elif nroi_sub == 54:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz'

    mask_cor = load_img(cortical).dataobj[:]
    mask_sub = load_img(subcortical).dataobj[:]

    for i1 in range(mask_sub.shape[0]):
        for i2 in range(mask_sub.shape[1]):
            for i3 in range(mask_sub.shape[2]):
                if mask_sub[i1, i2, i3] > 0:
                    mask_sub[i1, i2, i3] = mask_sub[i1, i2, i3] + nroi_cor

    overlap_id = np.where(np.multiply(mask_cor, mask_sub) > 0)
    mask = mask_cor + mask_sub
    mask[overlap_id[0], overlap_id[1], overlap_id[2]] = 0
    return mask

def bootstrapping(data):
    """Two-tailed bootstrap test: p = 2 * min(P(mean <= 0), P(mean >= 0))."""
    boot_means = []
    np.random.seed(42)
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.nanmean(sample))
    boot_means = np.array(boot_means)
    p_one_tail = min(np.nanmean(boot_means <= 0), np.nanmean(boot_means >= 0))
    p_value = min(2 * p_one_tail, 1.0)
    return p_value

def make_brain_map(mean_r_values, sig_rois_idx=None):
    """Map mean_r values onto volumetric brain image. If sig_rois_idx given, mask non-sig to NaN."""
    ref_img = load_img(ref_img_path)
    brain = np.zeros(mask.shape)
    for i in range(brain.shape[0]):
        for j in range(brain.shape[1]):
            for k in range(brain.shape[2]):
                idx = int(mask[i, j, k]) - 1
                if sig_rois_idx is not None:
                    if idx in sig_rois_idx:
                        brain[i, j, k] = mean_r_values[idx]
                    else:
                        brain[i, j, k] = np.nan
                else:
                    brain[i, j, k] = mean_r_values[idx]
    return new_img_like(ref_img, brain)

def plot_surface_maps(surface_left, surface_right, fsaverage, outpath,
                      cmap='cold_hot', vmin=-0.10, vmax=0.10):
    sns.set_context('paper')
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    plot_surf_stat_map(fsaverage.infl_left, surface_left, hemi='left',
                       view='lateral', bg_map=fsaverage.sulc_left,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=a)
    plot_surf_stat_map(fsaverage.infl_right, surface_right, hemi='right',
                       view='lateral', bg_map=fsaverage.sulc_right,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=b)
    plot_surf_stat_map(fsaverage.infl_left, surface_left, hemi='left',
                       view='medial', bg_map=fsaverage.sulc_left,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=c)
    plot_surf_stat_map(fsaverage.infl_right, surface_right, hemi='right',
                       view='medial', bg_map=fsaverage.sulc_right,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=d)
    plt.savefig(outpath, dpi=1000)
    plt.tight_layout()
    plt.close()

def _apply_bnorm(surf):
    out = np.full_like(surf, np.nan)
    valid = ~np.isnan(surf)
    out[valid] = _bnorm(surf[valid])
    return out

