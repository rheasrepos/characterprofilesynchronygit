# Code vs. write-up mismatches — Jin's `socialaha`

Running list of places where Jin's **public code** (`data/socialaha/code/`) and his **manuscript**
(`2026.05.13.724907v1`) disagree, found while building a faithful replication. Some may be intentional
(code updated after writing, or vice versa) — flag each for Jin rather than assuming a bug. Sources cited
by file/line and paper page/line.

_Last updated: 2026-07-15._

## Open

### 1. Hemodynamic lag: pattern-shift code (`step06`) shifts 3 TRs; everything else + Jin + paper say 4 — ⚑ one confirm left
- **Say 4:** paper (p.10, p.23), **Jin's email 2026-07-15 Q5** ("I shifted 4 TRs… brain ts[4:] matches the
  stimulus"), **`step01_load-brain`** (`hrf=4`), **`code_extractbold.py`** (`hrf=4`).
- **Says 3:** **`step06_compute_mtm_neural-pattern-shift`** — the script that generated the *published pattern
  shift* — uses **`hrf=3`**, slicing `loaded_data[...][:, hrf:TRs+hrf]`. The ROIsum pkl is the **raw** (unshifted)
  time series; the shift is applied at slice time, so `step06`'s `hrf=3` directly sets the pattern-shift alignment.
- **Our `06` cell 3 copies `step06` → `hrf=3`**, so it faithfully replicates Jin's *published pattern shift*,
  but not his stated intent / the paper (4). The BOLD pkls are present locally, so cell 3 regenerated the file.
- **Consequence:** 3 vs 4 shifts every `[-5,+3]` window by one TR relative to the aha stamps → **can change the
  0-ROI result**. The `06` null result on record used `hrf=3`.
- **Ask Jin (precise):** `step06` uses `hrf=3` while `step01`/`code_extractbold`/your email say 4 — which did the
  **published Fig 4 pattern shift** use? (If 3, keep; if 4, `step06` has a 1-TR bug → regenerate at 4.)
- **Tested (2026-07-16):** ran `06` both ways. Double threshold = **0 ROIs at every TR under both hrf=3 and
  hrf=4** → the sentiment pattern-shift null is **robust to the lag**; confirming with Jin is for correctness,
  not the conclusion. (Under hrf=4, `step08`-only surfaces a lone ROI [19] at TR−4, not in `step07`'s TR−4 set
  [81, 88], so it fails the double threshold.)

### 2. Two-tailed permutation p-value: code omits the +1 in the numerator
- **Paper** (Methods, p.22 IS-RSA & p.23 pattern-shift): `p = (1 + #{|null| ≥ |empirical|}) / (1 + n_perms)`.
- **Code** (`step07`/`step08` `twotail_p`, step08 line 96):
  `np.sum(np.abs(null) >= np.abs(real)) / (1 + len(null))` — **+1 only in the denominator**, not the numerator.
- **Impact:** minor; the code is slightly less conservative than the paper's formula. Our `06` currently
  mirrors the **code** (no +1 numerator).
- **Ask Jin:** which is canonical? (Trivial to switch.)

### 3. Pattern-shift FDR scope: per-TR over 100 cortical (code) vs "across 116 ROIs and 9 TRs" (paper)
- **Paper** (Methods, p.23): significance "corrected for multiple comparisons across **116 ROIs and 9
  assessed TRs**" (reads as a joint 116×9 correction).
- **Code** (`step08` `test_significance`): single-stage FDR-BH applied **per TR**, over the **100 cortical
  ROIs only** (`nroi_cor=100`), independently for each of the 9 TRs — not a joint 116×9 correction.
- **Internal code inconsistency too:** the top-of-file docstring says "**Two-stage FDR** (within-ROI across
  TPs, then within-TP across ROIs)," but the implemented `test_significance` is **single-stage per-TR**.
- **Impact:** materially changes how many ROIs survive. Our `06` mirrors the *implemented* code (per-TR over
  100 cortical) + Jin's double threshold, which is his final Fig-4e/f readout.
- **Ask Jin:** which correction is the intended one for the reported figures?

### 4. Subcortical ROIs: paper implies 116 tested, code corrects over 100 cortical
- **Paper** (Results, p.11): "No significant neural pattern shift was observed in **subcortical** regions"
  (implies subcortical were in the tested set of 116).
- **Code** (`step08` `test_significance`/`double_threshold`, `nroi_cor=100`): only ROIs 0–99 (cortical) are
  ever p-valued or FDR-selected; subcortical 100–115 never enter significance.
- **Ask Jin:** were subcortical ROIs formally tested (and null), or excluded from the correction by design?

### 5. IS-RSA threshold: posted `step04.py` two-sided q<0.01 vs published Figure 2 one-sided q<0.05
- **Code** (`step04.py`): two-sided bootstrap p, FDR **q<0.01**.
- **Published Figure 2:** one-sided p, **q<0.05** (confirmed in our `04a`: Jin's saved `.mat` p-values are
  exactly half the two-sided, and his sig ROIs sit at corrected-p up to ~0.045). One-sided p = two-sided/2.
- **Impact:** determines whether we "match" his six ROIs (we recover 6/6 under the figure method, 4/6 & 2/6
  under posted). Already handled as a dual readout in `04a`/`04b`/`04c`.
- **Ask Jin:** confirm the figure used one-sided q<0.05.

### 6. `scene_null.py` is absent — and our reconstruction FAILS its positive control ⚑ top ask for Jin
- **Missing file:** `step08` references `scene_null.py` ~10x (it is in Jin's git-committed `step08.py`, author
  Jin Ke), but the file itself is **not in the public repo**. The per-scene non-aha *draw* therefore had to be
  reconstructed from `step07`'s sampling logic.
- **Positive control (run 2026-07-18, `06` §6.6a/b):** fed **Jin's own impression-update data** — a known
  positive that his Fig 4 says should light up the mentalizing network — through our reconstructed null.
  Double threshold recovered **1 ROI: 15 = LH DorsalAttn (SPL_3), at TR 0** (plus step08-only [52, 78] at
  TR −5, pre-onset). Saved: `results/step6/06__impression_control_sig_hrf3.npy`.
- **Consequence:** the reconstructed null is **too conservative** — it barely recovers the known-positive, and
  in the wrong network. Therefore **every null in `06` (sentiment pattern-shift, perception-change) is
  UNDERPOWERED / INCONCLUSIVE, not evidence of absence.**
- **Ask Jin (top priority):** share the original `scene_null.py` so the null distribution matches his exactly.
  Until then `06` cannot be reported as a null result.
- **Scope note:** this does **not** touch the IS-RSA arm — `04b` (sentiment null), `07` (`like` -> [24,48,60]),
  and `05` (group mPFC [91]) are independent analyses and stand.

## Resolved / confirmed consistent
- **Null design** (non-aha moments, same run, outside any `[-5,+3]` aha window, 1000 iterations, per-scene,
  averaged to match aha count): code (`step08 compute_and_save_scene_null`) and paper (Methods p.23) **agree**
  *on the design as described*. Our `06` reconstruction matches that description — but see **open item 6**:
  matching the described design is not sufficient, because the reconstruction fails the positive control.
- **Scene count = 40** (exclude run 1 = no prior impression; run 7 = no aha categorization): code and paper agree.
- **Cohort = 29** (33 − 4 empty-transcript for behavior; both drop the same subjects): confirmed by Jin over email.
- **Pattern-shift window −5..+3 TR, "1 − Pearson" between consecutive TRs, average over multiple aha in a
  scene:** code and paper agree.

## Confirmed by Jin (email 2026-07-15)
- **`gX.char` coding:** 1=Jack, 2=Kate, 3=Randall, 4=Kevin, **5 = Kevin & Kate** (combined scene). Confirms the
  `id==5` slot in `compute_impression_updates` (kate+kevin); note averaging the two is still *our* modeling choice,
  which Jin did not separately endorse.
- **`scene` column = `order` column** in `socialaha_groupscene.csv`. `TR(scene)`/`TR(run)` are the aha-screenshot
  timestamps (0-indexed, python rule); HRF-shifted `time_series[36]` = the aha button press at `TR=36`.
- **No TRs cut in preprocessing.** Movie onset = 0 in the BIDS files; brain `ts[4:]` aligns to the stimulus
  (i.e., the 4-TR shift). Start of `sub-XXXX_task-01_smoothed.nii.gz` = start of the movie (scene 19 for group 1).
- **Post-scan character ratings** (warm/kind/intelligence/like…) live in `socialaha-fMRI/charactersurvey/`
  (per-subject `.mat`) — the source for `04c`/`07` `like`/PC1/positive_emotion. `socialaha-beh` is a **separate**
  behavioral study (not the fMRI participants). → verify `04c`'s survey loader points at `charactersurvey/`.
- **Event timestamps** (aha speech, scene play windows, character speech) are in OpenNeuro `events.tsv` per
  participant `/func/`; `boldexample.py` segments them — first 5 rows = movie watching, next 5 = verbal aha,
  last 4 = verbal character description. (No manual transcription needed for timing.)
- **Cross-participant voxel comparison** with unequal ROI voxel counts: Jin uses **SRM** to a 100-D common space
  (held-out CV for fitting); simpler alternative = subject+run-specific EPI mask with masked voxels left as NaN.

> Add new rows as they surface; keep the source citation (file/line + paper page/line) on each.
