# Results with named ROIs (Schaefer-100 17Net + Tian-S1)

Generated from saved result files + `results/roi_labels.csv`.

| Analysis | Significant ROIs (named) | Note |
|---|---|---|
| Jin impressions IS-RSA — after (validation gate) | 9 LH Somatomotor (Aud_1); 60 RH Somatomotor (S2_1); 64 RH DorsalAttn (SPL_1); 98 RH Temporoparietal (2) | posted q<.01 |
| 3-D sentiment IS-RSA (Fisher-combined) | — | NULL |
| 768-D embedding IS-RSA (Fisher-combined) | 9 LH Somatomotor (Aud_1); 60 RH Somatomotor (S2_1); 78 RH Limbic (TempPole_1) | reliability lift |
| 12-D multi-model IS-RSA (Fisher-combined) | 0 LH Visual (ExStr_1); 1 LH Visual (ExStr_2); 2 LH Visual (ExStr_3) | reliability lift |
| Survey `like` IS-RSA  ← PRIMARY POSITIVE | 24 LH Salience/VentAttn (FrMed_1); 48 LH DMN (PHC_1); 60 RH Somatomotor (S2_1) | posted q<.01 |
| Survey PC1 IS-RSA | 70 RH Salience/VentAttn (ParOper_1) | likely noise (0.055 reliab) |
| Survey positive_emotion IS-RSA | — | NULL |
| End-state sentiment (flat) IS-RSA | — | fig-match; posted=0 |
| End-state sentiment (concatenation) IS-RSA | 55 RH Visual (ExStrSup_3) | fig-match; posted=0 |
| Individual `like` IS-RSA (07) | 24 LH Salience/VentAttn (FrMed_1); 48 LH DMN (PHC_1); 60 RH Somatomotor (S2_1) | = 04c like |
| Pattern-shift ~ sentiment double-threshold (hrf3) | 0 ROIs (all TRs) | NULL both hrf |
| Your own step07 localizer (hrf3) | 72 ROIs — DMN×19, FrontoparietalControl×16, Salience/VentAttn×11, Visual×8, DorsalAttn×8, Limbic×5, Temporoparietal×3, Somatomotor×2 | replicates Jin Fig4d (see 06 for per-TR) |
| Pattern-shift ~ sentiment double-threshold (hrf4) | 0 ROIs (all TRs) | NULL both hrf |
| Your own step07 localizer (hrf4) | 74 ROIs — DMN×19, FrontoparietalControl×16, Visual×10, Salience/VentAttn×10, DorsalAttn×9, Limbic×5, Temporoparietal×3, Somatomotor×2 | replicates Jin Fig4d (see 06 for per-TR) |
| Group-level `like` (05/07) | 91 RH DMN (PFCm_1) | descriptive, 3 groups |
| Positive-valence SUBSCALE (04c individual, aggregate) | 65 RH DorsalAttn (SPL_2) | reliab 0.151 (> PC1/pos_emo, < like) |
| Positive-valence aggregate char_valence_composite (05 group) | 101 RH Amygdala | affect region; like=[91] DMN |
