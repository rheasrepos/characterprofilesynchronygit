# Open decisions for Hayoung — consolidated meeting brief

Every `ASK HAYOUNG!!` item across the notebooks, in one place, with the **default that was actually used**
and the result it produced — so each can be confirmed or redirected rather than re-litigated from scratch.

_Last updated: 2026-07-18._
_Source cells: `01` c49/c51 · `01b` c15/c16 · `02` c7 · `04a` c15 · `04b` c19 · `04b.1` c17 · `04c` c0/c9 · `05` c0/c8 · `07` c11._

---

## A. Already answered by work since — confirm and close

| # | Decision | Status |
|---|---|---|
| A1 | How to recreate Jin's brain **surface figures** (his conda env + Schaefer/Tian templates) — raised in `04b` c19 and `04b.1` c17 | **SOLVED.** `08_brain_figures` works: Jin's `build_nifti_mask` / `build_brain_volume` + nilearn surface render. Figures in `results/figures/08__*.png`. Close it. |
| A2 | Survey scope: run-resolved SONA = 16-item page; 35-item = fMRI post-scan only (`01` c49, `02` c7) | **CONFIRMED by Jin's email 2026-07-15** — post-scan character ratings live in `socialaha-fMRI/charactersurvey/`. Close unless Hayoung disputes. |
| A3 | Character column order Jack/Kate/Randall/Kevin (`01` c49, `02` c7) | **CONFIRMED by Jin 2026-07-15**: `gX.char` 1=Jack, 2=Kate, 3=Randall, 4=Kevin, **5 = Kevin & Kate combined**. Note: averaging the two for id=5 is *our* modelling choice, which Jin did **not** separately endorse — still worth flagging. |
| A4 | Scope: is the pattern-shift (step06–08) analysis in scope at all? (`03` c13/c15) | **DONE** — run in `06`. But see D1: its nulls are currently uninterpretable. |

## B. The headline framing decisions — these shape the paper

| # | Decision | Default used | Result under that default |
|---|---|---|---|
| B1 | **Headline target**: character-affect (positive emotion / affect cluster) vs participant-feeling (`01` c49) | character-affect | cv-R² 0.34 vs ~0 for `like` |
| B2 | **Headline representation**: 3-D valence score (`04b`, purest/interpretable) vs 768-d embedding (`04b.1`, more reliable) — and how to frame the pair (`04b.1` c17) | 3-D score headlines | `04b` = **0 ROIs**; `04b.1` = [9, 60] before / [9, 60, 78] combined |
| B3 | **Elevate the `like` (viewer-stance) analysis** alongside/above the sentiment IS-RSA, and which level headlines (`04c` c9, `07` c11) | reported beside `04b`, never pooled | `like` -> **[24, 48, 60]** (individual) and **[91] mPFC** (group) — currently the **strongest positive finding in the project** |
| B4 | How to read the **sentiment null** (0 FDR-sig ROIs): real dissociation vs low power at n=29 (`04b` c19) | lean: real — `04a` had power on the same 29 | **Strengthened since:** notebook `10` §3 shows behaviour carries 2–5× the model's unique variance, and §2 shows no brain-aligned model structure at any layer (\|r\|<0.07). Converging evidence for "real dissociation." |
| B5 | How to read the **embedding result** — a few regions where the score was null: genuine reliability lift or too thin at n=29 (`04b.1` c17) | reported as a qualified lift | **Important nuance found 2026-07-18:** ROIs 9 / 60 / 78 are **LH auditory, RH S2, temporal pole** — sensory/affective, *not* mentalizing. So it is not "the embedding recovers social structure." |

## C. Method choices — documented defaults, not silently chosen

| # | Decision | Default used | Note |
|---|---|---|---|
| C1 | Neural collapse over runs: mean ISC / final-run / aha-window (`04c` c0/c9, `07` c11) | mean ISC over 10 runs per character | |
| C2 | Scalar-rating similarity metric: AnnaK `−\|r_i − r_j\|` vs mean (`04c` c0/c9, `07` c11) | AnnaK `−\|Δ\|` | |
| C3 | Group-level neural quantity: group-mean ISC vs neural-pattern distance vs event discriminability (`05` c0/c8) | group-mean ISC | gave `like` -> ROI [91] |
| C4 | Brain load path: reuse Jin's precomputed ISC vs recompute from raw fMRI (`04a` c15) | Jin's precomputed ISC | `04a` is the gate either way — and it passes |
| C5 | Group-level bridge (3 scramble groups, cross-cohort) informative enough to report, or individual-only (`07` c11) | both reported, never pooled | 3 groups is coarse — genuinely Hayoung's call |
| C6 | Single averaged target vs 4 separate items; valence-only vs full VAD for lexicons; add a fused multi-model vector (`01` c51) | single averaged target, valence-only, no fused vector | |
| C7 | Drop `emotionally stable` from the affect composite (`01` c49) | kept | inert: 0.366 with vs 0.367 without |
| C8 | Participant aggregate equal-weighted (`01` c49) | equal-weighted | tuning weights barely helps (−0.08 → +0.02) |
| C9 | Flair pos/neg mapping fair? (`01` c49) | as-is | re-scaling barely moves it |
| C10 | Promote a halo item (`trustworthy` / `empathize`) to secondary validation target (`01b` c15/c16) | descriptive only | |

## D. Blocked on Jin — not Hayoung's to decide

| # | Item | Status |
|---|---|---|
| D1 | **`06` pattern-shift nulls are uninterpretable.** The reconstructed non-aha null fails its positive control (recovers 1 ROI, LH DorsalAttn, wrong network). | Blocked pending Jin's `scene_null.py`. See `CODE_VS_PAPER_MISMATCHES.md` item 6. **Do not report `06` as a null until resolved.** |
| D2 | IS-RSA threshold: one-sided q<0.05 (Fig 2) vs posted two-sided q<0.01 | In the email to Jin. Determines whether we "match" 6/6 or 4/6. |
| D3 | hrf 3 vs 4; permutation p +1 numerator; FDR scope; subcortical testing | All four in the email to Jin. |

---

## Suggested meeting order
1. **B3** — is `like` the headline? It's the strongest result and reframes the whole paper.
2. **B2 / B5** — which sentiment representation headlines, given the embedding only reaches sensory cortex.
3. **B4** — sign off on "real dissociation" given the new converging evidence from notebook `10`.
4. **D1** — agree that `06` is on hold until Jin sends `scene_null.py`.
5. Sweep **C1–C10** as a block (documented defaults; most are inert).
6. Close **A1–A4**.
