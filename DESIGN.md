# Character-Sentiment → Neural Synchrony — design doc & meeting brief

One doc for the meeting *and* the design rationale. It opens with the meeting-facing summary (headline,
what to show, decisions to make), then the full **reasoning chain** — for every analysis: *what we ran →
what we found → what that decided → what's next*, with the figure and numbers. It's the "why" companion to
`README.md` (the "what exists" reference).

> Figures embed with repo-relative paths (`results/figures/…`); they render in a local viewer
> (VS Code / Jupyter). `results/` is git-ignored, so on GitHub the links show as paths — regenerate by
> running the notebooks.

---

## TL;DR for the meeting

**Question.** Do within-person changes in expressed sentiment across the 10 runs track changes in neural
representation? **Status.** Full pipeline `00`→`06` runs on real data; IS-RSA verified line-by-line against
Jin's public code.

**Headline (the through-line).** Expressed sentiment reliably measures the **character's affect / valence
— not the viewer's stance — and it distills something *different* from Jin's impression embeddings.** The
difference shows up twice: **behaviorally** (sentiment barely carries over between runs) and **in the
brain** (impressions drive neural synchrony, sentiment doesn't). **The dissociation is the result**, not a
failure — and it's interpretable precisely because Aim 1 pinned down what the sentiment number is. **New (rerun):** the second half now has *positive* evidence — shared **`like` (viewer stance), from the survey**, shows an IS-RSA effect (`04c` like → 3 ROIs; `05` group like → 1 ROI) where transcript-sentiment did not. So: *character-affect sentiment doesn't drive synchrony; shared liking does show an effect* — **but** `like` is also the more reliably-measured item (§4c.5), so this is *suggestive, not clinched* (clean content test still pending).

**Figures to pull up (in order):** item-alignment heatmap (Fig 1a) → PC2/PC3 (Fig 1b) → behavioral-vs-fMRI
co-fluctuation (Fig 1c) → distance effect (Fig 3) → IS-RSA gate + sentiment (Fig 4a, 4b). All embedded below.

**Decisions I need (agenda, with my lean):**
1. Headline target — character-affect vs participant-feeling. *Lean: character-affect; stance = Track-B nuance.*
2. Reading the `04b` null (0 ROIs) — real dissociation vs low power (n=29). *Lean: real — the gate had power.*
3. Brain scope — IS-RSA only (`04b`), or also the pattern-shift test (`06`)?
4. Neural quantity for Step 5 (group-mean ISC vs pattern distance vs discriminability).
5. `04c` sub-choices — neural collapse over runs; scalar similarity metric.
6. Brain load path — reuse Jin's precomputed ISC vs recompute from raw.
7. Promote a halo item (`trustworthy`/`empathize`) to a secondary target? *Lean: keep descriptive.*

---

## The reasoning chain

**Two aims (Aim 1 is load-bearing for Aim 2):** (1) *what does sentiment-from-speech measure?* — pin the
construct; (2) *does it track neural representation?* — the same-subject swap of sentiment for Jin's
impressions. One measure, two independent ground truths (Track A group×run; Track B individual×end-state),
never pooled.

### Step 0 — Measure  (`00`)

- **Ran.** 6 classifiers → `valence = pos − neg`. Diagnostics: **D1** within/between SD + ICC(1); **D2**
  cross-model Spearman, % sign-agreement, Krippendorff's α.
- **Found.** D1: within-person variance dominates — SD ratios **2.0–2.3**, ICC **≈ 0.07–0.11** (~92%
  within-person). D2: moderate agreement — transformers ~0.84, VADER↔Flair 0.38, **Krippendorff α = 0.44**.

  ![Group-mean sentiment trajectory per character, mean ± SEM](results/figures/00__mean_errorbar.png)
  *Fig 0a — the sentiment signal moves across runs (not flat).*

  ![Variance: characters vs people](results/figures/00__variance_overview.png)
  *Fig 0b — within-person / across-people spread exceeds across-character spread (the D1 picture).*
- **Decided.** The signal moves within-person → the question is answerable; models aren't interchangeable →
  keep all six, let the data pick.
- **→ Next.** A model-selection contest against a behavioral ground truth (Step 1).

### Step 1 — Validate + characterize the construct, Track A  (`01`, `01b`, `01d`)

- **Ran.** Each model's group-level valence regressed on the SONA survey (leave-one-scramble-group-out
  cv-R²); 16-item alignment scan + construct sweep; PCA of the 16 traits; robustness (`01d`); halo (`01b`).
- **Found.**
  - **Twitter-RoBERTa wins, cv-R² = 0.34** (within-character Spearman **0.58**, permutation **p = 0.0005**,
    survives length control + FDR); both Twitter-domain models top-3.
  - Tracks positive emotion / emotionally stable / good relationship; best small target = **affect cluster
    R² ≈ 0.37**; viewer stance (`like`, empathize) ≈ 0.
  - PCA: PC1 = good-person axis (**43%**); **PC2 = character-affect vs viewer-stance**; PC3 =
    extraversion/impulsivity. Affect cluster (0.37) and positive-emotion item (0.34) both **beat PC1 (0.19)**.
  - Follow-ups: `emotionally stable` inert (0.366 vs 0.367); no stance subtype tracked (affective −0.02,
    cognitive +0.01, similarity −1.08); tuning aggregate weights barely helps (−0.08 → +0.02).

  ![Each survey item vs each model — cv-R² alignment heatmap](results/figures/01__item_alignment_heatmap.png)
  *Fig 1a — the models light up the affect items, not the viewer-stance items.*

  ![PC2 and PC3 loadings](results/figures/01__pca_pc2_pc3_loadings.png)
  *Fig 1b — PC2 = character-affect vs viewer-stance; PC3 = extraversion/impulsivity.*

  ![Behavioral vs fMRI co-fluctuation across runs](results/figures/01__behavioral_vs_fmri.png)
  *Fig 1c — the SONA and fMRI cohorts' trajectories move together (mean cell ρ ≈ 0.59).*

  *Supporting:* `results/figures/01__pca_pc1_loadings.png`, `01__pca_scree.png`,
  `01__trait_intercorrelation.png`, `results/step1b/01b__item_vs_model_heatmap.png`.
- **Decided.** The measure = **the character's affect / broad valence, not the viewer's stance.** Construct
  pinned → a brain result will be interpretable. Carry all candidate targets forward (`01__targets.csv`);
  the single headline target is a Hayoung call.
- **→ Next.** Confirm at the individual level in the brain cohort (Track B); take it to the brain.

### Step 2 — Validate, Track B (individual, same cohort)  (`02`)

- **Ran.** Same constructs vs the fMRI participants' *own* post-scan survey; Spearman + clustered bootstrap CI.
- **Found.** Character-valence composite **ρ ≈ 0.54 / 0.51**, positive emotion 0.40; unlike Track A,
  `like`/stance **modestly positive (~0.34–0.37)**; `val_mean` > `val_final`. *(Table: `02` "Reading Track
  B"; `results/step2/02__validation.csv`. No figure — it's a table.)*
- **Decided.** Character-affect reading replicates at the individual level in the brain cohort; the mild
  same-person stance alignment is a Track-B nuance. The measure is solid for the brain swap.
- **→ Next.** Relate the measure to the brain.

### Step 3 — Figure 1: distance effect  (`03`)

- **Ran.** Jin's Fig-1 with sentiment: between-subject cosine similarity per (group, run, character); does
  it decay with run-distance? (mixed-effects, clustered.)
- **Found.** A **shallow** decay — similarity **0.71 → 0.64** across distance 1→9, pooled ρ = **−0.04** —
  far weaker than Jin's steep USE-impression slope.

  ![Sentiment similarity vs run distance](results/figures/03__sentiment_distance_effect.png)
  *Fig 3 — sentiment barely decays with distance (vs impressions' clear negative slope).*
- **Decided.** Sentiment distills something **different** from an impression — closer to the character's
  *momentary* state. A dissociation *in degree*, predicting a weaker brain effect for sentiment.
- **→ Next.** Test that prediction with IS-RSA (Step 3 emits the sentiment similarity matrices).

### Step 4 — Figure 2: IS-RSA  (`04a` gate, `04b` sentiment, `04c` survey)

- **Ran.** Jin's IS-RSA ported verbatim (Fisher r↔z, 10k bootstrap w/ his legacy RNG, FDR-BH 0.01,
  before/after lag, Fisher combine-p; Schaefer-100 + Tian-16). `04a` = gate on Jin's impressions (33);
  `04b` = sentiment on the 29-overlap; `04c` = post-scan survey representation (scaffold).
- **Found (rerun, legacy RNG).** `04a` gate: Jin's impressions hit **4 ROIs after [9,60,64,98] / 2 before [64,99]** (max|r|≈0.077) — load + port validated. `04b` sentiment: **0 FDR-sig ROIs** (after/before/Fisher-combined; max|r|≈0.05). `04c` **survey** representations (same cohort, more reliable): **`like` → 3 ROIs [24,48,60]** (max|r| 0.175), `PC1` → 1 ROI [70] (0.169), `positive_emotion` → 0 survive FDR (max|r| 0.187).

  ![Validation gate — IS-RSA with Jin's own impressions](results/figures/04a__isrsa_impressions.png)
  *Fig 4a — impressions recover significant ROIs: the method has power on these 29.*

  ![Sentiment vs impressions IS-RSA, per ROI](results/figures/04b__isrsa_sentiment_vs_impressions.png)
  *Fig 4b — the punchline: impressions hit ROIs, sentiment hits none.*
- **Decided.** The **dissociation carries into the brain**: shared *impression* similarity predicts shared
  neural synchrony; shared *sentiment* similarity does not (at this power). The gate had power on the same
  29, so it's evidence of dissociation, not just low n (n = 29 stated as a caveat).
- **Robustness (§4.4) — a reframe.** Power: we were powered (~80%) only for a per-pair mean-r ≥ **~0.05**
  (impressions cleared it, sentiment didn't) → read the null as *"no effect above the ~0.05 floor,"* not
  zero. **Critically: the sentiment similarity RDM's split-half reliability is ≈ 0.030** and the **neural ISC reliability ≈ 0.127**, so the attainable IS-RSA ceiling ≈ √(0.127×0.030) ≈ **0.062** — barely above the ~0.05 detection floor (a near-zero detectable window). The attainable r ≤ √(neural × behavioral reliability) — so the `04b` null is **substantially a
  reliability-ceiling artifact**: the between-subject sentiment RDM is too unstable for IS-RSA to detect
  anything, almost regardless of the brain. **ASK HAYOUNG!!** — this means `04b` was likely the *wrong
  instrument*; the informative handles are **`06`** (within-person deltas — the actual question) and
  **`07`** (survey `like`, expected to be far more reliable than transcript-sentiment similarity). A 12-D
  multi-model representation is saved for a swap-and-rerun check.
- **Reliability, done correctly (`04c` §4c.5).** That 0.030 is a **run split-half** of *run-resolved*
  sentiment — so much of `04b`'s null is **run-noise**. Collapsed to end-state, the sentiment RDM's
  cross-character reliability is **≈ 0.20**, comparable to the survey representations' regime. But the reruns show the survey reps are **not** matched in reliability — `like` (0.196) ≫ `PC1` (0.055) >
  `positive_emotion` (0.035) — and **detection tracks reliability** (3 / 1 / 0 ROIs). So the `like` >
  `positive_emotion` brain contrast is **confounded with reliability**, not cleanly content. `like` and
  *end-state* sentiment have nearly matched reliability (~0.20), so the clean content test is an
  **end-state-sentiment IS-RSA vs `like`** — not yet run (`04b` used run-resolved sentiment, reliability 0.03).
  **ASK HAYOUNG!! / next analysis.**
- **→ Next.** (a) Gate reproduces Jin's `step04` verbatim (**no masking**; his impressions carry the 4 empty subjects as NaN, so `nanspearman` realizes his 29). If ROIs don't match his published set, suspect a stale `.npy` version — re-download his impressions/neural and re-run.
  (b) Run `04c` once its neural-collapse choice is set. (c) Decide IS-RSA-only vs adding Step 6/7 — noting
  the reliability-ceiling argument above pushes toward `06`/`07`.

### Step 5 — Cross-cohort group-level bridge  (`05`, scaffold)

- **Status.** Scaffolded, path-driven: group-averaged neural quantity vs group-level SONA targets (PC1 /
  positive emotion / like) via the 3 scramble groups. Neural quantity undecided. No figure yet.
- **Found (rerun).** Group-level `like` → **1 FDR-sig ROI [91]** (max|r| 0.299); `PC1`/`positive_emotion` → 0. Same direction as `04c`: the group-average *liking* co-varies with a group neural quantity, sentiment/PC1 don't.
- **Open.** Coarse (3 groups) — descriptive, reported beside `04b`/`04c`. Neural quantity is a Hayoung decision.

### Step 6 — Neural pattern-shift vs sentiment change  (`06`, scaffold)  ← the most direct test

- **Status.** Scaffolded: Jin's `step08` (neural pattern-shift ~ impression-update) with **our sentiment
  deltas** swapped for his USE impression-updates. Behavioral side runs now (1147 delta rows); neural side
  is path-gated with a preflight. No figure yet.
- **Why it matters.** The most *literal* test of the locked question — within-person *change* in sentiment
  vs within-person *change* in neural pattern.
- **Open.** Scope decision; needs extra brain inputs beyond IS-RSA: `pattern_shift/1TR_nearbytp.npy` (Jin
  `step06`; else `loaded_BOLD/`), `ahaannot_all.xlsx`, and a scene-shuffled permutation null
  (`scene_null.py`). The `06` preflight reports which resolve.

### Step 7 — Does shared *liking* drive shared brain activity?  (`07`, alt)

- **The reframe.** `04b` shows *character-affect sentiment* doesn't drive neural synchrony. But that's not
  the same as *viewer stance*: do people who **like** the characters similarly show similar brain activity?
- **Why it needs the survey, not the models.** Step 1 found the models don't read `like` (≈0), so this
  question can only be asked with the **survey's own `like` ratings** — a clean, separate handle.
- **Two levels (two questions):** **7.2** individual IS-RSA of neural ISC vs between-subject `like`
  similarity (Track B post-scan survey, *same cohort*, end-state) — the direct "like-alike people
  synchronize" test; **7.3** group-level bridge of a group neural quantity vs Track A run-resolved `like`
  (cross-cohort, coarser, correlational). Behavioral side runs now; brain side path-gated with a preflight.
- **→ Next (Hayoung).** Elevate alongside the sentiment result? Which level headlines? Same
  similarity-metric / neural-collapse choices as `04c`.

**The two-part story this sets up:** (1) *character-affect* sentiment does **not** drive neural synchrony
(`04b`, the dissociation); (2) does shared *viewer stance / liking* — measured directly by the survey —
**do so?** (`07`, open). Reported side by side, never pooled.

---

## Recommended path forward

1. Re-run `04a`/`04b` with the legacy RNG → confirm the gate matches Jin's published ROIs.
2. Settle the headline target (#1) and the `04c` collapse (#5), then run `04c`.
3. Decide brain scope (#3); if pattern-shift is in, place its extra inputs and wire `06`.
4. Pick the Step-5 neural quantity (#4), run the group-level bridge.
5. Draft the write-up around the **dissociation**: sentiment measures character-affect and, unlike
   impressions, does not drive neural synchrony — shown behaviorally (Fig 3) and neurally (Fig 4b).

## Loose ends / notes

- **Verification.** IS-RSA matches Jin's `step03`/`step04` exactly; fixed 3 discrepancies — bootstrap RNG →
  his legacy RNG; atlas → Schaefer-100 + Tian-16 (not AAL); added his `step04b` Fisher combine-p.
- **Docs.** Consolidated into `README.md` (reference) + this doc (design & brief). Pipeline renumbered so
  it reads `00·01·02·03·04·05` with no gap (Track B is now `02`).
- **Hard limit.** Only 3 scramble groups — the ceiling on group-level generalization; stated in limitations.

---

## Confirmatory vs exploratory (the forking-paths ledger)

We ran *many* analyses; declaring the a-priori set up front is what keeps the exploratory ones honest.

**Confirmatory — the a-priori plan (from Hayoung's original instruction + Jin's framework):**
- Target = `positive emotion` (Hayoung: "the last section… higher = more positive sentiment").
- Model-selection contest by leave-one-scramble-group-out cv-R² → pick the best model.
- Jin's Figure 1 (distance effect, `03`) and Figure 2 (IS-RSA, `04b`) with sentiment swapped for USE impressions.

**Exploratory — hypothesis-generating, treat as such until replicated on held-out data:**
- The 16-item alignment scan + full construct/target sweep; the affect-cluster target.
- PCA / PC1–PC3 dimensionality and the character-affect-vs-stance PC2.
- Track B nuances (`like`/stance positive at the individual level; `val_mean` > `val_final`).
- The good-person halo (`01b`); the composite variants (emotionally-stable in/out, optimized weights, stance subtypes).
- The `like`→brain analysis (`07`), the pattern-shift test (`06`), the survey IS-RSA (`04c`), the group bridge (`05`).

**Why it matters:** the confirmatory result is the model contest + the `04b` swap; everything else is a
lead. With only 3 scramble groups and n=29, the honest posture is "one pre-planned test + a rich map of
exploratory directions," not "many findings." Robustness for the headline (`04b`) is quantified in §4.4:
we were powered only for mean-r ≥ ~0.05, and the behavioral RDM's low reliability caps the attainable effect.

## ASK HAYOUNG!! — everything needing her input

*Core scientific decisions*
- **ASK HAYOUNG!!** — Headline target: character-affect (positive emotion / affect cluster) vs
  participant-feeling. *(lean: character-affect; stance = Track-B nuance)*
- **ASK HAYOUNG!!** — How to read the `04b` null (0 sig ROIs): real dissociation vs low power (n=29).
  *(lean: real — the gate `04a` had power on the same 29)*
- **ASK HAYOUNG!!** — Brain scope: IS-RSA only (`04b`), or also the pattern-shift test (`06`, the more
  direct within-person test)?
- **ASK HAYOUNG!!** — Neural quantity for Step 5: group-mean ISC vs neural-pattern distance vs event
  discriminability.
- **ASK HAYOUNG!!** — `04c` sub-choices: neural collapse over runs (mean / final-run / aha-window) and the
  scalar similarity metric (AnnaK `−|Δ|` vs mean).
- **ASK HAYOUNG!!** — Brain load path: reuse Jin's precomputed ISC vs recompute from raw fMRI.
- **ASK HAYOUNG!!** — Promote a halo item (`trustworthy` / `empathize`) to a secondary validation target,
  or keep purely descriptive? *(lean: keep descriptive)*

*Confirm-the-call (data suggests an answer; confirm it)*
- **ASK HAYOUNG!!** — Drop `emotionally stable` from the affect composite? *(inert: 0.366 with vs 0.367
  without)*
- **ASK HAYOUNG!!** — Leave the participant aggregate equal-weighted? *(tuning weights barely helps,
  −0.08 → +0.02, and overfits with 3 groups)*
- **ASK HAYOUNG!!** — Is the Flair pos/neg mapping fair? *(re-scaling barely moves it — likely its binary
  architecture, not our mapping)*
- **ASK HAYOUNG!!** — Is `|Δ valence|` the right operationalization of "sentiment change" for Step 6 (vs
  Jin's cosine impression-update)?

*Data / logistics*
- **ASK HAYOUNG!!** — Where is Jin's pattern-shift data (`1TR_nearbytp.npy` / `loaded_BOLD/`) and the aha
  annotations (`ahaannot_all.xlsx`), and can we get or generate the scene-shuffled permutation null?
- **ASK HAYOUNG!!** — How to recreate his brain *surface* figures (his conda env + Schaefer/Tian
  templates) — his own code guide says to ask if unclear.
- **ASK HAYOUNG!!** — Confirm survey scope: run-resolved SONA = the 16-item page; the 35-item survey is the
  fMRI post-scan only (no run-resolved 35-item data exists?).
- **ASK HAYOUNG!!** — Confirm the character column order (Jack / Kate / Randall / Kevin) in the `.mat` blocks.

---

## Reading list — to talk about this confidently

Mapped to where each is used; **Tier 1 = know cold** (you'll present/defend these).

**Tier 1 — core**
- **Parent paper** — Ke, Madhogarhia, Chun, Rosenberg, Leong & Song (2026, *bioRxiv*): the study, and the
  impression-update / IS-RSA / pattern-shift analyses you're re-running with sentiment. *(You're a co-author.)*
- **IS-RSA** — Finn et al. (2020, *NeuroImage*), "Idiosynchrony…": the IS-RSA framework + the **AnnaK vs
  nearest-neighbor** similarity models (this *is* your `04c` similarity-metric choice).
- **Inter-subject correlation (ISC)** — Nastase, Gazzola, Hasson & Keysers (2019, *SCAN*); origin: Hasson
  et al. (2004, *Science*). What ISC measures and why between-subject synchrony is meaningful.
- **RSA** — Kriegeskorte, Mur & Bandettini (2008, *Front. Syst. Neurosci.*): the RDM logic under Steps 3–5.
- **Winning model** — Barbieri et al. (2020, *EMNLP Findings*, TweetEval / Twitter-RoBERTa): your winner
  and the domain-match rationale.

**Tier 2 — be conversant (methods you used)**
- Sentiment models: VADER — Hutto & Gilbert (2014, *ICWSM*); SiEBERT — Hartmann et al. (2023, *IJRM*);
  BERTweet — Nguyen et al. (2020, *EMNLP*); zero-shot-via-NLI — Yin et al. (2019).
- Affect representation: Warriner, Kuperman & Brysbaert (2013); NRC-VAD — Mohammad (2018, *ACL*); dimensional
  affect — Russell (1980, circumplex) and **Tamir, Thornton, Contreras & Mitchell (2016, *PNAS*)** (3-D
  mental-state space — speaks directly to your PC1/PC2/PC3).
- Statistics: Shrout & Fleiss (1979) ICC; Hayes & Krippendorff (2007) / Krippendorff (2004) α; Benjamini &
  Hochberg (1995) FDR; Walther et al. (2016, *NeuroImage*) RDM reliability; Robinson (1950) ecological
  fallacy (your group-vs-individual caveat).
- Emotion dynamics: Hipson & Mohammad (2021, *PLOS ONE*) — utterance emotion dynamics (your trajectory framing).

**Tier 3 — background / the embedding angle**
- USE — Cer et al. (2018) (what Jin's impressions are). Brain–language-model alignment: Schrimpf et al.
  (2021, *PNAS*), Goldstein et al. (2022, *Nat. Neurosci.*), Toneva & Wehbe (2019, *NeurIPS*).
- Song et al. (2026, *Nat. Commun.*) — the earlier SocialAha paper (dataset lineage).
