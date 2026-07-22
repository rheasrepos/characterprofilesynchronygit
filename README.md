# Character Sentiment → Neural Synchrony

Undergraduate research project (CAB Lab, Summer 2026). In the SocialAha fMRI study, people watched a
scrambled TV episode and, between scenes, spoke about the characters. We turn **what they said about each
character into a positive-vs-negative "sentiment score,"** figure out **what that score actually captures**
(by comparing it to survey ratings), and then ask **whether that score is reflected in how similarly
people's brains respond.** For the brain part we re-run Jin Ke's published analyses, but feed in our
sentiment score in place of his original "impression" measure.

**Main question:** *When a person's expressed sentiment about a character changes across the 10 viewings,
does the way their brain represents that character change too?*

Parent paper: Ke, Madhogarhia, Chun, Rosenberg, Leong & Song (2026), *Neural dynamics of updating social
impressions during movie watching* (bioRxiv). Public code + data: https://github.com/jinke828/socialaha
(the author of this repo is a co-author on that paper).\

---

## The two goals

1. **What does the sentiment score actually measure?** A sentiment model boils speech down to one
   positive/negative number — but *which* human judgment does that number reflect? We answer this by
   comparing the score to survey ratings (Steps 1–2). Finding: the score reflects **how positive the
   character is portrayed** (their emotional state and being a "good person"), **not how much the viewer
   likes the character** (liking barely relates to it, at least at the group level).

2. **Is that score reflected in the brain?** Once we know what the score means, the brain steps (3–7) ask
   whether people who are more alike in the score also have more alike brain responses.

Goal 1 matters because it tells us how to read any brain result: "brain activity tracks *how positive the
character is portrayed*," not "brain activity tracks *a sentiment number*."

---

## The pipeline

Run in order — each notebook uses the files the previous ones saved (under `results/`).

### `00` — Build the sentiment score  *(done)*
- Score every spoken reflection with 6 off-the-shelf sentiment models (Twitter-RoBERTa, RoBERTa-ZS,
  VADER, Flair, SiEBERT, BERTweet); keep a positive and a negative number, and use **positive − negative**
  as the score. We keep all 6 so the data picks the best one.
- **Check 1 (does the score move?):** yes — a person's score changes a lot *across their own 10 viewings*
  (about 92% of the variation is within-person, not stable differences between people). That's the movement
  the main question needs.
- **Check 2 (do the models disagree?):** yes, enough that "which model best matches behavior/brain?" is a
  real question worth asking.
- `00b`, `00c`: word-list and embedding alternatives, built but **kept aside** (not part of the main flow).

### `01` (+ `01b`, `01d`) — Figure out what the score measures  *(done)*
- Compare each model's score to a **survey** where a separate group rated the characters each viewing.
- **Model contest:** Twitter-RoBERTa wins — its score explains ~**34%** of the survey ratings (a
  **cross-validated R²**: the share of variation explained on data the model wasn't fit to).
- **What it tracks:** "positive emotion," "emotionally stable," "good relationship" — i.e., the character
  being portrayed positively. The best small combo of these explains ~**37%**. In contrast, "how much the
  viewer likes the character" is ≈ 0.
- **PCA — principal component analysis (`§1.6b`):** you can squeeze the 16 trait ratings into one
  "overall good-person" summary number,
  but the plain "positive emotion" rating predicts the score just as well — so the fancy summary isn't
  needed. All candidate targets are saved together in `results/step1/01__targets.csv`.
- `01d`: extra robustness checks (the main result holds up). `01b`: what *else* the score happens to line
  up with (a broad "good-person halo").

### `02` — Double-check with the scanned people's own ratings  *(done)*
- The scan group filled out their own survey once, after the scan. We check the same things against those.
- **Result (36 people):** the score again lines up best with **how positive the character is** (about
  **0.54**); and here liking lines up somewhat too (~0.34–0.37) — a bit more than in the separate survey
  group. (The average score across viewings lines up better than the last-viewing score.)

### `03` — Does the score act like a lasting impression or a momentary feeling?  *(done)*
- Jin's original measure faded as viewings got further apart (a lasting, accumulating impression). Our
  sentiment score **barely fades** (similarity 0.71 → 0.64 across the gap). So the score behaves more like
  a **momentary feeling** than a lasting impression — evidence it captures something different from Jin's.

### `04a`/`04b`/`04c` — The brain test  *(run)*
The brain test — **IS-RSA (inter-subject representational similarity analysis)** — asks, per brain region:
**do pairs of people who are more alike in a measure also have more alike brain responses?** It lines up
two "who's-alike-with-whom" tables (called **RDMs, representational dissimilarity matrices**): one built
from the measure, one from the brain.
- **`04a` — sanity check:** re-run the test with **Jin's own measure**. It lights up a few brain regions,
  confirming the setup works. (Ours reproduces a *subset* of the exact regions he reported — 4 of 6 in one
  window, 2 of 6 in the other; all of ours are among his — as close as expected given small differences in
  people and random-number handling.)
- **`04b` — sentiment:** the transcript **sentiment score shows no significant brain regions.** *But* the
  score is too noisy to trust that null: split the data in half and the score's "who's-alike" pattern
  barely agrees with itself (its **reliability** ≈ **0.03**), and the biggest effect the noise could even
  allow (the **noise ceiling** — roughly √(reliability of the score × reliability of the brain)) is about
  **0.06** — so this is "can't detect," not "no effect." A big part of the noise is that the score bounces
  around viewing-to-viewing.
- **`04c` — survey ratings instead:** the scan group's own survey ratings are steadier, and here a few
  regions do light up — most for **"liking"** (3 regions), one for the overall-good-person summary, none
  for "positive emotion" (by the strict cutoff). **Important caveat:** "liking" was also the *most
  repeatably measured* survey item (repeatability ≈ 0.20 vs ~0.03–0.06 for the others), and how much each
  item lights up tracks how repeatable it is — so we can't yet separate "the brain cares about liking" from
  "liking was just measured most cleanly." Suggestive, not proven.

### `05` — Reach the separate survey group's ratings at the group level  *(run)*
- The separate survey group wasn't scanned, so their ratings can only touch the brain averaged into the 3
  viewing-order groups (coarse, low-power). Same hint as `04c`: group-average **"liking"** lines up with
  one brain region; the character-positivity ratings don't.

### `06` — The most direct version of the main question  *(built, brain half not yet run)*
- Asks it literally: on viewings where a person's score *changed* a lot, did their *brain pattern* change a
  lot? Uses the score-change files we already built. The behavioral half runs; the brain half still needs
  two files from Jin (`1TR_nearbytp.npy` and a permutation-null file), which aren't present yet.

### `07` — Does shared liking go with shared brain activity?  *(run)*
- The "liking" question on its own, using the survey's liking ratings directly (the models can't read
  liking). Reproduces `04c`/`05`: shared liking lines up with a few brain regions individually and one at
  the group level. Same caveat as `04c` (liking is also the most repeatable item).

**Order:** `00 → 00b → 00c → 01 → 01b → 01d → 02 → 03 → 04a → 04b → 04c → 05 → 06 → 07`.
(`01c` is just a pointer — its PCA moved into `01 §1.6b`; the old visualization notebook was retired and
its figures moved into the notebooks that make them. `06`/`07` are newer and partly awaiting data.)

**Paths to set (the only things you configure):** the transcript file + event-timing files (`00`); the
separate-survey folder (`01`); the scan-group survey folder (`02`, `04c`, `07`); Jin's brain-data folder
(`04a`–`07`).

---

## The two surveys we compare the score against

Every sentiment score comes from one place — what the scanned people said. We check it against two
*separate* surveys, kept apart because they measure different things:

| | who filled it out | when | used for |
|---|---|---|---|
| **Survey A** (`01`) | a *different* group of people (not scanned) | rated each viewing | figuring out *what the score measures*, viewing-by-viewing |
| **Survey B** (`02`, `04c`, `07`) | the **scanned people themselves** | once, after the scan | double-checking, and the only survey we can tie to the brain (same people) |

Jin used neither survey — his "impressions" came from the same speech we score, which is what lets us
swap our score in for his in a clean, same-people comparison in the brain steps.

---

## The data, and how we protect privacy

- **Survey A** (the separate raters): averaged into 3 groups (~12 people each) over the 10 viewings, for
  the 4 characters (Jack, Kate, Randall, Kevin).
- **Two separate groups of people.** The **scan group** (their speech, brains, and after-scan survey) are
  the same ~36 people. **Survey A raters** are *different* people; they only connect to the brain data
  averaged at the group level (Step 5).
- **Brain data** is Jin's shared intermediate data (plus the public OpenNeuro dataset). Our speech set and
  Jin's usable brain set are each 33 people but **not the same 33** — they overlap on **29**, which is who
  the brain steps use.
- **Privacy:** everything stays local; the raw spoken text is never printed (dropped right after scoring);
  only summary numbers and figures are shown; no participant data was shared with the AI.
- **No circular logic:** survey ratings are only ever the thing we compare *to* — they never go into the
  score itself.

---

## Figures — what each one shows

Each figure lives in the notebook that makes it, saved under `results/` (regenerates when you re-run).

| Figure | Notebook | What it shows |
|---|---|---|
| Sentiment over viewings (per character) | `00` | the score moves across viewings |
| Model-agreement heatmap | `00` | how much the 6 models agree |
| Within- vs between-person spread | `00` | most variation is within a person |
| One chart per person | `00` | `results/figures/participant_trajectories/00__<sub>.png` |
| All people, one character | `00` | `results/figures/00__contactsheet_<char>.png` |
| Average trajectory per character | `00` | `results/figures/00__mean_errorbar.png` |
| Which survey items the models track | `01` | `results/figures/01__item_alignment_heatmap.png` |
| Score vs survey, viewing-by-viewing | `01` | `results/figures/01__behavioral_vs_fmri.png` |
| The trait "summary axes" (PCA) | `01` | `01__pca_pc1_loadings.png`, `01__pca_pc2_pc3_loadings.png` |
| Good-person "halo" heatmap | `01b` | `results/step1b/01b__step1b_correlation_heatmap.png` |
| Does the score fade across viewings? | `03` | `results/figures/03__sentiment_distance_effect.png` |
| Brain test, per region | `04a`/`04b` | `04a__isrsa_impressions.png`, `04b__isrsa_sentiment_vs_impressions.png` |

Jin's published brain figures are colored 3-D brain surfaces (need his atlas templates + `nilearn`); our
per-region bar charts show the same numbers in a simpler form.

---

## How we checked our brain analysis matches Jin's

We pulled Jin's actual published code and compared ours line-by-line. Our brain test reproduces his method
faithfully. We found and fixed three differences: (1) our random-number method now matches his exactly;
(2) we use his brain atlas (116 regions: 100 cortical + 16 deep), not a different one we'd assumed; (3) we
added his step that combines the "before" and "after" results. Sanity check: with his own measure we
recover a subset of his reported regions (see `04a`).

---

## Open questions for the meeting (ASK HAYOUNG!!)

- **Which survey question do we say the score really captures?** Character-positivity is clearly strongest;
  liking is basically zero in the separate survey but shows up a little in the scan group's own survey.
  Which to lead with is a judgment call.
- **How to read the sentiment brain null (`04b`):** the score was too noisy/underpowered to detect
  anything, so "no regions" isn't strong evidence of "no effect." Agree on how to frame it.
- **The liking brain result (`04c`/`07`) is confounded with measurement quality** — liking is both a
  different thing *and* the most cleanly-measured survey item. The clean fix is to run the brain test with
  the *steadier, averaged* sentiment score (matched in quality to liking); not done yet.
- **`06` needs brain data — ASK HAYOUNG!!** It's the most direct test of the main question, but its brain
  half can't run until we have Jin's neural **pattern-shift** file (`data/brain/pattern_shift/1TR_nearbytp.npy`)
  and a permutation-null file. Neither is in this repo and neither resolved on the last run. Is it in Jin's
  shared data (his README lists a `pattern_shift/` folder), or do we generate it (his `step06`, which needs
  `data/brain/loaded_BOLD/`, + `scene_null.py`)?
- **A few smaller analysis choices** (how to average the brain over viewings; how to define "similarity")
  are flagged inside `04c`/`06`/`07`.
- **Parked (not core):** the separate mechanistic-interpretability idea and the audio analysis.

---

## Decision log (dated) — technical change history

*The durable record of what changed and why. Append new decisions here.*

**Foundations (Jul 1–6)**
- *Construct core:* `positive emotion` is a judgment of the **character's** state, not the viewer's;
  report both `like` and `positive emotion` and scan every item. Trait valence is tiered (affect
  statements / trait adjectives with impulsive & self-centered reverse-coded / neutral excluded / stance
  subtypes), all in `behavioral_constructs.py`; `emotionally stable` is a trait, flagged not silently used.
- *Models:* keep raw pos/neg per model, `valence = pos − neg`; six classifiers form the spine. Flair emits
  one label+confidence (mapped to the matching pole) — its saturation is a model property, not a pipeline
  choice. Added SiEBERT + BERTweet so "domain match helps" rests on two Twitter models + a large control.
  Everything runs locally.
- *Structure:* consolidated Step 0 (lexicons/embeddings quarantined to `00b`/`00c`); output provenance =
  notebook-prefixed files under `results/`; additions over replacement; IS-RSA notebooks grouped as `04a`/`04b`.
- *Statistics:* honest CV is **leave-one-scramble-group-out**, not 5-fold (5-fold leaked across the 3
  groups → optimistic 0.386; honest = 0.34). Reviewer robustness lives in `01d`.
- *Privacy:* never read raw participant files; only derived numbers/figures surface.

**Jul 8 — consolidation + brain run**
- Retired `02_visualizations` (figures inlined into `00`/`01`); PCA (`01c`) merged into `01 §1.6b` with a
  consolidated target file; removed a mislabeled orphan cell in `03`; Track B (`02`) run on real data
  (self-test bug that overwrote the real table fixed); `04c`/`05` made real path-driven (no dummy data);
  per-region IS-RSA figures + plain-English reads added to `04a`/`04b`.
- Verified against Jin's public code; fixed the random-number, atlas, and combine-p differences.

**Jul 8 (cont.) — margin notes, robustness, new scaffolds**
- Addressed the margin notes in `00`/`01`/`01b` as new cells (Krippendorff α=0.44; PC2/PC3; emotionally-
  stable is inert; no stance subtype tracked; full models×items table; trait correlations; `01b` heatmap +
  `like` version + FDR). Built `06` (pattern-shift) and `07` (liking→brain) scaffolds. Added `04b §4.4`
  robustness (power floor ≈0.05; noise ceilings). Merged the meeting brief into `DESIGN.md`; placed
  `ASK HAYOUNG!!` markers across notebooks.

**Reruns (Jul 9) — real brain findings.** Gate (`04a`): Jin's impressions → 4 regions after [9,60,64,98] /
2 before [64,99] (a **subset** of his published set — CLOSE). Sentiment (`04b`): **0 regions** (and via the
combined test). Repeatability: run-resolved sentiment score ≈ 0.030, brain ≈ 0.127 → most detectable effect
≈ 0.062 vs a ~0.05 floor ⇒ underpowered. **Survey ratings do detect:** `04c` liking → 3 regions [24,48,60],
overall-good-person → 1 [70], positive-emotion → 0; `05` group liking → 1 region [91]; `07` reproduces
these. **Reliability caveat:** the survey items are *not* equally repeatable (liking 0.196 ≫ good-person
0.055 > positive-emotion 0.035) and how much each detects tracks its repeatability — so liking-beats-the-
rest is confounded with measurement quality; the clean test is a brain test using the *averaged* sentiment
score (repeatability ≈0.20, matched to liking), not yet run. Exploratory throughout (29 people / 3 groups);
regions need names + replication.

---

## Running it / dependencies

`requirements.txt` (pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, openpyxl). Building the
score in `00` also needs transformers, torch, flair, vaderSentiment; `00c` needs tensorflow-hub. The brain
notebooks need only the basic scientific packages for the statistics; reproducing Jin's colored brain
surfaces additionally needs his environment (`nilearn` + the atlas templates).

---

## AI-use disclosure

Portions of the analysis code in this project were developed with the assistance of Claude (Anthropic),
used as an interactive tool for drafting and executing analysis notebooks, debugging, computing
statistics, generating figures, and writing documentation. All study design, construct definitions,
model-selection criteria, interpretation, and validation were performed and verified by the author, who
takes full responsibility for the content. The sentiment-analysis models and sentence embeddings used as
measurement instruments are distinct third-party tools and are cited separately. All AI-assisted analyses
are provided as reproducible notebooks so that every result can be independently re-run and verified. No
participant data was shared with the AI assistant.
