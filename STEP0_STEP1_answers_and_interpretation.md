# Step 0 + Step 1 — Answers, Interpretation, and Restructuring Blueprint

Prepared as write-up scaffolding. Part A and B answer the questions embedded in your notes,
section by section, with justifications. Part C is the concrete notebook-consolidation plan.
Part D collects the genuine open decisions and the analyses you flagged but haven't run.

Nothing here changes your results — it interprets what the notebooks already compute and tells you
which claims are load-bearing vs. descriptive.

---

## Part A — Step 0: building the character-sentiment vectors

### A0. What is the object, in one paragraph (for the methods section)
For each fMRI participant you have a spoken reflection per run. You transcribe it, split by character,
and score the *valence* of the text with several NLP sentiment models. One score per
(participant, character, run) is a **run-vector**; stacking a participant's 10 runs for one character
is a **profile** (a 10-entry trajectory over the watched narrative). Word count is carried as a length
covariate so a "more positive" score can't just mean "longer reflection." That's the whole of Step 0:
turn text into a run-resolved valence trajectory, then package it three ways (levels, centered, deltas).

### A1. "What do we even use the character profiles for?" (your biggest Step-0 question)
The profiles are the **input representation** for everything downstream. Concretely:

- **Step 1** validates the profile (does group-mean valence track the behavioral survey? → yes, cv-R²≈0.307).
- **Steps 3–4** feed profile-derived structure into the brain/IS-RSA analysis (does across-run sentiment
  change track neural change?).
- The three saved views map onto three different questions:
  - **levels** (raw 10-run values) = "what does the participant think about the character at run k" — the
    primary object, the one your mentors specified.
  - **centered** (each profile minus its own mean) = **shape only**, baseline removed. Used when the
    question is about *trajectory pattern*, not whether someone is globally positive. This is the standard
    move before correlation-based RDMs (Walther et al. 2016) — see A5.
  - **deltas** (run-to-run change, 9 entries) = the **change** signal, for the "does sentiment move, and
    does that movement track the brain" question (Utterance Emotion Dynamics; Hipson & Mohammad 2021).

So: levels are the measurement, centered/deltas are the two transforms that isolate the *change* question
your project is actually about. Say this explicitly in the write-up — it pre-empts "why three files?"

### A2. Which models output neu/neg/pos vs. only pos/neg (for your methods table)
| Model | Type | Classes | neutral? |
|---|---|---|---|
| Twitter-RoBERTa | transformer (Twitter-domain) | neg / neu / pos | **yes** (3-way) |
| RoBERTa zero-shot (MNLI) | transformer, NLI-as-classifier | pos / neg (candidate labels) | no |
| VADER | lexicon + rules | pos / neu / neg proportions | yes |
| Flair | DistilBERT single-label | POSITIVE **or** NEGATIVE + confidence | **no** (binary, one pole) |
| SiEBERT | RoBERTa-large, general | pos / neg | no (binary) |
| BERTweet | transformer (Twitter-domain) | neg / neu / pos | yes |
| Warriner / NRC-VAD (00b) | VAD lexicons | one **bipolar** valence axis | n/a (not pos/neg) |

The reason this matters: valence is computed as `pos − neg` for all classifiers, but **Flair and SiEBERT
can't express ambivalence** (no neutral, and Flair puts all confidence on one pole). That is exactly why
they underperform, and §1.7 proves the low rank is architectural, not a mapping artifact (see B12).

### A3. Model-selection rationale (fill the "MODEL SELECTION EXPLANATION" placeholder)
You didn't pick models ad hoc; the set is a **designed contrast** so that whichever wins, the win means
something:
- **Two Twitter-domain transformers** (Twitter-RoBERTa, BERTweet) — matched to informal, affect-laden
  spoken reflection. If domain-match helps, it should show up in *both*, not one lucky model.
- **A general large transformer** (SiEBERT) — a size-vs-domain control: it's large but not Twitter-tuned,
  so it isolates whether the edge is about domain or about model size.
- **A non-fine-tuned transformer** (RoBERTa zero-shot / MNLI) — sentiment with no sentiment training.
- **A lexicon+rules baseline** (VADER) — a training-free floor.
- **A binary classifier** (Flair) — included partly to *demonstrate* the cost of no-neutral architecture.
- **VAD lexicons** (Warriner, NRC-VAD; quarantined to 00b) — the cog-neuro field standard, kept as
  reference but flagged as ~23% coverage (partial-vocabulary), which conflicts with your full-coverage rule.

Framing for the paper: "We span architecture (lexicon → non-tuned transformer → fine-tuned → domain-matched)
and label geometry (binary vs. 3-way) so that model ranking is interpretable rather than incidental."

### A4. The two diagnostics D1 and D2 — what they establish
- **D1 (does sentiment move across runs?)** — within-person SD vs. between-person SD, reported as a ratio
  and as **ICC(1)** (Shrout & Fleiss 1979; Bartko 1966). A ratio ≫1 / low ICC means the variance is mostly
  *within-person* (i.e. the score changes run-to-run), which is the precondition for a change-over-time
  study. If sentiment were a fixed trait per person, there'd be nothing to track.
- **D2 (do the models agree enough to be worth comparing?)** — cross-model Spearman + **interval
  Krippendorff's α ≈ 0.44** (Krippendorff 2004; Hayes & Krippendorff 2007). Read: *moderate* agreement —
  related but **not interchangeable**. That's the justification for keeping several models rather than
  asserting one. (Your note "should prob do Krippendorff's α" — done, cell 0.4b/[18]. The "% sign
  agreement" line is the informal version; α is the citable one.)

### A5. "Why centered profiles (profile = its own mean = 0)? WHY?"
Because your question is about **trajectory shape / change**, not absolute level. If you correlate two raw
profiles, the correlation is dominated by the baseline offset (a globally positive person vs. a globally
negative one) — that swamps the run-to-run pattern you care about. Mean-centering removes the per-(person,
character) baseline and keeps only the shape, which is the standard preprocessing before correlation-based
RDMs in RSA (Walther et al. 2016, NeuroImage). Levels stay primary; centered is the version the RSA/brain
step actually consumes. (This same logic reappears in 1d.6, where centering isolates the *within-character*
effect — see B14.)

### A6. "Within- vs between-person spread" — how to explain it
Two sources of variance in the scores:
- **within-person**: how much one participant's rating of one character moves across the 10 runs (the
  narrative-driven change).
- **between-person**: how much participants differ from each other in their average rating of a character
  (stable individual differences).

The D1 ratio and ICC(1) partition these. You *want* within-person to be non-trivial (there's a signal that
moves) and you want to know the between-person spread because that's what individual-differences / IS-RSA
analyses feed on. The variance-overview figure (0.4c-v) does the parallel split for **characters vs people**:
how much do the four characters differ vs. how much do participants differ — i.e. is the signal mostly
"who's the character" or "who's the viewer."

### A7. The reliability / IS-RSA ceiling (§0.7) — what it means and why it lives here
Split-half over runs (odd vs. even), Spearman-Brown corrected, at two levels:
- **Measure reliability SB ≈ 0.685** — the mean valence per (subject × character) is a **reliable** measurement.
  This is *why* Step 1 can reach cv-R²≈0.307; you can't predict behavior with an unreliable score.
- **Similarity-structure reliability SB ≈ 0.229** — the reliability of *subject-pair similarity* derived from
  the score. This is the **ceiling on any IS-RSA effect**, and it's low **not because the score is bad** but
  because pairwise similarity is an inherently noisier, derived quantity.

Why report it at construction (Step 0) and not later: psychometric reliability is a property of the measure,
so it belongs where the measure is built. Stating the 0.685-vs-0.229 gradient up front explains the entire
downstream arc — strong measure, modest similarity ceiling — before a reviewer asks. Keep this; it's one of
the more defensible things in the pipeline.

### A8. The quarantine notebooks (00b lexicons, 00c embeddings) — one sentence each for the paper
- **00b (Warriner, NRC-VAD, + an EmoBank TF-IDF/Ridge proxy)**: dimensional VAD lexicons, kept for
  completeness but *partial coverage* (~23%), so they're reference models, not contenders. The EmoBank
  regressor is the modern full-coverage replacement (label it a bag-of-words proxy, weaker than a
  transformer VAD regressor).
- **00c (USE, GPT-2)**: **semantic** embeddings, not sentiment. USE is included specifically because the
  parent paper (Ke et al. / "Jin's paper") used it to represent impressions — you keep it to reproduce/
  benchmark against that representation at the brain step, not as a sentiment measure.

They're "quarantined" precisely so nobody mistakes a 512-d semantic vector or a 23%-coverage lexicon for
the validated scalar valence. That demarcation table at the top of `00` is doing important work — keep it.

---

## Part B — Step 1: the behavioral ground truth and the selection contest

### B0. The one-paragraph frame (for the methods section)
Two cohorts share only the 3 scramble **groups**, never individuals: the **fMRI cohort** (transcripts →
sentiment) and the **SONA behavioral cohort** (per-run survey ratings). Because they share only groups,
every Step-1 comparison is done at the **group level** — average within (group, character, run) on both
sides, giving a 3 × 4 × 10 = **120-cell** grid on each side, then regress. The result: expressed sentiment
best tracks the character's **positive emotional state**, not the viewer's **liking** — and that
dissociation is the finding, not a nuisance.

### B1. "Why is different-grain OK?" (capturing the answer you liked, cleaned up)
Mixing grains across the *project* (group-level here, participant-level at the brain step) is valid **as
long as both sides of any single comparison share a grain** — which they do. The three caveats are about
interpretation, not validity:
1. Group-level fit **does not license individual-level claims** (ecological fallacy / Simpson's paradox,
   Robinson 1950).
2. Averaging ~12 participants **inflates apparent effect sizes** — report these as group-level numbers.
3. Picking the winner at group level and reusing it at the participant level in the brain step **assumes**
   the best group representation is also the best individual one — an explicit assumption, not free.

### B2. "Why 120 rows? Is it at group level?"
Yes, group level. 3 scramble groups × 4 characters × 10 runs = 120 cells, on **both** the behavioral side
(mean survey rating per cell) and the model side (mean sentiment valence per cell). Characters and runs are
**never collapsed** — each character keeps its own 10-run trajectory per group. The two 120-row grids are
what the regression aligns. (If a cell has NaNs from skipped ratings, the mean uses available responses;
you never impute.)

### B3. "What cv-R² should I expect per model?" (calibrating expectations)
For a **single** sentiment number predicting a group survey trajectory, out-of-sample (leave-one-group-out):
- **~0.31 (0.307 exactly)** is the ceiling you actually hit (Twitter-RoBERTa on positive emotion) — strong
  for a univariate predictor.
- **~0.18–0.23** for the other transformers and VADER (respectable).
- **~0 or negative** for Flair/SiEBERT/lexicons on most items, and for essentially *all* models on
  `like`, `similar`, `extraverted`, `impulsive`.

Negative cv-R² is normal and just means "worse than predicting the mean" out-of-fold — expected for items
the signal genuinely can't read. Don't treat negatives as bugs; they're the boundary of the construct.

### B4. "Describe the selection contest — is it group level? one regression per model per trait?"
Yes to group level. There are **three nested scans**, and it's worth keeping them distinct in the write-up:
- **1.3 / 1.4 — the contest proper**: for each model, one leave-one-group-out regression of its group-level
  valence on the *chosen target* (`positive emotion`). Ranks models. Winner = Twitter-RoBERTa (cv-R²≈0.307,
  ρ≈0.588).
- **1.5 — the full item-alignment scan**: model × item, all 16 items one at a time. Maps *what the signal
  tracks*, not which model wins. This is the table you liked.
- **1.6 / 1.6b — the construct scan**: five pre-grouped constructs (char emotion, attitude, char-valence
  composite, participant-feeling aggregate, mixed) + a PCA-derived aggregate.

So the regression is run "for every model against the target" (contest) *and* "for every model against every
trait" (item scan) — both, at the group level. They answer different questions; label them so.

### B5. Reading the full item-alignment scan (the table you flagged as "really interesting")
The ranking is *not* a clean character-vs-participant split, and that's the point:
- **`positive emotion` stands alone at the top** (best cv-R²≈0.31, and the only item with a healthy mean
  across models ≈0.14). The signal reliably reads the character's **positive emotional state**.
- A secondary band of **evaluation/competence** items: good relationship, emotionally stable, warm and kind,
  intelligent, competent, rational, open-minded (~0.06–0.26).
- **Bottom (≈0 or negative)**: `like`, `empathize`, `similar`, `extraverted`, `impulsive` — viewer stance
  and non-valence personality descriptors the signal can't read.
- **Curiosity worth a sentence**: the *cognitive-trait* items (competent, intelligent, trustworthy) are best
  predicted by **Flair**, not the winner — a hint Flair tracks a slightly more "trait-like" signal than the
  transformers' cleaner valence. (Same pattern in 01b.)

Takeaway for the paper: "expressed sentiment most cleanly indexes the character's depicted positive affect,
secondarily broad evaluation, and not viewer stance or fine personality." Keeping both the character and
participant results in the notebook is what lets you show this contrast rather than assert it.

### B6. "How should this have changed later directions?" (your margin question)
Honestly, it should sharpen two things:
1. The **brain hypothesis** should be about the character's *depicted affect trajectory*, not the viewer's
   attitude — because that's the only thing the sentiment measure validly carries. If a later brain result
   is framed as "neural tracking of how much the viewer likes the character," the sentiment measure doesn't
   support that; reframe it as depicted-emotion tracking.
2. It licenses **dropping `like`/stance as a sentiment target** and instead treating the like-vs-positive-
   emotion dissociation as a *result* (a validity boundary), which is what you already do. It also motivates
   bringing in the non-valence impression features (USE) at the brain step to capture what valence can't.

### B7. PCA — scree, PC1, "does every trait contribute to PC1?", and what the components mean
- **Scree graph**: bar = variance each PC explains, line = cumulative. PC1≈43%, PC2≈11%, PC3≈7.5%
  (cumulative ≈61% by PC3). The elbow after PC1 says there's **one dominant evaluative axis** plus a couple
  of smaller interpretable ones.
- **PC1 loadings**: after sign-fixing to the overall-positive direction, essentially every trait loads
  positive except `impulsive` (negative). So yes — **almost every trait contributes to PC1**, and that's not
  a bug: it means PC1 *is* a general **"good-person"/positivity halo**. That's a substantive finding, matching
  the survey's own internal bundling (01b: positive emotion correlates 0.89 with emotionally stable, 0.70
  with rational behavior, etc.).
- **"Shouldn't the components tell me which traits contribute *nothing*?"** They do — look at loadings near
  0 on a given PC. On PC1 few are near zero (that's the halo). Where discrimination lives is **PC2/PC3**:
  - **PC2 (≈11%)** = character **affect statements** (positive emotion, rational behavior, emotionally
    stable, good relationship) vs. **viewer stance** (like, empathize, understand, similar) + impulsive.
    This is *character-state vs viewer-stance* — the exact dissociation the project rests on. **This is the
    PC worth reporting.** (Your note "PC2 seems the more participant-forward axis" — right.)
  - **PC3 (≈7.5%)** = extraverted/impulsive — a temperament axis.
- **"What is the consolidated target / affect cluster?"** The consolidated target file
  (`01__targets.csv`) carries *every* candidate so downstream steps read one source of truth:
  `positive_emotion` (single item), `affect_cluster` (positive emotion + emotionally stable + good
  relationship — the strongest small composite), `char_valence_composite` (all positively-keyed char traits),
  `like`, and unsupervised `PC1`. The **affect cluster** is just that 3-item mean; it's the best-performing
  compact composite (cv-R²≈0.355).

### B8. "What if we did PCA with the brain data?" (your flagged idea)
Legitimate and interesting, but it's a **different** analysis with a different interpretation, so flag it as
future work rather than folding it in:
- PCA on the **survey traits** (what you did) asks "how many evaluative dimensions do people use." PCA on the
  **brain data** would ask "how many dominant spatial/temporal modes are in the neural response," which is a
  dimensionality-reduction step for encoding/RSA, not a construct analysis.
- A more targeted version that connects the two: use the survey **PC2** (character-vs-viewer axis) as a
  regressor against neural patterns, or do a cross-decomposition (PLS/CCA) between trait-PCs and brain-PCs.
  That directly tests "does the brain separate depicted-affect from viewer-stance the way the survey does."
  Worth proposing to Hayoung; don't shoehorn into Step 1.

### B9. The "each model vs BOTH targets" table (posemo_nonPCA / PC1_PCA / affect_cluster)
It ranks every model against three versions of the target at once so no single target choice is load-bearing:
the raw `positive_emotion` item, the unsupervised `PC1` aggregate, and the `affect_cluster` composite.
Reading it: Twitter-RoBERTa wins on all three (0.307 / 0.207 / 0.355); the transformers lead; PC1 is a bit
*lower* than the single item because PC1 blends in traits the signal doesn't track (that's expected and is
the point of B7). The message: **the winner and the ordering are stable across target definitions**, so your
choice of `positive_emotion` as the headline isn't cherry-picked.

### B10. The behavioral-vs-fMRI co-fluctuation figures (the 4×3 grids) — what they show
Per (character × group) cell, overlay the two cohorts' z-scored 10-run trajectories: fMRI-cohort sentiment
vs. SONA-cohort survey rating. Per-cell Spearman asks **"do the two independent cohorts fluctuate together
across runs?"** Mean cell ρ ≈ **0.61** with `positive emotion`. This is arguably your strongest single piece
of validation: two *different sets of people* produce matching run-by-run trajectories for the same
characters, so the sentiment signal is capturing something real about the stimulus, not an artifact of one
cohort. (Kevin group 1 is the weak/negative cell — worth a footnote; small n and Kevin is the low-signal
character.)

### B11. "mean cell rho with PC1 = 0.445 vs positive-emotion ~0.59" — why lower
Exactly as predicted: PC1 is the broad good-person axis, which **mixes in traits the sentiment signal
doesn't read** (competence, personality), so co-fluctuation with PC1 is diluted relative to the pure
positive-emotion item. This is confirmation that `positive_emotion` is the *right* headline target — the
signal aligns with depicted affect specifically, not with general evaluation. Report both numbers; the drop
is evidence, not a problem.

### B12. §1.7 Flair robustness — what the code does (fill your "description for the Flair code")
Flair only ever emits one label + a confidence, so your baseline mapping put the confidence on the winning
pole and 0 on the other — which **cannot express graded valence**. §1.7 reconstructs a *graded* score:
recover P(positive) from the chosen-pole confidence (`p_pos = fp if fp>0 else 1−fn`), rescale to [−1, 1]
(`2·p_pos − 1`), and re-run the identical leave-one-group-out regression vs. positive emotion. Result:
cv-R² barely moves (≈0.118 → 0.120). **Conclusion: Flair's low rank is its binary/over-confident
architecture, not your pos/neg mapping** — and the same logic covers SiEBERT (also binary, no neutral).
This directly answers your "double-check the Flair range conversion is fair" note: yes, it's fair, because
a fairer conversion changes nothing.

### B13. §1.9 follow-up results — what they mean
- **variance explained PC1..3 = [0.43, 0.107, 0.075], cumulative [0.43, 0.537, 0.612]** → one dominant axis
  (43%) + two minor interpretable ones; ~61% of trait variance in 3 dims (see B7).
- **PC2 poles: [positive emotion, rational behavior] vs [similar, impulsive]** → character-state vs
  viewer-stance/temperament — the reportable second axis.
- **PC3 poles: [understand, empathize] vs [impulsive, extraverted]** → a cognitive-stance vs temperament axis.
- **affect cluster WITH vs WITHOUT emotionally stable: 0.355 vs 0.354** → `emotionally stable` is **inert**;
  drop it for construct purity (it's a trait, not an affect statement) or keep it — the number doesn't move.
- **stance subtypes (affective −0.008, cognitive +0.027, similarity −1.076)** and **participant stance
  equal-weight −0.06 vs optimized −0.081** → the model tracks *none* of the viewer-stance subtypes, and
  tuning the weights doesn't rescue it. So the participant-stance null is about **the construct**, not about
  how you combine items. (With only 3 groups, weight-tuning also overfits — don't lean on the optimized number.)

### B14. §1d — the reviewer-robustness notebook, and "should the null have come earlier?"
`01d` is the honesty layer, and it matters because the 120 cells are **not independent** (3 groups, ~12
shared people, 10-run autocorrelated series). What it establishes:
- **Group-aware CV**: leave-one-group-out R²≈0.307 (vs naive 5-fold ≈0.320) — holds when a whole stimulus
  stream is held out. **Leave-one-character-out is negative** — the model tracks *dynamics*, not a held-out
  character's absolute level. State this explicitly; it's a real limit.
- **Within-character effect (1d.6)**: after centering out character-level differences, ρ≈0.58, R²≈0.40 —
  the effect is genuine *run-to-run* change (your actual research question), not just "Kevin is low."
- **Clustering-correct null (1d.2)**: circular-shift permutation p≈0.0005.
- **Noise ceiling (1d.3)**: split-half reliability ≈0.91, ceiling ≈0.96; winner is ~62% of ceiling → not
  noise-limited.
- **Length control (1d.4)**: 0.588 → 0.583 controlling word count → not a length artifact.
- **FDR (1d.5)**: 15/16 items survive BH q<.05; positive emotion strongest.
- **Mixed model (1d.7) + cluster bootstrap (1d.8) + generalization audit (1d.9)**.

**On ordering** (your question): the *null and reliability* logically belong **with the headline result**,
not in a separate appendix notebook — a reader wants "here's the effect, and here's the honest version"
together. But you should **not** move the null *before* the contest; the sequence is: build ground truth →
run the contest → immediately report the clustering-correct versions of that same contest. My
recommendation in Part C folds 01d in as the back half of the Step-1 result section (as "honest numbers"),
not as a prequel and not as a separate file.

### B15. §1.10 embedding validation — what to add
Right now it shows the 768-d RoBERTa embedding also validates against behavior (cv-R²≈0.443, ρ≈0.68),
slightly beating the 3-d score on the identical test — closing the gap that the embedding's validity was
*asserted* (same model's penultimate layer) rather than *demonstrated*. What to add:
- The **caveat sentence** it already flags: 768 features across 3 outer folds is a demanding fit; treat as
  **supportive, not definitive** (report the ridge alpha chosen and note the CI is wide).
- A one-line **interpretation of the gap**: the embedding carries non-valence structure the scalar throws
  away, so a modest gain is expected — but the scalar remains the *interpretable* object, which is why it,
  not the embedding, is the headline measure. (Otherwise a reviewer asks "why not just use the embedding.")

### B16. 01b — "why recompute instead of loading the group-level trajectories? why not run-level too?"
- **Why recompute**: 01b needs **all 16 items** (`gt_all`), whereas the saved `01__ground_truth_group_level.csv`
  already has them — so 01b *could* load it. The recompute is defensive redundancy (auto-locates the folder,
  self-contained), but it's genuinely redundant with 01.1. In the consolidated notebook (Part C) 01b reads the
  saved ground-truth file instead of re-loading the .mat — one loader, one source of truth.
- **"Should we also do run-level? participant / character level?"** — Good instinct, and they answer
  different questions:
  - **group-level** (what you have): the only grain where the two cohorts are comparable. Keep as primary.
  - **participant-level**: *within the fMRI cohort you can't* validate against SONA (different people), but
    you can look at run-level fluctuation per fMRI participant (you already make the per-participant
    trajectory figures in 0.4c). That's descriptive, not a validation.
  - **character-level**: 1d.9(a) already gives the within-character effect per character (4 semi-independent
    replications). That *is* your character-level view.
  - **run-level "does it match in fluctuation"**: that's exactly the co-fluctuation figure (B10), which is
    run-resolved. So you have run-level fluctuation at the group grain; you don't have a valid
    participant-grain cross-cohort test, and that's a data-design fact, not an omission.

### B17. "Other behavioral data from the fMRI cohort itself — shouldn't we run that?"
Yes — and that's **Track B** (the post-scan 35-item survey from the fMRI cohort, end-state, individual).
It's referenced in your notes and the demarcation table but isn't in these six notebooks (it's the `02`
notebook). It matters because it's the *same-people* validation that SONA can't give you. The cross-cohort
agreement check (01d-adjacent in your notes: liking Spearman +0.82, character-affect +0.54, n=12) is the
bridge that says the SONA cohort is a valid stand-in for the fMRI cohort's own ratings — which is what lets
you use SONA as ground truth at all. **Track B should be treated as part of Step 1's validation story**, per
your note. See Part C for where it slots.

### B18. The BH-FDR table (80/96 survive) — what it means
16 items × 6 models = 96 group-level correlations; Benjamini-Hochberg controls the false-discovery rate
across all 96. **80 survive q<.05** — the sentiment↔survey relationships are broadly real, not multiple-
comparisons noise. The top of the list (positive emotion×Twitter-RoBERTa r=0.588, emotionally stable, competent
×Flair, open-minded, intelligent…) shows the **halo**: the signal correlates with a whole evaluative bundle,
not one item. This is the multiple-comparison guard a reviewer will ask for; keep it, and report it as
"broad evaluative alignment survives FDR," with the caveat that group-level correlations are still
dependency-inflated.

### B19. "Which items co-move with `like` vs `positive emotion`" — the interesting table
This is a genuinely nice result and worth a paragraph:
- **`positive emotion`** co-moves most with **emotionally stable (0.87), good relationship (0.70),
  rational behavior (0.68)** — a *character-state* cluster.
- **`like`** co-moves most with **trustworthy (0.85), empathize (0.84), agreeable (0.83), warm and kind
  (0.81), open-minded (0.80), understand (0.77)** — a *warmth / relational* cluster.
- **empathize↔like** are highly correlated; `like` is embedded in a warmth-and-trust bundle, while
  `positive emotion` sits in an emotional-stability bundle.

Interpretation: the survey itself separates **"is the character in a good state"** (positive emotion) from
**"do I warmly regard the character"** (like). Your sentiment models track the *former* and miss the
*latter* — so the like-vs-positive-emotion dissociation isn't a quirk of two items, it's two coherent
construct clusters, and the NLP signal lands squarely on the character-state one. **Why anchor on positive
emotion, not like** (your 01b note): because positive emotion is the item the models actually track
(cv-R²≈0.307 vs ~0 for like), so it's the meaningful anchor to ask "what else co-moves." A like-anchored halo
would mostly reflect survey structure, not the model.

### B20. Is there a like-vs-positive-emotion confound? (your flagged worry)
Not a confound in the damaging sense — it's the **finding**. positive emotion and like load on the same PC1
(both positive) but split on PC2 and are tracked differently (B9/B11/B19). The thing to *state* so it can't
be read as a confound: the models do **not** predict `like`, so any downstream claim must be about depicted
character affect, never viewer attitude. The dissociation is clean and reported, which is the opposite of a
hidden confound.

### B21. Cross-cohort agreement (your closing note) — what to do with it
"SONA Track-A ratings vs fMRI-cohort Track-B ratings: liking ρ=+0.82, character-affect ρ=+0.54 (n=12
group×char)" — this **validates using SONA as the ground truth for the fMRI cohort's sentiment**, especially
for liking (0.82). It's the linchpin that justifies the whole cross-cohort design, so it should sit **early**
in Step 1 (right after you introduce the two cohorts), not buried at the end of 01d — a reader needs to
believe the cohorts agree *before* you use one to validate the other. Caveat to keep: n=12, group-level,
character-mean agreement.

---

## Part C — Restructuring blueprint (what to merge, cut, reorder)

Guiding principles: (1) one loader / one source of truth per object; (2) main result and its honest
(clustering-correct) version live together; (3) descriptive "halo" analysis and robustness are subsections,
not separate files; (4) cross-cohort agreement comes *before* the cohort is used as ground truth.

### Step 0
- **Keep `00` as-is** structurally — it's already well-ordered (data → score → baselines → diagnostics →
  profiles → reliability). Only fill the prose placeholders (model selection A3, "what are profiles for" A1).
- **Merge `00c` into `00b`** → `00b_quarantine_lexicons_and_embeddings.ipynb`. Both are "non-sentiment
  reference representations"; 00c is 3 cells. (Done — see deliverables.)
- Rename the merged notebook's intro to make the two families explicit: dimensional VAD lexicons (partial
  coverage) and semantic embeddings (USE/GPT-2, for the brain step).

### Step 1 → one consolidated notebook
Target order for `01_step1_ground_truth_CONSOLIDATED.ipynb`:

1. **1.0 Framing + demarcation** (from 01 cell 0): two cohorts, group-level, the positive-emotion vs like
   construct note.
2. **1.0b Cross-cohort agreement** (NEW placement — from your closing note / the 02 bridge): SONA vs fMRI-cohort
   ratings agree (liking 0.82). *Establish this before using SONA as ground truth.* (Stub cell if the data
   lives in 02; otherwise move the computation here.)
3. **1.1 Build group-level ground truth** (01 cells 1–3) + **1.1b structure check** (01 cells 4–6). This is
   the single loader; everything else reads its saved CSV.
4. **1.2 Aggregate models to group level** (01 cells 7–8).
5. **1.3 Target comparison** (01 cells 9–11).
6. **1.4 The contest / model ranking** (01 cells 12–13).
7. **1.4-honest = former 1d** (01d cells): group-aware CV, permutation null, noise ceiling, length control,
   within-character effect, mixed model, bootstrap, generalization audit. Folded in **right after the contest**
   as "the honest numbers," not a separate notebook and not before the contest.
8. **1.5 Full item-alignment scan** (01 cells 14–16) + **FDR** (from 01d.5, dedup with 01b's FDR).
9. **1.6 Five constructs** (01 cells 17–19).
10. **1.6b PCA + consolidated target** (01 cells 20–36) + **PC2/PC3** (01 cells 43–44 from 1.9a/b).
11. **1.7 What else does the signal track — the halo** = former **01b** (its 1b.3/1b.4/1b.5), but reading the
    saved `01__ground_truth_group_level.csv` instead of re-loading the .mat. This is where the
    like-vs-positive-emotion co-movement table (B19) and BH-FDR halo (B18) live.
12. **1.8 Behavioral vs fMRI co-fluctuation** (01 cells 40–41; PC1 version 33–34 as a subsection).
13. **1.9 Flair robustness** (01 cells 37–39).
14. **1.10 Embedding validation** (01 cells 53–54) with the added caveat (B15).
15. **1.11 Findings + open decisions for Hayoung** (consolidate the three scattered "ASK HAYOUNG" cells from
    01, 01b into one decision list).

Redundancies to cut when you execute the merge in your own environment:
- 01b's re-loader of the .mat (use the saved ground-truth CSV).
- The two FDR computations (01b.5b and 01d.5) — keep one.
- The `model_group` rebuild is repeated in nearly every self-contained cell; that's fine for standalone
  execution but in a single ordered notebook you can build it once in 1.2 and reuse.

I have **not** physically merged 01/01b/01d into one file blindly, because several cells are only
"self-contained" by re-reading CSVs and a naive concatenation would run the loaders many times and could mask
a missing-file error. Instead I've built a **combined notebook that preserves every code cell verbatim in the
order above, with section headers**, so you can run it top-to-bottom and delete the now-redundant reloads as
you verify each section. See deliverables.

---

## Part D — Genuine open decisions + analyses you flagged but haven't run

Decisions that are actually Hayoung's (not resolvable from the data):
1. **Headline target**: character-affect (positive emotion / affect cluster) vs. participant-feeling.
   *Data leans character-affect.*
2. **Drop `emotionally stable`** from the affect composite? *Inert (0.355 vs 0.354) — cosmetic.*
3. **Participant aggregate weighting**: equal vs tuned? *Tuning barely helps and overfits with 3 groups.*
4. **Flair mapping** fair? *Yes — §1.7 shows re-scaling changes nothing.*
5. **Survey scope + character column order** (16-item run-resolved SONA vs 35-item post-scan; Jack/Kate/
   Randall/Kevin) — confirm against instruction.pdf p.6.
6. **Promote a halo item** (trustworthy/empathize) to a secondary target, or keep descriptive? *Lean descriptive.*

Analyses you flagged as wanted — my recommendation on each:
- **Run-level / participant-level / character-level fluctuation** — you already have run-level (co-fluctuation)
  and character-level (1d.9a); participant-grain cross-cohort isn't possible (different people). *Do: add a
  short markdown noting this so it reads as a design fact, not a gap.*
- **PCA on the brain data** — real future work; better framed as PLS/CCA between trait-PCs and brain-PCs
  (B8). *Propose to Hayoung, don't fold into Step 1.*
- **Aggregate of any non-zero-explained trait** — worth a quick check but risks overfitting with 3 groups;
  report as exploratory only.
- **PC2 as the character-vs-viewer axis** — this is under-exploited and is arguably your cleanest structural
  result; consider promoting it from a §1.9 follow-up to a named result.
- **Track B (fMRI-cohort 35-item survey)** — bring into the Step-1 validation narrative (B17/B21).

---

*Caveat on all numbers here: I read them from your notes and notebook outputs; I could not re-execute the
notebooks (your data isn't in this workspace), so verify the consolidated notebook runs top-to-bottom once
in your environment.*
