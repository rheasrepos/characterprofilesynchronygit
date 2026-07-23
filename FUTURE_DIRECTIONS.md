# FUTURE_DIRECTIONS.md — the parking lot

Ideas deliberately **out of scope** for this thesis. Captured so they're not lost, so
the writing phase can start clean. Nothing here changes the headline finding:

> Encoder-only NLP sentiment tracks *depicted character affect*; human synchrony tracks
> *viewer stance/liking*. A double dissociation. That's the snapshot this paper reports.

Rule for this list: an idea only earns a place in the paper's Discussion if writing the
argument *demands* it. Otherwise it lives here until grad school.

---

## Methods extensions (the "one more pipeline" urges)
- **CCA / non-linear CCA** (e.g. L2-regularized Ridge CCA): move from the macro double
  dissociation to a micro node-to-region alignment map. Cool, but it's a different paper —
  answers "*which* units align," not "*whether* the two systems dissociate."
- **Temporal lags / sliding-window synchrony**: does the affect–synchrony relationship
  shift with lag? Currently a single end-state / run-level view. A whole temporal study.
- **Video / multimodal models** vs text-only sentiment: does a model that sees the scene
  track something closer to viewer stance than encoder-only text does? Directly probes the
  research-program question (artificial vs biological social processing).
- **Decoder / generative LLMs** vs encoder-only: deferred on purpose. New embeddings/LLMs
  were explicitly frozen for this thesis.

## Power / data
- **Larger, independent cohort**: cross-cohort agreement (liking r=0.82, char-affect 0.54)
  rests on n=12, group-level, character-mean. Replication with a bigger fMRI sample is the
  clean confirmation — not a same-thesis patch.

## Blocked on others (not on you)
- **`06` pattern-shift regeneration**: blocked on Jin's missing `scene_null.py`, and the
  regen cells are 700+ min. Revisit only if Jin supplies the null. Parked, not abandoned.

## Inference / stats to revisit later
- Reconcile the subject-level permutation citation (code cites Chen et al. 2016; the
  reading was Nastase et al. 2019) — this is a *writing* fix, do it in the Methods, not a
  new analysis.
- hrf 3-vs-4 lag choice: on the Jin question list, not a solo exploration.

---

## Explicitly CLOSED — do not relitigate
These were tested and settled. Re-running them is the pipeline trap in disguise.
- **Shared-liking IS-RSA variants**: exhausted. Single `like` is the fairest; aggregate
  and PC2 are *provably* worse. Do not add more variants.
- **More embeddings / LLMs**: frozen for this thesis.
- **`06` heavy regen**: see above — blocked, and expensive.
