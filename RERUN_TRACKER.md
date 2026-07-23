# Re-run tracker — changes this session & what must be re-executed

## Audit result (character/group comparison concern)
**No mistake found — nothing to fix.** Every downstream analysis holds character fixed and content-aligns
runs:
- `04b`, `04b.1`, `07` build similarity **per character** (`for ch`), pairs formed **within group** only.
- `04a`, `04b`, `06`, `03` match brain-vs-behavior **condition-by-condition** on (run, character); the
  scramble is undone by `rearrange_new` / `event_idxs`.
- `02` merges on `["pid","Character"]`; `03` cell 19's second-order RSA ravels characters but both sides
  share the identical (char, pair) layout, so it's element-matched, not pooled.

Optional refinement (not a bug): the IS-RSA correlates brain-sim vs behavior-sim across the 40 (run×char)
conditions per pair, which includes between-character variance. This is the parent paper's method,
faithfully reproduced. If you ever want a strict within-character-only version, center per character
before the RSA — but that would deviate from the Jin reproduction, so keep it as a sensitivity check only.

## Changes made this session
1. **Stale-number fixes** in `01`, `01b`, `01d` (markdown + two hardcoded output echoes). Cosmetic —
   the numbers already matched the code (0.307 / 0.588). **No re-run needed.**
2. **Consolidated Step 1** (`01_step1_..._CONSOLIDATED.ipynb`) + **merged `00b`**. Already re-run and
   verified byte-identical to originals. **Done.**
3. **RoBERTa-ZS → 3-way** (`00`, Model 2: added `"neutral"` candidate label; `neu` now real). **REQUIRES RE-RUN.**
4. **Reliability contest** cells `0.7b` / `0.7c` added to `00`. Run once after scoring exists.

## Re-run order & impact (for change #3 and #4)
| Notebook | Re-run? | Why / impact |
|---|---|---|
| `00` | **Yes** | Regenerates scores with 3-way ZS + runs the reliability contest. Needs the transformers. Twitter-RoB / VADER / SiEBERT / BERTweet / Flair are deterministic → **unchanged**; only RoBERTa-ZS changes. |
| `00b` | No | Lexicons/embeddings independent of ZS. |
| `01` consolidated (+ `01b`/`01d` if kept) | **Yes** | RoBERTa-ZS row in the contest/tables changes. **Winner (Twitter-RoB) unchanged — 0.307 / 0.588 stay.** |
| `04b` (12-D rep, cells ~12/14) + `04b.1` | **Yes** | The 12-D vector concatenates all 6 models' `[pos,neg]`, including ZS. The 3-D Twitter-RoB `SENT` (from `03`) is unaffected. |
| `03`, `04a`, `04c`, `05`, `06`, `07`, `02` | No | Use the Twitter-RoB winner, the survey, or Jin's data — none read RoBERTa-ZS. |

## Added (optional, run locally)
- **`00` §0.7b/0.7c** — reliability contest (scalar classifiers + fused) and ceiling levers. Runs on the
  scored CSV; no model needed.
- **`03b` §3b.6/3b.6b** — multi-model **embedding** contest: extracts penultimate embeddings from
  Twitter-RoBERTa / BERTweet / SiEBERT / RoBERTa-ZS-base, then runs the same validity (§1.10) + structure
  reliability (§0.7) tests. **Needs each model + transcripts → run locally**; caches to
  `results/embeddings/03b__emb_*.pkl`. This is a semantic-space contest, separate from the pos/neg/neu
  question (which does not apply to embeddings).

## Neutral (pos/neg/neu) — settled
Counterfactual on your re-scored data (strip neutral = `(pos−neg)/(pos+neg)`, exact for softmax models):
keeping neutral helps or is harmless for **every** model (Twitter +0.022, ZS +0.040, BERTweet/VADER ~+0.007);
Flair/SiEBERT are binary architectures with no neutral to add. ZS was the only mis-configured model, now
fixed (0.226 → 0.265, 2nd place). **Nothing further to strip or enable.**

## Sanity check on the re-run
After re-running `00`, confirm the non-ZS models came back **identical** (they should, they're
deterministic). If any of them moved, something else changed and is worth investigating before trusting
the new numbers.
