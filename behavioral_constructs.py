"""Shared behavioral-construct definitions for the two ground-truth tracks.

ONE source of truth for item labels and valence coding, imported by BOTH:
  * Track A (01)  -- SONA behavioral survey, group x run, 16 items.
  * Track B (02) -- fMRI participants' own post-scan survey, individual x end-state, 35 items.

Keeping this upstream is what stops the two tracks' constructs from drifting apart. Each track loads
its own DATA (different files, cohorts, resolutions) but builds targets from these shared definitions.

------------------------------------------------------------------------------------------------------
SOUNDNESS: every item is classified by HOW the survey elicits it, not post-hoc, so the valence coding
is transparent and defensible. Three character tiers plus a stance family:

  TIER 1  AFFECT STATEMENTS  -- items worded as an explicit bipolar contrast ("X vs opposite"), so they
                               are unambiguously valence-coded by design (positive emotion, good vs bad
                               relationship, rational vs irrational, good vs bad job, content vs
                               discontent). These are the *strictest* valence items.
  TIER 2  TRAIT ADJECTIVES   -- "how well does <trait> describe the character": evaluative but not framed
                               as valence. Positive-keyed (warm, competent, ...) and negative-keyed
                               (impulsive, self-centered, reverse-coded).
  NEUTRAL                    -- descriptors with no clear evaluative direction (extraverted, assertive,
                               reserved) and magnitude-only items (big vs small influence). Excluded from
                               every valence composite.
  STANCE (participant)       -- the viewer's own relation to the character, split by SUBTYPE because they
                               are heterogeneous: affective attitude (like, empathize, care, want-to-be-
                               friends/character, attractive), cognitive (understand), similarity
                               (similar), and a negative pole (annoying, reverse-coded).

`positive emotion` (TIER 1) is the clean single anchor. `emotionally stable` is TIER 2 (a trait, not an
affect statement) -- flagged so its inclusion in any composite is a visible choice, not a default.

Scale: 1..7 Likert (4 = neutral). Reverse-code = (SCALE_MAX + 1) - x.
"""
import pandas as pd

SCALE_MAX = 7

# --- TIER 1: explicit bipolar valence statements ("X vs opposite") ---
AFFECT_STATEMENTS_POS = ["positive emotion", "good relationship", "rational behavior", "good job", "content"]

# --- TIER 2: trait adjectives with a keyed valence direction ---
TRAIT_POS = ["warm and kind", "intelligent", "agreeable", "emotionally stable", "open-minded",
             "trustworthy", "competent", "friendly", "self-disciplined", "optimistic", "humorous",
             "sincere and honest", "determined", "caring and supportive", "ambitious"]
TRAIT_NEG = ["impulsive", "self-centered"]                      # reverse-coded

# --- NEUTRAL: no clear evaluative direction; excluded from valence ---
NEUTRAL = ["extraverted", "assertive", "reserved", "big influence"]

# Character valence = TIER 1 + TIER 2 (negatives reverse-coded). Membership identical to before.
CHAR_POSITIVE = AFFECT_STATEMENTS_POS + TRAIT_POS
CHAR_NEGATIVE = TRAIT_NEG

# --- STANCE (participant) by subtype -- heterogeneous, so kept separable ---
STANCE_AFFECTIVE = ["like", "empathize", "care what happens", "want to be friends",
                    "want to be character", "attractive"]
STANCE_COGNITIVE = ["understand"]
STANCE_SIMILARITY = ["similar"]
STANCE_NEGATIVE = ["annoying"]                                 # reverse-coded
STANCE_POSITIVE = STANCE_AFFECTIVE + STANCE_COGNITIVE + STANCE_SIMILARITY  # full "last section" bundle

# The 16 page-one items collected run-resolved in the SONA survey (Track A scope)
ITEMS_16 = ["warm and kind", "intelligent", "agreeable", "extraverted", "impulsive",
            "emotionally stable", "open-minded", "trustworthy", "competent",
            "rational behavior", "positive emotion", "good relationship",
            "empathize", "understand", "like", "similar"]

# single-item anchors present in BOTH surveys -- kept identical across tracks
CHAR_EMOTION_ITEM = "positive emotion"
LIKE_ITEM = "like"


def _rev(s):
    return (SCALE_MAX + 1) - s


def _scoped(cols, scope):
    return [c for c in cols if scope is None or c in scope]


def valence_composite(df, scope=None):
    """Character valence: mean of TIER 1 + TIER 2 items, negatives reverse-coded.
    `scope` restricts to a given item set (pass ITEMS_16 for the Track-A-matched composite)."""
    pos = _scoped(CHAR_POSITIVE, scope)
    neg = _scoped(CHAR_NEGATIVE, scope)
    parts = [df[pos]]
    if neg:
        parts.append(_rev(df[neg]))
    return pd.concat(parts, axis=1).mean(axis=1)


def affect_statement_composite(df, scope=None):
    """STRICTER valence: only TIER-1 explicit bipolar statements (positive emotion, good relationship,
    rational, ...). Excludes trait adjectives. Use to test whether the sentiment tracks the
    explicitly-valenced items specifically, without the trait-adjective soundness worry."""
    pos = _scoped(AFFECT_STATEMENTS_POS, scope)
    return df[pos].mean(axis=1)


def stance_composite(df, scope=None, subtypes=("affective", "cognitive", "similarity")):
    """Participant stance: mean over the requested subtype families, negative pole reverse-coded.
    Default = the full 'last section' bundle (Hayoung's recommended target). Pass e.g.
    subtypes=('affective',) to isolate the affective-attitude items (like/empathize/...)."""
    fam = {"affective": STANCE_AFFECTIVE, "cognitive": STANCE_COGNITIVE, "similarity": STANCE_SIMILARITY}
    pos = [c for st in subtypes for c in fam[st]]
    pos = _scoped(pos, scope)
    neg = _scoped(STANCE_NEGATIVE, scope)
    parts = [df[pos]]
    if neg:
        parts.append(_rev(df[neg]))
    return pd.concat(parts, axis=1).mean(axis=1)
