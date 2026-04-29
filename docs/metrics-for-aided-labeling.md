# Metrics for aided labeling

Notes from a session evaluating the `tuned_merge_v4` recommendation model
inside voxa. The takeaway: **the standard precision / recall / F1 numbers
we surface today are the wrong question for an aided-labeling tool.**
This doc records what we found, why the conventional metrics mislead
here, and which metrics would map onto actual user pain.

## What the current numbers say

Compare-mode result on `annotated/munich_water_pump` (the model's own
training scene), at IoU threshold 0.3:

| Metric | Value |
|---|---|
| GT cuboids | 126 |
| Pred cuboids | 411 |
| TP / FP / FN | 20 / 391 / 106 |
| Precision | 0.049 |
| Recall | 0.159 |
| F1 | 0.074 |
| Mean IoU (over TPs only) | 0.724 |

Per class:

| Class | GT | TP | FP | FN |
|---|---|---|---|---|
| pipe | 70 | 13 | 227 | 57 |
| tank | 8 | 7 | 139 | 1 |
| equipment | 48 | 0 | 0 | 48 |
| structural | 0 | 0 | 25 | 0 |

These look catastrophic. They aren't quite — but they aren't wrong
either. The metrics tell a true story of *1:1 detection performance*,
which isn't the story the tool needs.

## Why mean IoU is 0.72 while precision is 0.05

`iou_mean` is computed only over the 20 successful 1:1 matches
(`backend/main.py::compare`). It says: *when the model produces a box
that crosses the 0.3 match threshold, the box is tight* — 72% overlap
on average. It says nothing about the 391 boxes that didn't match or
the 106 GT objects nothing matched against.

Precision and recall are computed over all 411 predictions and 126 GT
respectively, with greedy 1:1 matching. They say: *as a 1:1 detector,
the model is bad.*

Both numbers are correct. They measure different things.

## What's actually happening: over-segmentation

The model breaks single GT objects into multiple predicted segments.
~3.3 predictions per GT on munich. Concretely, one long pipe gets
split into ~5 short segments:

- 1 segment may overlap the GT box ≥ 0.3 → counted as TP at IoU ~0.7
  (a single segment along the pipe still has decent box overlap with
  the long GT box).
- The other 4 segments cover ~1/5 of the GT each, IoU < 0.3 → FPs.
- The user, however, sees five candidate boxes near the pipe. They
  pick one, adjust, delete the others. **That's faster than drawing
  from scratch** — but the metric scores it as 4 FPs and 1 TP.

The tank class makes this clearest: 8 GT, 7 hit (87% recall in coverage
terms) but 139 FPs because each GT tank produced ~17 candidate boxes.
The model *found* nearly every tank; it just didn't produce a clean
1:1 inventory.

## Why the conventional metrics are wrong here

Standard PR/F1 at IoU 0.3 was designed for **autonomous detection** —
the system is the final step, and over-prediction is a real failure
because nobody filters it. In aided labeling the user is the filter.
Penalizing over-segmentation 1:1 is double-counting a problem that
doesn't translate to user pain.

What the user actually cares about:

1. **Did the model find a usable starting point near every GT object?**
   If yes, drawing-from-scratch becomes adjusting-from-near-correct.
2. **How much noise must the user wade through?** The cost of
   over-prediction is real, but it's *delete time*, not exclusion from
   match counts.

PR/F1 don't separate these.

## Metrics that fit the aided-labeling use case

Three metrics worth tracking, replacing or complementing the current
panel:

### Coverage @ τ — "did the model find each GT?"

For each GT, take the **best IoU against any pred of the same class**,
ignoring the 1:1 matching constraint. Then:

```
coverage(τ) = |{ g ∈ GT : max_p IoU(g, p) ≥ τ }| / |GT|
```

Useful values of τ:
- **0.3** — "the model found a usable starting point" (matches current threshold)
- **0.1** — "the model found something in the neighborhood that needs resizing"

The gap between coverage(0.1) and coverage(0.3) tells you how much
post-edit resizing the user will do.

### Mean best-IoU per GT — overall tightness

```
best_iou_mean = mean_g [ max_p IoU(g, p of same class) ]
```

A single number indicating average recommendation tightness across
all GT. Smoother than thresholded coverage, harder to game.

### Predictions per GT — noise cost

```
pred_per_gt = |Pred| / |GT|
```

Captures how much delete-the-extras work the user faces. On munich
this is 3.3 — meaning roughly 2.3 spurious boxes per GT to clean up.
If this number is high, the model is over-segmenting; user time is
dominated by deletion.

### What to keep from the current set

- **Per-class breakdown** — the most actionable signal in our run.
  Reveals "equipment is entirely missed" much more clearly than
  aggregate F1.
- **Mean IoU over TPs** — useful as a "when it gets it, how good?"
  diagnostic, but should be labeled as conditional on match. Drop
  the headline status.

## What to drop or de-emphasize

- **F1 as the headline metric.** F1 conflates over- and under-prediction
  into one number that doesn't map onto user pain — labeling time isn't
  symmetric in those failure modes. Move it to a "rigor / model card"
  subsection, not the at-a-glance panel.
- **Aggregate precision / recall** without the per-class breakdown.

## The metric we don't have but should

**Time to label a scene, with recommendations vs. from scratch.** This
is the actual quantity the tool optimizes. Numbers above are proxies.
If we ever label 2–3 scenes both ways, the result is a model card
nothing else can produce.

Practically: pick one scene without GT (e.g., one of the unlabeled
annotated tier scans), have someone label it from scratch, time it.
Then label another comparable scene with recommendations on, time it.
Compare. Repeat as scenes accrue.

## Why the model is the primary bottleneck (not the UI)

The bundle's own card claims `r@50 = 76.8%` on munich (per-segment,
point-level recall). Our cuboid-level metrics are much worse because
projecting per-point segments to AABBs amplifies fragmentation. Even
on its own training scene, cuboid-level coverage @ 0.3 is around 16%
under 1:1 matching. On other scenes (single-scene-trained model →
domain shift) it's worse: navvis_vlx3 returned 25 instances total,
mostly classified as "structural" — the model couldn't find pipes or
tanks.

The recommendation pipeline is the *intended* fix: voxa serves bad
recommendations → user corrects → corrections become training pairs →
retrain on N scenes → recommendations improve. The model card itself
says multi-scene training "isn't wired up yet — once the second scene
(`smart_ais`) is corrected through voxa, this script gets a small
extension that concatenates per-pair training sets." That's the loop.

## Recommended next steps

1. **Don't overinvest in metric work right now.** The existing
   precision/recall numbers are technically correct and not actively
   harmful — just misread. A single coverage column would close the
   biggest gap.
2. **Add coverage @ 0.3 and 0.1 to Compare** — small backend change
   (`backend/main.py::compare`), small UI add. Removes the misleading
   F1 and surfaces the actual aided-labeling signal.
3. **Run one timed labeling pass on a real scene.** That gives the
   only metric that genuinely matters. Update this doc with results.
4. **Don't retrain the model until at least one more scene's worth of
   corrected labels exists.** Single-scene training is the root cause
   of poor recommendations on other scenes.
