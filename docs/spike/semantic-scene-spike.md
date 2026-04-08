# Spike: Tier 1 Semantic Scene Gate

**Issue:** [#2](https://github.com/joaodotwork/metalcut/issues/2) (closed)
**PR:** [#6](https://github.com/joaodotwork/metalcut/pull/6)
**Date:** 2026-04-08
**Branch:** `spike/semantic-scene` (merged)
**Status:** Negative result — infrastructure shipped as opt-in scaffolding

---

## Context

Phase 3 of the cut-refinement spike (#1) reached F1 = 49.4% on the anime test clip with two-pass lookahead adaptive thresholding. The remaining 34 false positives were characterized as **isolated single-frame spikes** where explosion/action frame changes score identically to real scene changes — frames where the pixel-difference metrics (quick MAD, histogram correlation, Canny edge diff) cannot distinguish "explosion N→N+1" from "editorial cut to a different shot."

The hypothesis: a lightweight semantic descriptor — something that captures *what the scene looks like* rather than *how much it changed pixel-wise* — should filter these FPs without sacrificing recall.

## Goal

Add a per-frame "scene fingerprint" that the hard-cut detector can use as a precision-side filter. Specifically:

- Reject hard-cut candidates whose predecessor and successor are perceptually nearly identical (same composition + same color palette).
- Preserve all candidates where either the composition or the palette has actually shifted.
- Zero processing overhead when the feature is disabled.
- Negligible overhead when enabled (no second decode pass, no extra GPU↔CPU transfers).

## Approach (Tier 1)

Two cheap descriptors, computed only when the existing detailed-stage gate already fires (so we already have CPU-side `np.ndarray` frames):

1. **Perceptual hash (`FrameMetrics.phash`)** — DCT-based 64-bit pHash. Resize to 32×32, run `cv2.dct`, keep the top-left 8×8 low-frequency block, threshold at the median, pack into a 64-bit integer. Hamming distance between two hashes measures perceptual similarity (range 0–64).
2. **Color palette descriptor (`FrameMetrics.color_palette`)** — HSV histogram with 8 bins on H + 8 bins on S, concatenated and L2-normalized. Cheap alternative to k-means dominant colors. L2 distance between two descriptors measures palette change.

**Semantic gate logic** (`CutDetector.detect_cuts()`):

```python
if use_semantic:
    if palette_dist < palette_change_threshold AND phash_hamming < phash_hamming_threshold:
        reject candidate  # within-scene motion, not a cut
```

The gate is intentionally conservative — it rejects only when *both* descriptors say "perceptually identical." Either descriptor on its own crossing the threshold is enough to keep the candidate.

**Default thresholds:** `palette_change_threshold = 0.3`, `phash_hamming_threshold = 12` (out of 64 bits).

**Zero-overhead integration:** pHash and palette are computed inside `combined_difference()` immediately after the existing `.get()` call that pulls UMat → ndarray for histogram/edge analysis. No second decode, no extra GPU↔CPU transfer, no separate per-frame storage — descriptors live in `all_metrics[i]` alongside the existing scores.

## Methodology

1. Annotated two test videos with the GUI annotator (Y/N review + browse mode for missed cuts):
   - **Eva trailer** (`13nSISwxrY4`) — anime, fast cuts, action sequences
   - **B&W film** (`hQtH7MS2Rec`) — high-contrast monochrome, slower editorial pace
2. Ran detection with `sensitivity=0.7 score_mode=max` baseline.
3. Re-ran detection with `--semantic` enabled, identical other parameters.
4. Compared the two output sets byte-for-byte.
5. Inspected per-cut pHash and palette distributions to confirm the gate had anything to filter.

## Results

### Eva trailer (`13nSISwxrY4`)

| Run         | Detected | TPs | FPs | Missed | Precision | Recall |
|-------------|---------:|----:|----:|-------:|----------:|-------:|
| Baseline    |       39 |  35 |   4 |     93 |     89.7% |  27.3% |
| `--semantic`|       39 |  35 |   4 |     93 |     89.7% |  27.3% |

### B&W film (`hQtH7MS2Rec`)

| Run         | Detected | TPs | FPs | Missed | Precision | Recall |
|-------------|---------:|----:|----:|-------:|----------:|-------:|
| Baseline    |       35 |  27 |   8 |     39 |     77.1% |  40.9% |
| `--semantic`|       35 |  27 |   8 |     39 |     77.1% |  40.9% |

**Detection sets are byte-identical with and without `--semantic` on both videos.** The semantic gate filters zero candidates.

## Why Zero Impact

Per-cut pHash distribution at all 39 Eva detections:

- **True positives**: pHash hamming median **28**, range 12–40 (out of 64 bits)
- **The one unambiguous false positive**: pHash 24, palette distance 0.985 — *higher* palette distance than the TP median (0.743)

The gate's logic is "reject if palette **and** pHash are both small," i.e. only when the candidate is perceptually nearly identical to its predecessor. On both test videos, every detected cut crosses the score threshold *because* it produces a large visual difference, and that pixel-level change correlates with large pHash deltas (median 28 of 64 bits differ — well above the 12-bit gate threshold). There is no candidate that scores high on pixel-difference but low on perceptual distance, so the gate never fires.

The FP pattern from #2 — *"explosion/action frame changes score identically to real scene changes"* — would manifest as: high pixel-difference score + low pHash distance (same composition) + low palette distance (same colors). **Neither test video produces such a candidate.** Two possibilities:

- The original anime/Psycho test clips referenced in #1's Phase 3 had a content quirk that the current test corpus does not reproduce, or
- The two-pass lookahead + IQR damping (#1, Phase 3) already eliminated those FPs upstream, before they ever reached the semantic gate.

The second is plausible — Phase 3 cut FPs from 76 → 34 by recognizing sustained-action neighborhoods, which is exactly the pattern explosions produce.

## What We Found Instead

While annotating, two completely different recall problems surfaced — neither described in #1 nor #2:

### 1. Burst-cut ceiling

The Eva trailer has cut bursts every 1–2 frames (e.g. 14.26 → 14.31 → 14.39 → 14.43), faster than `min_cut_distance = 0.15s` permits. The detector physically cannot fire densely enough to catch them — the second cut in each pair gets suppressed by the distance gate. This caps recall at ~27% on burst-heavy content regardless of any other tuning.

**Fix surface:** loosen `min_cut_distance`, or replace the fixed-distance gate with something content-aware (e.g. let consecutive cuts through if both score well above adaptive threshold). Needs its own issue.

### 2. Subtle-cut floor

The B&W film has hard cuts producing pixel-difference scores of **0–2**, well below `quick_gate = 10`. These cuts never become candidates at all — the quick stage rejects them before any detailed analysis runs. Visually they're real editorial cuts but the frame content is similar enough (matched composition, monochrome, slow motion) that the MAD signal is tiny.

**Fix surface:** lower `quick_gate` for low-dynamic-range content, or use a different signal entirely (e.g. histogram or pHash *as the primary score* on monochrome). Needs its own issue.

## Decision

**Ship the implementation as opt-in scaffolding.** The infrastructure is sound, the integration is zero-overhead-when-off, and the negative result is documented here and in PR #6. When test content matching #2's stated FP pattern becomes available, the gate can be re-validated and thresholds tuned without further plumbing work.

Concretely, what shipped:

- `FrameMetrics.phash()`, `FrameMetrics.color_palette()`, `FrameMetrics.hamming_distance()`
- Semantic gate inside `CutDetector.detect_cuts()` (gated on `config.use_semantic`)
- `DetectionConfig` fields: `use_semantic`, `phash_size`, `palette_bins`, `palette_change_threshold`, `phash_hamming_threshold`
- `--semantic` CLI flag
- New `phash` and `palDst` columns in `--diagnose` output (only when `--semantic` is on)
- Annotator browse mode for marking missed cuts (false negatives) — enables real recall measurement, not just precision
- Removed misleading "frame reading slower than FPS" warning from `VideoReader` (it was measuring round-trip time including downstream processing, not actual I/O)

## Follow-ups

These are not in scope for this spike but were uncovered during evaluation:

1. **Burst-cut ceiling** — `min_cut_distance` is too coarse for fast-cut content. Needs either a lower default, a content-aware override, or replacement with a per-pair confidence gate.
2. **Subtle-cut floor** — `quick_gate = 10` rejects valid low-MAD cuts on monochrome / matched-composition content. Needs investigation: lower the gate, or add a fallback signal that doesn't rely on pixel MAD alone.
3. **Tier 2 / Tier 3 semantic** — if the FP pattern from #2 reappears on richer test content, the next escalation would be CLIP-style embeddings or a tiny scene-classifier head. Tier 1 was deliberately the cheapest possible filter; Tier 2+ would need real GPU work and was deferred until Tier 1 proved its value (which it didn't, on this corpus).

## Out of Scope

- Tuning the existing semantic thresholds against more test videos (no candidates to filter, so tuning is moot)
- CLIP/DINOv2/embedding-based scene similarity (Tier 2+)
- Audio-based cut detection
- Multi-shot scene segmentation (vs. per-cut detection)
