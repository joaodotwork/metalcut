# Spike: Cut Detection Refinement

**Issue:** [#1](https://github.com/joaodotwork/metalcut/issues/1)
**Date:** 2026-03-12
**Branch:** `spike/cut-refinement`

---

## Current State

The detection pipeline has two layers:

1. **FrameMetrics** (`src/core/metrics.py`) — per-frame scoring
   - Quick stage: downscale to 64x36, mean absolute difference (0-100)
   - Detailed stage (if quick > 10.0): histogram correlation + Canny edge diff, weighted 0.4/0.4/0.2

2. **CutDetector** (`src/core/detector.py`) — temporal scoring + thresholding
   - Keeps a 5-frame history of scores
   - Temporal boost: `current * (1.0 + max(0, trend)) * (1.0 + std)`
   - Final score: `quick_score * (1.0 + 0.5 * temporal_score)`
   - Cut fires when final score > `detailed_threshold`

## Findings

### P0 — Bugs

#### 1. UMat conversion bug in `combined_difference()`
**File:** `src/core/metrics.py:134-169`

`combined_difference()` passes frames straight to `histogram_difference()` and `edge_difference()`, but both are typed `np.ndarray` and call `.shape` directly — which crashes on `cv2.UMat`. The `quick_difference()` method handles UMat fine, but the detailed path does not.

**Fix:** Convert UMat to ndarray before calling histogram/edge methods, or make those methods UMat-aware. Converting to ndarray is simpler since `calcHist` and `Canny` don't benefit from UMat in this context (small intermediate results).

#### 2. Temporal score explosion
**File:** `src/core/detector.py:55-71, 104`

The boost formula is unbounded: `current * (1.0 + max(0, trend)) * (1.0 + std)`. When consecutive frames have high variance (e.g., fast action sequences), `std` can be large, multiplying the score by 3-5x or more. Then line 104 applies *another* multiplicative boost: `quick_score * (1.0 + 0.5 * temporal_score)`.

This double multiplication causes false positives in high-motion content.

**Fix:** Clamp the temporal boost. For example:
- Cap `std` contribution: `min(std, 1.0)`
- Cap overall temporal_score: `min(temporal_score, current * 2.0)`
- Or switch to an additive boost instead of multiplicative

### P1 — Config & Thresholds

#### 3. Config not wired
**File:** `config/default_config.json`, `src/cli/main.py`

The config file exists with `quick_threshold`, `detailed_threshold`, and `min_cut_distance`, but `main.py` constructs `CutDetector` purely from CLI args. The config is never loaded.

**Fix:** Load config as defaults, let CLI args override. Add a `--config` flag.

#### 4. Hardcoded magic numbers
**Files:** `src/core/metrics.py`, `src/core/detector.py`

| Value | Location | Purpose |
|-------|----------|---------|
| `10.0` | metrics.py:151 | Quick gate for detailed analysis |
| `(64, 36)` | metrics.py:12 | Downscale resolution |
| `100, 200` | metrics.py:121 | Canny thresholds |
| `64` | metrics.py:58 | Histogram bins |
| `0.4/0.4/0.2` | metrics.py:161-164 | Combined weights |
| `5` | detector.py:37 | Temporal history window |
| `0.5` | detector.py:104 | Temporal boost factor |
| `15.0-35.0` | detector.py:22 | Quick threshold range |
| `20.0-50.0` | detector.py:23 | Detailed threshold range |

**Fix:** Extract into a `DetectionConfig` dataclass, populated from `default_config.json` with CLI overrides. This unblocks tuning via the evaluation harness (item 5).

### P2 — Evaluation & Tuning

#### 5. Evaluation harness
**Files:** `tools/annotate_cuts.py`, `output/annotations/`

The annotator collects `good_cuts` and `false_positives` per video, but there's no script to consume this feedback. No annotation data exists yet.

**Plan:**
- Create `tools/evaluate.py` that loads annotation JSONs + detection JSONs
- Compute precision (good / (good + false_positives)) and recall (good / total annotated)
- Use a matching window (e.g., ±0.5s) to pair detected cuts with annotations
- Output a summary table per video and aggregate

This is the foundation for tuning all the magic numbers in item 4.

#### 6. Adaptive thresholding
**File:** `src/core/detector.py`

Current thresholds are static per run. Dark scenes, fast action, and static content all get the same thresholds.

**Approach:** Use a rolling baseline from `recent_scores` — a cut is detected when the current score deviates significantly from the local baseline, not just when it exceeds a fixed number. This naturally adapts to content characteristics.

**Dependency:** Needs evaluation harness (item 5) to validate that adaptive thresholds improve precision/recall vs. fixed.

### P3 — New Capabilities

#### 7. Dissolve/fade detection
Currently only hard cuts (abrupt frame changes) are detected. Dissolves and fades produce gradual score ramps that stay below the hard-cut threshold.

**Approach:**
- Detect sustained mid-range scores over N frames (gradual transition signature)
- Track luminance mean for fade-to-black / fade-from-black
- Add a `transition_type` field to output: `"hard_cut"`, `"dissolve"`, `"fade"`

**Dependency:** Adaptive thresholding (item 6) — dissolve detection needs the rolling baseline to distinguish gradual transitions from slow camera movement.

#### 8. Frame skipping for high-FPS content
No sampling strategy exists — every frame is processed. For 60fps+ content this is wasteful since consecutive frames are near-identical.

**Approach:**
- Skip every Nth frame based on FPS (e.g., process every 2nd frame at 60fps, every 4th at 120fps)
- When a skip detects a candidate, backtrack and check skipped frames for precise cut location
- Make skip factor configurable

---

## Proposed Order of Work

```
Phase 1 — Fix & Foundation
  ├── [P0] #1 UMat conversion bug
  ├── [P0] #2 Temporal score explosion
  ├── [P1] #3 Wire config file
  └── [P1] #4 Extract magic numbers to DetectionConfig

Phase 2 — Measure
  └── [P2] #5 Evaluation harness

Phase 3 — Tune
  └── [P2] #6 Adaptive thresholding (validated by harness)

Phase 4 — Extend
  ├── [P3] #7 Dissolve/fade detection
  └── [P3] #8 Frame skipping
```

Phase 1 items are independent and can be tackled in any order. Phase 2 depends on Phase 1 (config must be wired so the harness can test different parameter sets). Phase 3 and 4 each depend on the previous phase.

---

## Progress Log

### Phase 1 — Complete

All 4 items fixed. Additional work:
- Added `score_mode: "max"` — takes strongest individual signal instead of weighted average. Catches cuts where quick+histogram are high but edge is low.
- Added `--diagnose` CLI tool for per-frame score dumping on a time range.
- YouTube downloader updated for SABR protocol (yt-dlp breaking change).

**Baseline → Phase 1 results (97-min anime film):**
- Cuts detected: 1,499 → 30 (sensitivity 0.5, weighted mode)
- Processing time: 330s → 9s

### Phase 2 — Complete

Evaluation harness built (`tools/evaluate.py`). First annotation pass on test video (100 detections, sensitivity 0.7, max mode):

| Metric | Value |
|--------|-------|
| Precision | 24.0% |
| Recall | 100.0% |
| F1 | 38.7% |

**Key finding:** 100% recall (all real cuts found), but 76 false positives — mostly in explosion/high-motion sequences where sustained high scores mimic cuts.

### Phase 3 — In Progress

**Goal:** Adaptive thresholding to distinguish high-motion continuous shots from editorial cuts. The rolling baseline should rise during action sequences so only genuine scene changes break through.

**Constraint:** Must NOT use min_cut_distance to suppress false positives — rapid-fire editorial cuts (e.g., Hitchcock's Psycho shower scene) are legitimate and must be preserved.

## Out of Scope

- Multi-GPU / distributed processing
- Audio-based cut detection
- Real-time streaming input
- UI/UX changes to the annotator beyond what's needed for evaluation
