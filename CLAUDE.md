# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**metalcut** — GPU-accelerated video cut detection for Apple Silicon, using OpenCV with Metal/OpenCL. Detects hard cuts, dissolves, and fades from a video file using a two-pass pipeline: every frame is scored in Pass 1, then Pass 2 runs adaptive-threshold hard-cut detection plus dissolve and fade detectors over the full score history.

## Setup & Running

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run cut detection (defaults to score_mode=max)
python -m src.cli.main --input video.mp4 --sensitivity 0.7

# Save transitions (hard cuts + dissolves + fades) to JSON
python -m src.cli.main --input video.mp4 --sensitivity 0.7 --output-json

# Dump per-frame scores for a time range
python -m src.cli.main --input video.mp4 --diagnose "16.47-21.10"

# Enable the optional Tier 1 semantic scene gate (pHash + color palette FP filter)
python -m src.cli.main --input video.mp4 --semantic

# Download test videos (reads URLs from data/urls.txt)
python tools/download_videos.py --urls data/urls.txt --output-dir data/videos

# Review detected cuts with GUI annotator (Y/N review + browse mode for missed cuts)
python tools/annotate_cuts.py

# Evaluate detection precision/recall against annotation files
python tools/evaluate.py
```

There are no tests in this project. Validation is done via the annotator + `tools/evaluate.py` against per-video annotation JSON in `output/annotations/`.

## Architecture

The project uses relative imports via `python -m src.cli.main` (no `__init__.py` files exist).

### Detection Pipeline (`src/core/`)

`CutDetector` runs a **two-pass pipeline**:

**Pass 1 — `score_frame()` (called per frame):** Each frame is optionally uploaded to GPU via `MetalAccelerator.process_frame()` (converts to `cv2.UMat`) and scored by `FrameMetrics.combined_difference()`. The combined method itself has two stages:
1. **Quick stage**: Downscales frames to `downscale_width × downscale_height` (default 64×36), computes mean absolute difference (score 0-100).
2. **Detailed stage** (only if quick > `quick_gate`, default 10.0): Adds histogram correlation and Canny edge difference. Two score modes:
   - `max` (default): `max(quick, hist, weighted)` — strongest individual signal wins, so a single weak metric can't drag down a real cut.
   - `weighted`: classic weighted average, defaulting to `0.4 quick + 0.4 hist + 0.2 edge`.

   When `use_semantic` is enabled, the same call also computes pHash and HSV color-palette descriptors for both frames, piggybacking on the GPU→CPU transfer that already happened. Per-frame mean luminance is tracked separately for the fade detector.

**Pass 2 — `detect_transitions()` (called once after scoring):** Three detectors run over the full score history:
- **Hard cuts** (`detect_cuts()`): Adaptive thresholding with symmetrical ±`lookahead_frames` lookahead. The neighborhood p75 + `adaptive_margin` sets the bar, but the margin is dampened proportionally to the neighborhood's IQR/p75 ratio — high spread (rapid-fire editorial cuts) gets less margin, low spread (sustained action) gets the full margin. If `use_semantic` is on, candidates are then filtered through a pHash hamming + palette distance gate (rejected only when *both* `palette_dist < palette_change_threshold` AND `phash_hamming < phash_hamming_threshold`).
- **Dissolves** (`_detect_dissolves()`): Sustained runs of mid-range scores (`dissolve_score_floor` ≤ score ≤ `detailed_threshold`) lasting ≥`dissolve_min_frames`. Filtered by proximity to hard cuts (post-cut settling) and bell-curve shape (peak must not be in first 25% of the run, which excludes camera pans).
- **Fades** (`_detect_fades()`): Mean-luminance trends to/from black — finds dark regions (luminance < `fade_luminance_threshold`) preceded or followed by a luminance change of ≥`fade_luminance_drop` over ≥`fade_min_frames`.

There is no temporal score boost — Phase 3 of the cut-refinement spike replaced it with the adaptive lookahead approach (see `docs/spike/cut-refinement-plan.md`).

Sensitivity (0.0-1.0) inversely scales `quick_threshold` and `detailed_threshold`: higher sensitivity = lower thresholds = more cuts detected.

### Configuration (`src/core/config.py`)

`DetectionConfig` is a single dataclass holding every tunable parameter (thresholds, weights, lookahead, dissolve/fade params, semantic params). It's loaded by `src/cli/main.py` from `config/default_config.json` via `DetectionConfig.from_config_dict()`, with CLI args (`--sensitivity`, `--score-mode`, `--semantic`, `--min-cut-distance`) taking priority over file values. `quick_threshold` and `detailed_threshold` are computed as properties from `sensitivity` and the threshold base/range fields.

### GPU Acceleration (`src/core/accelerator.py`)

Uses OpenCV's OpenCL backend which maps to Metal on macOS. Frames are converted to `cv2.UMat` for GPU-resident processing. Falls back to CPU (`np.ndarray`) automatically on failure. Both `UMat` and `ndarray` paths must be handled throughout the codebase.

### I/O (`src/io/`)

- `VideoReader`: Buffered frame reader with `deque`-based buffer, used as context manager. Releases capture on iterator exhaustion.
- `ClipWriter`: Writes detected segments as separate MP4 clips. Supports parallel writing via a background thread and `Queue`.

### Key Design Considerations

- All frame processing code must handle both `cv2.UMat` (GPU) and `np.ndarray` (CPU) types — check `isinstance` before calling `.shape` vs `.get().shape`. The detailed-stage metrics (`histogram_difference`, `edge_difference`) and the semantic features (`phash`, `color_palette`) require `np.ndarray`, so `combined_difference()` calls `.get()` once when the quick gate fires and reuses the CPU frames for everything downstream.
- `VideoReader.read_frames()` releases the capture in its `finally` block, so each reader instance supports only one iteration. Pass 1 (scoring), the optional preview pass, and the optional clip-creation pass each open a fresh reader.
- The two-pass detector stores all per-frame scores, metrics, luminance, and frame numbers in lists on the `CutDetector` instance — Pass 2 walks these in place. `reset()` clears them between runs.
- Detection is content-based and frame-difference-based. The semantic gate (`use_semantic`) is opt-in scaffolding that currently filters zero cuts on our test videos; it stays in the codebase for future validation when matching test content is available (see `docs/spike/semantic-scene-spike.md`).
- Output goes to `output/json/` (transitions, including hard cuts + dissolves + fades) and `output/annotations/` (annotator review feedback used by `tools/evaluate.py`).
