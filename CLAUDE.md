# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**metalcut** — GPU-accelerated video cut detection for Apple Silicon, using OpenCV with Metal/OpenCL. Processes video frames through a two-stage detection pipeline: a fast downscaled difference check, followed by histogram and edge analysis for candidates above the quick threshold.

## Setup & Running

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run cut detection
python -m src.cli.main --input video.mp4 --sensitivity 0.5

# Download test videos (reads URLs from data/urls.txt)
python tools/download_videos.py --urls data/urls.txt --output-dir data/videos

# Review detected cuts with GUI annotator (uses tkinter file dialogs)
python tools/annotate_cuts.py
```

There are no tests in this project.

## Architecture

The project uses relative imports via `python -m src.cli.main` (no `__init__.py` files exist).

### Detection Pipeline (`src/core/`)

`CutDetector` orchestrates detection: each frame is optionally uploaded to GPU via `MetalAccelerator.process_frame()` (converts to `cv2.UMat`), then scored by `FrameMetrics.combined_difference()`. The combined method uses a two-stage approach:
1. **Quick stage**: Downscales frames to 64x36, computes mean absolute difference (score 0-100)
2. **Detailed stage** (only if quick > 10.0): Adds histogram correlation and Canny edge difference, weighted 0.4/0.4/0.2

`CutDetector` adds temporal scoring on top — it tracks recent scores and boosts the final score based on gradient/variance trends. A cut fires when the boosted score exceeds `detailed_threshold`.

Sensitivity (0.0-1.0) inversely scales thresholds: higher sensitivity = lower thresholds = more cuts detected.

### GPU Acceleration (`src/core/accelerator.py`)

Uses OpenCV's OpenCL backend which maps to Metal on macOS. Frames are converted to `cv2.UMat` for GPU-resident processing. Falls back to CPU (`np.ndarray`) automatically on failure. Both `UMat` and `ndarray` paths must be handled throughout the codebase.

### I/O (`src/io/`)

- `VideoReader`: Buffered frame reader with `deque`-based buffer, used as context manager. Releases capture on iterator exhaustion.
- `ClipWriter`: Writes detected segments as separate MP4 clips. Supports parallel writing via a background thread and `Queue`.

### Key Design Considerations

- All frame processing code must handle both `cv2.UMat` (GPU) and `np.ndarray` (CPU) types — check `isinstance` before calling `.shape` vs `.get().shape`
- `VideoReader.read_frames()` releases the capture in its `finally` block, so each reader instance supports only one iteration
- Config in `config/default_config.json` is not currently loaded by the CLI — thresholds are derived from the `--sensitivity` flag
- Output goes to `output/json/` (cut timestamps) and `output/annotations/` (review feedback)
