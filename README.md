# metalcut

GPU-accelerated video cut detection for Apple Silicon.

## Installation

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Architecture

```mermaid
flowchart LR
    Video[Video File] --> Reader[VideoReader\nbuffered I/O]
    Reader --> |frame| Accel{MetalAccelerator}
    Accel --> |cv2.UMat\nGPU| Detect[CutDetector]
    Accel --> |ndarray\nCPU fallback| Detect

    subgraph Detection Pipeline
        Detect --> Quick[Quick Difference\n64×36 downscale]
        Quick --> |score > 10| Detailed[Histogram +\nEdge Analysis]
        Quick --> |score ≤ 10| Skip[Skip]
        Detailed --> Temporal[Temporal Scoring\ntrend + variance boost]
        Skip --> Temporal
    end

    Temporal --> |cut detected| Cuts[Cut Timestamps]
    Cuts --> JSON[JSON Output]
    Cuts --> Writer[ClipWriter\nthreaded]
    Writer --> Clips[Video Clips]
```

## Usage

```bash
python -m src.cli.main --input video.mp4 --sensitivity 0.5
```

