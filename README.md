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
flowchart TD
    Video[Video File] --> Reader["VideoReader
    buffered I/O"]
    Reader --> |frame| Accel{MetalAccelerator}
    Accel --> |"cv2.UMat
    GPU"| Detect[CutDetector]
    Accel --> |"ndarray
    CPU fallback"| Detect

    subgraph Detection Pipeline
        Detect --> Quick["Quick Difference
        64×36 downscale"]
        Quick --> |"score > 10"| Detailed["Histogram +
        Edge Analysis"]
        Quick --> |"score ≤ 10"| Skip[Skip]
        Detailed --> Temporal["Temporal Scoring
        trend + variance boost"]
        Skip --> Temporal
    end

    Temporal --> |cut detected| Cuts[Cut Timestamps]
    Cuts --> JSON[JSON Output]
    Cuts --> Writer["ClipWriter
    threaded"]
    Writer --> Clips[Video Clips]
```

## Usage

```bash
python -m src.cli.main --input video.mp4 --sensitivity 0.5
```

