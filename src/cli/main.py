import argparse
import logging
import time
from pathlib import Path
from typing import Optional, List
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.logging import RichHandler
import sys
import cv2
import json

from ..core.config import DetectionConfig
from ..core.detector import CutDetector
from ..core.accelerator import MetalAccelerator
from ..io.video_reader import VideoReader
from ..io.video_writer import ClipWriter
from ..utils.visualization import create_preview

logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False):
    """Configure logging with rich formatting."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def diagnose_range(input_path: str, time_range: str, config: DetectionConfig):
    """Dump per-frame scores for a time range using two-pass lookahead.

    Scores the entire video first so that the adaptive threshold at each
    frame reflects full lookahead context (frames before AND after).

    Args:
        input_path: Path to video file.
        time_range: Time range string, e.g. '16.47-21.10'.
        config: Detection config.
    """
    start_str, end_str = time_range.split("-")
    start_time, end_time = float(start_str), float(end_str)

    accelerator = MetalAccelerator()
    detector = CutDetector(config=config, use_gpu=accelerator.metal_available)

    # Pass 1: Score all frames
    with VideoReader(input_path) as reader:
        detector.set_video_params(reader.fps)
        total_frames = int(reader.duration * reader.fps)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Scoring frames...", total=total_frames)
            for frame in reader.read_frames():
                detector.score_frame(frame, frame_number=reader.frame_count)
                progress.update(task, advance=1)

    # Pass 2: Get diagnostics with lookahead thresholds
    diagnostics = detector.get_diagnostics()

    print(f"\n{'time':>8s}  {'frame':>6s}  {'quick':>6s}  {'hist':>6s}  {'edge':>6s}  {'score':>7s}  {'nbhood':>7s}  {'thresh':>7s}  {'cut':>3s}")
    print("-" * 80)

    for row in diagnostics:
        if row['time'] < start_time:
            continue
        if row['time'] > end_time:
            break

        m = row['metrics']
        quick = m.get('quick_score', 0)
        hist = m.get('histogram_score', 0)
        edge = m.get('edge_score', 0)
        marker = "<<<" if row['is_cut'] else ""

        # Show neighborhood p75 for frames above base threshold
        if row['score'] > config.detailed_threshold:
            nbhood = row['threshold'] - config.adaptive_margin
            nbhood_str = f"{nbhood:7.1f}"
        else:
            nbhood_str = "      -"

        print(f"{row['time']:8.3f}  {row['frame']:6d}  {quick:6.1f}  {hist:6.1f}  {edge:6.1f}  {row['score']:7.1f}  {nbhood_str}  {row['threshold']:7.1f}  {marker}")

    print()


def process_video(input_path: str,
                 output_dir: str,
                 config: Optional[DetectionConfig] = None,
                 sensitivity: float = 0.5,
                 min_cut_distance: float = 0.5,
                 preview: bool = False,
                 create_clips: bool = False,
                 debug: bool = False) -> tuple:
    """Process video with two-pass detection.

    Pass 1: Score every frame (fast — GPU-accelerated).
    Pass 2: Detect all transitions (hard cuts, dissolves, fades).
    Pass 3 (optional): Re-read video to create clips at known cut points.

    Returns:
        (cuts, transitions) — cuts is a list of timestamps (floats) for
        backwards compat; transitions is the full list of transition dicts.
    """
    # Initialize components
    accelerator = MetalAccelerator()
    if config is None:
        config = DetectionConfig(sensitivity=sensitivity, min_cut_distance=min_cut_distance)
    detector = CutDetector(
        config=config,
        use_gpu=accelerator.metal_available
    )

    # Pass 1: Score all frames
    with VideoReader(input_path) as reader:
        detector.set_video_params(reader.fps)
        total_frames = int(reader.duration * reader.fps)
        fps = reader.fps
        duration = reader.duration

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Scoring frames...", total=total_frames)
            for frame in reader.read_frames():
                detector.score_frame(frame, frame_number=reader.frame_count)
                progress.update(task, advance=1)

    # Pass 2: Detect all transitions
    transitions = detector.detect_transitions()

    # Log results by type
    type_counts = {}
    for t in transitions:
        ttype = t['type']
        type_counts[ttype] = type_counts.get(ttype, 0) + 1
        if ttype == 'hard_cut':
            logger.info(f"Hard cut at {t['timestamp']:.2f}s")
        elif ttype == 'dissolve':
            logger.info(f"Dissolve at {t['timestamp']:.2f}s (duration: {t['duration']:.2f}s, avg_score: {t['avg_score']:.1f})")
        elif ttype in ('fade_to_black', 'fade_from_black'):
            logger.info(f"{ttype.replace('_', ' ').title()} at {t['timestamp']:.2f}s (duration: {t['duration']:.2f}s)")

    # Extract hard cut timestamps for backwards compat
    cuts = [t['timestamp'] for t in transitions if t['type'] == 'hard_cut']

    # Pass 3 (optional): Create clips at known cut points (hard cuts only)
    if create_clips:
        cut_frame_set = set(t['frame'] for t in transitions if t['type'] == 'hard_cut')

        with VideoReader(input_path) as reader:
            writer = ClipWriter(output_dir, fps=fps, parallel=True)
            current_clip_frames = []
            clip_start_time = 0.0

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Creating clips...", total=total_frames)

                for frame in reader.read_frames():
                    frame_num = reader.frame_count
                    current_clip_frames.append(frame.copy())

                    if frame_num in cut_frame_set:
                        cut_time = frame_num / fps
                        if current_clip_frames:
                            writer.create_clip_from_frames(
                                current_clip_frames,
                                clip_start_time,
                                cut_time,
                                fps
                            )
                        clip_start_time = cut_time
                        current_clip_frames = []

                    progress.update(task, advance=1)

                # Handle last clip
                if current_clip_frames:
                    writer.create_clip_from_frames(
                        current_clip_frames,
                        clip_start_time,
                        duration,
                        fps
                    )

    return cuts, transitions

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config file. Falls back to config/default_config.json.

    Returns:
        Configuration dict, or empty dict if no config found.
    """
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).parent.parent.parent / "config" / "default_config.json"

    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="metalcut - GPU-accelerated video cut detection for Apple Silicon")

    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-dir", "-o", default="output",
                       help="Output directory for clips")
    parser.add_argument("--config", default=None,
                       help="Path to config JSON (default: config/default_config.json)")
    parser.add_argument("--sensitivity", "-s", type=float, default=None,
                       help="Detection sensitivity (0.0-1.0), overrides config")
    parser.add_argument("--min-cut-distance", "-d", type=float, default=None,
                       help="Minimum distance between cuts (seconds), overrides config")
    parser.add_argument("--preview", "-p", action="store_true",
                       help="Show preview window")
    parser.add_argument("--create-clips", "-c", action="store_true",
                       help="Create video clips")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--output-json", action="store_true",
                       help="Save cuts data to JSON file")
    parser.add_argument("--score-mode", choices=["weighted", "max"], default=None,
                       help="Score mode: 'weighted' (default) or 'max' (best-of signals)")
    parser.add_argument("--diagnose", type=str, default=None,
                       help="Dump per-frame scores for a time range, e.g. '16.47-21.10'")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    # Build DetectionConfig from file + CLI overrides
    raw_config = load_config(args.config)
    detection_config = DetectionConfig.from_config_dict(
        raw_config,
        sensitivity=args.sensitivity,
        min_cut_distance=args.min_cut_distance,
        score_mode=args.score_mode,
    )

    if args.debug:
        logger.debug(f"Raw config: {raw_config}")
        logger.debug(f"DetectionConfig: {detection_config}")

    try:
        # Validate input
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input video not found: {input_path}")
            return 1

        # Diagnose mode — dump scores for a time range and exit
        if args.diagnose:
            diagnose_range(str(input_path), args.diagnose, detection_config)
            return 0

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process video
        start_time = time.time()
        
        cuts, transitions = process_video(
            str(input_path),
            str(output_dir),
            config=detection_config,
            preview=args.preview,
            create_clips=args.create_clips,
            debug=args.debug
        )

        processing_time = time.time() - start_time

        # Generate JSON output if requested
        if args.output_json:
            # Build clean transition list for JSON (remove internal fields)
            json_transitions = []
            for t in transitions:
                entry = {'timestamp': t['timestamp'], 'type': t['type']}
                if 'duration' in t:
                    entry['duration'] = t['duration']
                if 'avg_score' in t:
                    entry['avg_score'] = t['avg_score']
                json_transitions.append(entry)

            json_output = {
                "video_path": str(input_path),
                "parameters": {
                    "sensitivity": detection_config.sensitivity,
                    "min_cut_distance": detection_config.min_cut_distance,
                    "quick_threshold": detection_config.quick_threshold,
                    "detailed_threshold": detection_config.detailed_threshold,
                },
                "processing_time": processing_time,
                "cuts": cuts,
                "transitions": json_transitions,
                "metadata": {
                    "total_cuts": len(cuts),
                    "total_transitions": len(transitions),
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            # Create output directory if it doesn't exist
            json_dir = output_dir / "json"
            json_dir.mkdir(parents=True, exist_ok=True)

            # Generate JSON filename
            json_path = json_dir / f"cuts_{input_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"

            # Save JSON file
            try:
                with open(json_path, 'w') as f:
                    json.dump(json_output, f, indent=2)
                logger.info(f"Cuts data saved to: {json_path}")
            except Exception as e:
                logger.error(f"Error saving JSON file: {e}")

        # Print summary
        type_counts = {}
        for t in transitions:
            type_counts[t['type']] = type_counts.get(t['type'], 0) + 1

        logger.info("\nProcessing Summary:")
        logger.info(f"Input video: {input_path}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Transitions detected: {len(transitions)}")
        for ttype, count in sorted(type_counts.items()):
            logger.info(f"  {ttype}: {count}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        logger.error("Error processing video")
        logger.debug("Detailed error:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
