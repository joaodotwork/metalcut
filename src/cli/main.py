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

def process_video(input_path: str,
                 output_dir: str,
                 sensitivity: float = 0.5,
                 min_cut_distance: float = 0.5,
                 preview: bool = False,
                 create_clips: bool = False,
                 debug: bool = False) -> List[float]:
    """Process video and detect cuts."""
    # Initialize components
    accelerator = MetalAccelerator()
    detector = CutDetector(
        sensitivity=sensitivity,
        min_cut_distance=min_cut_distance,
        use_gpu=accelerator.metal_available
    )
    
    cuts = []
    frames_since_cut = 0
    current_clip_frames = []
    
    try:
        # Initialize video reader
        with VideoReader(input_path) as reader:
            detector.set_video_params(reader.fps)
            
            # Initialize clip writer if needed
            writer = None
            if create_clips:
                writer = ClipWriter(
                    output_dir,
                    fps=reader.fps,
                    parallel=True
                )
            
            # Setup progress bar
            total_frames = int(reader.duration * reader.fps)
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Processing video...", total=total_frames)
                
                # Process frames
                for frame in reader.read_frames():
                    # Detect cut
                    is_cut, metrics = detector.detect_cut(frame)
                    
                    if debug and metrics:
                        logger.debug(f"Frame {reader.frame_count}: {metrics}")
                    
                    # Handle cut detection
                    if is_cut:
                        cut_time = reader.frame_count / reader.fps
                        cuts.append(cut_time)
                        logger.info(f"Cut detected at {cut_time:.2f}s")
                        
                        # Create clip if needed
                        if create_clips and current_clip_frames:
                            clip_start = (reader.frame_count - len(current_clip_frames)) / reader.fps
                            writer.create_clip_from_frames(
                                current_clip_frames,
                                clip_start,
                                cut_time,
                                reader.fps
                            )
                            current_clip_frames = []
                    
                    # Store frame for clip creation
                    if create_clips:
                        current_clip_frames.append(frame.copy())
                    
                    # Show preview if enabled
                    if preview:
                        preview_frame = create_preview(frame, metrics)
                        cv2.imshow('Preview', preview_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Update progress
                    progress.update(task, advance=1)
                
                # Handle last clip
                if create_clips and current_clip_frames:
                    clip_start = (reader.frame_count - len(current_clip_frames)) / reader.fps
                    writer.create_clip_from_frames(
                        current_clip_frames,
                        clip_start,
                        reader.duration,
                        reader.fps
                    )
    
    finally:
        if preview:
            cv2.destroyAllWindows()
    
    return cuts

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

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    # Load config, then let CLI args override
    config = load_config(args.config)
    detection_cfg = config.get("detection", {})

    sensitivity = args.sensitivity if args.sensitivity is not None else detection_cfg.get("sensitivity", 0.5)
    min_cut_distance = args.min_cut_distance if args.min_cut_distance is not None else detection_cfg.get("min_cut_distance", 0.5)

    if args.debug:
        logger.debug(f"Config: {config}")
        logger.debug(f"Effective sensitivity={sensitivity}, min_cut_distance={min_cut_distance}")

    try:
        # Validate input
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input video not found: {input_path}")
            return 1
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process video
        start_time = time.time()
        
        cuts = process_video(
            str(input_path),
            str(output_dir),
            sensitivity=sensitivity,
            min_cut_distance=min_cut_distance,
            preview=args.preview,
            create_clips=args.create_clips,
            debug=args.debug
        )
        
        processing_time = time.time() - start_time
        
        # Generate JSON output if requested
        if args.output_json:
            json_output = {
                "video_path": str(input_path),
                "parameters": {
                    "sensitivity": sensitivity,
                    "min_cut_distance": min_cut_distance
                },
                "processing_time": processing_time,
                "cuts": cuts,
                "metadata": {
                    "total_cuts": len(cuts),
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
        logger.info("\nProcessing Summary:")
        logger.info(f"Input video: {input_path}")
        logger.info(f"Cuts detected: {len(cuts)}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        
        if cuts:
            logger.info("\nCut timestamps:")
            for i, cut in enumerate(cuts, 1):
                logger.info(f"  {i}. {cut:.2f}s")
        
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
