import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

logger = logging.getLogger(__name__)

class ClipWriter:
    """Efficient video clip writer with parallel processing."""
    
    def __init__(self, 
                 output_dir: str,
                 fps: Optional[float] = None,
                 codec: str = 'mp4v',
                 parallel: bool = True,
                 max_workers: int = 2):
        """Initialize clip writer.
        
        Args:
            output_dir: Directory for output clips
            fps: Frames per second (if None, uses source FPS)
            codec: Video codec (default: mp4v)
            parallel: Enable parallel processing
            max_workers: Number of parallel workers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.parallel = parallel
        self.max_workers = max_workers
        
        self._frame_buffer = Queue(maxsize=100)
        self._current_writer = None
        self._current_path = None
        self._writing_thread = None
        self._stop_flag = threading.Event()
        
        logger.info(f"Initialized ClipWriter: {output_dir}")
        logger.info(f"Parallel processing: {parallel} ({max_workers} workers)")
    
    def _get_output_path(self, start_time: float, end_time: float) -> Path:
        """Generate output path for clip."""
        filename = f"clip_{start_time:.2f}-{end_time:.2f}.mp4"
        return self.output_dir / filename
    
    def _init_writer(self, frame_size: Tuple[int, int], fps: float, path: Path):
        """Initialize video writer."""
        self._current_writer = cv2.VideoWriter(
            str(path),
            self.codec,
            fps,
            frame_size,
            True  # isColor
        )
        
        if not self._current_writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {path}")
            
        self._current_path = path
    
    def _write_frame_worker(self):
        """Worker thread for parallel frame writing."""
        while not self._stop_flag.is_set() or not self._frame_buffer.empty():
            try:
                frame = self._frame_buffer.get(timeout=1.0)
                if frame is not None and self._current_writer is not None:
                    self._current_writer.write(frame)
                self._frame_buffer.task_done()
            except:
                continue
    
    def start_clip(self, 
                   frame_size: Tuple[int, int],
                   start_time: float,
                   end_time: float,
                   source_fps: float):
        """Start writing a new clip.
        
        Args:
            frame_size: Size of video frames (width, height)
            start_time: Start time in seconds
            end_time: End time in seconds
            source_fps: Source video FPS
        """
        # Close any existing writer
        self.finish_clip()
        
        # Get output path
        path = self._get_output_path(start_time, end_time)
        
        # Initialize writer
        self._init_writer(frame_size, self.fps or source_fps, path)
        
        if self.parallel:
            self._stop_flag.clear()
            self._writing_thread = threading.Thread(target=self._write_frame_worker)
            self._writing_thread.start()
        
        logger.info(f"Started clip: {path}")
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to current clip.
        
        Args:
            frame: Video frame to write
        """
        if self._current_writer is None:
            raise RuntimeError("No active clip writer")
        
        if self.parallel:
            # Add frame to buffer for parallel processing
            try:
                self._frame_buffer.put(frame.copy(), timeout=1.0)
            except:
                logger.warning("Frame buffer full, dropping frame")
        else:
            # Direct writing
            self._current_writer.write(frame)
    
    def finish_clip(self):
        """Finish writing current clip."""
        if self._current_writer is not None:
            if self.parallel:
                # Wait for buffer to empty
                self._stop_flag.set()
                if self._writing_thread is not None:
                    self._writing_thread.join()
                
                # Clear buffer
                while not self._frame_buffer.empty():
                    try:
                        frame = self._frame_buffer.get_nowait()
                        if frame is not None:
                            self._current_writer.write(frame)
                    except:
                        continue
            
            # Release writer
            self._current_writer.release()
            self._current_writer = None
            
            logger.info(f"Finished clip: {self._current_path}")
            self._current_path = None
    
    def create_clip_from_frames(self, 
                              frames: List[np.ndarray],
                              start_time: float,
                              end_time: float,
                              fps: float):
        """Create clip from list of frames.
        
        Args:
            frames: List of frames
            start_time: Start time in seconds
            end_time: End time in seconds
            fps: Frames per second
        """
        if not frames:
            return
        
        frame_size = (frames[0].shape[1], frames[0].shape[0])
        self.start_clip(frame_size, start_time, end_time, fps)
        
        for frame in frames:
            self.write_frame(frame)
        
        self.finish_clip()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish_clip()

