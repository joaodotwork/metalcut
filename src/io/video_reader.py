import cv2
import numpy as np
from typing import Generator, Optional, Tuple, Dict
from pathlib import Path
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class VideoReader:
    """Efficient video frame reader with buffering."""
    
    def __init__(self, 
                 path: str, 
                 buffer_size: int = 32,
                 target_size: Optional[Tuple[int, int]] = None):
        """Initialize video reader.
        
        Args:
            path: Path to video file
            buffer_size: Size of frame buffer
            target_size: Optional target size for frames (width, height)
        """
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.target_size = target_size
        self.buffer = deque(maxlen=buffer_size)
        
        self._cap = None
        self._frame_count = 0
        self._fps = 0
        self._duration = 0
        self._initialize()
    
    def _initialize(self):
        """Initialize video capture and metadata."""
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
            
        try:
            self._cap = cv2.VideoCapture(str(self.path))
            
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.path}")
            
            # Get video metadata
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._duration = total_frames / self._fps if self._fps > 0 else 0
            
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Opened video: {self.path}")
            logger.info(f"FPS: {self._fps:.2f}")
            logger.info(f"Duration: {self._duration:.2f}s")
            logger.info(f"Size: {width}x{height}")
            
        except Exception as e:
            logger.error(f"Error initializing video reader: {e}")
            raise
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if target size is set."""
        if self.target_size is None:
            return frame
            
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def read_frames(self, batch_size: int = 1) -> Generator[np.ndarray, None, None]:
        """Read frames from video with buffering.
        
        Args:
            batch_size: Number of frames to read at once
            
        Yields:
            Video frames
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Video capture not initialized")
        
        frames_read = 0
        batch_start_time = time.time()
        
        try:
            while True:
                # Fill buffer if needed
                while len(self.buffer) < self.buffer_size:
                    ret, frame = self._cap.read()
                    if not ret:
                        break
                    
                    if frame is not None:
                        frame = self._resize_frame(frame)
                        self.buffer.append(frame)
                        self._frame_count += 1
                
                # Yield frames from buffer
                while self.buffer and frames_read < batch_size:
                    yield self.buffer.popleft()
                    frames_read += 1
                
                # Reset counters and check performance
                if frames_read >= batch_size:
                    batch_time = time.time() - batch_start_time
                    fps = batch_size / batch_time if batch_time > 0 else 0
                    
                    if fps < self._fps * 0.8:  # Performance warning
                        logger.warning(f"Frame reading running slower than video FPS: {fps:.2f} fps")
                    
                    frames_read = 0
                    batch_start_time = time.time()
                
                # Check if we've reached the end
                if not self.buffer and self._cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
        
        except Exception as e:
            logger.error(f"Error reading frames: {e}")
            raise
        
        finally:
            self._cap.release()
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame at timestamp or None if not found
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        
        try:
            # Calculate frame number from timestamp
            frame_number = int(timestamp * self._fps)
            
            # Set position and read frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self._cap.read()
            
            if ret and frame is not None:
                return self._resize_frame(frame)
            return None
            
        except Exception as e:
            logger.error(f"Error getting frame at {timestamp}s: {e}")
            return None
    
    @property
    def fps(self) -> float:
        """Get video FPS."""
        return self._fps
    
    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        return self._duration
    
    @property
    def frame_count(self) -> int:
        """Get number of frames processed."""
        return self._frame_count
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()

