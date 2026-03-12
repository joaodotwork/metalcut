from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
from .metrics import FrameMetrics
from .accelerator import MetalAccelerator

class CutDetector:
    """Efficient cut detector optimized for hard cuts."""
    
    def __init__(self, 
                 sensitivity: float = 0.5,
                 min_cut_distance: float = 0.5,
                 use_gpu: bool = True):
        """Initialize cut detector.
        
        Args:
            sensitivity: Detection sensitivity (0.0-1.0)
            min_cut_distance: Minimum distance between cuts in seconds
            use_gpu: Whether to use GPU acceleration when available
        """
        # Scale thresholds based on sensitivity
        self.quick_threshold = 15.0 + (1.0 - sensitivity) * 20.0  # 15-35 range
        self.detailed_threshold = 20.0 + (1.0 - sensitivity) * 30.0  # 20-50 range
        
        self.min_cut_distance = min_cut_distance
        self.metrics = FrameMetrics()
        self.accelerator = MetalAccelerator() if use_gpu else None
        
        # State tracking
        self.previous_frame = None
        self.last_cut_frame = 0
        self.frame_count = 0
        self.fps = None
        
        # Detection history for temporal consistency
        self.recent_scores = []
        self.max_history = 5
        
    def _should_check_cut(self, frame_number: int) -> bool:
        """Check if enough frames have passed since last cut."""
        if self.fps is None or self.last_cut_frame == 0:
            return True
            
        frames_since_last = frame_number - self.last_cut_frame
        min_frames = int(self.fps * self.min_cut_distance)
        
        return frames_since_last >= min_frames
        
    def _update_history(self, score: float):
        """Update detection history."""
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.max_history:
            self.recent_scores.pop(0)
            
    def _get_temporal_score(self) -> float:
        """Calculate temporal-aware score based on recent history.

        Returns a bounded boost factor (0.0 to 1.0) based on whether the
        current score represents a sudden jump relative to recent history.
        """
        if len(self.recent_scores) < 2:
            return 0.0

        current = self.recent_scores[-1]
        avg = np.mean(self.recent_scores[:-1])
        std = np.std(self.recent_scores[:-1])

        # How many standard deviations above the recent average?
        # High deviation = likely a real cut, not sustained motion.
        if std > 0:
            z_score = (current - avg) / std
        else:
            z_score = 0.0 if current <= avg else 2.0

        # Clamp to [0, 1] — acts as a bounded boost factor
        temporal_boost = min(1.0, max(0.0, z_score / 3.0))

        return temporal_boost
        
    def detect_cut(self, frame: Union[np.ndarray, cv2.UMat]) -> Tuple[bool, Dict[str, float]]:
        """Detect if current frame is a cut point."""
        self.frame_count += 1
        metrics_dict = {}
        
        # Process frame through accelerator if available
        if self.accelerator and self.accelerator.metal_available:
            frame = self.accelerator.process_frame(frame)
        
        # First frame handling
        if self.previous_frame is None:
            # Handle UMat or ndarray appropriately
            if isinstance(frame, cv2.UMat):
                self.previous_frame = cv2.UMat(frame)  # Create new UMat from existing one
            else:
                self.previous_frame = frame.copy()
            return False, {'first_frame': True}
        
        # Quick check first
        quick_score, quick_metrics = FrameMetrics.combined_difference(
            self.previous_frame, frame
        )
        metrics_dict.update(quick_metrics)
        
        # Update history and get temporal score
        self._update_history(quick_score)
        temporal_score = self._get_temporal_score()
        metrics_dict['temporal_score'] = temporal_score
        
        # Final decision — additive boost, bounded by temporal_score in [0, 1]
        is_cut = False
        final_score = quick_score + (temporal_score * self.detailed_threshold * 0.25)
        metrics_dict['final_score'] = final_score
        
        if final_score > self.detailed_threshold:
            is_cut = True
            self.last_cut_frame = self.frame_count
        
        # Update previous frame
        if isinstance(frame, cv2.UMat):
            self.previous_frame = cv2.UMat(frame)  # Create new UMat from existing one
        else:
            self.previous_frame = frame.copy()
        
        return is_cut, metrics_dict

    def set_video_params(self, fps: float):
        """Set video parameters for temporal analysis."""
        self.fps = fps
        
    def reset(self):
        """Reset detector state."""
        self.previous_frame = None
        self.last_cut_frame = 0
        self.frame_count = 0
        self.recent_scores.clear()
        
    @property
    def current_frame(self) -> int:
        """Get current frame number."""
        return self.frame_count
        
    @property
    def last_cut_time(self) -> float:
        """Get time of last cut in seconds."""
        if self.fps is None:
            return 0.0
        return self.last_cut_frame / self.fps

