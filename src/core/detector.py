from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
from .metrics import FrameMetrics
from .accelerator import MetalAccelerator
from .config import DetectionConfig

class CutDetector:
    """Efficient cut detector optimized for hard cuts."""

    def __init__(self,
                 config: Optional[DetectionConfig] = None,
                 use_gpu: bool = True,
                 # Legacy kwargs for backwards compat with CLI
                 sensitivity: Optional[float] = None,
                 min_cut_distance: Optional[float] = None):
        if config is None:
            config = DetectionConfig()
        if sensitivity is not None:
            config.sensitivity = sensitivity
        if min_cut_distance is not None:
            config.min_cut_distance = min_cut_distance

        self.config = config
        self.quick_threshold = config.quick_threshold
        self.detailed_threshold = config.detailed_threshold
        self.min_cut_distance = config.min_cut_distance

        self.metrics = FrameMetrics(config)
        self.accelerator = MetalAccelerator() if use_gpu else None

        # State tracking
        self.previous_frame = None
        self.last_cut_frame = 0
        self.frame_count = 0
        self.fps = None

        # Detection history for temporal consistency
        self.recent_scores = []
        self.max_history = config.temporal_window

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

        if std > 0:
            z_score = (current - avg) / std
        else:
            z_score = 0.0 if current <= avg else 2.0

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
            if isinstance(frame, cv2.UMat):
                self.previous_frame = cv2.UMat(frame)
            else:
                self.previous_frame = frame.copy()
            return False, {'first_frame': True}

        # Score the frame pair
        quick_score, quick_metrics = self.metrics.combined_difference(
            self.previous_frame, frame
        )
        metrics_dict.update(quick_metrics)

        # Update history and get temporal score
        self._update_history(quick_score)
        temporal_score = self._get_temporal_score()
        metrics_dict['temporal_score'] = temporal_score

        # Final decision — additive boost, bounded by temporal_score in [0, 1]
        is_cut = False
        boost = self.config.temporal_boost_factor
        final_score = quick_score + (temporal_score * self.detailed_threshold * boost)
        metrics_dict['final_score'] = final_score

        if final_score > self.detailed_threshold:
            is_cut = True
            self.last_cut_frame = self.frame_count

        # Update previous frame
        if isinstance(frame, cv2.UMat):
            self.previous_frame = cv2.UMat(frame)
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
