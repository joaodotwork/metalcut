from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
from .metrics import FrameMetrics
from .accelerator import MetalAccelerator
from .config import DetectionConfig

class CutDetector:
    """Two-pass cut detector: score all frames, then decide with lookahead.

    Pass 1: Call score_frame() for every frame — computes per-frame scores.
    Pass 2: Call detect_cuts() — uses symmetrical lookahead context to
             distinguish real cuts from high-motion sequences.
    """

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
        self.detailed_threshold = config.detailed_threshold
        self.min_cut_distance = config.min_cut_distance

        self.metrics = FrameMetrics(config)
        self.accelerator = MetalAccelerator() if use_gpu else None

        # State tracking
        self.previous_frame = None
        self.frame_count = 0
        self.fps = None

        # Two-pass storage
        self.all_scores = []
        self.all_metrics = []
        self.frame_numbers = []  # external frame numbers for timestamp calculation

    def score_frame(self, frame: Union[np.ndarray, cv2.UMat],
                    frame_number: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        """Score a single frame against its predecessor.

        Call this for every frame in order. Scores are stored internally
        for the subsequent detect_cuts() pass.

        Args:
            frame: Video frame (ndarray or UMat).
            frame_number: External frame number (e.g. from VideoReader.frame_count)
                          used for timestamp calculation. If None, uses internal counter.
        """
        self.frame_count += 1
        fn = frame_number if frame_number is not None else self.frame_count

        # Process frame through accelerator if available
        if self.accelerator and self.accelerator.metal_available:
            frame = self.accelerator.process_frame(frame)

        # First frame
        if self.previous_frame is None:
            if isinstance(frame, cv2.UMat):
                self.previous_frame = cv2.UMat(frame)
            else:
                self.previous_frame = frame.copy()
            self.all_scores.append(0.0)
            self.all_metrics.append({'first_frame': True})
            self.frame_numbers.append(fn)
            return 0.0, {'first_frame': True}

        # Score the frame pair
        score, metrics = self.metrics.combined_difference(
            self.previous_frame, frame
        )
        self.all_scores.append(score)
        self.all_metrics.append(metrics)
        self.frame_numbers.append(fn)

        # Update previous frame
        if isinstance(frame, cv2.UMat):
            self.previous_frame = cv2.UMat(frame)
        else:
            self.previous_frame = frame.copy()

        return score, metrics

    def _neighborhood_threshold(self, idx: int) -> float:
        """Compute adaptive threshold for frame idx using symmetrical lookahead.

        Looks at N frames before AND after the candidate. If the neighborhood
        is "hot" (many high scores), the threshold rises — only genuine scene
        changes that tower above the action will fire. If the neighborhood is
        calm, the base threshold governs and isolated spikes pass through.

        Uses IQR-based damping: when the neighborhood has high spread (bimodal
        distribution of high spikes + near-zero frames), the margin is reduced.
        This preserves rapid-fire editorial cuts where high scores are
        interspersed with calm frames, while still suppressing sustained
        high-motion sequences where scores are uniformly elevated.
        """
        scores = self.all_scores
        n = len(scores)
        lookahead = self.config.lookahead_frames

        window_start = max(0, idx - lookahead)
        window_end = min(n, idx + lookahead + 1)
        neighborhood = scores[window_start:idx] + scores[idx + 1:window_end]

        if len(neighborhood) < 3:
            return self.detailed_threshold

        p75 = float(np.percentile(neighborhood, self.config.adaptive_percentile))
        p25 = float(np.percentile(neighborhood, 25))
        iqr = p75 - p25

        # Dampen margin when neighborhood has high relative spread.
        # High spread (IQR/p75 → 1.0) = bimodal = rapid-fire cuts → less margin
        # Low spread (IQR/p75 → 0.0) = uniform elevation = action → full margin
        if p75 > 0:
            spread = min(1.0, iqr / p75)
        else:
            spread = 0.0
        damping = max(0.0, 1.0 - spread)

        adaptive = p75 + self.config.adaptive_margin * damping
        return max(self.detailed_threshold, adaptive)

    def detect_cuts(self) -> List[Tuple[int, float]]:
        """Determine cut points from stored scores using lookahead context.

        Must be called after scoring all frames with score_frame().

        Returns:
            List of (frame_number, timestamp) tuples for detected cuts.
        """
        if self.fps is None:
            raise ValueError("fps not set — call set_video_params() first")

        scores = self.all_scores
        min_frames = int(self.fps * self.min_cut_distance)
        base_threshold = self.detailed_threshold

        cuts = []
        last_cut_frame = -min_frames  # allow first frame

        for i, score in enumerate(scores):
            if score <= base_threshold:
                continue

            if (i - last_cut_frame) < min_frames:
                continue

            effective_threshold = self._neighborhood_threshold(i)

            if score > effective_threshold:
                timestamp = self.frame_numbers[i] / self.fps
                cuts.append((i, timestamp))
                last_cut_frame = i

        return cuts

    def get_diagnostics(self) -> List[Dict]:
        """Return per-frame diagnostics including lookahead thresholds.

        For use by the diagnose CLI tool after both passes are complete.
        """
        if self.fps is None:
            raise ValueError("fps not set — call set_video_params() first")

        rows = []
        base_threshold = self.detailed_threshold

        # First compute all cut decisions (need last_cut_frame tracking)
        min_frames = int(self.fps * self.min_cut_distance)
        cut_frames = set()
        last_cut_frame = -min_frames

        for i, score in enumerate(self.all_scores):
            if score <= base_threshold:
                continue
            if (i - last_cut_frame) < min_frames:
                continue
            thresh = self._neighborhood_threshold(i)
            if score > thresh:
                cut_frames.add(i)
                last_cut_frame = i

        # Build diagnostic rows
        for i, (score, metrics, fn) in enumerate(zip(self.all_scores, self.all_metrics, self.frame_numbers)):
            thresh = self._neighborhood_threshold(i) if score > base_threshold else base_threshold
            rows.append({
                'frame': fn,
                'time': fn / self.fps,
                'score': score,
                'metrics': metrics,
                'threshold': thresh,
                'is_cut': i in cut_frames,
            })

        return rows

    def set_video_params(self, fps: float):
        """Set video parameters for temporal analysis."""
        self.fps = fps

    def reset(self):
        """Reset detector state."""
        self.previous_frame = None
        self.frame_count = 0
        self.all_scores.clear()
        self.all_metrics.clear()
        self.frame_numbers.clear()

    @property
    def current_frame(self) -> int:
        """Get current frame number."""
        return self.frame_count
