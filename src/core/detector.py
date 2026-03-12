from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
from .metrics import FrameMetrics
from .accelerator import MetalAccelerator
from .config import DetectionConfig

class CutDetector:
    """Two-pass cut detector with dissolve/fade detection.

    Pass 1: Call score_frame() for every frame — computes per-frame scores
             and tracks luminance.
    Pass 2: Call detect_transitions() — finds hard cuts (with lookahead),
             dissolves (sustained mid-range scores), and fades (luminance
             trending to/from black).
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
        self.all_luminance = []
        self.frame_numbers = []  # external frame numbers for timestamp calculation

    def score_frame(self, frame: Union[np.ndarray, cv2.UMat],
                    frame_number: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        """Score a single frame against its predecessor.

        Call this for every frame in order. Scores and luminance are stored
        internally for the subsequent detect_transitions() pass.

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

        # Compute luminance (works for both UMat and ndarray)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminance = float(cv2.mean(gray)[0])
        self.all_luminance.append(luminance)

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

        if p75 > 0:
            spread = min(1.0, iqr / p75)
        else:
            spread = 0.0
        damping = max(0.0, 1.0 - spread)

        adaptive = p75 + self.config.adaptive_margin * damping
        return max(self.detailed_threshold, adaptive)

    def detect_cuts(self) -> List[Tuple[int, float]]:
        """Determine hard cut points from stored scores using lookahead context.

        Must be called after scoring all frames with score_frame().

        Returns:
            List of (frame_index, timestamp) tuples for detected hard cuts.
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

    def _detect_dissolves(self, hard_cut_indices: set) -> List[Dict]:
        """Find dissolve transitions — sustained mid-range score runs.

        A dissolve blends two shots over multiple frames, producing a run of
        elevated difference scores that stay below the hard-cut threshold.

        Filters out:
        - Runs adjacent to hard cuts (post-cut settling)
        - Runs without bell-curve shape (pans produce flat plateaus,
          post-cut settling produces monotonic declines)
        """
        scores = self.all_scores
        n = len(scores)
        floor = self.config.dissolve_score_floor
        ceiling = self.detailed_threshold
        min_len = self.config.dissolve_min_frames
        proximity = min_len  # frames of exclusion zone around hard cuts

        dissolves = []
        i = 0
        while i < n:
            if scores[i] >= floor:
                # Start of a potential dissolve run
                run_start = i
                hit_ceiling = False
                while i < n and scores[i] >= floor:
                    if scores[i] > ceiling:
                        hit_ceiling = True
                        break
                    i += 1

                if hit_ceiling:
                    # This run contains a hard-cut-level spike — not a dissolve
                    i += 1
                    continue

                run_len = i - run_start
                if run_len >= min_len:
                    # Filter 1: Skip runs adjacent to hard cuts (post-cut settling)
                    near_cut = any(
                        abs(ci - run_start) <= proximity or abs(ci - (i - 1)) <= proximity
                        for ci in hard_cut_indices
                    )
                    if near_cut:
                        continue

                    # Filter 2: Require bell-curve shape — peak not in first 25%
                    # (settling decays from the start; dissolves ramp up first)
                    run_scores = scores[run_start:i]
                    peak_offset = int(np.argmax(run_scores))
                    if peak_offset < run_len * 0.25:
                        continue

                    midpoint = (run_start + i) // 2
                    dissolves.append({
                        'frame_index': midpoint,
                        'frame': self.frame_numbers[midpoint],
                        'timestamp': self.frame_numbers[midpoint] / self.fps,
                        'type': 'dissolve',
                        'duration': (self.frame_numbers[i - 1] - self.frame_numbers[run_start]) / self.fps,
                        'avg_score': float(np.mean(scores[run_start:i])),
                        'frame_start': self.frame_numbers[run_start],
                        'frame_end': self.frame_numbers[i - 1],
                    })
            else:
                i += 1

        return dissolves

    def _detect_fades(self) -> List[Dict]:
        """Find fade transitions (to/from black) based on luminance trends.

        Fade-to-black: luminance drops significantly over N frames to near-zero.
        Fade-from-black: luminance rises from near-zero over N frames.
        """
        lum = self.all_luminance
        n = len(lum)
        dark_thresh = self.config.fade_luminance_threshold
        min_frames = self.config.fade_min_frames
        min_drop = self.config.fade_luminance_drop

        fades = []
        i = 0
        while i < n:
            if lum[i] < dark_thresh:
                # Found a dark region
                dark_start = i
                while i < n and lum[i] < dark_thresh:
                    i += 1
                dark_end = i

                # Check fade-to-black: luminance declining before dark region
                if dark_start >= min_frames:
                    pre_start = dark_start - min_frames
                    pre_lum = lum[pre_start:dark_start]
                    lum_drop = pre_lum[0] - pre_lum[-1]

                    if lum_drop >= min_drop:
                        fades.append({
                            'frame_index': dark_start,
                            'frame': self.frame_numbers[dark_start],
                            'timestamp': self.frame_numbers[dark_start] / self.fps,
                            'type': 'fade_to_black',
                            'duration': (self.frame_numbers[dark_start] - self.frame_numbers[pre_start]) / self.fps,
                            'luminance_start': float(pre_lum[0]),
                            'luminance_end': float(pre_lum[-1]),
                        })

                # Check fade-from-black: luminance rising after dark region
                if dark_end + min_frames <= n:
                    post_end = dark_end + min_frames
                    post_lum = lum[dark_end:post_end]
                    lum_rise = post_lum[-1] - post_lum[0]

                    if lum_rise >= min_drop:
                        fades.append({
                            'frame_index': dark_end,
                            'frame': self.frame_numbers[dark_end],
                            'timestamp': self.frame_numbers[dark_end] / self.fps,
                            'type': 'fade_from_black',
                            'duration': (self.frame_numbers[post_end - 1] - self.frame_numbers[dark_end]) / self.fps,
                            'luminance_start': float(post_lum[0]),
                            'luminance_end': float(post_lum[-1]),
                        })
            else:
                i += 1

        return fades

    def detect_transitions(self) -> List[Dict]:
        """Detect all transitions: hard cuts, dissolves, and fades.

        Must be called after scoring all frames with score_frame().

        Returns:
            List of transition dicts, each with at least:
                'timestamp', 'type' ('hard_cut'|'dissolve'|'fade_to_black'|'fade_from_black')
            Sorted by timestamp.
        """
        if self.fps is None:
            raise ValueError("fps not set — call set_video_params() first")

        transitions = []

        # Hard cuts (detected first so dissolve filter can use their positions)
        hard_cuts = self.detect_cuts()
        hard_cut_indices = set(idx for idx, _ in hard_cuts)
        for frame_idx, timestamp in hard_cuts:
            transitions.append({
                'frame_index': frame_idx,
                'frame': self.frame_numbers[frame_idx],
                'timestamp': timestamp,
                'type': 'hard_cut',
            })

        # Dissolves (filtered by proximity to hard cuts and bell-curve shape)
        transitions.extend(self._detect_dissolves(hard_cut_indices))

        # Fades
        transitions.extend(self._detect_fades())

        # Sort by timestamp
        transitions.sort(key=lambda t: t['timestamp'])

        return transitions

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
            lum = self.all_luminance[i] if i < len(self.all_luminance) else 0.0
            rows.append({
                'frame': fn,
                'time': fn / self.fps,
                'score': score,
                'metrics': metrics,
                'threshold': thresh,
                'luminance': lum,
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
        self.all_luminance.clear()
        self.frame_numbers.clear()

    @property
    def current_frame(self) -> int:
        """Get current frame number."""
        return self.frame_count
