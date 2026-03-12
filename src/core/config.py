from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    """All tunable detection parameters in one place.

    Populated from config/default_config.json, with CLI overrides.
    """

    # Sensitivity (0.0-1.0) — higher = more cuts detected
    sensitivity: float = 0.5

    # Minimum distance between cuts in seconds
    min_cut_distance: float = 0.5

    # Quick stage: downscale resolution for fast difference check
    downscale_width: int = 64
    downscale_height: int = 36

    # Quick gate: minimum quick_score to trigger detailed analysis
    quick_gate: float = 10.0

    # Canny edge detection thresholds
    canny_low: int = 100
    canny_high: int = 200

    # Histogram bins for histogram_difference
    histogram_bins: int = 64

    # Combined score weights (must sum to 1.0)
    weight_quick: float = 0.4
    weight_histogram: float = 0.4
    weight_edge: float = 0.2

    # Score mode: "weighted" (classic weighted average) or "max" (best-of signals)
    score_mode: str = "max"

    # Temporal scoring
    temporal_window: int = 5  # Number of recent scores to track
    temporal_boost_factor: float = 0.25  # Additive boost scaling

    # Lookahead adaptive thresholding — scores all frames first, then decides
    # with symmetrical context (N frames before AND after each candidate)
    lookahead_frames: int = 15  # Frames to look ahead/behind for neighborhood context
    adaptive_percentile: float = 75.0  # Percentile of neighborhood scores
    adaptive_margin: float = 12.0  # Score must exceed neighborhood_percentile + margin to be a cut

    # Dissolve detection — sustained mid-range scores over N frames
    dissolve_min_frames: int = 8  # Minimum frames of elevated scores for a dissolve
    dissolve_score_floor: float = 3.0  # Minimum score to count as "active"

    # Fade detection — luminance trending to/from black
    fade_luminance_threshold: float = 15.0  # Mean luminance below this = "dark" (0-255)
    fade_min_frames: int = 15  # Minimum frames for a fade transition (~0.5s at 24fps)
    fade_luminance_drop: float = 30.0  # Luminance must change by at least this much

    # Threshold ranges derived from sensitivity:
    #   quick_threshold = quick_base + (1 - sensitivity) * quick_range
    #   detailed_threshold = detailed_base + (1 - sensitivity) * detailed_range
    quick_threshold_base: float = 15.0
    quick_threshold_range: float = 20.0
    detailed_threshold_base: float = 20.0
    detailed_threshold_range: float = 30.0

    @property
    def quick_threshold(self) -> float:
        return self.quick_threshold_base + (1.0 - self.sensitivity) * self.quick_threshold_range

    @property
    def detailed_threshold(self) -> float:
        return self.detailed_threshold_base + (1.0 - self.sensitivity) * self.detailed_threshold_range

    @classmethod
    def from_config_dict(cls, config: dict, **overrides) -> "DetectionConfig":
        """Build from a parsed config JSON dict with optional overrides.

        Args:
            config: Parsed JSON config (with 'detection' key).
            **overrides: Direct field overrides (e.g. from CLI args).
        """
        detection = config.get("detection", {})

        # Map config keys to dataclass fields
        kwargs = {}
        field_map = {
            "sensitivity": "sensitivity",
            "min_cut_distance": "min_cut_distance",
            "quick_gate": "quick_gate",
            "downscale_width": "downscale_width",
            "downscale_height": "downscale_height",
            "canny_low": "canny_low",
            "canny_high": "canny_high",
            "histogram_bins": "histogram_bins",
            "weight_quick": "weight_quick",
            "weight_histogram": "weight_histogram",
            "weight_edge": "weight_edge",
            "temporal_window": "temporal_window",
            "temporal_boost_factor": "temporal_boost_factor",
            "quick_threshold_base": "quick_threshold_base",
            "quick_threshold_range": "quick_threshold_range",
            "detailed_threshold_base": "detailed_threshold_base",
            "detailed_threshold_range": "detailed_threshold_range",
            "score_mode": "score_mode",
            "lookahead_frames": "lookahead_frames",
            "adaptive_percentile": "adaptive_percentile",
            "adaptive_margin": "adaptive_margin",
            "dissolve_min_frames": "dissolve_min_frames",
            "dissolve_score_floor": "dissolve_score_floor",
            "fade_luminance_threshold": "fade_luminance_threshold",
            "fade_min_frames": "fade_min_frames",
            "fade_luminance_drop": "fade_luminance_drop",
        }

        for config_key, field_name in field_map.items():
            if config_key in detection:
                kwargs[field_name] = detection[config_key]

        # CLI overrides take priority
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value

        return cls(**kwargs)
