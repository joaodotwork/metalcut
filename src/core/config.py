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

    # Semantic scene analysis (Tier 1) — optional FP reduction via pHash + color palette
    use_semantic: bool = False
    phash_size: int = 8  # Hash grid size (8 → 64-bit hash)
    palette_bins: int = 8  # Bins per HSV channel for palette descriptor
    palette_change_threshold: float = 0.3  # Min palette distance to confirm a cut
    phash_hamming_threshold: int = 12  # Min hamming distance to confirm a cut

    # Threshold ranges derived from sensitivity:
    #   quick_threshold = quick_base + (1 - sensitivity) * quick_range
    #   detailed_threshold = detailed_base + (1 - sensitivity) * detailed_range
    quick_threshold_base: float = 15.0
    quick_threshold_range: float = 20.0
    detailed_threshold_base: float = 20.0
    detailed_threshold_range: float = 30.0

    # High-sensitivity relief — above the knee, both the detailed-threshold
    # floor and the adaptive neighborhood margin ramp down toward these minima,
    # so low-contrast hard cuts (visually similar adjacent shots, score < 20)
    # become catchable. At or below the knee, behavior is identical to the plain
    # linear sensitivity scale above, so default/medium-sensitivity runs are
    # unchanged. See docs/spike/cut-refinement-plan.md.
    high_sensitivity_knee: float = 0.7
    detailed_threshold_min: float = 8.0
    adaptive_margin_min: float = 8.0

    def _knee_ramp(self, value_at_knee: float, minimum: float) -> float:
        """Interpolate value_at_knee → minimum as sensitivity rises knee → 1.0.

        Returns value_at_knee unchanged at or below the knee, so the relief is
        a no-op for default/medium sensitivity.
        """
        s, knee = self.sensitivity, self.high_sensitivity_knee
        if s <= knee or knee >= 1.0:
            return value_at_knee
        t = (s - knee) / (1.0 - knee)
        return value_at_knee + t * (minimum - value_at_knee)

    @property
    def quick_threshold(self) -> float:
        return self.quick_threshold_base + (1.0 - self.sensitivity) * self.quick_threshold_range

    @property
    def detailed_threshold(self) -> float:
        # Linear scale clamped at the knee, then ramped toward the floor minimum
        # above it. min(sensitivity, knee) makes this equal the plain linear value
        # at/below the knee and the linear value *at* the knee once above it.
        s_clamped = min(self.sensitivity, self.high_sensitivity_knee)
        value_at_knee = self.detailed_threshold_base + (1.0 - s_clamped) * self.detailed_threshold_range
        return self._knee_ramp(value_at_knee, self.detailed_threshold_min)

    @property
    def effective_adaptive_margin(self) -> float:
        """Adaptive neighborhood margin after high-sensitivity relief."""
        return self._knee_ramp(self.adaptive_margin, self.adaptive_margin_min)

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
            "high_sensitivity_knee": "high_sensitivity_knee",
            "detailed_threshold_min": "detailed_threshold_min",
            "adaptive_margin_min": "adaptive_margin_min",
            "score_mode": "score_mode",
            "lookahead_frames": "lookahead_frames",
            "adaptive_percentile": "adaptive_percentile",
            "adaptive_margin": "adaptive_margin",
            "dissolve_min_frames": "dissolve_min_frames",
            "dissolve_score_floor": "dissolve_score_floor",
            "fade_luminance_threshold": "fade_luminance_threshold",
            "fade_min_frames": "fade_min_frames",
            "fade_luminance_drop": "fade_luminance_drop",
            "use_semantic": "use_semantic",
            "phash_size": "phash_size",
            "palette_bins": "palette_bins",
            "palette_change_threshold": "palette_change_threshold",
            "phash_hamming_threshold": "phash_hamming_threshold",
        }

        for config_key, field_name in field_map.items():
            if config_key in detection:
                kwargs[field_name] = detection[config_key]

        # CLI overrides take priority
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value

        return cls(**kwargs)
