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

    # Temporal scoring
    temporal_window: int = 5  # Number of recent scores to track
    temporal_boost_factor: float = 0.25  # Additive boost scaling

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
        }

        for config_key, field_name in field_map.items():
            if config_key in detection:
                kwargs[field_name] = detection[config_key]

        # CLI overrides take priority
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value

        return cls(**kwargs)
