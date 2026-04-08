import cv2
import numpy as np
from typing import Tuple, Optional, Union
import logging

from .config import DetectionConfig

class FrameMetrics:
    """Efficient frame comparison metrics optimized for hard cut detection."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def quick_difference(self, frame1: Union[np.ndarray, cv2.UMat],
                        frame2: Union[np.ndarray, cv2.UMat]) -> float:
        """Ultra-fast frame difference check using downscaled frames."""
        try:
            downscale_size = (self.config.downscale_width, self.config.downscale_height)

            # Get dimensions from UMat or ndarray
            if isinstance(frame1, cv2.UMat):
                height, width = frame1.get().shape[:2]
            else:
                height, width = frame1.shape[:2]

            # Ensure frames are same size
            if isinstance(frame2, cv2.UMat):
                height2, width2 = frame2.get().shape[:2]
                if (width, height) != (width2, height2):
                    frame2 = cv2.resize(frame2, (width, height))
            else:
                if frame1.shape != frame2.shape:
                    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

            # Downscale frames for speed
            small1 = cv2.resize(frame1, downscale_size, interpolation=cv2.INTER_LINEAR)
            small2 = cv2.resize(frame2, downscale_size, interpolation=cv2.INTER_LINEAR)

            # Convert to grayscale if needed
            if len(small1.get().shape if isinstance(small1, cv2.UMat) else small1.shape) == 3:
                small1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
                small2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference
            diff = cv2.absdiff(small1, small2)

            # Get mean value (handling UMat)
            if isinstance(diff, cv2.UMat):
                mean_diff = cv2.mean(diff)[0]
            else:
                mean_diff = np.mean(diff)

            # Normalize score to 0-100 range
            return float(mean_diff * 100.0 / 255.0)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in quick_difference: {e}")
            return 0.0

    def histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            float: Difference score between 0 and 100
        """
        try:
            bins = self.config.histogram_bins
            score = 0.0

            if len(frame1.shape) == 3:  # Color image
                for channel in range(3):
                    hist1 = cv2.calcHist([frame1], [channel], None, [bins], [0, 256])
                    hist2 = cv2.calcHist([frame2], [channel], None, [bins], [0, 256])

                    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

                    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    score += (1.0 - corr) / 3.0
            else:  # Grayscale image
                hist1 = cv2.calcHist([frame1], [0], None, [bins], [0, 256])
                hist2 = cv2.calcHist([frame2], [0], None, [bins], [0, 256])

                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

                score = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            return float(score * 100.0)

        except Exception as e:
            print(f"Error in histogram_difference: {e}")
            return 0.0

    def edge_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate difference in edge patterns between frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            float: Difference score between 0 and 100
        """
        try:
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frame1, frame2

            edges1 = cv2.Canny(gray1, self.config.canny_low, self.config.canny_high)
            edges2 = cv2.Canny(gray2, self.config.canny_low, self.config.canny_high)

            diff = cv2.absdiff(edges1, edges2)

            return float(np.mean(diff) * 100.0 / 255.0)

        except Exception as e:
            print(f"Error in edge_difference: {e}")
            return 0.0

    def color_palette(self, frame: np.ndarray) -> np.ndarray:
        """Extract a compact color palette descriptor from HSV histogram.

        Returns a normalized 1D vector (palette_bins * 2 elements) capturing
        the hue and saturation distribution. Fast alternative to k-means
        dominant color extraction.
        """
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bins = self.config.palette_bins
            h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
            descriptor = np.concatenate([h_hist, s_hist]).flatten()
            cv2.normalize(descriptor, descriptor)
            return descriptor
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in color_palette: {e}")
            return np.zeros(self.config.palette_bins * 2, dtype=np.float32)

    def phash(self, frame: np.ndarray) -> int:
        """Compute perceptual hash via DCT.

        Resizes to (hash_size*4)^2, applies DCT, keeps low-frequency
        hash_size^2 coefficients, thresholds at median → 64-bit hash.
        Hamming distance between hashes measures perceptual similarity.
        """
        try:
            hash_size = self.config.phash_size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                                 interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            dct_low = dct[:hash_size, :hash_size]
            median = np.median(dct_low)
            bits = (dct_low > median).flatten()
            # Pack bits into integer (up to 64 bits for hash_size=8)
            hash_val = 0
            for bit in bits:
                hash_val = (hash_val << 1) | int(bit)
            return hash_val
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in phash: {e}")
            return 0

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Count differing bits between two hashes."""
        return bin(h1 ^ h2).count('1')

    def combined_difference(self, frame1: Union[np.ndarray, cv2.UMat],
                           frame2: Union[np.ndarray, cv2.UMat]) -> Tuple[float, dict]:
        """Calculate combined difference score using multiple metrics.

        Args:
            frame1: First frame (ndarray or UMat)
            frame2: Second frame (ndarray or UMat)

        Returns:
            Tuple[float, dict]: Combined score and individual metrics
        """
        quick_score = self.quick_difference(frame1, frame2)

        metrics = {'quick_score': quick_score}

        if quick_score > self.config.quick_gate:
            # histogram_difference and edge_difference require np.ndarray
            f1 = frame1.get() if isinstance(frame1, cv2.UMat) else frame1
            f2 = frame2.get() if isinstance(frame2, cv2.UMat) else frame2
            hist_score = self.histogram_difference(f1, f2)
            edge_score = self.edge_difference(f1, f2)

            metrics.update({
                'histogram_score': hist_score,
                'edge_score': edge_score
            })

            weighted_score = (
                self.config.weight_quick * quick_score +
                self.config.weight_histogram * hist_score +
                self.config.weight_edge * edge_score
            )

            if self.config.score_mode == "max":
                # Don't let weak metrics drag down strong signals
                combined_score = max(quick_score, hist_score, weighted_score)
            else:
                combined_score = weighted_score

            metrics['weighted_score'] = weighted_score

            # Piggyback semantic features on the CPU frames we already have
            if self.config.use_semantic:
                metrics['phash_1'] = self.phash(f1)
                metrics['phash_2'] = self.phash(f2)
                metrics['palette_1'] = self.color_palette(f1)
                metrics['palette_2'] = self.color_palette(f2)
        else:
            combined_score = quick_score

        return combined_score, metrics
