import cv2
import numpy as np
from typing import Tuple, Optional, Union
import logging

class FrameMetrics:
    """Efficient frame comparison metrics optimized for hard cut detection."""
    
    @staticmethod
    def quick_difference(frame1: Union[np.ndarray, cv2.UMat], 
                        frame2: Union[np.ndarray, cv2.UMat],
                        downscale_size: Tuple[int, int] = (64, 36)) -> float:
        """Ultra-fast frame difference check using downscaled frames."""
        try:
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

    @staticmethod
    def histogram_difference(frame1: np.ndarray, frame2: np.ndarray, 
                           bins: int = 64) -> float:
        """Calculate histogram difference between frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            bins: Number of histogram bins
            
        Returns:
            float: Difference score between 0 and 100
        """
        try:
            # Calculate histograms for each channel
            score = 0.0
            
            if len(frame1.shape) == 3:  # Color image
                for channel in range(3):
                    hist1 = cv2.calcHist([frame1], [channel], None, [bins], [0, 256])
                    hist2 = cv2.calcHist([frame2], [channel], None, [bins], [0, 256])
                    
                    # Normalize histograms
                    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                    
                    # Compare histograms (correlation method)
                    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    score += (1.0 - corr) / 3.0  # Average across channels
            else:  # Grayscale image
                hist1 = cv2.calcHist([frame1], [0], None, [bins], [0, 256])
                hist2 = cv2.calcHist([frame2], [0], None, [bins], [0, 256])
                
                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                
                score = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Convert to 0-100 range
            return float(score * 100.0)
            
        except Exception as e:
            print(f"Error in histogram_difference: {e}")
            return 0.0

    @staticmethod
    def edge_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate difference in edge patterns between frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            float: Difference score between 0 and 100
        """
        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frame1, frame2
            
            # Detect edges
            edges1 = cv2.Canny(gray1, 100, 200)
            edges2 = cv2.Canny(gray2, 100, 200)
            
            # Compare edge patterns
            diff = cv2.absdiff(edges1, edges2)
            
            # Normalize score to 0-100 range
            return float(np.mean(diff) * 100.0 / 255.0)
            
        except Exception as e:
            print(f"Error in edge_difference: {e}")
            return 0.0

    @staticmethod
    def combined_difference(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, dict]:
        """Calculate combined difference score using multiple metrics.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Tuple[float, dict]: Combined score and individual metrics
        """
        # Calculate individual metrics
        quick_score = FrameMetrics.quick_difference(frame1, frame2)
        
        # Only calculate more expensive metrics if quick score indicates potential cut
        metrics = {'quick_score': quick_score}
        
        if quick_score > 10.0:  # Threshold for additional analysis
            hist_score = FrameMetrics.histogram_difference(frame1, frame2)
            edge_score = FrameMetrics.edge_difference(frame1, frame2)
            
            metrics.update({
                'histogram_score': hist_score,
                'edge_score': edge_score
            })
            
            # Weighted combination of metrics
            combined_score = (
                0.4 * quick_score +
                0.4 * hist_score +
                0.2 * edge_score
            )
        else:
            combined_score = quick_score
        
        return combined_score, metrics
