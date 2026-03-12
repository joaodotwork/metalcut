import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MetalAccelerator:
    """GPU acceleration using Metal for Apple Silicon."""
    
    def __init__(self):
        """Initialize Metal acceleration support."""
        self.metal_available = self._init_metal()
        self.use_gpu = self.metal_available
        self._frame_count = 0
        self._last_log_time = 0
        
        if self.metal_available:
            logger.info("Metal acceleration enabled")
        else:
            logger.info("Metal acceleration not available, using CPU")
    
    def _init_metal(self) -> bool:
        """Initialize Metal support through OpenCV."""
        try:
            # Enable OpenCL (Metal backend on macOS)
            cv2.ocl.setUseOpenCL(True)
            has_metal = cv2.ocl.haveOpenCL()
            
            if has_metal:
                # Get Metal device info
                device = cv2.ocl.Device_getDefault()
                logger.info(f"Metal device: {device.name()}")
                logger.info(f"Compute units: {device.maxComputeUnits()}")
                return True
            return False
            
        except Exception as e:
            logger.warning(f"Failed to initialize Metal: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Union[cv2.UMat, np.ndarray]:
        """Process frame using Metal acceleration if available.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame (UMat if GPU, ndarray if CPU)
        """
        self._frame_count += 1
        
        if not self.use_gpu:
            return frame
            
        try:
            # Convert to GPU memory
            gpu_frame = cv2.UMat(frame)
            
            # Log performance every 100 frames
            if self._frame_count % 100 == 0:
                logger.debug(f"Processing frame {self._frame_count} with Metal")
            
            return gpu_frame
            
        except Exception as e:
            logger.warning(f"GPU processing failed, falling back to CPU: {e}")
            self.use_gpu = False
            return frame
    
    def compute_difference(self, 
                         frame1: Union[cv2.UMat, np.ndarray],
                         frame2: Union[cv2.UMat, np.ndarray],
                         method: str = 'quick') -> Tuple[float, dict]:
        """Compute frame difference using Metal acceleration.
        
        Args:
            frame1: First frame
            frame2: Second frame
            method: Difference method ('quick' or 'detailed')
            
        Returns:
            Tuple of (difference_score, metrics_dict)
        """
        try:
            if not self.use_gpu:
                return 0.0, {}
            
            metrics = {}
            
            # Ensure frames are on GPU
            gpu_frame1 = frame1 if isinstance(frame1, cv2.UMat) else cv2.UMat(frame1)
            gpu_frame2 = frame2 if isinstance(frame2, cv2.UMat) else cv2.UMat(frame2)
            
            if method == 'quick':
                # Quick difference using downsized frames
                small1 = cv2.resize(gpu_frame1, (64, 36))
                small2 = cv2.resize(gpu_frame2, (64, 36))
                
                diff = cv2.absdiff(small1, small2)
                score = float(cv2.mean(diff)[0] * 100.0 / 255.0)
                
                metrics['quick_score'] = score
                
            else:  # detailed
                # Full resolution difference
                diff = cv2.absdiff(gpu_frame1, gpu_frame2)
                
                # Multiple metrics
                metrics['full_diff'] = float(cv2.mean(diff)[0] * 100.0 / 255.0)
                
                # Edge difference
                edges1 = cv2.Canny(gpu_frame1, 100, 200)
                edges2 = cv2.Canny(gpu_frame2, 100, 200)
                edge_diff = cv2.absdiff(edges1, edges2)
                metrics['edge_diff'] = float(cv2.mean(edge_diff)[0] * 100.0 / 255.0)
                
                # Combined score
                score = (metrics['full_diff'] + metrics['edge_diff']) / 2.0
            
            return score, metrics
            
        except Exception as e:
            logger.warning(f"GPU difference calculation failed: {e}")
            self.use_gpu = False
            return 0.0, {'error': str(e)}
    
    def optimize_histogram(self, frame: Union[cv2.UMat, np.ndarray]) -> np.ndarray:
        """Compute optimized histogram using Metal.
        
        Args:
            frame: Input frame
            
        Returns:
            Computed histogram
        """
        try:
            if not self.use_gpu:
                return cv2.calcHist([frame], [0], None, [256], [0, 256])
            
            gpu_frame = frame if isinstance(frame, cv2.UMat) else cv2.UMat(frame)
            
            # Compute histogram on GPU
            hist = cv2.calcHist([gpu_frame], [0], None, [256], [0, 256])
            
            return hist
            
        except Exception as e:
            logger.warning(f"GPU histogram calculation failed: {e}")
            self.use_gpu = False
            return cv2.calcHist([frame], [0], None, [256], [0, 256])
    
    def release(self):
        """Release GPU resources."""
        try:
            if self.use_gpu:
                cv2.ocl.finish()
                logger.debug("Released GPU resources")
        except Exception as e:
            logger.warning(f"Error releasing GPU resources: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

