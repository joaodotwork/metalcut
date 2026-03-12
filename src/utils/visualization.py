import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import time

def create_preview(frame: np.ndarray,
                  metrics: Dict[str, float],
                  scale: float = 0.5,
                  show_details: bool = True) -> np.ndarray:
    """Create preview frame with detection information.
    
    Args:
        frame: Input frame
        metrics: Detection metrics
        scale: Preview scale factor
        show_details: Show detailed metrics
        
    Returns:
        Annotated frame
    """
    # Create copy of frame
    preview = frame.copy()
    
    # Scale frame if needed
    if scale != 1.0:
        width = int(preview.shape[1] * scale)
        height = int(preview.shape[0] * scale)
        preview = cv2.resize(preview, (width, height))
    
    # Add metrics visualization
    if metrics:
        # Get main score
        score = metrics.get('final_score', metrics.get('quick_score', 0.0))
        
        # Add score bar
        bar_height = 5
        bar_width = preview.shape[1]
        bar_x = 0
        bar_y = preview.shape[0] - bar_height
        
        # Draw background bar
        cv2.rectangle(
            preview,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (0, 0, 0),
            -1
        )
        
        # Draw score bar
        score_width = int(bar_width * min(score / 100.0, 1.0))
        cv2.rectangle(
            preview,
            (bar_x, bar_y),
            (bar_x + score_width, bar_y + bar_height),
            (0, int(255 * min(score / 50.0, 1.0)), 0),
            -1
        )
        
        # Add detailed metrics if enabled
        if show_details:
            y_pos = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            
            # Format metrics for display
            display_metrics = {
                'Score': f"{score:.1f}",
                'Quick': f"{metrics.get('quick_score', 0.0):.1f}",
                'Temporal': f"{metrics.get('temporal_score', 0.0):.1f}"
            }
            
            if 'edge_score' in metrics:
                display_metrics['Edge'] = f"{metrics['edge_score']:.1f}"
            if 'histogram_score' in metrics:
                display_metrics['Hist'] = f"{metrics['histogram_score']:.1f}"
            
            # Draw metrics
            for label, value in display_metrics.items():
                text = f"{label}: {value}"
                
                # Draw background for better readability
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                cv2.rectangle(
                    preview,
                    (10, y_pos - text_height),
                    (10 + text_width, y_pos + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    preview,
                    text,
                    (10, y_pos),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
                
                y_pos += 25
    
    return preview

def create_debug_view(frame1: np.ndarray,
                     frame2: np.ndarray,
                     metrics: Dict[str, float]) -> np.ndarray:
    """Create debug view showing frame comparison.
    
    Args:
        frame1: First frame
        frame2: Second frame
        metrics: Detection metrics
        
    Returns:
        Debug visualization
    """
    # Ensure frames are the same size
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Calculate difference image
    diff = cv2.absdiff(frame1, frame2)
    
    # Create composite view
    height = max(frame1.shape[0], frame2.shape[0])
    width = frame1.shape[1] + frame2.shape[1] + diff.shape[1]
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add frames and difference
    composite[:frame1.shape[0], :frame1.shape[1]] = frame1
    composite[:frame2.shape[0], frame1.shape[1]:frame1.shape[1]+frame2.shape[1]] = frame2
    composite[:diff.shape[0], frame1.shape[1]+frame2.shape[1]:] = \
        cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR) if len(diff.shape) == 2 else diff
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "Frame 1", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(composite, "Frame 2", (frame1.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(composite, "Difference", (frame1.shape[1] + frame2.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Add metrics
    y_pos = height - 10
    for label, value in metrics.items():
        text = f"{label}: {value:.2f}"
        cv2.putText(composite, text, (10, y_pos), font, 0.6, (255, 255, 255), 1)
        y_pos -= 25
    
    return composite

def create_thumbnail(frame: np.ndarray,
                    size: Tuple[int, int] = (320, 180)) -> np.ndarray:
    """Create thumbnail from frame.
    
    Args:
        frame: Input frame
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def draw_timeline(width: int,
                 height: int,
                 cuts: list,
                 duration: float,
                 current_time: Optional[float] = None) -> np.ndarray:
    """Create timeline visualization.
    
    Args:
        width: Timeline width
        height: Timeline height
        cuts: List of cut timestamps
        duration: Video duration
        current_time: Current playback time
        
    Returns:
        Timeline visualization
    """
    timeline = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw background
    cv2.rectangle(timeline, (0, 0), (width, height), (32, 32, 32), -1)
    
    # Draw cuts
    for cut in cuts:
        x = int((cut / duration) * width)
        cv2.line(timeline, (x, 0), (x, height), (0, 255, 0), 2)
    
    # Draw current time
    if current_time is not None:
        x = int((current_time / duration) * width)
        cv2.line(timeline, (x, 0), (x, height), (255, 255, 255), 2)
    
    # Add time markers
    font = cv2.FONT_HERSHEY_SIMPLEX
    marker_interval = max(int(duration / 10), 1)  # Show up to 10 time markers
    
    for t in range(0, int(duration) + 1, marker_interval):
        x = int((t / duration) * width)
        # Draw marker line
        cv2.line(timeline, (x, height-5), (x, height), (128, 128, 128), 1)
        # Add time label
        time_str = f"{t}s"
        (text_width, text_height), _ = cv2.getTextSize(time_str, font, 0.4, 1)
        cv2.putText(timeline, time_str,
                   (x - text_width//2, height-7),
                   font, 0.4, (128, 128, 128), 1)
    
    return timeline

def create_summary_view(frames: list,
                       cuts: list,
                       duration: float,
                       max_thumbnails: int = 6) -> np.ndarray:
    """Create summary view with thumbnails and timeline.
    
    Args:
        frames: List of key frames
        cuts: List of cut timestamps
        duration: Video duration
        max_thumbnails: Maximum number of thumbnails to show
        
    Returns:
        Summary visualization
    """
    if not frames:
        return None
    
    # Create thumbnails
    thumb_width = 320
    thumb_height = 180
    thumbnails = []
    
    # Select evenly spaced frames
    step = max(len(frames) // max_thumbnails, 1)
    for i in range(0, len(frames), step):
        if len(thumbnails) >= max_thumbnails:
            break
        thumb = create_thumbnail(frames[i], (thumb_width, thumb_height))
        thumbnails.append(thumb)
    
    # Create composite image
    timeline_height = 40
    spacing = 10
    
    width = max(thumb_width * len(thumbnails) + spacing * (len(thumbnails)-1),
               thumb_width * 2)  # Minimum width
    height = thumb_height + timeline_height + spacing
    
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add thumbnails
    x = 0
    for thumb in thumbnails:
        y = 0
        composite[y:y+thumb_height, x:x+thumb_width] = thumb
        x += thumb_width + spacing
    
    # Add timeline
    timeline = draw_timeline(width, timeline_height, cuts, duration)
    composite[height-timeline_height:, :] = timeline
    
    return composite

def draw_metrics_overlay(frame: np.ndarray,
                        metrics: Dict[str, float],
                        position: str = 'top-left') -> np.ndarray:
    """Draw metrics overlay on frame.
    
    Args:
        frame: Input frame
        metrics: Metrics to display
        position: Overlay position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        
    Returns:
        Frame with overlay
    """
    overlay = frame.copy()
    
    # Configure text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    padding = 10
    line_height = 25
    
    # Calculate text dimensions
    text_lines = [f"{k}: {v:.2f}" for k, v in metrics.items()]
    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                 for text in text_lines]
    
    max_width = max(size[0] for size in text_sizes) + padding * 2
    total_height = len(text_lines) * line_height + padding * 2
    
    # Determine overlay position
    if position.startswith('top'):
        y_start = padding
    else:
        y_start = frame.shape[0] - total_height
        
    if position.endswith('right'):
        x_start = frame.shape[1] - max_width
    else:
        x_start = 0
    
    # Draw background
    cv2.rectangle(overlay,
                 (x_start, y_start),
                 (x_start + max_width, y_start + total_height),
                 (0, 0, 0),
                 -1)
    
    # Draw metrics
    y = y_start + padding + line_height
    for text in text_lines:
        cv2.putText(overlay,
                   text,
                   (x_start + padding, y),
                   font,
                   font_scale,
                   (255, 255, 255),
                   font_thickness)
        y += line_height
    
    # Blend overlay with original frame
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame

def create_detection_preview(frame: np.ndarray,
                           is_cut: bool,
                           score: float,
                           threshold: float) -> np.ndarray:
    """Create preview with cut detection visualization.
    
    Args:
        frame: Input frame
        is_cut: Whether frame is a cut point
        score: Detection score
        threshold: Detection threshold
        
    Returns:
        Preview visualization
    """
    preview = frame.copy()
    
    # Add detection indicator
    if is_cut:
        color = (0, 255, 0)  # Green for cuts
        text = "CUT DETECTED"
    else:
        color = (0, 0, 255)  # Red for no cut
        text = "NO CUT"
    
    # Draw indicator box
    height, width = preview.shape[:2]
    box_width = 200
    box_height = 40
    x = width - box_width - 10
    y = 10
    
    cv2.rectangle(preview,
                 (x, y),
                 (x + box_width, y + box_height),
                 color,
                 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(preview,
                text,
                (x + 10, y + 25),
                font,
                0.7,
                color,
                2)
    
    # Add score meter
    meter_width = width - 20
    meter_height = 20
    x = 10
    y = height - meter_height - 10
    
    # Background
    cv2.rectangle(preview,
                 (x, y),
                 (x + meter_width, y + meter_height),
                 (64, 64, 64),
                 -1)
    
    # Score bar
    score_width = int(meter_width * (score / 100.0))
    cv2.rectangle(preview,
                 (x, y),
                 (x + score_width, y + meter_height),
                 (0, int(255 * min(score / threshold, 1.0)), 0),
                 -1)
    
    # Threshold marker
    threshold_x = x + int(meter_width * (threshold / 100.0))
    cv2.line(preview,
             (threshold_x, y),
             (threshold_x, y + meter_height),
             (255, 255, 255),
             2)
    
    return preview
