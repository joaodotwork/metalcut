import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
import time
from functools import wraps

# Custom theme for rich output
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "debug": "dim cyan",
    "cut": "green",
    "metric": "blue",
    "time": "magenta"
})

console = Console(theme=CUSTOM_THEME)

class MetricsLogger:
    """Logger for performance metrics and statistics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'frames_processed': 0,
            'cuts_detected': 0,
            'processing_time': 0,
            'fps': 0
        }
    
    def update(self, **kwargs):
        """Update metrics."""
        self.metrics.update(kwargs)
        self.metrics['processing_time'] = time.time() - self.start_time
        if self.metrics['processing_time'] > 0:
            self.metrics['fps'] = self.metrics['frames_processed'] / self.metrics['processing_time']
    
    def summary(self) -> str:
        """Get metrics summary."""
        return (
            f"\nProcessing Summary:\n"
            f"  Frames Processed: {self.metrics['frames_processed']}\n"
            f"  Cuts Detected: {self.metrics['cuts_detected']}\n"
            f"  Processing Time: {self.metrics['processing_time']:.2f}s\n"
            f"  Average FPS: {self.metrics['fps']:.2f}"
        )

def setup_logging(output_dir: Optional[str] = None,
                 debug: bool = False,
                 log_file: bool = True) -> logging.Logger:
    """Configure logging with rich formatting.
    
    Args:
        output_dir: Directory for log files
        debug: Enable debug logging
        log_file: Enable file logging
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('cut_detector')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with rich formatting
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=debug,
        rich_tracebacks=True,
        tracebacks_show_locals=debug
    )
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file and output_dir:
        log_dir = Path(output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_file = log_dir / f'cut_detector-{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(file_handler)
    
    return logger

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('cut_detector')
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}"
            )
            raise
    
    return wrapper

class ProgressLogger:
    """Custom progress logger with rich output."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 1.0  # seconds
    
    def update(self, amount: int = 1):
        """Update progress."""
        self.current += amount
        current_time = time.time()
        
        # Update display if enough time has passed
        if current_time - self.last_update >= self.update_interval:
            self._display_progress()
            self.last_update = current_time
    
    def _display_progress(self):
        """Display progress with rich formatting."""
        elapsed = time.time() - self.start_time
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        # Calculate speed and ETA
        speed = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / speed if speed > 0 else 0
        
        console.print(
            f"{self.desc}: "
            f"[cyan]{percentage:.1f}%[/] "
            f"({self.current}/{self.total}) "
            f"[magenta]{speed:.1f} fps[/] "
            f"ETA: [yellow]{remaining:.1f}s[/]",
            end='\r'
        )
    
    def finish(self):
        """Complete progress logging."""
        self._display_progress()
        console.print()  # New line

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Enhanced error logging with context."""
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(str(error))
    
    if logger.level <= logging.DEBUG:
        logger.exception("Detailed traceback:")

