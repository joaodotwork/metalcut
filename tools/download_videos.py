import yt_dlp
import logging
from pathlib import Path
from typing import List, Optional, Dict
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.logging import RichHandler
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("downloader")

class VideoDownloader:
    def __init__(self, output_dir: str = "data/videos", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Thread-safe progress tracking
        self.progress_dict = {}
        self.progress_lock = threading.Lock()
        
        # Base yt-dlp options
        # Uses SABR protocol formats to work around YouTube's SABR streaming enforcement.
        # See https://github.com/yt-dlp/yt-dlp/issues/12482
        self.base_opts = {
            'format': 'bv[height<=720][protocol=sabr]+ba[protocol=sabr]/b',
            'extractor_args': {'youtube': ['formats=duplicate']},
            'quiet': True,
            'no_warnings': True,
            'concurrent_fragment_downloads': 4,
        }
    
    def _get_progress_hook(self, video_id: str):
        """Create a progress hook for a specific video."""
        def progress_hook(d):
            with self.progress_lock:
                if video_id not in self.progress_dict:
                    return
                    
                task_id = self.progress_dict[video_id]['task_id']
                progress = self.progress_dict[video_id]['progress']
                
                if d['status'] == 'downloading':
                    if 'total_bytes' in d:
                        percentage = (d['downloaded_bytes'] / d['total_bytes']) * 100
                        progress.update(task_id, completed=percentage)
                elif d['status'] == 'finished':
                    progress.update(task_id, completed=100)
                    
        return progress_hook
    
    def _download_single(self, url: str, progress: Progress) -> bool:
        """Download a single video with progress tracking."""
        try:
            video_id = url.split("v=")[1]
            output_path = self.output_dir / f"{video_id}.mp4"
            
            # Skip if already downloaded
            if output_path.exists():
                logger.info(f"Skipping {video_id} (already downloaded)")
                return True
            
            # Create progress bar for this video
            with self.progress_lock:
                task_id = progress.add_task(
                    f"Downloading {video_id}...",
                    total=100
                )
                self.progress_dict[video_id] = {
                    'task_id': task_id,
                    'progress': progress
                }
            
            # Configure yt-dlp options for this download
            ydl_opts = self.base_opts.copy()
            ydl_opts.update({
                'outtmpl': str(output_path),
                'progress_hooks': [self._get_progress_hook(video_id)]
            })
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            logger.info(f"Downloaded: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
        finally:
            # Cleanup progress tracking
            with self.progress_lock:
                if video_id in self.progress_dict:
                    progress.remove_task(self.progress_dict[video_id]['task_id'])
                    del self.progress_dict[video_id]
    
    def download_videos(self, urls: List[str]):
        """Download multiple videos in parallel with progress tracking."""
        successful = 0
        failed = 0
        start_time = time.time()
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn()
        ) as progress:
            # Create thread pool
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all downloads
                future_to_url = {
                    executor.submit(self._download_single, url, progress): url
                    for url in urls
                }
                
                # Wait for completion
                for future in future_to_url:
                    url = future_to_url[future]
                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Download failed for {url}: {e}")
                        failed += 1
        
        # Print summary
        duration = time.time() - start_time
        logger.info("\nDownload Summary:")
        logger.info(f"Total videos processed: {len(urls)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total time: {duration:.2f}s")
        if successful > 0:
            logger.info(f"Average time per video: {duration/successful:.2f}s")

def load_urls(filepath: str) -> List[str]:
    """Load URLs from file, skipping empty lines and comments."""
    urls = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and 'youtube.com' in line:
                    urls.append(line)
        return urls
    except Exception as e:
        logger.error(f"Error loading URLs: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos for processing")
    parser.add_argument("--urls", default="data/urls.txt",
                       help="Path to file containing YouTube URLs")
    parser.add_argument("--output-dir", default="data/videos",
                       help="Directory to save downloaded videos")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel downloads")
    
    args = parser.parse_args()
    
    # Load URLs
    urls = load_urls(args.urls)
    if not urls:
        logger.error("No valid URLs found")
        return 1
    
    logger.info(f"Found {len(urls)} URLs to download")
    logger.info(f"Using {args.workers} parallel workers")
    
    # Download videos
    downloader = VideoDownloader(args.output_dir, max_workers=args.workers)
    downloader.download_videos(urls)
    
    return 0

if __name__ == "__main__":
    exit(main())
