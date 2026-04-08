import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from rich import print

class CutAnnotator:
    def __init__(self, video_path: str, cuts_file: str):
        self.video = cv2.VideoCapture(video_path)
        self.video_id = Path(video_path).stem
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.cuts = self.load_cuts(cuts_file)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.feedback = {
            'video_id': self.video_id,
            'timestamp': datetime.now().isoformat(),
            'good_cuts': [],  # Can now contain timestamps or ranges
            'false_positives': [],
            'missed_cuts': []  # Cuts found during browse that the detector missed
        }
        self.window_name = 'Cut Review'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def review_cut(self, timestamp: float):
        """Show cut with context window."""
        try:
            # Initialize frame numbers and marks
            self.current_frame_number = int(timestamp * self.fps)
            self.mark_in = None
            self.mark_out = None
            
            # Create main window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            def create_three_panel_view(center_frame_number):
                """Create three panel view with before/center/after frames."""
                # Get the frames
                before_frame_number = max(0, center_frame_number - 3)
                after_frame_number = center_frame_number + 3
                
                # Read frames
                self.video.set(cv2.CAP_PROP_POS_FRAMES, before_frame_number)
                ret, before_frame = self.video.read()
                self.video.set(cv2.CAP_PROP_POS_FRAMES, center_frame_number)
                ret, center_frame = self.video.read()
                self.video.set(cv2.CAP_PROP_POS_FRAMES, after_frame_number)
                ret, after_frame = self.video.read()
                
                if not all([before_frame is not None, center_frame is not None, after_frame is not None]):
                    return None
                
                # Create comparison view
                height = before_frame.shape[0]
                width = before_frame.shape[1] * 3
                comparison = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add frames side by side
                comparison[:, :before_frame.shape[1]] = before_frame
                comparison[:, before_frame.shape[1]:before_frame.shape[1]*2] = center_frame
                comparison[:, before_frame.shape[1]*2:] = after_frame
                
                # Add labels and visual markers
                font = cv2.FONT_HERSHEY_SIMPLEX
                overlay = comparison.copy()
                
                # Add semi-transparent overlays for top and bottom info
                cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)  # Top bar
                cv2.rectangle(overlay, (0, height-100), (width, height), (0, 0, 0), -1)  # Bottom bar
                
                # Blend overlay
                alpha = 0.7
                comparison = cv2.addWeighted(overlay, alpha, comparison, 1 - alpha, 0)
                
                # Add panel labels and frame info
                cv2.putText(comparison, "Before", (10, 40), font, 1, (255, 255, 255), 2)
                current_time = center_frame_number / self.fps
                cv2.putText(comparison, f"Current Frame ({current_time:.2f}s)", 
                            (before_frame.shape[1] + 10, 40), font, 1, (255, 255, 255), 2)
                cv2.putText(comparison, "After", (before_frame.shape[1]*2 + 10, 40), font, 1, (255, 255, 255), 2)
                
                # Add marking status
                if self.mark_in is not None:
                    cv2.putText(comparison, f"Mark In: {self.mark_in/self.fps:.2f}s", 
                                (10, height-70), font, 0.8, (0, 255, 0), 2)
                if self.mark_out is not None:
                    cv2.putText(comparison, f"Mark Out: {self.mark_out/self.fps:.2f}s", 
                                (width-300, height-70), font, 0.8, (0, 255, 0), 2)
                
                # Add instructions with clearer arrow key descriptions
                instructions = [
                    "Left/Right: Previous/Next frame | I: Mark In | O: Mark Out",
                    "Y: Correct cut | N: Incorrect cut | Space: Play context | ESC: Exit"
                ]
                for i, text in enumerate(instructions):
                    cv2.putText(comparison, text, 
                                (10, height-35+i*25), font, 0.8, (255, 255, 255), 2)
                
                # Highlight current panel
                cv2.rectangle(comparison, 
                            (before_frame.shape[1], 0),
                            (before_frame.shape[1]*2, height),
                            (0, 255, 0), 2)
                
                return comparison

            
            while True:
                # Update both windows
                display = create_three_panel_view(self.current_frame_number)
                if display is None:
                    return False
                
                cv2.imshow(self.window_name, display)
                                
                key = cv2.waitKey(1) & 0xFF
                
                # Handle keyboard input
                if key == ord('y'):
                    if self.mark_in is not None and self.mark_out is not None:
                        self.feedback['good_cuts'].append({
                            'start': float(self.mark_in / self.fps),
                            'end': float(self.mark_out / self.fps),
                            'original_timestamp': timestamp
                        })
                    else:
                        self.feedback['good_cuts'].append(timestamp)
                    return True
                elif key == ord('n'):
                    self.feedback['false_positives'].append(timestamp)
                    return True
                elif key == ord(' '):
                    self._show_context(timestamp)
                elif key == ord('i'):
                    self.mark_in = self.current_frame_number
                elif key == ord('o'):
                    self.mark_out = self.current_frame_number
                elif key in [2, ord('a')]:  # Left arrow or 'a'
                    self.current_frame_number = max(0, self.current_frame_number - 1)
                elif key in [3, ord('d')]:  # Right arrow or 'd'
                    self.current_frame_number += 1
                elif key == 27:  # ESC
                    return False

        except Exception as e:
            print(f"Error reviewing cut at {timestamp:.2f}s: {str(e)}")
            return False
        finally:
            cv2.destroyWindow('Instructions')
   
    def _show_context(self, timestamp: float, window: int = 2):
        """Show video context around cut point."""
        start_frame = int(max(0, (timestamp - window)) * self.fps)
        end_frame = int((timestamp + window) * self.fps)
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while self.video.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = self.video.read()
            if not ret:
                break
            
            # Add timeline
            height, width = frame.shape[:2]
            current_time = self.video.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
            progress = (current_time - (timestamp - window)) / (window * 2)
            
            # Draw timeline at bottom
            cv2.rectangle(frame, (0, height-30), (width, height), (0, 0, 0), -1)
            timeline_width = width - 40
            position = int(20 + timeline_width * progress)
            cv2.rectangle(frame, (20, height-20), (width-20, height-15), (128, 128, 128), -1)
            cv2.circle(frame, (position, height-17), 5, (0, 255, 0), -1)
            
            cv2.imshow('Context View (Press Q to return)', frame)
            if cv2.waitKey(int(1000/self.fps)) & 0xFF == ord('q'):
                break
                
        cv2.destroyWindow('Context View (Press Q to return)')

    def browse_for_missed(self):
        """Browse the video freely to mark missed cuts.

        Controls:
            Left/Right arrow or A/D: step 1 frame
            Shift behavior via J/L: jump 1 second
            Comma/Period: jump 10 seconds
            M: mark current position as a missed cut
            Space: play/pause
            ESC: done
        """
        frame_num = 0
        playing = False
        marked_frames = set()

        while True:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.video.read()
            if not ret:
                frame_num = max(0, frame_num - 1)
                continue

            display = frame.copy()
            h, w = display.shape[:2]
            current_time = frame_num / self.fps

            # Top bar with time/frame info
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)

            cv2.putText(display, f"{current_time:.2f}s  frame {frame_num}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show marked count
            marked_text = f"Missed cuts marked: {len(self.feedback['missed_cuts'])}"
            cv2.putText(display, marked_text,
                        (w - 400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Highlight if this frame is marked
            if frame_num in marked_frames:
                cv2.putText(display, "MARKED", (w // 2 - 60, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Timeline bar
            progress = frame_num / max(1, self.total_frames)
            bar_x = int(20 + (w - 40) * progress)
            cv2.rectangle(display, (20, h - 55), (w - 20, h - 50), (128, 128, 128), -1)
            cv2.circle(display, (bar_x, h - 52), 6, (0, 255, 0), -1)

            # Draw marks on timeline
            for mf in marked_frames:
                mx = int(20 + (w - 40) * (mf / max(1, self.total_frames)))
                cv2.line(display, (mx, h - 60), (mx, h - 45), (0, 255, 255), 2)

            # Instructions
            cv2.putText(display, "A/D: frame | J/L: 1s | ,/.: 10s | M: mark cut | Space: play | ESC: done",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(self.window_name, display)

            wait_time = int(1000 / self.fps) if playing else 1
            key = cv2.waitKey(wait_time) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('m'):
                if frame_num not in marked_frames:
                    marked_frames.add(frame_num)
                    self.feedback['missed_cuts'].append(round(current_time, 3))
                    self.feedback['good_cuts'].append(round(current_time, 3))
            elif key == ord(' '):
                playing = not playing
            elif key in [2, ord('a')]:  # Left / A
                playing = False
                frame_num = max(0, frame_num - 1)
            elif key in [3, ord('d')]:  # Right / D
                playing = False
                frame_num = min(self.total_frames - 1, frame_num + 1)
            elif key == ord('j'):  # Back 1s
                playing = False
                frame_num = max(0, frame_num - int(self.fps))
            elif key == ord('l'):  # Forward 1s
                playing = False
                frame_num = min(self.total_frames - 1, frame_num + int(self.fps))
            elif key == ord(','):  # Back 10s
                playing = False
                frame_num = max(0, frame_num - int(self.fps * 10))
            elif key == ord('.'):  # Forward 10s
                playing = False
                frame_num = min(self.total_frames - 1, frame_num + int(self.fps * 10))
            elif playing:
                frame_num = min(self.total_frames - 1, frame_num + 1)
                if frame_num >= self.total_frames - 1:
                    playing = False

    def save_feedback(self, output_file: str):
        """Save annotation feedback."""
        with open(output_file, 'w') as f:
            json.dump(self.feedback, f, indent=2)

    @staticmethod
    def load_cuts(cuts_file: str) -> list:
        """Load cut timestamps from file."""
        with open(cuts_file, 'r') as f:
            data = json.load(f)
        return data.get('cuts', [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.video is not None:
            self.video.release()
        cv2.destroyAllWindows()

def main():
    # File selection
    root = tk.Tk()
    root.withdraw()

    print("[cyan]Select video file...[/cyan]")
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.mkv *.avi")]
    )
    if not video_path:
        return

    print("[cyan]Select cuts JSON file...[/cyan]")
    cuts_file = filedialog.askopenfilename(
        title="Select Cuts JSON File",
        filetypes=[("JSON files", "*.json")]
    )
    if not cuts_file:
        return

    with CutAnnotator(video_path, cuts_file) as annotator:
        # Create output directory
        output_dir = Path("output/annotations")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"feedback_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Review cuts
        total_cuts = len(annotator.cuts)
        reviewed = 0

        for i, cut in enumerate(annotator.cuts, 1):
            # Show progress in window title
            cv2.setWindowTitle(annotator.window_name, 
                f'Cut Review - {i}/{total_cuts} ({(i/total_cuts)*100:.1f}%) - Y (correct) or N (incorrect)')
            
            if not annotator.review_cut(cut):  # ESC pressed
                break
                
            reviewed += 1

        # Browse mode for missed cuts
        print("[cyan]Entering browse mode — scrub through the video to mark missed cuts.[/cyan]")
        print("[cyan]Press ESC when done.[/cyan]")
        cv2.setWindowTitle(annotator.window_name, 'Browse — mark missed cuts (M to mark, ESC to finish)')
        annotator.browse_for_missed()

        # Save results
        annotator.save_feedback(output_file)

        # Show final statistics in a window
        missed = len(annotator.feedback['missed_cuts'])
        stats_text = [
            f"Annotation Complete",
            f"Reviewed: {reviewed}/{total_cuts} detected cuts",
            f"Correct cuts: {len(annotator.feedback['good_cuts']) - missed}",
            f"Incorrect cuts: {len(annotator.feedback['false_positives'])}",
            f"Missed cuts marked: {missed}",
            f"",
            f"Results saved to: {output_file.name}",
            f"",
            f"Press any key to exit"
        ]
        
        # Create stats image
        stats_img = np.zeros((300, 600, 3), dtype=np.uint8)
        for i, text in enumerate(stats_text):
            cv2.putText(stats_img, text, (20, 40 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Annotation Complete', stats_img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()