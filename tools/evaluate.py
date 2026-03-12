"""Evaluation harness for cut detection.

Compares detection results against human annotations to compute
precision, recall, and F1 score.

Usage:
    # Evaluate a single detection run against its annotations
    python tools/evaluate.py --detections output/json/cuts_test_*.json \
                             --annotations output/annotations/feedback_*.json

    # Evaluate with custom matching window
    python tools/evaluate.py --detections output/json/cuts_test_*.json \
                             --annotations output/annotations/feedback_*.json \
                             --window 0.3

    # Auto-discover: find all detection/annotation pairs by video ID
    python tools/evaluate.py --auto
"""

import argparse
import json
import glob
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("evaluate")
console = Console()


@dataclass
class EvalResult:
    """Evaluation result for a single video."""
    video_id: str
    total_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_pairs: List[Tuple[float, float]] = field(default_factory=list)
    unmatched_detections: List[float] = field(default_factory=list)
    unmatched_annotations: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def load_detections(path: str) -> Dict:
    """Load a detection JSON file."""
    with open(path) as f:
        return json.load(f)


def load_annotations(path: str) -> Dict:
    """Load an annotation feedback JSON file.

    Handles both simple timestamp lists and range-based annotations.
    Returns normalized lists of good_cuts (as timestamps) and false_positives.
    """
    with open(path) as f:
        data = json.load(f)

    good_cuts = []
    for cut in data.get("good_cuts", []):
        if isinstance(cut, dict):
            # Range-based: use midpoint or original_timestamp
            good_cuts.append(cut.get("original_timestamp", (cut["start"] + cut["end"]) / 2))
        else:
            good_cuts.append(float(cut))

    false_positives = [float(t) for t in data.get("false_positives", [])]

    return {
        "video_id": data.get("video_id", "unknown"),
        "good_cuts": sorted(good_cuts),
        "false_positives": sorted(false_positives),
    }


def match_cuts(detected: List[float],
               good_cuts: List[float],
               false_positives: List[float],
               window: float = 0.5) -> EvalResult:
    """Match detected cuts against annotations.

    A detected cut is a true positive if it falls within `window` seconds
    of an annotated good cut. Each annotation can only match once (greedy,
    closest-first).

    Args:
        detected: List of detected cut timestamps.
        good_cuts: Annotated correct cut timestamps.
        false_positives: Annotated false positive timestamps.
        window: Matching tolerance in seconds.

    Returns:
        EvalResult with precision/recall metrics.
    """
    result = EvalResult(video_id="")

    # All annotated cuts (good + false_positive) form the ground truth
    # good_cuts = real cuts that should be detected (true positives if matched)
    # false_positives = timestamps the annotator marked as NOT real cuts
    #
    # A detection is:
    #   - true positive: matches a good_cut within window
    #   - false positive: matches a false_positive annotation, OR doesn't match anything
    # A good_cut that isn't matched by any detection is a false negative (missed cut)

    remaining_good = list(good_cuts)
    result.total_detected = len(detected)

    for det_time in sorted(detected):
        # Find closest good cut within window
        best_match = None
        best_dist = float("inf")

        for i, gt_time in enumerate(remaining_good):
            dist = abs(det_time - gt_time)
            if dist <= window and dist < best_dist:
                best_match = i
                best_dist = dist

        if best_match is not None:
            gt_time = remaining_good.pop(best_match)
            result.true_positives += 1
            result.matched_pairs.append((det_time, gt_time))
        else:
            result.false_positives += 1
            result.unmatched_detections.append(det_time)

    # Remaining unmatched good cuts are false negatives (missed)
    result.false_negatives = len(remaining_good)
    result.unmatched_annotations = remaining_good

    return result


def find_pairs(detections_dir: str = "output/json",
               annotations_dir: str = "output/annotations") -> List[Tuple[str, str]]:
    """Auto-discover detection/annotation pairs by video ID."""
    det_files = glob.glob(f"{detections_dir}/cuts_*.json")
    ann_files = glob.glob(f"{annotations_dir}/feedback_*.json")

    # Index annotations by video_id
    ann_by_video = {}
    for ann_path in ann_files:
        with open(ann_path) as f:
            data = json.load(f)
        vid = data.get("video_id", "")
        if vid:
            ann_by_video.setdefault(vid, []).append(ann_path)

    pairs = []
    for det_path in sorted(det_files):
        det_data = load_detections(det_path)
        # Extract video ID from video_path
        vid = Path(det_data.get("video_path", "")).stem
        if vid in ann_by_video:
            # Use the latest annotation
            latest_ann = sorted(ann_by_video[vid])[-1]
            pairs.append((det_path, latest_ann))

    return pairs


def print_result(result: EvalResult, verbose: bool = False):
    """Print evaluation result as a rich table."""
    table = Table(title=f"Evaluation: {result.video_id}")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Detected cuts", str(result.total_detected))
    table.add_row("True positives", str(result.true_positives))
    table.add_row("False positives", str(result.false_positives))
    table.add_row("False negatives (missed)", str(result.false_negatives))
    table.add_row("Precision", f"{result.precision:.1%}")
    table.add_row("Recall", f"{result.recall:.1%}")
    table.add_row("F1 Score", f"{result.f1:.1%}")

    console.print(table)

    if verbose and result.unmatched_detections:
        console.print("\n[yellow]Unmatched detections (false positives):[/yellow]")
        for t in result.unmatched_detections:
            console.print(f"  {t:.2f}s")

    if verbose and result.unmatched_annotations:
        console.print("\n[red]Missed cuts (false negatives):[/red]")
        for t in result.unmatched_annotations:
            console.print(f"  {t:.2f}s")


def print_summary(results: List[EvalResult]):
    """Print aggregate summary across all videos."""
    if not results:
        return

    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    table = Table(title="Aggregate Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Videos evaluated", str(len(results)))
    table.add_row("Total true positives", str(total_tp))
    table.add_row("Total false positives", str(total_fp))
    table.add_row("Total false negatives", str(total_fn))
    table.add_row("Precision", f"{precision:.1%}")
    table.add_row("Recall", f"{recall:.1%}")
    table.add_row("F1 Score", f"{f1:.1%}")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate cut detection against annotations")
    parser.add_argument("--detections", nargs="+", help="Detection JSON file(s)")
    parser.add_argument("--annotations", nargs="+", help="Annotation feedback JSON file(s)")
    parser.add_argument("--auto", action="store_true",
                       help="Auto-discover detection/annotation pairs")
    parser.add_argument("--window", type=float, default=0.5,
                       help="Matching window in seconds (default: 0.5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show unmatched detections and missed cuts")

    args = parser.parse_args()

    if args.auto:
        pairs = find_pairs()
        if not pairs:
            console.print("[red]No matching detection/annotation pairs found.[/red]")
            console.print("Run the annotator first:")
            console.print("  python tools/annotate_cuts.py")
            return 1
    elif args.detections and args.annotations:
        pairs = list(zip(args.detections, args.annotations))
    else:
        parser.error("Provide --detections and --annotations, or use --auto")

    results = []

    for det_path, ann_path in pairs:
        det_data = load_detections(det_path)
        ann_data = load_annotations(ann_path)

        detected_cuts = det_data.get("cuts", [])
        video_id = Path(det_data.get("video_path", "")).stem

        result = match_cuts(
            detected=detected_cuts,
            good_cuts=ann_data["good_cuts"],
            false_positives=ann_data["false_positives"],
            window=args.window,
        )
        result.video_id = video_id

        print_result(result, verbose=args.verbose)
        results.append(result)

        console.print(f"  Detection: {det_path}")
        console.print(f"  Annotation: {ann_path}")
        console.print(f"  Window: ±{args.window}s")
        console.print()

    if len(results) > 1:
        print_summary(results)

    return 0


if __name__ == "__main__":
    exit(main())
