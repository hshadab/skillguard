"""
Stratified sampling of unlabeled skills from scan-report.json.

Samples proportionally from each category field, and within each category
samples across confidence bands (low/medium/high) for diversity.
Excludes skills already present in the existing labeled dataset.

Usage:
    python training/stratified_sample.py \
        --scan-report scan-report.json \
        --existing training/real-labels.json \
        --target-count 500 \
        --output training/stratified-candidates.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load_existing_names(path: str) -> set[str]:
    """Load skill names from existing labeled dataset."""
    p = Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return {e.get("skill_name", "") for e in data} - {""}
    return set()


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling of unlabeled skills for labeling"
    )
    parser.add_argument(
        "--scan-report", type=str, default="scan-report.json",
        help="Path to scan-report.json",
    )
    parser.add_argument(
        "--existing", type=str, default="training/real-labels.json",
        help="Path to existing labeled dataset (skills here are excluded)",
    )
    parser.add_argument(
        "--target-count", type=int, default=500,
        help="Total number of skills to sample",
    )
    parser.add_argument(
        "--output", type=str, default="training/stratified-candidates.json",
        help="Output path for sampled scan-report subset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    verbose = not args.quiet

    # Load scan report
    data = json.loads(Path(args.scan_report).read_text())
    results = data.get("results", [])

    # Exclude already-labeled skills
    existing = load_existing_names(args.existing)
    candidates = [r for r in results if r["skill_name"] not in existing]
    if verbose:
        print(f"Scan report: {len(results)} total, {len(candidates)} unlabeled")

    # Group by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        cat = r.get("category", "unknown") or "unknown"
        by_category[cat].append(r)

    if verbose:
        print(f"Categories: {len(by_category)}")
        for cat, skills in sorted(by_category.items()):
            print(f"  {cat}: {len(skills)}")

    # Compute per-category quota (proportional to category size)
    total_candidates = len(candidates)
    target = min(args.target_count, total_candidates)

    sampled = []
    for cat, skills in by_category.items():
        quota = max(1, round(target * len(skills) / total_candidates))

        # Split into confidence bands based on max score
        bands: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}
        for s in skills:
            scores = s.get("scores", {})
            max_score = max(scores.values()) if isinstance(scores, dict) and scores else 0.0
            if max_score < 0.5:
                bands["low"].append(s)
            elif max_score < 0.8:
                bands["medium"].append(s)
            else:
                bands["high"].append(s)

        # Sample evenly across bands, then fill remainder from largest band
        per_band = max(1, quota // 3)
        cat_sampled = []
        for band_name in ("low", "medium", "high"):
            pool = bands[band_name]
            n = min(per_band, len(pool))
            cat_sampled.extend(random.sample(pool, n))

        # Fill remaining quota from all skills in category
        remaining = quota - len(cat_sampled)
        if remaining > 0:
            already = {s["skill_name"] for s in cat_sampled}
            pool = [s for s in skills if s["skill_name"] not in already]
            n = min(remaining, len(pool))
            cat_sampled.extend(random.sample(pool, n))

        sampled.extend(cat_sampled)

    # Trim to exact target count if over-sampled due to rounding
    if len(sampled) > target:
        sampled = random.sample(sampled, target)

    # Output in scan-report format so fetch_and_label.py can consume it
    output_data = {"results": sampled}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output_data, indent=2))

    if verbose:
        print(f"\nSampled {len(sampled)} skills -> {args.output}")
        # Show distribution
        cat_counts: dict[str, int] = defaultdict(int)
        for s in sampled:
            cat_counts[s.get("category", "unknown") or "unknown"] += 1
        for cat, count in sorted(cat_counts.items()):
            print(f"  {cat}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
