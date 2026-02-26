"""
Generate a prioritized review queue for human verification of LLM labels.

Loads training/real-labels.json, runs `skillguard extract-features` and
`skillguard check` on each skill to get the MLP model prediction, then
outputs a sorted queue prioritized by:
  1. All DANGEROUS-labeled skills (most impactful to verify)
  2. MLP/LLM disagreements (model confusion indicates possible label error)
  3. Very short LLM reasoning (low labeler confidence)

Usage:
    python training/generate_review_queue.py
    python training/generate_review_queue.py --labels training/real-labels.json \
        --output training/review-queue.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_model_prediction(
    bin_path: str, skill_md: str
) -> dict | None:
    """Run skillguard check on a skill_md and return classification + scores."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp.write(skill_md)
            tmp_path = tmp.name

        result = subprocess.run(
            [bin_path, "check", "--input", tmp_path, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        Path(tmp_path).unlink(missing_ok=True)

        if result.returncode not in (0, 1, 2):
            return None
        data = json.loads(result.stdout.strip())
        evaluation = data.get("evaluation", data)
        return {
            "classification": evaluation.get("classification", ""),
            "scores": evaluation.get("scores", {}),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        Path(tmp_path).unlink(missing_ok=True)
        return None


def compute_priority(entry: dict) -> tuple[int, int, int]:
    """Compute sort key for review priority (lower = higher priority).

    Returns (tier, sub_priority, index) where:
      tier 0: DANGEROUS-labeled skills
      tier 1: MLP/LLM label disagreements
      tier 2: Short reasoning (< 50 chars)
      tier 3: Everything else
    """
    llm_label = entry.get("llm_label", "")
    model_label = entry.get("model_prediction", "")
    reasoning = entry.get("reasoning", "")

    # Tier 0: All DANGEROUS labels — most impactful to verify
    if llm_label == "DANGEROUS":
        return (0, 0, 0)

    # Tier 1: MLP/LLM disagreement — possible label error
    if model_label and model_label != llm_label:
        return (1, 0, 0)

    # Tier 2: Short reasoning — low labeler confidence
    if len(reasoning) < 50:
        return (2, 0, 0)

    # Tier 3: Everything else
    return (3, 0, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate prioritized human review queue for label verification"
    )
    parser.add_argument(
        "--labels", type=str, default="training/real-labels.json",
        help="Path to LLM-labeled dataset",
    )
    parser.add_argument(
        "--output", type=str, default="training/review-queue.json",
        help="Output path for review queue",
    )
    parser.add_argument(
        "--bin", type=str, default="target/release/skillguard",
        help="Path to skillguard binary",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    # Find the skillguard binary
    bin_path = shutil.which("skillguard") or args.bin
    if not Path(bin_path).exists():
        alt = "target/debug/skillguard"
        if Path(alt).exists():
            bin_path = alt
        else:
            print(
                f"ERROR: skillguard binary not found at '{bin_path}'. "
                "Build it first: cargo build --release",
                file=sys.stderr,
            )
            return 1

    # Load labeled data
    data = json.loads(Path(args.labels).read_text())
    if not isinstance(data, list):
        print("ERROR: Expected JSON array", file=sys.stderr)
        return 1

    if verbose:
        print(f"Loaded {len(data)} labeled skills from {args.labels}")

    queue = []
    errors = 0

    for i, entry in enumerate(data):
        skill_name = entry.get("skill_name", "")
        llm_label = entry.get("llm_label", "")
        reasoning = entry.get("reasoning", "")
        skill_md = entry.get("skill_md", "")

        if not skill_md.strip():
            continue

        # Get model prediction
        prediction = get_model_prediction(bin_path, skill_md)
        if prediction is None:
            errors += 1
            model_label = ""
            model_scores = {}
        else:
            model_label = prediction["classification"]
            model_scores = prediction["scores"]

        review_entry = {
            "skill_name": skill_name,
            "llm_label": llm_label,
            "reasoning": reasoning,
            "model_prediction": model_label,
            "model_scores": model_scores,
            "disagreement": bool(model_label and model_label != llm_label),
            "short_reasoning": len(reasoning) < 50,
        }

        queue.append(review_entry)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)}...")

    # Sort by priority
    queue.sort(key=compute_priority)

    # Add priority tier labels for readability
    for entry in queue:
        tier = compute_priority(entry)[0]
        entry["review_tier"] = [
            "DANGEROUS_LABEL",
            "MLP_LLM_DISAGREEMENT",
            "SHORT_REASONING",
            "STANDARD",
        ][tier]

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(queue, indent=2))

    if verbose:
        # Summary
        tier_counts = {}
        for entry in queue:
            t = entry["review_tier"]
            tier_counts[t] = tier_counts.get(t, 0) + 1

        print(f"\nReview queue: {len(queue)} skills -> {args.output}")
        print(f"  Extraction errors: {errors}")
        print(f"\nPriority tiers:")
        for tier_name in ["DANGEROUS_LABEL", "MLP_LLM_DISAGREEMENT", "SHORT_REASONING", "STANDARD"]:
            count = tier_counts.get(tier_name, 0)
            if count > 0:
                print(f"  {tier_name}: {count}")

        n_disagree = sum(1 for e in queue if e["disagreement"])
        print(f"\nTotal disagreements: {n_disagree}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
