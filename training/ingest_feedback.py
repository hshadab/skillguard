"""
Feedback ingestion pipeline for SkillGuard.

Reads feedback.jsonl, groups disputes by skill, re-labels with LLM labeler
using dispute context, outputs candidates for human review.
NOT auto-retrain — human approves before entering training set.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

from llm_label import label_skill

DEFAULT_FEEDBACK_PATH = "/var/data/skillguard-cache/feedback.jsonl"
DEFAULT_OUTPUT = "training/feedback-candidates.json"
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def load_feedback(feedback_path: str, quiet: bool = False) -> list[dict]:
    """Load feedback entries from a JSONL file.

    Each line is a JSON object with keys:
      timestamp, skill_name, reported_classification,
      expected_classification, comment
    """
    path = Path(feedback_path)
    if not path.exists():
        if not quiet:
            print(f"Feedback file not found: {feedback_path}", file=sys.stderr)
        return []

    entries = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                if not quiet:
                    print(f"  WARN: skipping malformed line {lineno}: {e}", file=sys.stderr)
    return entries


def group_by_skill(entries: list[dict]) -> dict[str, list[dict]]:
    """Group feedback entries by skill_name, preserving order."""
    grouped = defaultdict(list)
    for entry in entries:
        skill_name = entry.get("skill_name", "")
        if skill_name:
            grouped[skill_name].append(entry)
    return dict(grouped)


def build_dispute_context(disputes: list[dict]) -> str:
    """Build a human-readable summary of disputes for a single skill.

    This context is appended to the skill_md when re-labeling so the LLM
    can consider the community feedback.
    """
    lines = [
        "",
        "--- DISPUTE CONTEXT (community feedback) ---",
        f"Number of disputes: {len(disputes)}",
        "",
    ]
    for i, d in enumerate(disputes, 1):
        reported = d.get("reported_classification", "unknown")
        expected = d.get("expected_classification", "unknown")
        comment = d.get("comment") or "(no comment)"
        timestamp = d.get("timestamp", "unknown")
        lines.append(f"Dispute #{i} ({timestamp}):")
        lines.append(f"  Model said: {reported}")
        lines.append(f"  User expected: {expected}")
        lines.append(f"  Comment: {comment}")
        lines.append("")
    lines.append("--- END DISPUTE CONTEXT ---")
    return "\n".join(lines)


def find_skill_md(skill_name: str, label_files: list[Path]) -> str | None:
    """Search existing label files for the skill_md of a given skill.

    Returns the skill_md text if found, or None.
    """
    for label_file in label_files:
        if not label_file.exists():
            continue
        try:
            data = json.loads(label_file.read_text())
            if isinstance(data, list):
                for entry in data:
                    if entry.get("skill_name") == skill_name:
                        md = entry.get("skill_md", "")
                        if md and len(md.strip()) > 10:
                            return md
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def discover_label_files() -> list[Path]:
    """Find all label/batch JSON files in the training directory."""
    training_dir = Path(__file__).parent
    files = []
    # Check real-labels.json first (most likely to have content)
    real_labels = training_dir / "real-labels.json"
    if real_labels.exists():
        files.append(real_labels)
    # Then check label-batch-*.json and labels-batch-*.json
    for pattern in ["label-batch-*.json", "labels-batch-*.json"]:
        files.extend(sorted(training_dir.glob(pattern)))
    return files


def ingest_feedback(
    feedback_path: str,
    output_path: str,
    model: str,
    quiet: bool = False,
) -> int:
    """Main ingestion pipeline.

    Returns the number of candidates generated.
    """
    verbose = not quiet

    # Step 1: Load feedback
    entries = load_feedback(feedback_path, quiet=quiet)
    if not entries:
        if verbose:
            print("No feedback entries found. Nothing to do.")
        return 0

    if verbose:
        print(f"Loaded {len(entries)} feedback entries from {feedback_path}")

    # Step 2: Group by skill
    grouped = group_by_skill(entries)
    if verbose:
        print(f"Found disputes for {len(grouped)} unique skill(s)")

    # Step 3: Discover existing label files for skill_md lookup
    label_files = discover_label_files()
    if verbose:
        print(f"Found {len(label_files)} label file(s) for skill_md lookup")

    # Step 4: Re-label each disputed skill with dispute context
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic SDK not installed. Run: pip install anthropic", file=sys.stderr)
        return 0

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    candidates = []
    skipped = 0
    errors = 0

    for skill_name, disputes in grouped.items():
        # Find the skill_md for this skill
        skill_md = find_skill_md(skill_name, label_files)
        if not skill_md:
            if verbose:
                print(f"  SKIP {skill_name}: no skill_md found in label files")
            skipped += 1
            continue

        # Build dispute context and append to skill_md for re-labeling
        dispute_context = build_dispute_context(disputes)
        augmented_md = skill_md + "\n" + dispute_context

        try:
            result = label_skill(client, skill_name, augmented_md, model=model)

            # Collect the original model label from the most recent dispute
            original_label = disputes[-1].get("reported_classification", "unknown")

            candidate = {
                "skill_name": skill_name,
                "original_label": original_label,
                "relabeled": result["label"],
                "reasoning": result["reasoning"],
                "num_disputes": len(disputes),
                "dispute_summary": [
                    {
                        "reported": d.get("reported_classification"),
                        "expected": d.get("expected_classification"),
                        "comment": d.get("comment"),
                        "timestamp": d.get("timestamp"),
                    }
                    for d in disputes
                ],
                "label_changed": result["label"] != original_label.upper(),
            }
            candidates.append(candidate)

            if verbose:
                changed = "CHANGED" if candidate["label_changed"] else "same"
                print(
                    f"  {skill_name}: {original_label} -> {result['label']} "
                    f"({changed}, {len(disputes)} dispute(s))"
                )

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  ERROR {skill_name}: {e}", file=sys.stderr)

    # Step 5: Write candidates for human review
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "feedback_path": feedback_path,
            "total_feedback_entries": len(entries),
            "unique_skills_disputed": len(grouped),
            "candidates_generated": len(candidates),
            "skipped_no_skill_md": skipped,
            "errors": errors,
            "model": model,
            "note": (
                "These are CANDIDATES for human review. "
                "Do NOT auto-add to training set — verify each label "
                "before merging into training/real-labels.json."
            ),
        },
        "candidates": candidates,
    }

    output.write_text(json.dumps(output_data, indent=2))

    if verbose:
        print()
        print(f"Results: {len(candidates)} candidates, {skipped} skipped, {errors} errors")
        print(f"Written to: {output_path}")
        print()
        changed_count = sum(1 for c in candidates if c["label_changed"])
        print(f"  {changed_count} label(s) changed after considering dispute context")
        print(f"  {len(candidates) - changed_count} label(s) unchanged (model stands by original)")
        print()
        print("NEXT STEP: Review candidates and manually merge approved labels")
        print("           into training/real-labels.json before retraining.")

    return len(candidates)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest feedback disputes and generate re-label candidates for human review"
    )
    parser.add_argument(
        "--feedback-path",
        type=str,
        default=DEFAULT_FEEDBACK_PATH,
        help=f"Path to feedback.jsonl (default: {DEFAULT_FEEDBACK_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for candidates JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model for re-labeling (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    n = ingest_feedback(
        feedback_path=args.feedback_path,
        output_path=args.output,
        model=args.model,
        quiet=args.quiet,
    )

    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
