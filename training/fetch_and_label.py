"""
Fetch SKILL.md content and label via Claude API.

Pipeline:
1. Reads scan-report.json for the full skill list (names)
2. Merges existing label-batch-*.json files for skills we already have content for
3. Fetches SKILL.md for skills we don't have content for yet (via GitHub raw URLs)
4. Labels each through the LLM labeler
5. Outputs training/real-labels.json — master labeled dataset

Usage:
    python training/fetch_and_label.py --scan-report scan-report.json
    python training/fetch_and_label.py --existing-only  # Just label the 159 we already have
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

from llm_label import label_batch


AWESOME_LIST_URL = (
    "https://raw.githubusercontent.com/VoltAgent/awesome-openclaw-skills/main/README.md"
)


def load_existing_skills(training_dir: str = "training") -> list[dict]:
    """Load skills with content from existing label-batch-*.json files."""
    skills = []
    seen = set()
    training_path = Path(training_dir)

    for batch_file in sorted(training_path.glob("label-batch-*.json")):
        try:
            data = json.loads(batch_file.read_text())
            for entry in data:
                name = entry.get("skill_name", "")
                skill_md = entry.get("skill_md", "")
                if name and skill_md.strip() and name not in seen:
                    skills.append({
                        "skill_name": name,
                        "skill_md": skill_md,
                        "author": entry.get("author", "unknown"),
                    })
                    seen.add(name)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: error reading {batch_file}: {e}", file=sys.stderr)

    return skills


def load_scan_report(path: str) -> list[dict]:
    """Load skill names from scan-report.json."""
    data = json.loads(Path(path).read_text())
    results = data.get("results", [])
    return [
        {"skill_name": r["skill_name"], "author": r.get("author", "unknown")}
        for r in results
    ]


def github_skill_url(author: str, skill_name: str) -> str:
    """Construct raw GitHub URL for a skill's SKILL.md."""
    return (
        f"https://raw.githubusercontent.com/{author}/skills/main/"
        f"skills/{skill_name}/SKILL.md"
    )


def fetch_skill_md(
    author: str,
    skill_name: str,
    github_token: str | None = None,
    timeout: int = 15,
) -> str | None:
    """Fetch SKILL.md content from GitHub."""
    url = github_skill_url(author, skill_name)
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        return None
    except requests.RequestException:
        return None


def fetch_missing_skills(
    scan_skills: list[dict],
    existing_names: set[str],
    limit: int = 0,
    delay: float = 0.2,
    github_token: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Fetch SKILL.md for skills not already in existing data."""
    missing = [s for s in scan_skills if s["skill_name"] not in existing_names]
    if limit > 0:
        missing = missing[:limit]

    fetched = []
    for i, skill in enumerate(missing):
        name = skill["skill_name"]
        author = skill.get("author", "openclaw")

        content = fetch_skill_md(author, name, github_token=github_token)
        if content:
            fetched.append({
                "skill_name": name,
                "skill_md": content,
                "author": author,
            })
            if verbose:
                print(f"  [{i+1}/{len(missing)}] Fetched {name} ({len(content)} chars)")
        else:
            if verbose:
                print(f"  [{i+1}/{len(missing)}] MISS {name}")

        time.sleep(delay)

    return fetched


def main():
    parser = argparse.ArgumentParser(description="Fetch and label skills pipeline")
    parser.add_argument(
        "--scan-report", type=str, default="scan-report.json",
        help="Path to scan-report.json"
    )
    parser.add_argument(
        "--output", type=str, default="training/real-labels.json",
        help="Output path for labeled dataset"
    )
    parser.add_argument(
        "--existing-only", action="store_true",
        help="Only label existing skills (no fetching)"
    )
    parser.add_argument(
        "--fetch-limit", type=int, default=0,
        help="Max skills to fetch (0 = all missing)"
    )
    parser.add_argument(
        "--label-model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model for labeling"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls"
    )
    parser.add_argument(
        "--skip-labeling", action="store_true",
        help="Only fetch, don't label (useful for building content dataset first)"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    import os
    github_token = os.environ.get("GITHUB_TOKEN")

    # Step 1: Load existing skills with content
    if verbose:
        print("Loading existing skills from label-batch files...")
    existing = load_existing_skills()
    existing_names = {s["skill_name"] for s in existing}
    if verbose:
        print(f"  Found {len(existing)} skills with content")

    all_skills = list(existing)

    # Step 2: Optionally fetch missing skills
    if not args.existing_only:
        if verbose:
            print(f"\nLoading scan report from {args.scan_report}...")
        scan_skills = load_scan_report(args.scan_report)
        if verbose:
            print(f"  {len(scan_skills)} skills in scan report")
            n_missing = sum(1 for s in scan_skills if s["skill_name"] not in existing_names)
            print(f"  {n_missing} missing content")

        if verbose:
            print("\nFetching missing SKILL.md files...")
        fetched = fetch_missing_skills(
            scan_skills,
            existing_names,
            limit=args.fetch_limit,
            github_token=github_token,
            verbose=verbose,
        )
        all_skills.extend(fetched)
        if verbose:
            print(f"  Fetched {len(fetched)} new skills")

    if verbose:
        print(f"\nTotal skills with content: {len(all_skills)}")

    # Step 3: Label via LLM
    if not args.skip_labeling:
        if verbose:
            print(f"\nStarting LLM labeling with {args.label_model}...")
        results = label_batch(
            all_skills,
            output_path=args.output,
            model=args.label_model,
            delay=args.delay,
            verbose=verbose,
        )
        if verbose:
            # Summary
            label_counts = {}
            for r in results:
                lbl = r.get("llm_label", "UNKNOWN")
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            print(f"\nLabel distribution:")
            for lbl, count in sorted(label_counts.items()):
                print(f"  {lbl}: {count}")
    else:
        # Just save the fetched skills without labels
        output = Path(args.output.replace("real-labels", "skills-with-content"))
        output.write_text(json.dumps(all_skills, indent=2))
        if verbose:
            print(f"Saved {len(all_skills)} skills to {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
