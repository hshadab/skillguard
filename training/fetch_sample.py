#!/usr/bin/env python3
"""Fetch SKILL.md content for the 200 sampled skills from ClawHub."""

import json
import time
import urllib.request
import urllib.error
import urllib.parse
import sys
import os

CLAWHUB_API = "https://clawhub.ai/api/v1"
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "..", "label-sample.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "label-sample-with-content.json")


def fetch_skill_md(slug: str) -> str | None:
    """Fetch SKILL.md content for a skill slug."""
    url = f"{CLAWHUB_API}/skills/{urllib.parse.quote(slug, safe='')}/file?path=SKILL.md"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  FAILED {slug}: {e}", file=sys.stderr)
        return None


def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    print(f"Fetching SKILL.md for {len(sample)} skills...")

    # Resume support: load existing progress if any
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        fetched = {r["skill_name"] for r in existing if r.get("skill_md")}
        print(f"  Resuming: {len(fetched)} already fetched")
    else:
        existing = []
        fetched = set()

    results = list(existing)
    new_count = 0

    for i, skill in enumerate(sample):
        name = skill["skill_name"]
        if name in fetched:
            continue

        print(f"  [{i+1}/{len(sample)}] {name}...", end=" ", flush=True)
        md = fetch_skill_md(name)

        entry = dict(skill)
        entry["skill_md"] = md or ""
        entry["fetch_ok"] = md is not None
        results.append(entry)

        if md:
            print(f"OK ({len(md)} chars)")
            new_count += 1
        else:
            print("FAILED")

        time.sleep(0.3)

        # Save progress every 20 skills
        if new_count % 20 == 0 and new_count > 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r.get("fetch_ok"))
    print(f"\nDone: {ok} fetched, {len(results) - ok} failed")


if __name__ == "__main__":
    main()
