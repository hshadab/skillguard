"""
Bulk LLM labeling of OpenClaw skills using the Claude API.

Reads skills with `skill_md` content from JSON, calls Claude with a structured
safety-classification prompt, and returns SAFE | CAUTION | DANGEROUS + reasoning
per skill.

Features:
- Resume-safe: skips already-labeled skills
- Rate limiting with exponential backoff
- Structured JSON output parsing
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

VALID_LABELS = {"SAFE", "CAUTION", "DANGEROUS"}

SYSTEM_PROMPT = """\
You are a security analyst classifying OpenClaw/ClawHub AI agent skills for safety.

Given a SKILL.md file (the full skill definition), classify it into exactly one of three categories:

- SAFE: The skill is a benign utility (calculator, formatter, file converter, documentation helper, etc.). No concerning patterns.
- CAUTION: The skill accesses sensitive resources (APIs, credentials, environment variables, file system) but for legitimate purposes. New/unvetted author, moderate risk.
- DANGEROUS: The skill contains clear risk indicators: credential harvesting, privilege escalation, data exfiltration, reverse shells, obfuscated payloads, persistence mechanisms, or instructions that trick an LLM into leaking secrets.

Important guidelines:
- Skills that use shell commands for legitimate automation (git, npm, docker, etc.) are typically SAFE or CAUTION, not DANGEROUS.
- Skills by established authors with good documentation that access APIs are typically CAUTION.
- Focus on INTENT and CAPABILITY, not just the presence of shell commands.
- A skill that accesses env vars for API keys to call a legitimate service is CAUTION, not DANGEROUS.
- A skill that accesses env vars to exfiltrate them to an external server is DANGEROUS.

Respond with ONLY valid JSON in this exact format:
{"label": "SAFE|CAUTION|DANGEROUS", "reasoning": "One sentence explaining why."}
"""

USER_PROMPT_TEMPLATE = """\
Classify the following SKILL.md:

Skill name: {skill_name}

```
{skill_md}
```
"""


def label_skill(
    client: anthropic.Anthropic,
    skill_name: str,
    skill_md: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 3,
) -> dict:
    """Label a single skill using Claude API.

    Returns dict with keys: label, reasoning.
    Raises on persistent failure.
    """
    # Truncate very long skill_md to stay within context
    if len(skill_md) > 80_000:
        skill_md = skill_md[:80_000] + "\n\n[TRUNCATED]"

    user_msg = USER_PROMPT_TEMPLATE.format(skill_name=skill_name, skill_md=skill_md)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()

            # Parse JSON response
            # Handle potential markdown code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(text)
            label = result.get("label", "").upper()
            reasoning = result.get("reasoning", "")

            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label '{label}', expected one of {VALID_LABELS}")

            return {"label": label, "reasoning": reasoning}

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Failed to parse LLM response for '{skill_name}' after {max_retries} attempts: {e}")

        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 2)  # 4, 8, 16 seconds
            print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue

        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"API error for '{skill_name}': {e}")

    raise RuntimeError(f"Exhausted retries for '{skill_name}'")


def label_batch(
    skills: list[dict],
    output_path: str,
    model: str = "claude-sonnet-4-20250514",
    delay: float = 0.5,
    verbose: bool = True,
) -> list[dict]:
    """Label a batch of skills, writing results incrementally.

    Each skill dict must have 'skill_name' and 'skill_md' keys.
    Results are appended to output_path as they complete.

    Returns list of labeled results.
    """
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    output = Path(output_path)

    # Load existing results for resume
    existing = {}
    if output.exists():
        try:
            data = json.loads(output.read_text())
            if isinstance(data, list):
                for entry in data:
                    existing[entry["skill_name"]] = entry
        except (json.JSONDecodeError, KeyError):
            pass

    results = list(existing.values())
    labeled_names = set(existing.keys())
    skipped = 0
    errors = 0

    for i, skill in enumerate(skills):
        name = skill["skill_name"]
        skill_md = skill.get("skill_md", "")

        if name in labeled_names:
            skipped += 1
            continue

        if not skill_md or len(skill_md.strip()) < 10:
            if verbose:
                print(f"  [{i+1}/{len(skills)}] SKIP {name} (no content)")
            continue

        try:
            result = label_skill(client, name, skill_md, model=model)
            entry = {
                "skill_name": name,
                "llm_label": result["label"],
                "reasoning": result["reasoning"],
                "skill_md": skill_md,
            }
            results.append(entry)
            labeled_names.add(name)

            # Save incrementally
            output.write_text(json.dumps(results, indent=2))

            if verbose:
                print(f"  [{i+1}/{len(skills)}] {result['label']:10s} {name}")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [{i+1}/{len(skills)}] ERROR {name}: {e}", file=sys.stderr)

        # Rate limiting delay
        time.sleep(delay)

    if verbose:
        print(f"\nLabeling complete: {len(results)} labeled, {skipped} skipped, {errors} errors")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bulk LLM labeling of skills")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSON file with skills (must have skill_name and skill_md fields)"
    )
    parser.add_argument(
        "--output", type=str, default="training/real-labels.json",
        help="Output JSON file for labeled data"
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model to use for labeling"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load input
    input_data = json.loads(Path(args.input).read_text())
    if isinstance(input_data, dict) and "skills" in input_data:
        skills = input_data["skills"]
    elif isinstance(input_data, list):
        skills = input_data
    else:
        print(f"ERROR: Unexpected input format", file=sys.stderr)
        return 1

    verbose = not args.quiet
    if verbose:
        print(f"Loaded {len(skills)} skills from {args.input}")
        n_with_content = sum(1 for s in skills if s.get("skill_md", "").strip())
        print(f"  {n_with_content} have skill_md content")

    results = label_batch(
        skills,
        output_path=args.output,
        model=args.model,
        delay=args.delay,
        verbose=verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
