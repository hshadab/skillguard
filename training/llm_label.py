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

- SAFE: Benign utility with no meaningful risk. Includes shell wrappers for well-known tools \
(git, npm, docker, cargo, make) that run predefined commands, formatters, converters, \
documentation helpers, and skills that read 1-2 env vars solely for their own config \
(e.g., an API key for the specific service the skill wraps).
- CAUTION: Accesses sensitive resources for legitimate purposes but with moderate risk. \
Includes skills that read 3+ env vars (especially generic ones like PATH, HOME, SSH_AUTH_SOCK), \
execute user-provided strings as shell commands, make authenticated API calls to third-party \
services, or write to the file system beyond their own workspace. New/unvetted author.
- DANGEROUS: Contains clear attack indicators: download-and-execute patterns (curl|bash), \
reverse shells, credential harvesting combined with exfiltration, privilege escalation \
(sudo, chmod +s), obfuscated payloads (base64 decode + eval), persistence mechanisms \
(crontab, systemd, launchd), or instructions that trick an LLM into leaking secrets.

Boundary rules (SAFE vs CAUTION):
- SAFE: A git wrapper that runs `git clone`, `git pull`, `git push` with hardcoded args.
- SAFE: A skill that reads $OPENAI_API_KEY to call the OpenAI API it was built to wrap.
- CAUTION: A skill that reads $HOME, $PATH, or enumerates environment variables generically.
- CAUTION: A skill that takes a user-supplied string and passes it to `exec()` or `sh -c`.

Boundary rules (CAUTION vs DANGEROUS):
- CAUTION: A skill that runs `curl` to fetch data from a known API and saves it locally.
- DANGEROUS: A skill that runs `curl <url> | bash` or downloads a script then executes it.
- DANGEROUS: A skill that reads credentials AND sends them to an external endpoint.

Focus on INTENT and CAPABILITY, not just the presence of keywords.

## Examples

Skill: "git-auto-commit" — runs `git add .`, `git commit -m`, `git push` with user-provided message.
{"label": "SAFE", "reasoning": "Shell wrapper for standard git commands with user-supplied commit message; no credential access or network exfiltration."}

Skill: "multi-api-hub" — reads OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, AWS_ACCESS_KEY; routes user queries to multiple LLM providers.
{"label": "CAUTION", "reasoning": "Reads 4+ API keys for legitimate multi-provider routing, but broad credential access increases blast radius if compromised."}

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
