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

## Classification Rules

### DANGEROUS — clear attack indicators
Classify as DANGEROUS if ANY of these are present:
- Download-and-execute patterns (curl|bash, wget+exec, fetch-then-run)
- Reverse shells or bind shells
- Credential harvesting combined with exfiltration to external endpoints
- Privilege escalation (sudo, chmod +s, setuid)
- Obfuscated payloads (base64 decode + eval, hex-encoded shellcode)
- Persistence mechanisms (crontab, systemd unit, launchd plist)
- Instructions that trick an LLM into leaking secrets or ignoring safety guardrails

### CAUTION — any of these 6 triggers (otherwise SAFE)
Classify as CAUTION if ANY of these are true:
1. Executes shell commands (subprocess, os.system, exec, child_process, backtick) — even benign ones
2. Makes outbound network requests (HTTP, DNS, sockets) beyond the declared API it wraps
3. Reads or writes files outside its own directory scope
4. Accesses environment variables that contain credentials or secrets (not mere config like PORT or NODE_ENV)
5. Requests permissions beyond what the description advertises
6. Has no source code available for inspection (binary-only or obfuscated)

### SAFE — benign utility
Classify as SAFE only when NONE of the CAUTION triggers above apply. Includes:
- Pure formatters, converters, linters, documentation helpers
- Shell wrappers for well-known tools (git, npm, docker, cargo, make) that run \
predefined commands with hardcoded arguments and do NOT execute arbitrary user strings
- Skills that read a single API key solely for the specific service they wrap \
(e.g., $OPENAI_API_KEY to call the OpenAI API) with no other credential access

## Boundary Examples

SAFE vs CAUTION:
- SAFE: A git wrapper that runs `git clone`, `git pull`, `git push` with hardcoded args.
- SAFE: A markdown-to-HTML converter that reads stdin and writes stdout.
- CAUTION: A git wrapper that additionally runs user-provided strings via `sh -c`.
- CAUTION: A skill that reads $HOME, $SSH_AUTH_SOCK, or enumerates env vars generically.
- CAUTION: A skill that uses `subprocess.run()` to invoke a linter (trigger #1: shell command).
- CAUTION: A skill that fetches data from a third-party API not mentioned in its description.

CAUTION vs DANGEROUS:
- CAUTION: A skill that runs `curl` to fetch data from a known API and saves it locally.
- DANGEROUS: A skill that runs `curl <url> | bash` or downloads a script then executes it.
- DANGEROUS: A skill that reads credentials AND sends them to an external endpoint.

Focus on INTENT and CAPABILITY, not just the presence of keywords. When in doubt \
between SAFE and CAUTION, prefer CAUTION.

## Examples

Skill: "git-auto-commit" — runs `git add .`, `git commit -m`, `git push` with user-provided message.
{"label": "SAFE", "reasoning": "Shell wrapper for standard git commands with hardcoded operations; no credential access or network exfiltration."}

Skill: "eslint-runner" — runs `npx eslint` via subprocess on user-specified files.
{"label": "CAUTION", "reasoning": "Executes shell commands via subprocess (trigger #1), even though the command is a well-known linter."}

Skill: "multi-api-hub" — reads OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, AWS_ACCESS_KEY; routes user queries to multiple LLM providers.
{"label": "CAUTION", "reasoning": "Accesses multiple credential-bearing environment variables (trigger #4) for multi-provider routing."}

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
