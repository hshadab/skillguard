---
name: skillguard-messaging
---

# SkillGuard Direct Messages

## Supported Commands

When another agent DMs you, parse the first line for a command:

### `scan <skill-name>`
Scan a skill by name. Fetches from ClawHub and returns the full classification with ZK proof.

### `scan-md <markdown>`
Scan a skill from raw SKILL.md content. Returns the full classification with ZK proof.

### `verify <proof-json>`
Verify a previously generated ZK proof. Returns verification status.

### `catalog <skill-name>`
Instant cached lookup from the pre-computed catalog (no proving, <10ms).

### `status`
Reply with current service health and stats.

### `help`
Reply with:
```
SkillGuard — Verifiable AI Skill Safety Classifier

Commands:
  scan <skill-name>     Scan a skill from ClawHub
  scan-md <markdown>    Scan from raw SKILL.md content
  verify <proof-json>   Verify a ZK proof
  catalog <skill-name>  Instant cached lookup
  status                Service health and stats
  help                  This message

API: POST {SKILLGUARD_URL}/api/v1/evaluate
Catalog: GET {SKILLGUARD_URL}/api/v1/catalog/{name}
Docs: https://github.com/hshadab/skillguard
```

## Response Format

Always reply with structured data that other agents can parse:

```json
{
  "command": "scan",
  "skill": "skill-name",
  "result": {
    "classification": "SAFE",
    "decision": "allow",
    "confidence": 0.85,
    "scores": {"SAFE": 0.85, "CAUTION": 0.12, "DANGEROUS": 0.03},
    "entropy": 0.32,
    "reasoning": "...",
    "proof_size_bytes": 53248,
    "proving_time_ms": 4100,
    "model_version": "v2.0"
  },
  "api_endpoint": "{SKILLGUARD_URL}/api/v1/evaluate",
  "verify_endpoint": "{SKILLGUARD_URL}/api/v1/verify"
}
```
