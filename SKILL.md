---
name: skillguard
author: hshadab
description: Verifiable AI skill safety classifier — evaluate any agent skill for safety with a cryptographic ZK proof
version: 2.1.0
---

# SkillGuard

A safety classifier for AI agent skills, powered by zero-knowledge machine learning (ZKML).

## What it does

SkillGuard evaluates agent skills and classifies them as **SAFE**, **CAUTION**, or **DANGEROUS**. Every classification includes a cryptographic zero-knowledge proof that the neural network produced the stated result — no trust required.

## MCP Tool

**`skillguard_evaluate`** — Classify a skill by name or by its full SKILL.md content.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `skill_name` | string | Skill name for minimal evaluation |
| `skill_md` | string | Full SKILL.md markdown content for richer feature extraction |

Provide one or the other. If both are given, `skill_md` takes priority.

### Example

Evaluate a skill by name:

```json
{
  "skill_name": "web-search"
}
```

Evaluate raw skill content:

```json
{
  "skill_md": "---\nname: my-skill\n---\n# My Skill\n\nRuns `echo hello` to greet the user."
}
```

### Response

Returns classification, decision, confidence, per-class scores, reasoning, raw model logits, entropy, model hash, and a verifiable ZK proof bundle.

## How it works

1. Extracts 45 security-relevant features from the skill (commands, file types, obfuscation patterns, metadata signals, cross-features)
2. Runs a 3-layer MLP (4,979 parameters) using fixed-point integer arithmetic inside a ZK circuit
3. Generates a Jolt/Dory SNARK proof that the inference was computed correctly
4. Returns the classification along with the proof for independent verification

## Setup

```bash
claude mcp add skillguard -- cargo run --release --bin skillguard-mcp
```

Or install from source:

```bash
cargo install --git https://github.com/hshadab/skillguard --bin skillguard-mcp
```
