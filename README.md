# SkillGuard

Standalone skill safety classifier for [OpenClaw](https://openclaw.org)/[ClawHub](https://clawhub.ai) skills. Extracted from [ClawGuard](https://github.com/hshadab/clawguard) as a self-contained service — no zero-knowledge proofs, no receipts, no other guardrail models. Just the classifier, an HTTP server, and a ClawHub client.

## What This Does (Plain English)

OpenClaw is a framework for building AI agent "skills" — small packages of instructions and scripts that tell an AI what to do. Anyone can publish a skill to ClawHub, the public skill registry. That's useful, but it also means someone could publish a skill that steals your API keys, opens a backdoor on your machine, or exfiltrates data to a remote server.

SkillGuard is a safety scanner that checks skills before you install them. It works like an antivirus for AI agent plugins.

When you submit a skill (either as raw data or by name from ClawHub), SkillGuard:

1. **Extracts 22 features** from the skill's content — things like "does it call `exec()`?", "does it access environment variables?", "does it contain reverse shell patterns?", "does it try to download executables?", "does its instructions tell the AI to leak your secrets?"
2. **Feeds those features into a small neural network** (a 3-layer MLP with 1,924 parameters) that was trained on labeled examples of safe, suspicious, and malicious skills.
3. **Returns a classification** — one of SAFE, CAUTION, DANGEROUS, or MALICIOUS — along with a decision (allow, flag, or deny), a confidence score, and a human-readable explanation.

The classifier runs entirely locally. No data is sent to any external service (other than ClawHub if you use the evaluate-by-name endpoint). The model weights are embedded directly in the binary — there's no model file to download or manage.

Typical use cases:

- **CI gating**: Block dangerous skills from being installed in production pipelines.
- **Pre-install hooks**: Wire into the OpenClaw CLI so every `clawhub install` gets scanned first.
- **Registry scanning**: Run as a service that continuously evaluates newly published skills.
- **Manual review**: Point it at a SKILL.md file on disk and get a safety report.

## Safety Classifications

| Classification | Decision | Meaning |
|---|---|---|
| **SAFE** | allow | No concerning patterns detected |
| **CAUTION** | allow | Minor concerns noted (e.g., API key references in a legitimate API skill) |
| **DANGEROUS** | deny | Significant risk: credential exposure, excessive permissions, privilege escalation |
| **MALICIOUS** | deny | Active malware indicators: reverse shells, obfuscated payloads, data exfiltration |

Low-confidence DANGEROUS/MALICIOUS results produce a **flag** decision instead of a hard deny, letting a human make the final call.

## Features Analyzed

The classifier examines 22 features extracted from a skill's SKILL.md file, bundled scripts, and registry metadata:

| # | Feature | What it measures |
|---|---|---|
| 0 | `shell_exec_count` | Calls to `exec()`, `spawn()`, `system()`, `subprocess.run()`, etc. |
| 1 | `network_call_count` | Uses of `fetch()`, `curl`, `axios`, `requests.get()`, etc. |
| 2 | `fs_write_count` | File writes via `writeFile()`, `>` redirect, `open('w')`, etc. |
| 3 | `env_access_count` | Reads from `process.env`, `os.environ`, `$ENV_VAR`, `.env` files |
| 4 | `credential_patterns` | Mentions of `api_key`, `password`, `secret`, `token`, `Bearer` in SKILL.md |
| 5 | `external_download` | Links to `.exe`, `.sh`, `.zip` downloads or `curl -O` patterns |
| 6 | `obfuscation_score` | Use of `eval()`, `atob()`, `base64.b64decode()`, `new Function()`, dynamic imports |
| 7 | `privilege_escalation` | `sudo`, `chmod +x`, `chmod 777`, `chown root` |
| 8 | `persistence_mechanisms` | `crontab`, `systemctl enable`, `@reboot`, `.bashrc` modification |
| 9 | `data_exfiltration_patterns` | HTTP POST to external URLs, webhook patterns, `curl --data` |
| 10 | `skill_md_line_count` | Size of the SKILL.md instructions |
| 11 | `script_file_count` | Number of bundled script files |
| 12 | `dependency_count` | `npm install`, `pip install`, `require()`, `import` statements |
| 13 | `author_account_age_days` | How long the author's ClawHub account has existed |
| 14 | `author_skill_count` | Total skills published by the author |
| 15 | `stars` | Registry star count (log-scaled) |
| 16 | `downloads` | Registry download count (log-scaled) |
| 17 | `has_virustotal_report` | Whether a VirusTotal report was provided |
| 18 | `vt_malicious_flags` | Number of VirusTotal engines flagging the skill |
| 19 | `password_protected_archives` | Bundled `.zip`/`.rar`/`.7z` files with password mentions |
| 20 | `reverse_shell_patterns` | `nc -e`, `/dev/tcp/`, `bash -i >&`, `socat`, `mkfifo` |
| 21 | `llm_secret_exposure` | SKILL.md instructions that tell the AI to output/pass/send secrets |

## Quickstart

### Build from source

```bash
git clone https://github.com/hshadab/skillguard.git
cd skillguard
cargo build --release
```

### Run the server

```bash
# Default: 0.0.0.0:8080, 60 req/min rate limit
cargo run -- serve

# Custom bind address and rate limit
cargo run -- serve --bind 127.0.0.1:3000 --rate-limit 120
```

### Check a skill locally

```bash
# From a SKILL.md file
cargo run -- check --input path/to/SKILL.md

# From a JSON skill definition (with JSON output)
cargo run -- check --input skill.json --format json

# With an optional VirusTotal report
cargo run -- check --input skill.json --vt-report vt-report.json
```

The `check` command exits with code 0 for allow/flag decisions and code 1 for deny.

## API Endpoints

### `GET /health`

Health check. Returns server status, version, model hash, and uptime.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_hash": "sha256:abc123...",
  "uptime_seconds": 3600
}
```

### `POST /api/v1/evaluate`

Evaluate a skill from full skill data.

```bash
curl -X POST http://localhost:8080/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "skill": {
      "name": "my-skill",
      "version": "1.0.0",
      "author": "someone",
      "description": "Does something",
      "skill_md": "# My Skill\n\nInstructions here.",
      "scripts": [],
      "files": []
    }
  }'
```

### `POST /api/v1/evaluate/name`

Evaluate a skill by its ClawHub registry name. SkillGuard fetches the skill data from the ClawHub API automatically.

```bash
curl -X POST http://localhost:8080/api/v1/evaluate/name \
  -H 'Content-Type: application/json' \
  -d '{"skill": "weather-helper"}'

# With a specific version
curl -X POST http://localhost:8080/api/v1/evaluate/name \
  -H 'Content-Type: application/json' \
  -d '{"skill": "weather-helper", "version": "1.0.0"}'
```

### Response format

All evaluation endpoints return the same flat JSON structure:

```json
{
  "success": true,
  "evaluation": {
    "skill_name": "weather-helper",
    "classification": "SAFE",
    "decision": "allow",
    "confidence": 0.85,
    "scores": {
      "SAFE": 0.72,
      "CAUTION": 0.18,
      "DANGEROUS": 0.07,
      "MALICIOUS": 0.03
    },
    "reasoning": "No concerning patterns detected"
  },
  "processing_time_ms": 3
}
```

On error:

```json
{
  "success": false,
  "error": "Failed to fetch skill 'nonexistent': HTTP 404",
  "processing_time_ms": 120
}
```

### `GET /stats`

Usage statistics since server start.

```bash
curl http://localhost:8080/stats
```

```json
{
  "uptime_seconds": 7200,
  "model_hash": "sha256:abc123...",
  "requests": { "total": 150, "errors": 3 },
  "classifications": { "safe": 120, "caution": 15, "dangerous": 10, "malicious": 5 },
  "decisions": { "allow": 135, "deny": 12, "flag": 3 },
  "endpoints": { "evaluate": 50, "evaluate_by_name": 95, "stats": 5 }
}
```

## Docker

```bash
# Build
docker build -t skillguard .

# Run
docker run -p 8080:8080 skillguard

# Custom rate limit
docker run -p 8080:8080 skillguard /app/skillguard serve --bind 0.0.0.0:8080 --rate-limit 120
```

The image is a two-stage build (Rust builder + Debian slim runtime). No RISC-V toolchain needed — there's no ZK proving involved.

### Deploy on Render

1. Push to GitHub.
2. Create a new Web Service on Render.
3. Set the build command to `docker build -t skillguard .`
4. Set the start command to `/app/skillguard serve --bind 0.0.0.0:8080 --rate-limit 30`
5. Expose port 8080.

## Pre-Install Hook

An example shell script is included at `examples/skillguard-install-hook.sh` for wiring SkillGuard into the OpenClaw CLI install flow. It supports two modes:

**Hosted API mode** (recommended for teams):
```bash
SKILLGUARD_API_URL=https://your-skillguard.onrender.com \
  ./examples/skillguard-install-hook.sh weather-helper
```

**Local binary mode** (requires `skillguard` and `clawhub` in PATH):
```bash
./examples/skillguard-install-hook.sh weather-helper 1.0.0
```

The hook:
- Blocks skills classified as DANGEROUS or MALICIOUS (exit 1).
- Prompts the user interactively for flagged skills.
- Proceeds automatically for SAFE/CAUTION skills (exit 0).

## Architecture

```
src/
├── lib.rs         classify() and model_hash() entry points
├── main.rs        CLI: serve and check subcommands
├── model.rs       Embedded 1,924-param MLP (22→32→32→4, ReLU activations)
├── skill.rs       Skill struct, SkillFeatures, classification/decision types
├── patterns.rs    22 compiled regex pattern sets for feature extraction
├── scores.rs      ClassScores with softmax normalization
├── clawhub.rs     ClawHub registry API client (async, reqwest-based)
└── server.rs      Axum HTTP server with rate limiting and access logging
```

### Model details

- **Architecture**: 3-layer MLP — 22 input features, two 32-neuron hidden layers with ReLU, 4 output classes
- **Parameters**: 1,924 total (22×32 + 32 + 32×32 + 32 + 32×4 + 4)
- **Arithmetic**: Fixed-point at scale=7 (values multiplied by 128)
- **Inference**: Uses [onnx-tracer](https://github.com/ICME-Lab/jolt-atlas/tree/main/onnx-tracer) from the ICME Lab jolt-atlas project
- **Deterministic hash**: Model weights are embedded in the binary; `model_hash()` produces a stable SHA-256 digest for auditability

### Access log

Every evaluation is logged to `skillguard-access.jsonl` (configurable via `--access-log`):

```json
{"timestamp":"2026-02-10T12:00:00Z","endpoint":"evaluate_by_name","skill_name":"weather-helper","classification":"SAFE","decision":"allow","confidence":0.85,"processing_time_ms":3}
```

## Relationship to ClawGuard

SkillGuard is a focused extraction from [ClawGuard](https://github.com/hshadab/clawguard). ClawGuard is a broader system that gates all OpenClaw agent actions through ONNX guardrail models with zero-knowledge proof verification. It includes multiple models (action gatekeeper, PII shield, scope guard, policy rules), ZK proof generation with JOLT Atlas, cryptographic receipts, and more.

SkillGuard strips all of that away and keeps only the skill safety classifier — making it faster to build, simpler to deploy, and easier to audit.

**What was removed**: `zkml-jolt-core`, `jolt-core`, `ark-bn254`, `ark-serialize`, `base64`, `dirs`, `toml`, the proving pipeline, the receipt system, the enforcement engine, and all non-skill models.

**What was kept**: The 22-feature skill classifier, the regex pattern library, the ClawHub API client, and the HTTP server (rewritten for flat JSON responses).

## License

MIT
