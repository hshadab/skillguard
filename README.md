# SkillGuard

[![CI](https://github.com/hshadab/skillguard/actions/workflows/ci.yml/badge.svg)](https://github.com/hshadab/skillguard/actions/workflows/ci.yml)

Standalone skill safety classifier for [OpenClaw](https://openclaw.org)/[ClawHub](https://clawhub.ai) AI agent skills. A single-binary safety scanner with an embedded neural network, an HTTP API, a CLI, and an ecosystem crawler — no external model files, no cloud dependencies, no zero-knowledge proofs. Supports [x402](https://www.x402.org/) pay-per-request on Base mainnet so agents can pay $0.001 USDC per evaluation without needing an API key.

---

## What This Does (Plain English)

### The Problem

OpenClaw is an open framework for building AI agent "skills" — small packages of instructions (written in a `SKILL.md` file) and optional scripts that tell an AI assistant what to do. Think of them like browser extensions, but for AI agents. Anyone can write a skill and publish it to [ClawHub](https://clawhub.ai), the public skill registry.

This is powerful, but it creates the same risk that browser extension stores, npm, and PyPI all face: **someone can publish something malicious**. A skill could contain a reverse shell that gives an attacker remote access to your machine. It could include instructions that trick the AI into leaking your API keys through the context window. It could download and execute a binary from a suspicious server. It could install a cron job that persists after the skill is removed.

### The Solution

SkillGuard is a safety scanner that sits between the user and the registry. Before you install a skill, SkillGuard reads its content, looks for known-bad patterns, and tells you whether it's safe. It works like an antivirus, but purpose-built for AI agent plugins.

### How It Works, Step by Step

Here's exactly what happens when you submit a skill to SkillGuard:

**Step 1: Feature Extraction (patterns.rs, skill.rs)**

SkillGuard reads the skill's `SKILL.md` file and any bundled scripts. It runs 22 sets of compiled regex patterns across the text to count suspicious signals. For example:

- How many times does the code call `exec()`, `subprocess.run()`, `Invoke-Expression`, or `std::process::Command`? (shell execution)
- Does it use `curl`, `fetch()`, `Invoke-WebRequest`, or `requests.get()`? (network calls)
- Does it contain `nc -e`, `/dev/tcp/`, `socat`, or `mkfifo`? (reverse shell patterns)
- Does the SKILL.md contain phrases like "pass the API key" or "include your token in the request"? (LLM secret exposure — the skill is trying to trick the AI into leaking credentials)
- Does it contain `eval()`, `atob()`, `document.write()`, `innerHTML`, or zero-width Unicode characters? (obfuscation)
- Does it reference `crontab`, `systemctl enable`, `.bashrc`, or `schtasks`? (persistence mechanisms)

It also pulls in metadata: how old the author's account is, how many skills they've published, the skill's star count, download count, and whether a VirusTotal report has flagged anything.

The result is a 22-number feature vector. Each value is normalized to the range [0, 128] using thresholds calibrated from training data.

**Step 2: Neural Network Classification (model.rs, lib.rs)**

The 22 features feed into a small neural network — a 3-layer MLP (multi-layer perceptron) with 1,924 parameters total. The architecture is 22 inputs → 32 neurons → 32 neurons → 4 outputs, with ReLU activations in the hidden layers. The weights were trained on labeled examples of safe, suspicious, and malicious skills and are hardcoded directly into the binary (no model file to download or lose).

The model runs inference using [onnx-tracer](https://github.com/ICME-Lab/jolt-atlas/tree/main/onnx-tracer) with fixed-point arithmetic (scale=7, meaning values are multiplied by 128). The four output values correspond to the four safety classes: SAFE, CAUTION, DANGEROUS, MALICIOUS.

**Step 3: Softmax + Confidence (scores.rs)**

The raw output scores are passed through softmax normalization so they sum to 1.0 and can be read as probabilities. SkillGuard also computes a confidence score — the margin between the top class and the runner-up, divided by 128. High confidence means the model is decisive; low confidence means the model is uncertain.

**Step 4: Decision (skill.rs)**

The classification is translated into an actionable decision using hard-coded thresholds:

| Classification | Score Threshold | Decision |
|---|---|---|
| MALICIOUS | score > 0.7 | **deny** — hard block |
| DANGEROUS | score > 0.6 | **deny** — hard block |
| DANGEROUS or MALICIOUS | score < 0.6 | **flag** — uncertain, ask a human |
| CAUTION | any | **allow** — minor concerns, probably fine |
| SAFE | any | **allow** — no concerns |

The result is returned as a JSON response (or printed to the terminal in CLI mode) containing the classification, decision, confidence, per-class probability scores, and a human-readable reasoning string.

### What Each Module Does

| Module | Purpose |
|---|---|
| `main.rs` | CLI entry point. Parses subcommands (`serve`, `check`, `crawl`, `scan`), sets up structured logging, dispatches to the right handler. |
| `lib.rs` | Public API. Exposes `classify()` (run inference) and `model_hash()` (SHA-256 of model weights for auditability). |
| `model.rs` | The neural network. All 1,924 weights and biases are hardcoded as `i32` arrays. Builds the model graph using onnx-tracer. |
| `patterns.rs` | Pre-compiled regex patterns. Each `LazyLock<Vec<Regex>>` or `LazyLock<Regex>` is compiled once and reused across all classifications. Covers shell exec, network calls, file writes, env access, credentials, reverse shells, persistence, obfuscation, exfiltration, downloads, privilege escalation, dependencies, archives, and LLM secret exposure. Also includes PowerShell, Rust `std::process::Command`, Go `exec.Command`, DOM injection, and zero-width Unicode patterns. |
| `skill.rs` | Data structures (`Skill`, `SkillFeatures`, `SafetyClassification`, `SafetyDecision`) and the feature extraction pipeline. Also parses YAML frontmatter from SKILL.md files for `name` and `description` fields. |
| `scores.rs` | Softmax normalization of raw model output into per-class probabilities. |
| `server.rs` | Axum HTTP server. Routes: `GET /health`, `POST /api/v1/evaluate`, `POST /api/v1/evaluate/name`, `GET /stats`. Includes per-IP rate limiting (token bucket via `governor`), JSONL access logging with size-based rotation, bearer token auth middleware on `/api/v1/*` routes, x402 payment middleware for pay-per-request on Base mainnet, and usage metrics tracking. |
| `clawhub.rs` | Async client for the ClawHub registry API. Fetches skill metadata (author, stars, downloads) and SKILL.md content. Used by the `evaluate/name` endpoint to look up skills by slug. |
| `crawler.rs` | Parses the [awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills) README to extract skill entries. Converts GitHub `tree` URLs to `raw.githubusercontent.com` URLs. Fetches SKILL.md files with concurrency control (`tokio::sync::Semaphore`) and optional GitHub token auth. Writes a `manifest.json` alongside the fetched files. |
| `batch.rs` | Batch scanning pipeline. Two modes: directory (read local SKILL.md files) and live (fetch from awesome list + classify in one pass). Produces JSON, CSV, or text summary reports. Supports filtering by classification. |

---

## Safety Classifications

| Classification | Decision | What it means |
|---|---|---|
| **SAFE** | allow | The skill's content has no detectable concerning patterns. The model is confident it poses no risk. |
| **CAUTION** | allow | Minor concerns were found — for example, a weather API skill that legitimately mentions "API key" in its instructions, or a deployment tool that uses `sudo`. These are common in functional skills and don't warrant blocking. |
| **DANGEROUS** | deny | Significant risk patterns detected. The skill may expose credentials, request excessive permissions, escalate privileges, or install persistence mechanisms. This warrants human review before installation. |
| **MALICIOUS** | deny | Active malware indicators found: reverse shells, heavily obfuscated payloads, data exfiltration to external servers, or instructions that deliberately trick the AI into leaking secrets. Installation should be blocked. |

When the model classifies a skill as DANGEROUS or MALICIOUS but with low confidence (top score < 0.6), the decision is downgraded to **flag** — the skill is suspicious, but the model isn't sure enough to hard block it. This lets a human make the final call.

---

## Features Analyzed

The classifier examines 22 features extracted from a skill's `SKILL.md` file, bundled scripts, and registry metadata. Here is what each one measures and why it matters:

| # | Feature | What it measures | Why it matters |
|---|---|---|---|
| 0 | `shell_exec_count` | Calls to `exec()`, `spawn()`, `system()`, `subprocess.run()`, `Invoke-Expression`, `std::process::Command`, `exec.Command()` | Shell execution is the primary mechanism for running arbitrary commands. Legitimate skills rarely need it; malicious skills almost always use it. |
| 1 | `network_call_count` | Uses of `fetch()`, `curl`, `axios`, `requests.get()`, `Invoke-WebRequest`, `XMLHttpRequest`, `urllib` | Network calls can be benign (API skills) or malicious (phoning home to a C2 server). The count combined with other signals helps distinguish the two. |
| 2 | `fs_write_count` | File writes via `writeFile()`, `>` redirect, `open('w')`, `fs.write()`, `>>` append | Writing to the filesystem can install backdoors, modify config files, or drop payloads. |
| 3 | `env_access_count` | Reads from `process.env`, `os.environ`, `$ENV_VAR`, `.env`, `dotenv`, `getenv` | Accessing environment variables can be legitimate (reading config) or malicious (harvesting secrets stored in env vars). |
| 4 | `credential_patterns` | Mentions of `api_key`, `password`, `secret`, `token`, `Bearer`, `private_key` in the SKILL.md | Frequent credential references in instructions may indicate the skill is designed to capture or leak authentication material. |
| 5 | `external_download` | URLs ending in `.exe`, `.sh`, `.zip`, `.bin`, `.deb`, `.rpm`, or `curl -O`/`wget -O` patterns | Downloading executables from external sources is a primary delivery mechanism for malware payloads. |
| 6 | `obfuscation_score` | `eval()`, `atob()`, `base64.b64decode()`, `new Function()`, `__import__`, `document.write()`, `.innerHTML`, zero-width Unicode characters | Code obfuscation is the single strongest indicator of malicious intent. Legitimate code has no reason to encode itself in base64 or use zero-width characters. |
| 7 | `privilege_escalation` | `sudo`, `chmod +x/777/755`, `chown root` | Requesting root privileges or making files executable suggests the skill wants to operate outside its intended sandbox. |
| 8 | `persistence_mechanisms` | `crontab`, `systemctl enable`, `@reboot`, `schtasks`, `.bashrc`, `.zshrc`, `.profile`, `autostart` | Installing persistence means the skill's code survives a reboot or session end — a hallmark of backdoors. |
| 9 | `data_exfiltration_patterns` | HTTP POST to external URLs, webhook references, `curl --data` | Sending data to external servers is the primary way attackers extract stolen credentials, files, or other sensitive information. |
| 10 | `skill_md_line_count` | Number of lines in the SKILL.md file | Very short or very long instructions can be suspicious. Extremely short skills may be hiding their true purpose; extremely long ones may be burying malicious instructions. |
| 11 | `script_file_count` | Number of bundled script files | More bundled scripts means a larger attack surface. |
| 12 | `dependency_count` | `npm install`, `pip install`, `cargo add`, `require()`, `import` statements | Each dependency is a potential supply chain risk. |
| 13 | `author_account_age_days` | Days since the author's ClawHub account was created | New accounts publishing complex skills are higher risk. Established authors with a track record are more trustworthy. |
| 14 | `author_skill_count` | Total skills published by the author | Prolific authors with many skills have more reputation to lose. |
| 15 | `stars` | Registry star count (log-scaled) | Community endorsement. Skills with many stars have been reviewed by other users. |
| 16 | `downloads` | Registry download count (log-scaled) | Usage signal. Widely used skills are less likely to be malicious (they'd have been caught). |
| 17 | `has_virustotal_report` | Whether a VirusTotal report was provided | External scanning signal. |
| 18 | `vt_malicious_flags` | Combined VirusTotal signal: `malicious_count + suspicious_count / 2` | How many antivirus engines flagged the skill's content. |
| 19 | `password_protected_archives` | Bundled `.zip`/`.rar`/`.7z` files combined with the word "password" in the SKILL.md | Password-protected archives are a classic malware delivery technique — the password defeats automated scanning. |
| 20 | `reverse_shell_patterns` | `nc -e`, `/dev/tcp/`, `bash -i >&`, `socat`, `mkfifo`, `ncat`, Python/Perl socket patterns | Reverse shells are the single most direct indicator of malicious intent — they give an attacker interactive remote access to the machine. |
| 21 | `llm_secret_exposure` | SKILL.md instructions that tell the AI to "pass the API key", "include the token", "send the password", "output the credential" | This is a novel attack vector specific to AI agent skills: the attacker doesn't write code to steal secrets — they write instructions that convince the AI to leak them through the context window. |

---

## Quickstart

### Install

```bash
# Option 1: Build from source
git clone https://github.com/hshadab/skillguard.git
cd skillguard
cargo build --release

# Option 2: Install via cargo
cargo install --git https://github.com/hshadab/skillguard.git

# Option 3: Download pre-built binary from GitHub Releases
# See https://github.com/hshadab/skillguard/releases
```

### Check a single skill

```bash
# Scan a SKILL.md file and get a human-readable summary
skillguard check --input path/to/SKILL.md

# Get JSON output (for piping to other tools)
skillguard check --input skill.json --format json

# Include a VirusTotal report for additional signal
skillguard check --input skill.json --vt-report vt-report.json
```

The `check` command exits with code 0 for allow/flag decisions and code 1 for deny. This makes it easy to use in CI pipelines: `skillguard check --input SKILL.md || exit 1`.

### Run the HTTP server

```bash
# Default: binds to 127.0.0.1:8080, 60 requests/min per IP
skillguard serve

# Expose externally with higher rate limit
skillguard serve --bind 0.0.0.0:8080 --rate-limit 120

# With API key authentication
SKILLGUARD_API_KEY=your-secret-key skillguard serve
```

**CLI options for `serve`:**

| Flag | Default | Description |
|---|---|---|
| `--bind` | `127.0.0.1:8080` | Address to bind to. Use `0.0.0.0:8080` to expose externally. |
| `--rate-limit` | `60` | Requests per minute per IP. Set to `0` to disable. |
| `--access-log` | `skillguard-access.jsonl` | Path for the JSONL access log. |
| `--max-log-bytes` | `50000000` (50 MB) | Rotate the access log after this size. `0` = no rotation. |

### Crawl the ecosystem

```bash
# Fetch all SKILL.md files from the awesome-openclaw-skills list
skillguard crawl

# Fetch only 20 skills, save to a custom directory
skillguard crawl --limit 20 --output-dir my-skills/

# Use a GitHub token for higher rate limits (5,000 vs 60 requests/hr)
GITHUB_TOKEN=ghp_... skillguard crawl
```

**CLI options for `crawl`:**

| Flag | Default | Description |
|---|---|---|
| `--awesome-url` | awesome-openclaw-skills README | URL of the awesome list to parse |
| `--limit` | `0` (no limit) | Maximum skills to fetch |
| `--concurrency` | `5` | Concurrent fetch requests |
| `--delay-ms` | `200` | Delay between fetches in ms |
| `--output-dir` | `crawled-skills/` | Where to save fetched SKILL.md files |

### Batch scan

```bash
# Scan a directory of crawled skills and print a summary
skillguard scan --input-dir crawled-skills --format summary

# Scan directly from the awesome list (no crawl step needed)
skillguard scan --from-awesome --limit 50 --format json --output report.json

# Only show dangerous/malicious results
skillguard scan --input-dir crawled-skills --filter DANGEROUS,MALICIOUS

# Export as CSV for spreadsheet analysis
skillguard scan --input-dir crawled-skills --format csv --output results.csv
```

**CLI options for `scan`:**

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | `crawled-skills/` | Directory of SKILL.md files to scan |
| `--from-awesome` | false | Fetch and scan from awesome list (live mode) |
| `--format` | `summary` | Output: `json`, `csv`, or `summary` |
| `--output` | stdout | Output file path |
| `--filter` | show all | Comma-separated classifications to include |
| `--concurrency` | `5` | Concurrent classifications |
| `--limit` | `0` (no limit) | Max skills to scan (live mode only) |

---

## API Endpoints

### `GET /health`

Health check. No authentication required.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_hash": "sha256:af562e81...",
  "uptime_seconds": 3600
}
```

### `POST /api/v1/evaluate`

Evaluate a skill from full skill data. Requires auth via API key or x402 payment (if configured). See [x402 Pay-Per-Request](#x402-pay-per-request) for details on response tiers.

```bash
curl -X POST http://localhost:8080/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-key' \
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

Evaluate a skill by its ClawHub registry name. SkillGuard fetches the skill data from the ClawHub API automatically. Requires auth via API key or x402 payment (if configured).

```bash
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

Usage statistics since server start. No authentication required.

```json
{
  "uptime_seconds": 7200,
  "model_hash": "sha256:af562e81...",
  "requests": { "total": 150, "errors": 3 },
  "classifications": { "safe": 120, "caution": 15, "dangerous": 10, "malicious": 5 },
  "decisions": { "allow": 135, "deny": 12, "flag": 3 },
  "endpoints": { "evaluate": 50, "evaluate_by_name": 95, "stats": 5 }
}
```

---

## API Authentication

Set `SKILLGUARD_API_KEY` to enable bearer token authentication on `/api/v1/*` endpoints:

```bash
SKILLGUARD_API_KEY=your-secret-key skillguard serve
```

When set, all `/api/v1/*` requests must include an `Authorization: Bearer <key>` header. Requests without the header or with the wrong key get a `401 Unauthorized` response. The `/health` and `/stats` endpoints remain open regardless.

If `SKILLGUARD_API_KEY` is not set, all endpoints are open. This is the default for backward compatibility and local development.

---

## x402 Pay-Per-Request

### What This Is (Plain English)

SkillGuard supports [x402](https://www.x402.org/), a payment protocol built on HTTP 402. Instead of getting an API key, an AI agent can pay $0.001 USDC per request to evaluate a skill. The payment happens on [Base](https://base.org/) (an Ethereum L2), and the agent gets back a classification and decision. No signup, no API key, no account — just pay and get an answer.

This means any agent can discover SkillGuard through the [x402 Bazaar](https://www.x402.org/), hit the evaluate endpoint, include a payment header, and get a safety classification. The facilitator (Coinbase CDP) verifies and settles the payment on-chain.

### How It Works

1. Agent sends `POST /api/v1/evaluate` without an API key
2. SkillGuard returns `402 Payment Required` with a `PAYMENT-REQUIRED` header containing the price ($0.001 USDC), the wallet address, and the facilitator URL
3. Agent signs a USDC payment on Base and re-sends the request with a `PAYMENT-SIGNATURE` header
4. The x402 middleware forwards the payment to the facilitator for verification and settlement
5. If payment is valid, the request goes through and the agent gets a **basic response** (classification + decision only — no scores, confidence, or reasoning)

API key holders continue to get the **full response** with scores, confidence, and reasoning. The two auth methods coexist on the same endpoints.

### Configuration

| Env Var | Required | Example | Purpose |
|---------|----------|---------|---------|
| `SKILLGUARD_PAY_TO` | Yes (to enable x402) | `0x742d35Cc6634C0532925a3b844Bc9e7595f2bD1e` | Your Base wallet address to receive USDC payments |
| `SKILLGUARD_FACILITATOR_URL` | No | `https://api.cdp.coinbase.com/platform/v2/x402` | Facilitator URL (defaults to CDP mainnet) |
| `SKILLGUARD_API_KEY` | No | `sk-...` | API key auth (works alongside x402) |

### Enable x402

```bash
SKILLGUARD_PAY_TO=0xYourBaseWalletAddress skillguard serve --bind 0.0.0.0:8080
```

Or with both API key and x402:

```bash
SKILLGUARD_PAY_TO=0xYourBaseWalletAddress \
  SKILLGUARD_API_KEY=your-secret-key \
  skillguard serve --bind 0.0.0.0:8080
```

### Pricing

| Endpoint | Price |
|----------|-------|
| `POST /api/v1/evaluate` | $0.001 USDC (1000 base units) |
| `POST /api/v1/evaluate/name` | $0.001 USDC (1000 base units) |
| `GET /health`, `GET /stats`, `GET /` | Free |

### Response Tiers

**API key users** get the full response:

```json
{
  "success": true,
  "evaluation": {
    "skill_name": "weather-helper",
    "classification": "SAFE",
    "decision": "allow",
    "confidence": 0.85,
    "scores": { "safe": 0.72, "caution": 0.18, "dangerous": 0.07, "malicious": 0.03 },
    "reasoning": "No concerning patterns detected"
  },
  "processing_time_ms": 3
}
```

**x402 payers** get a basic response:

```json
{
  "success": true,
  "basic_evaluation": {
    "skill_name": "weather-helper",
    "classification": "SAFE",
    "decision": "allow"
  },
  "processing_time_ms": 3
}
```

### Auth Flow Summary

| Request has... | Behavior |
|----------------|----------|
| Valid API key | Pass through, full response |
| Invalid API key | 401 Unauthorized |
| No API key, valid x402 payment | Payment settled, basic response |
| No API key, no payment | 402 Payment Required |
| No API key, no `pay_to` configured | Endpoints open (backward compatible) |

---

## Docker

```bash
# Build the image
docker build -t skillguard .

# Run (binds to 0.0.0.0:8080 inside the container)
docker run -p 8080:8080 skillguard

# With auth and custom rate limit
docker run -p 8080:8080 \
  -e SKILLGUARD_API_KEY=your-key \
  skillguard /app/skillguard serve --bind 0.0.0.0:8080 --rate-limit 120
```

The Dockerfile is a two-stage build: Rust 1.88 builder + Debian bookworm-slim runtime. The binary is self-contained (model weights embedded), so the runtime image only needs `ca-certificates` for HTTPS.

### Deploy on Render

1. Push to GitHub.
2. Create a new Web Service on Render.
3. Set the build command to `docker build -t skillguard .`
4. Set the start command to `/app/skillguard serve --bind 0.0.0.0:8080 --rate-limit 30`
5. Set environment variables:
   - `SKILLGUARD_API_KEY` — your API key for bearer token auth
   - `SKILLGUARD_PAY_TO` — your Base wallet address to enable x402 payments
   - `SKILLGUARD_FACILITATOR_URL` — (optional) facilitator URL, defaults to CDP mainnet
6. Expose port 8080.

---

## Pre-Install Hook

An example shell script at `examples/skillguard-install-hook.sh` wires SkillGuard into the OpenClaw CLI install flow. It supports two modes:

**Hosted API mode** (recommended for teams):
```bash
SKILLGUARD_API_URL=https://your-skillguard.onrender.com \
  SKILLGUARD_API_KEY=your-key \
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
- Passes `SKILLGUARD_API_KEY` as a Bearer token when set.

---

## Deployment

### systemd

A systemd unit file is provided at `deploy/skillguard.service`. Copy it to `/etc/systemd/system/`, adjust paths and environment variables, then:

```bash
sudo systemctl enable --now skillguard
```

### Reverse proxy

Production deployments should run behind a reverse proxy for TLS termination. Example configs are included:

- **nginx**: `deploy/nginx-skillguard.conf` — reverse proxy with TLS, HTTP→HTTPS redirect, and request size limits.
- **Caddy**: `deploy/Caddyfile` — minimal two-line config with automatic TLS.

---

## Architecture

```
src/
├── lib.rs         classify() and model_hash() entry points
├── main.rs        CLI: serve, check, crawl, scan subcommands
├── model.rs       Embedded 1,924-param MLP (22→32→32→4, ReLU activations)
├── skill.rs       Skill struct, SkillFeatures, YAML frontmatter parsing, decision logic
├── patterns.rs    Compiled regex pattern sets for feature extraction
├── scores.rs      ClassScores with softmax normalization
├── clawhub.rs     ClawHub registry API client (async, reqwest-based)
├── crawler.rs     Awesome-openclaw-skills list parser and GitHub fetcher
├── batch.rs       Batch scanning pipeline with JSON/CSV/summary output
└── server.rs      Axum HTTP server with auth, rate limiting, and access logging
tests/
└── integration.rs Integration tests for HTTP endpoints, auth, and classification
deploy/
├── skillguard.service     systemd unit file
├── nginx-skillguard.conf  nginx reverse proxy config with TLS
└── Caddyfile              Caddy reverse proxy config
examples/
└── skillguard-install-hook.sh  Pre-install hook for OpenClaw CLI
data/
└── registry.json               Model metadata (hash, version, architecture)
```

### Data flow

```
                                   ┌─────────────────────────────────┐
                                   │  Input: SKILL.md + scripts +   │
                                   │         metadata               │
                                   └────────────┬────────────────────┘
                                                │
                                   ┌────────────▼────────────────────┐
                                   │  patterns.rs                    │
                                   │  Run 22 regex pattern sets      │
                                   │  Count matches, detect signals  │
                                   └────────────┬────────────────────┘
                                                │
                                   ┌────────────▼────────────────────┐
                                   │  skill.rs                       │
                                   │  Build 22-element feature vec   │
                                   │  Normalize to [0, 128]          │
                                   └────────────┬────────────────────┘
                                                │
                                   ┌────────────▼────────────────────┐
                                   │  model.rs → lib.rs              │
                                   │  MLP inference: 22→32→32→4      │
                                   │  Fixed-point arithmetic         │
                                   └────────────┬────────────────────┘
                                                │
                                   ┌────────────▼────────────────────┐
                                   │  scores.rs                      │
                                   │  Softmax → probabilities        │
                                   │  Confidence = margin / 128      │
                                   └────────────┬────────────────────┘
                                                │
                                   ┌────────────▼────────────────────┐
                                   │  skill.rs                       │
                                   │  Classification → Decision      │
                                   │  SAFE/CAUTION → allow           │
                                   │  DANGEROUS/MALICIOUS → deny     │
                                   │  Low confidence → flag          │
                                   └────────────┬────────────────────┘
                                                │
                                   ┌────────────▼────────────────────┐
                                   │  Output: classification,        │
                                   │  decision, confidence, scores,  │
                                   │  reasoning                      │
                                   └─────────────────────────────────┘
```

### Model details

- **Architecture**: 3-layer MLP — 22 input features, two 32-neuron hidden layers with ReLU, 4 output classes.
- **Parameters**: 1,924 total (22×32 + 32 + 32×32 + 32 + 32×4 + 4).
- **Arithmetic**: Fixed-point at scale=7 (values multiplied by 128). No floating-point during inference.
- **Inference**: Uses [onnx-tracer](https://github.com/ICME-Lab/jolt-atlas/tree/main/onnx-tracer) from the ICME Lab jolt-atlas project.
- **Deterministic hash**: `model_hash()` computes a SHA-256 digest over the serialized model bytecode. The hash is stable across runs and platforms for auditability.
- **Accuracy**: 86.7% exact class match, 90.0% decision match (allow/deny/flag) on the validation set.

### Logging

SkillGuard uses structured logging via [`tracing`](https://docs.rs/tracing). Log output goes to stderr:

```bash
RUST_LOG=skillguard=debug skillguard serve   # debug logging for SkillGuard only
RUST_LOG=error skillguard serve              # errors only
```

### Access log

Every evaluation is logged to `skillguard-access.jsonl` (configurable via `--access-log`):

```json
{"timestamp":"2026-02-10T12:00:00Z","endpoint":"evaluate_by_name","skill_name":"weather-helper","classification":"SAFE","decision":"allow","confidence":0.85,"processing_time_ms":3}
```

Auto-rotates when the file exceeds `--max-log-bytes` (default 50 MB). Set to `0` to disable rotation.

---

## Limitations & Threat Model

SkillGuard is a defense-in-depth layer, not a guarantee. Understanding its boundaries matters.

**What it detects:**
- Reverse shells, data exfiltration, obfuscated payloads, privilege escalation, persistence mechanisms.
- Social engineering via SKILL.md: instructions that trick the AI into leaking secrets or credentials.
- PowerShell attacks: `Invoke-Expression`, `-EncodedCommand`, `Invoke-WebRequest`.
- Cross-language shell execution: Rust `std::process::Command`, Go `exec.Command`.
- DOM injection payloads: `document.write`, `innerHTML`.
- Low-reputation authors: new accounts, zero downloads, no community trust signals.
- Password-protected archives bundled with skills.
- Zero-width Unicode characters used for obfuscation.

**What it does NOT detect:**
- **Novel attack vectors** the model hasn't seen. Zero-day techniques without recognizable patterns may bypass it.
- **Compromised ClawHub API.** If the registry itself is tampered with, `evaluate-by-name` responses can't be trusted. Use `/api/v1/evaluate` with locally fetched data instead.
- **Obfuscation beyond regex reach.** Multi-stage payloads, encrypted blobs decoded at runtime, steganography.
- **Supply chain attacks via dependencies.** SkillGuard counts dependency installs but doesn't resolve or audit transitive dependencies.
- **Post-classification changes.** A skill that passes today could be updated with malicious content tomorrow. Re-scan on updates.

**Model accuracy:** ~90% decision accuracy. Roughly 1 in 10 skills may get the wrong decision. SkillGuard should complement — not replace — human review for high-stakes deployments.

See [SECURITY.md](SECURITY.md) for a detailed breakdown of evasion techniques, detection boundaries, and deployment recommendations.

---

## Relationship to ClawGuard

SkillGuard is a focused extraction from [ClawGuard](https://github.com/hshadab/clawguard). ClawGuard is a broader system that gates all OpenClaw agent actions through ONNX guardrail models with zero-knowledge proof verification — multiple models, ZK proof generation with JOLT Atlas, cryptographic receipts, and an enforcement engine.

SkillGuard strips all of that away and keeps only the skill safety classifier. No `zkml-jolt-core`, no `jolt-core`, no `ark-bn254`, no proving pipeline, no receipt system. Just the 22-feature classifier, the regex pattern library, the ClawHub client, the ecosystem crawler, the batch scanner, and the HTTP server.

---

## License

MIT
