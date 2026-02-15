# SkillGuard: SSL for Agent Skills

> Every classification is a verifiable certificate. Like SSL proves server identity, SkillGuard proofs prove classification integrity — powered by [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) zero-knowledge machine learning proofs.

[![CI](https://github.com/hshadab/skillguard/actions/workflows/ci.yml/badge.svg)](https://github.com/hshadab/skillguard/actions/workflows/ci.yml)

**Live:** [https://skillguard.onrender.com](https://skillguard.onrender.com)

---

## What Is This?

SkillGuard is **SSL for the AI agent skill supply chain**. It answers a simple question: **"Is this skill safe to install?"** — and backs every answer with a cryptographic certificate anyone can verify.

AI agents on platforms like [OpenClaw](https://openclaw.ai) can install community-created "skills" — small packages of code and instructions that give an agent new abilities (calling APIs, writing files, running scripts, etc.). Some skills might be malicious: they could steal credentials, open reverse shells, or trick the AI into leaking secrets.

SkillGuard inspects each skill and classifies it as **SAFE**, **CAUTION**, **DANGEROUS**, or **MALICIOUS**. It then makes a decision: **ALLOW**, **FLAG**, or **DENY**.

Just as an SSL certificate proves a server is who it claims to be, every SkillGuard classification comes with a **zero-knowledge machine learning proof** — a cryptographic certificate proving the classification was computed correctly by a specific model. Anyone can verify this proof without trusting the SkillGuard operator and without seeing the model's internal weights.

### How It Works

1. **Skill submitted** — A developer publishes a skill to [ClawHub](https://clawhub.ai), or submits data directly via API.

2. **Features extracted** — SkillGuard reads the skill's documentation, scripts, and metadata, then extracts 28 numeric features that capture security-relevant signals (shell execution calls, reverse shell patterns, credential access, obfuscation techniques, entropy analysis, author reputation, download counts, etc.).

3. **Classified with proof** — The 28 features feed into a small neural network (3-layer MLP, 2,116 parameters). The entire forward pass runs inside a SNARK virtual machine ([Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas)), producing a ~53 KB cryptographic proof that the classification was computed correctly.

4. **Anyone verifies** — Anyone can verify a proof by posting it to `/api/v1/verify`. Verification is free, takes milliseconds, and requires no API key.

5. **Classification made** — The result (ALLOW, FLAG, or DENY) plus the proof become a tamperproof safety certificate for the skill. Payment is handled via the [x402 protocol](https://www.x402.org/) at $0.001 USDC on Base per classification.

---

## Quick Start

### Build

```bash
git clone https://github.com/hshadab/skillguard.git
cd skillguard
cargo build --release
```

Requires Rust nightly (arkworks const generics dependency).

### Developer Setup

```bash
# Copy environment template and configure
cp .env.example .env

# Install pre-commit hooks (fmt + clippy)
make setup-hooks
```

### Serve

```bash
# Start the server (ZKML prover initializes in background)
./target/release/skillguard serve --bind 0.0.0.0:8080

# With API key authentication
SKILLGUARD_API_KEY=your-secret-key ./target/release/skillguard serve --bind 0.0.0.0:8080

# With x402 payments enabled (USDC on Base)
SKILLGUARD_PAY_TO=0xYourBaseWallet ./target/release/skillguard serve --bind 0.0.0.0:8080
```

### Classify a Skill

There is a single endpoint that handles all classifications. Every response includes a zkML proof.

**By name** (fetches from [ClawHub](https://clawhub.ai)):
```bash
curl -X POST https://skillguard.onrender.com/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"skill": "4claw"}'
```

**With full skill data:**
```bash
curl -X POST https://skillguard.onrender.com/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "skill": {
      "name": "hello-world",
      "version": "1.0.0",
      "author": "dev",
      "description": "A safe greeting skill",
      "skill_md": "# Hello World\nSays hello to the user.",
      "scripts": [],
      "files": []
    }
  }'
```

### Verify a Proof

```bash
curl -X POST https://skillguard.onrender.com/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{"proof_b64": "...", "program_io": {...}}'
```

### CLI

```bash
# Classify a local SKILL.md file with proof
skillguard check --input SKILL.md --prove --format json
```

---

## API Reference

| Method | Path | Auth | Price | Description |
|--------|------|------|-------|-------------|
| POST | `/api/v1/evaluate` | API key or x402 | $0.001 USDC | Classify a skill with mandatory zkML proof (auto-detects name lookup vs full data) |
| POST | `/api/v1/verify` | None | Free | Verify a zkML proof |
| GET | `/health` | None | Free | Health check (includes `zkml_enabled`, `pay_to`) |
| GET | `/stats` | None | Free | Usage statistics and proof counts |
| GET | `/openapi.json` | None | Free | OpenAPI 3.1 specification |
| GET | `/.well-known/ai-plugin.json` | None | Free | AI agent discovery manifest |
| GET | `/.well-known/llms.txt` | None | Free | LLM-readable API description |
| GET | `/` | None | Free | Web dashboard |

The `/api/v1/evaluate` endpoint accepts two request formats:
- **Name lookup:** `{"skill": "skill-slug"}` — fetches skill data from ClawHub, then classifies
- **Full skill data:** `{"skill": {"name": "...", "version": "...", ...}}` — classifies directly

Both formats return the same response with classification, confidence, scores, reasoning, and a zkML proof bundle. The proof is mandatory — if the prover is still initializing, the endpoint returns an error until it is ready.

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | 3-layer MLP: 28 inputs, 2x32 hidden (ReLU), 4 outputs. 2,116 parameters. Fixed-point integer arithmetic. |
| Proving | [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) SNARK with Dory commitment (BN254 curve). ~53 KB proofs, ~4s proving time. |
| Payment | [x402](https://www.x402.org/) HTTP 402 protocol. $0.001 USDC on Base. [OpenFacilitator](https://openfacilitator.io). |
| Server | Axum async HTTP. LRU per-IP rate limiting (IPv6 /64 aggregation), constant-time API key auth, CORS, graceful shutdown, JSONL access logging. |
| Runtime | Docker on Render. Rust nightly. Pre-generated Dory SRS bundled in image. |

### Feature List

The classifier extracts 28 features from each skill:

| # | Feature | What It Measures |
|---|---------|-----------------|
| 1 | `shell_exec_count` | Shell/process execution calls (exec, spawn, subprocess, Process.Start, Runtime.exec, etc.) |
| 2 | `network_call_count` | HTTP/network requests (fetch, curl, wget, axios, reqwest, aiohttp, httpx) |
| 3 | `fs_write_count` | File system writes (writeFile, `>`, `>>`) |
| 4 | `env_access_count` | Environment variable access (process.env, os.environ) |
| 5 | `credential_patterns` | Mentions of API keys, passwords, secrets, tokens |
| 6 | `external_download` | Downloads executables or archives from URLs |
| 7 | `obfuscation_score` | Obfuscation techniques (eval, atob, base64, String.fromCharCode, marshal.loads) |
| 8 | `privilege_escalation` | Sudo, chmod 777, chown root |
| 9 | `persistence_mechanisms` | Crontab, systemd, launchd, autostart, registry Run keys, init.d |
| 10 | `data_exfiltration_patterns` | POST/PUT to external URLs, webhooks, DNS exfil, netcat piping |
| 11 | `skill_md_line_count` | Lines of documentation |
| 12 | `script_file_count` | Number of script files |
| 13 | `dependency_count` | Package install / import statements |
| 14 | `author_account_age_days` | How old the author's account is |
| 15 | `author_skill_count` | Total skills the author has published |
| 16 | `stars` | Repository stars |
| 17 | `downloads` | Download count |
| 18 | `has_virustotal_report` | Whether a VirusTotal report was provided |
| 19 | `vt_malicious_flags` | Combined VirusTotal malicious + suspicious flags |
| 20 | `password_protected_archives` | Bundled password-protected zip/rar/7z files |
| 21 | `reverse_shell_patterns` | Reverse shell patterns (nc -e, socat, /dev/tcp/, pty.spawn, ruby -rsocket) |
| 22 | `llm_secret_exposure` | Instructions that trick the AI into leaking secrets or prompt injection |
| 23 | `entropy_score` | Shannon entropy of script bytes (high = encrypted/encoded) |
| 24 | `non_ascii_ratio` | Ratio of non-ASCII bytes (catches homoglyphs, encoded payloads) |
| 25 | `max_line_length` | Longest script line (long = minified/obfuscated) |
| 26 | `comment_ratio` | Comment lines / total lines (malware rarely has comments) |
| 27 | `domain_count` | Unique external domains referenced |
| 28 | `string_obfuscation_score` | Hex escapes + join() + chr() pattern counts |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SKILLGUARD_API_KEY` | Bearer token for API authentication. If unset, all endpoints are open. | (none) |
| `SKILLGUARD_PAY_TO` | Ethereum address to receive x402 USDC payments on Base. | (none) |
| `SKILLGUARD_FACILITATOR_URL` | x402 facilitator URL. Production Render deployment overrides to `https://facilitator.x402.rs`. | `https://pay.openfacilitator.io` |
| `SKILLGUARD_EXTERNAL_URL` | Public base URL (for x402 resource URLs behind TLS proxies). | (none) |
| `SKILLGUARD_SKIP_PROVER` | Set to `1` to disable the ZKML prover. | `0` |
| `SKILLGUARD_PRICE_USDC_MICRO` | Price per classification in USDC micro-units (6 decimals). `1000` = $0.001. | `1000` |
| `RUST_LOG` | Log level filter. | `info` |

See `.env.example` for a documented template of all variables.

---

## Links

- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) — ZKML proving stack
- [Jolt](https://github.com/a16z/jolt) — SNARK VM by a16z
- [x402 Protocol](https://www.x402.org/) — HTTP 402 payment protocol
- [OpenClaw](https://openclaw.ai) — Open framework for AI agent skills
- [ClawHub](https://clawhub.ai) — Registry for OpenClaw skills
- [Novanet](https://novanet.xyz) — Verifiable inference network

---

## License

MIT
