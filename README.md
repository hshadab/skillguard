# SkillGuard

> Unforgeable cryptographic safety guardrails for OpenClaw agent skills, powered by [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) zero-knowledge machine learning proofs

[![CI](https://github.com/hshadab/skillguard/actions/workflows/ci.yml/badge.svg)](https://github.com/hshadab/skillguard/actions/workflows/ci.yml)

**Live:** [https://skillguard.onrender.com](https://skillguard.onrender.com)

---

## What Is This?

SkillGuard is a safety classifier for AI agent skills. It answers a simple question: **"Is this skill safe to install?"**

AI agents on platforms like [OpenClaw](https://openclaw.org) can install community-created "skills" — small packages of code and instructions that give an agent new abilities (calling APIs, writing files, running scripts, etc.). Some skills might be malicious: they could steal credentials, open reverse shells, or trick the AI into leaking secrets.

SkillGuard inspects each skill and classifies it as **SAFE**, **CAUTION**, **DANGEROUS**, or **MALICIOUS**. It then makes a decision: **ALLOW**, **FLAG**, or **DENY**.

What makes SkillGuard different from a regular classifier is that every classification comes with a **zero-knowledge machine learning proof** — a cryptographic certificate proving the classification was computed correctly by a specific model. Anyone can verify this proof without trusting the SkillGuard operator and without seeing the model's internal weights.

### How It Works

1. **Feature extraction** — SkillGuard reads the skill's documentation, scripts, and metadata, then extracts 22 numeric features that capture security-relevant signals (shell execution calls, reverse shell patterns, credential access, obfuscation techniques, author reputation, download counts, etc.).

2. **Neural network classification** — The 22 features feed into a small neural network (3-layer MLP, 1,924 parameters) that outputs probabilities for each safety class. All arithmetic uses fixed-point integers so the computation is deterministic and provable.

3. **Zero-knowledge machine learning proof** — The entire neural network forward pass runs inside a SNARK virtual machine ([Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas)). This produces a ~53 KB cryptographic proof that the classification was computed correctly. The proof reveals the inputs and outputs but not the model weights.

4. **Verification** — Anyone can verify a proof by posting it to `/api/v1/verify`. Verification is free, takes milliseconds, and requires no API key.

5. **Payment** — SkillGuard uses the [x402 protocol](https://www.x402.org/) for pay-per-request pricing. AI agents or users pay $0.001 USDC on Base per classification. Payment is settled on-chain via `transferWithAuthorization`. API key holders bypass payment.

---

## Quick Start

### Build

```bash
git clone https://github.com/hshadab/skillguard.git
cd skillguard
cargo build --release
```

Requires Rust nightly (arkworks const generics dependency).

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

There is a single endpoint that handles all classifications. It automatically generates a zkML proof when the prover is ready.

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
| POST | `/api/v1/evaluate` | API key or x402 | $0.001 USDC | Classify a skill (auto-detects name lookup vs full data, includes zkML proof when prover is ready) |
| POST | `/api/v1/verify` | None | Free | Verify a zkML proof |
| GET | `/health` | None | Free | Health check (includes `zkml_enabled`, `pay_to`) |
| GET | `/stats` | None | Free | Usage statistics and proof counts |
| GET | `/openapi.json` | None | Free | OpenAPI 3.1 specification |
| GET | `/` | None | Free | Web dashboard |

The `/api/v1/evaluate` endpoint accepts two request formats:
- **Name lookup:** `{"skill": "skill-slug"}` — fetches skill data from ClawHub, then classifies
- **Full skill data:** `{"skill": {"name": "...", "version": "...", ...}}` — classifies directly

Both formats return the same response with classification, confidence, scores, reasoning, and an optional zkML proof bundle.

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | 3-layer MLP: 22 inputs, 2x32 hidden (ReLU), 4 outputs. 1,924 parameters. Fixed-point integer arithmetic. |
| Proving | [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) SNARK with Dory commitment (BN254 curve). ~53 KB proofs, ~4s proving time. |
| Payment | [x402](https://www.x402.org/) HTTP 402 protocol. $0.001 USDC on Base. [OpenFacilitator](https://openfacilitator.io). |
| Server | Axum async HTTP. LRU per-IP rate limiting (IPv6 /64 aggregation), constant-time API key auth, CORS, graceful shutdown, JSONL access logging. |
| Runtime | Docker on Render. Rust nightly. Pre-generated Dory SRS bundled in image. |

### Feature List

The classifier extracts 22 features from each skill:

| # | Feature | What It Measures |
|---|---------|-----------------|
| 1 | `shell_exec_count` | Shell/process execution calls (exec, spawn, subprocess) |
| 2 | `network_call_count` | HTTP/network requests (fetch, curl, wget, axios) |
| 3 | `fs_write_count` | File system writes (writeFile, `>`, `>>`) |
| 4 | `env_access_count` | Environment variable access (process.env, os.environ) |
| 5 | `credential_patterns` | Mentions of API keys, passwords, secrets, tokens |
| 6 | `external_download` | Downloads executables or archives from URLs |
| 7 | `obfuscation_score` | Obfuscation techniques (eval, atob, base64, new Function) |
| 8 | `privilege_escalation` | Sudo, chmod 777, chown root |
| 9 | `persistence_mechanisms` | Crontab, systemd, launchd, autostart entries |
| 10 | `data_exfiltration_patterns` | POST/PUT to external URLs, webhooks |
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
| 21 | `reverse_shell_patterns` | Reverse shell patterns (nc -e, socat, /dev/tcp/) |
| 22 | `llm_secret_exposure` | Instructions that trick the AI into leaking secrets |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SKILLGUARD_API_KEY` | Bearer token for API authentication. If unset, all endpoints are open. | (none) |
| `SKILLGUARD_PAY_TO` | Ethereum address to receive x402 USDC payments on Base. | (none) |
| `SKILLGUARD_FACILITATOR_URL` | x402 facilitator URL. | `https://pay.openfacilitator.io` |
| `SKILLGUARD_EXTERNAL_URL` | Public base URL (for x402 resource URLs behind TLS proxies). | (none) |
| `SKILLGUARD_SKIP_PROVER` | Set to `1` to disable the ZKML prover. | `0` |
| `RUST_LOG` | Log level filter. | `info` |

---

## Links

- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) — ZKML proving stack
- [Jolt](https://github.com/a16z/jolt) — SNARK VM by a16z
- [x402 Protocol](https://www.x402.org/) — HTTP 402 payment protocol
- [OpenClaw](https://openclaw.org) — Open framework for AI agent skills
- [ClawHub](https://clawhub.ai) — Registry for OpenClaw skills
- [Novanet](https://novanet.xyz) — Verifiable inference network

---

## License

MIT
