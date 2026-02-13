# SkillGuard ZKML

> Powered by [Jolt Atlas zkML](https://github.com/ICME-Lab/jolt-atlasc)

[![CI](https://github.com/hshadab/skillguard/actions/workflows/ci.yml/badge.svg)](https://github.com/hshadab/skillguard/actions/workflows/ci.yml)

Provably correct AI safety classifications with agentic commerce. A verifiable skill safety classifier that produces cryptographic SNARK proofs via [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) and accepts pay-per-request via [x402](https://www.x402.org/) on Base.

**Live:** [https://skillguard.onrender.com](https://skillguard.onrender.com)

---

## How It Works (Plain English)

SkillGuard answers a simple question: **"Is this AI agent skill safe to run?"**

AI agents on platforms like [OpenClaw](https://openclaw.org) can install community-created "skills" -- small packages of code and instructions that give an agent new abilities. Some of those skills might be malicious: they could steal credentials, open reverse shells, or trick the AI into leaking secrets. SkillGuard is the safety checkpoint.

Here is what happens when a skill is submitted for evaluation:

### Step 1: Feature Extraction

SkillGuard reads the skill's documentation (`SKILL.md`), scripts, and metadata, then extracts 22 numeric features that capture security-relevant signals. These include things like:

- How many shell execution calls does the code make?
- Does it download external executables?
- Are there reverse shell patterns (`nc -e`, `/dev/tcp/`)?
- Does it access environment variables or credential-like strings?
- Does the documentation try to trick an LLM into leaking secrets?
- How old is the author's account? How many downloads does the skill have?

Each feature is normalized to a 0-128 integer scale. A brand-new author with zero downloads and a script full of `eval()` calls will produce very different numbers than a well-known author with thousands of downloads and clean documentation.

### Step 2: Neural Network Classification

The 22 features feed into a small neural network (a 3-layer MLP with 1,924 parameters). The network has been trained on labeled examples of safe, cautious, dangerous, and malicious skills. It outputs 4 raw scores -- one for each safety class.

All arithmetic is done in **fixed-point integers** (no floating point). This is important for the next step: you can't prove floating-point computations inside a SNARK because floating point is non-deterministic across platforms. Integer math is identical everywhere.

The raw scores go through a softmax function to produce probabilities (e.g., 72% SAFE, 15% CAUTION, 10% DANGEROUS, 3% MALICIOUS). The highest class wins. A confidence margin is computed from the gap between the top two scores.

### Step 3: Zero-Knowledge Proof (Optional)

This is the part that makes SkillGuard different from a regular classifier.

When you call the `/api/v1/evaluate/prove` endpoint, the MLP forward pass runs inside a **SNARK virtual machine** ([Jolt](https://github.com/a16z/jolt)). Jolt traces every arithmetic operation the neural network performs, then produces a cryptographic proof that all the multiplications, additions, and ReLU activations were done correctly.

The proof says: *"Given these 22 input features, this specific model (identified by its SHA-256 hash) produced these 4 output scores. Here is a ~53 KB proof that you can verify in milliseconds."*

What the proof does **not** reveal: the model's internal weights. The verifier sees the inputs, the outputs, and the proof -- but not how the model arrived at its answer. This is the "zero-knowledge" property.

The proving scheme is **Dory** (a polynomial commitment scheme based on bilinear pairings on the BN254 curve). Dory uses a structured reference string (SRS) that is generated once and bundled with the deployment. Proving takes about 4 seconds on a single CPU core.

### Step 4: Verification

Anyone can verify a proof by posting it to `/api/v1/verify`. Verification is free, requires no API key, and runs in milliseconds. This means a downstream agent, a marketplace, or an auditor can independently confirm that a classification was computed honestly -- without trusting SkillGuard's operator and without re-running the model.

### Step 5: Payment (Optional)

SkillGuard supports the [x402 protocol](https://www.x402.org/), which lets AI agents pay per request using USDC on Base. An agent that doesn't have an API key can pay $0.001 for a classification or $0.005 for a classification with proof. Payment is settled on-chain. API key holders bypass payment entirely.

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | 3-layer MLP: 22 inputs, 2x32 hidden (ReLU), 4 outputs. 1,924 parameters. Fixed-point arithmetic (scale=7, all weights are i32). |
| Proving | [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) SNARK with Dory commitment scheme (BN254 curve, Keccak transcript). ~53 KB proofs, ~4s proving time. |
| Payment | [x402](https://www.x402.org/) HTTP 402 protocol. USDC on Base (eip155:8453). CDP facilitator. |
| Server | Axum async HTTP server. LRU per-IP rate limiting, constant-time API key auth, CORS, graceful shutdown, JSONL access logging, persistent metrics. |
| Runtime | Docker on Render. Rust nightly (required by arkworks const generics). Pre-generated Dory SRS bundled in image. |

### Feature List

The classifier extracts these 22 features from each skill:

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
# Start the server (ZKML prover initializes at startup)
./target/release/skillguard serve --bind 0.0.0.0:8080

# With API key authentication
SKILLGUARD_API_KEY=your-secret-key ./target/release/skillguard serve --bind 0.0.0.0:8080

# With x402 payments enabled
SKILLGUARD_PAY_TO=0xYourBaseWallet ./target/release/skillguard serve --bind 0.0.0.0:8080
```

### Classify (no proof)

```bash
curl -X POST http://localhost:8080/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"skill": {"name": "hello", "version": "1.0.0", "author": "dev", "description": "test", "skill_md": "# Hello", "scripts": [], "files": []}}'
```

### Classify + Prove

```bash
curl -X POST http://localhost:8080/api/v1/evaluate/prove \
  -H 'Content-Type: application/json' \
  -d '{"skill": {"name": "hello", "version": "1.0.0", "author": "dev", "description": "test", "skill_md": "# Hello", "scripts": [], "files": []}}'
```

### Verify

```bash
# Submit the proof_b64 and program_io from the prove response
curl -X POST http://localhost:8080/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{"proof_b64": "...", "program_io": {...}}'
```

### CLI

```bash
skillguard check --input SKILL.md --prove --format json
```

---

## API Reference

| Method | Path | Auth | Price | Description |
|--------|------|------|-------|-------------|
| POST | `/api/v1/evaluate` | API key or x402 | $0.001 USDC | Classification only |
| POST | `/api/v1/evaluate/name` | API key or x402 | $0.001 USDC | Classification by ClawHub name |
| POST | `/api/v1/evaluate/prove` | API key or x402 | $0.005 USDC | Classification + ZK proof |
| POST | `/api/v1/verify` | None | Free | Verify a proof |
| GET | `/health` | None | Free | Health check (includes `zkml_enabled`) |
| GET | `/stats` | None | Free | Usage statistics + proof counts |
| GET | `/openapi.json` | None | Free | OpenAPI 3.1 specification |
| GET | `/` | None | Free | Web dashboard |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SKILLGUARD_API_KEY` | Bearer token for API authentication. If unset, all endpoints are open. | (none) |
| `SKILLGUARD_PAY_TO` | Ethereum address to receive x402 USDC payments on Base. | (none) |
| `SKILLGUARD_FACILITATOR_URL` | x402 facilitator URL. | `https://facilitator.x402.rs` |
| `SKILLGUARD_SKIP_PROVER` | Set to `1` to disable the ZKML prover (saves memory). Prove endpoints return an error. | `0` |
| `RUST_LOG` | Log level filter. | `info` |

---

## Why ZKML + Agentic Commerce

- **Trust without access**: Agents can verify a classification was computed correctly without seeing the model weights or re-running inference.
- **Composable verification**: Proofs are portable. A downstream agent can verify a safety classification without trusting the classifier operator.
- **Pay-per-use**: No accounts needed. Agents pay per request via x402, settled on-chain in USDC on Base.
- **Deterministic reproducibility**: The model hash (`sha256:...`) identifies exactly which model produced a classification. Combined with the proof, this gives full auditability.

---

## Links

- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) -- ZKML proving stack
- [Jolt](https://github.com/a16z/jolt) -- SNARK VM by a16z
- [x402 Protocol](https://www.x402.org/) -- HTTP 402 payment protocol
- [OpenClaw](https://openclaw.org) -- Open framework for AI agent skills

---

## License

MIT
