# SkillGuard ZKML

[![CI](https://github.com/hshadab/skillguard/actions/workflows/ci.yml/badge.svg)](https://github.com/hshadab/skillguard/actions/workflows/ci.yml)

Provably correct AI safety classifications with agentic commerce. A verifiable skill safety classifier that produces cryptographic SNARK proofs via [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) and accepts pay-per-request via [x402](https://www.x402.org/) on Base.

---

## What This Demonstrates

1. **ZKML via Jolt Atlas** -- A neural network classifier runs inside a Jolt SNARK VM. Every classification can produce a proof that the model was evaluated correctly, without revealing the model weights.
2. **x402 Payments on Base** -- Agents pay $0.001 USDC per evaluation or $0.005 per proof on Base mainnet. No API key needed -- just pay and get an answer.

---

## How It Works

```
Feature Extraction -> MLP in Jolt SNARK VM -> Proof + Classification -> x402 Settlement
```

1. A skill's `SKILL.md` and scripts are scanned for 22 security-relevant features (shell exec, reverse shells, obfuscation, etc.)
2. The 22-feature vector feeds into a 3-layer MLP (22->32->32->4, 1,924 params) running inside the Jolt SNARK VM
3. The prover produces a cryptographic proof that the MLP was evaluated correctly
4. The classification (SAFE/CAUTION/DANGEROUS/MALICIOUS) and proof bundle are returned
5. Anyone can verify the proof without re-running the model

Softmax normalization happens in plaintext after proving -- only the deterministic MLP forward pass is proved.

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | 3-layer MLP: 22 inputs, 2x32 hidden (ReLU), 4 outputs. 1,924 parameters. Fixed-point arithmetic (scale=7). |
| Proving | [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) SNARK with HyperKZG commitment scheme, Keccak transcript, BN254 curve. |
| Payment | [x402](https://www.x402.org/) HTTP 402 protocol. USDC on Base (eip155:8453). CDP facilitator. |
| Server | Axum HTTP server with per-IP rate limiting, auth middleware, and JSONL access logging. |

---

## Quick Start

### Build

```bash
git clone https://github.com/hshadab/skillguard.git
cd skillguard
cargo build --release
```

### Serve

```bash
# Start the server (ZKML prover initializes at startup)
./target/release/skillguard serve --bind 0.0.0.0:8080

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

### CLI with proof

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
| GET | `/` | None | Free | Web dashboard |

---

## Why ZKML + Agentic Commerce

- **Trust without access**: Agents can verify a classification was computed correctly without accessing the model weights or re-running inference.
- **Composable verification**: Proofs are portable. A downstream agent can verify a safety classification without trusting the classifier operator.
- **Pay-per-use**: No accounts, no API keys needed. Agents pay per request via x402, settled on-chain in USDC on Base.
- **On-chain settlement**: Every payment is a Base transaction. Verifiable, auditable, composable with other on-chain protocols.

---

## Links

- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) -- ZKML proving stack
- [x402 Protocol](https://www.x402.org/) -- HTTP 402 payment protocol
- [OpenClaw](https://openclaw.org) -- Open framework for AI agent skills

---

## License

MIT
