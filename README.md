# SkillGuard: SSL for Agent Skills

> Every classification is a verifiable certificate. Like SSL proves server identity, SkillGuard proofs prove classification integrity — powered by [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) zero-knowledge machine learning proofs.

[![CI](https://github.com/hshadab/skillguard/actions/workflows/ci.yml/badge.svg)](https://github.com/hshadab/skillguard/actions/workflows/ci.yml)

**Live:** [https://skillguard.onrender.com](https://skillguard.onrender.com)

---

## Table of Contents
- [Plain English: What Does This Do?](#plain-english-what-does-this-do)
- [What Is This?](#what-is-this)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [How the Model Works](#how-the-model-works-plain-english)
- [Architecture](#architecture)
- [Model Training](#model-training)
- [Batch Scanning](#batch-scanning)
- [MCP Server](#mcp-server)
- [Environment Variables](#environment-variables)
- [Links](#links)
- [License](#license)

---

## Plain English: What Does This Do?

AI agents are starting to install third-party plugins ("skills") from open marketplaces, just like humans install apps from an app store. The problem: **there's no app review process.** Any skill could contain a reverse shell, steal your API keys, or trick the AI into leaking secrets.

SkillGuard is a safety scanner for these skills. You send it a skill, and it tells you whether it's **safe**, **suspicious**, or **dangerous** — similar to how an antivirus scans a file before you open it, except:

1. **It's built for AI agent code**, not desktop malware. It understands patterns like prompt injection, LLM secret exfiltration, and obfuscated shell payloads that traditional scanners miss.
2. **Every answer comes with a cryptographic proof.** You don't have to trust the SkillGuard server. The proof mathematically guarantees the neural network actually produced that classification — anyone can verify it independently in milliseconds.
3. **It's fast enough for agents to use in real time.** Classification with a full cryptographic proof takes ~4 seconds. For instant lookups, use the catalog endpoint which returns pre-computed results in <10ms.

### Why would an agent use this?

SkillGuard is a **first line of defense** — not a complete solution, but a fast, cheap, verifiable check that catches the obvious threats before anything else runs.

An AI agent about to install a community skill from [OpenClaw](https://openclaw.ai) can call SkillGuard first:
- **ALLOW** → install the skill
- **FLAG** → ask the human for approval
- **DENY** → refuse to install

Without SkillGuard, the agent has no signal at all — it either trusts everything (dangerous) or trusts nothing (useless). SkillGuard gives it a concrete, verifiable answer in under 100ms. It won't catch every attack (novel obfuscation, time-delayed payloads, runtime-only behavior), but it reliably catches reverse shells, credential harvesters, obfuscated payloads, and other common patterns. Pair it with sandboxing, human review for flagged skills, and runtime monitoring for a proper defense-in-depth setup.

### Current model performance

SkillGuard is a **first line of defense**, not a complete security solution. The model is tuned to catch dangerous skills reliably — the metric that matters most is whether dangerous skills get through, not whether safe skills get the perfect label.

**Safety metrics (what matters):**

| Metric | v2.3 (current) | v3.0 (target) |
|--------|---------------:|:--------------|
| DANGEROUS catch rate (MLP) | **91.3%** | ≥ 95% |
| DANGEROUS recall (i32 holdout) | **95.9%** | ≥ 95% |
| Rule-based safety floor | 7 danger + 1 safe rule | — |
| Binary DANGEROUS-vs-rest accuracy | 84.7% | ≥ 95% |

**Detailed per-class metrics:**

| Metric | v2.3 (current) | v3.0 (target) |
|--------|---------------:|:--------------|
| DANGEROUS recall / F1 | 91.3% / 0.76 | ≥ 95% / ≥ 0.82 |
| SAFE recall / F1 | 48.2% / 0.52 | ≥ 75% / ≥ 0.75 |
| CAUTION recall / F1 | 57.2% / 0.61 | ≥ 65% / ≥ 0.78 |
| 3-class accuracy | 63.0% | ≥ 80% |
| Regression tests | 111 passing | 111 passing |

v2.3 uses a **three-layer defense**: (1) the MLP classifier tuned for high DANGEROUS recall with danger-sensitive loss (weight=20), (2) a deterministic **danger floor** (7 rules) that overrides to DANGEROUS when pattern matching detects unambiguous threats — reverse shells, data exfiltration, credential harvesting, curl|bash, privilege escalation with downloads, and LLM secret exposure, and (3) a **safe floor** that prevents false positives by downgrading DANGEROUS classifications when all risk features are zero.

The 63.0% 3-class accuracy reflects the deliberate trade-off: the model is tuned to maximize DANGEROUS recall (91.3%, up from 80.2%) at the cost of SAFE/CAUTION confusion. The i32 fixed-point model achieves 95.9% DANGEROUS recall on the holdout set.

The v3.0 training pipeline will scale to 1,400+ labeled skills for further accuracy improvements.

---

## What Is This?

SkillGuard is **SSL for the AI agent skill supply chain**. It answers a simple question: **"Is this skill safe to install?"** — and backs every answer with a cryptographic certificate anyone can verify.

AI agents on platforms like [OpenClaw](https://openclaw.ai) can install community-created "skills" — small packages of code and instructions that give an agent new abilities (calling APIs, writing files, running scripts, etc.). Some skills might be malicious: they could steal credentials, open reverse shells, or trick the AI into leaking secrets.

SkillGuard inspects each skill and classifies it as **SAFE**, **CAUTION**, or **DANGEROUS**. It then makes a decision: **ALLOW**, **FLAG**, or **DENY**.

Just as an SSL certificate proves a server is who it claims to be, every SkillGuard classification comes with a **zero-knowledge machine learning proof** — a cryptographic certificate proving the classification was computed correctly by a specific model. Anyone can verify this proof without trusting the SkillGuard operator and without seeing the model's internal weights.

### How It Works

1. **Skill submitted** — A developer publishes a skill to [ClawHub](https://clawhub.ai), or submits data directly via API.

2. **Features extracted** — SkillGuard reads the skill's documentation, scripts, and metadata, then extracts 45 numeric features that capture security-relevant signals (shell execution calls, reverse shell patterns, credential access, obfuscation techniques, entropy analysis, author reputation, download counts, interaction terms, density ratios, cross-features like credential+exfiltration co-occurrence, etc.). When a skill only has a SKILL.md file (no separate scripts), SkillGuard extracts code blocks from the markdown and analyzes them as if they were script files.

3. **Classified with proof** — The 45 features feed into a small neural network (3-layer MLP, 4,979 parameters). The entire forward pass runs inside a SNARK virtual machine ([Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas)), producing a ~53 KB cryptographic proof that the classification was computed correctly.

4. **Anyone verifies** — Anyone can verify a proof by posting it to `/api/v1/verify`. Verification is free, takes milliseconds, and requires no API key.

5. **Classification made** — The result (ALLOW, FLAG, or DENY) plus the proof become a tamperproof safety certificate for the skill.
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
| POST | `/api/v1/evaluate` | None | Free | Classify a skill with mandatory zkML proof (auto-detects name lookup vs full data) |
| GET | `/api/v1/catalog/{name}` | None | Free | Instant cached classification lookup |
| POST | `/api/v1/verify` | None | Free | Verify a zkML proof |
| GET | `/health` | None | Free | Health check (includes `zkml_enabled`, `model_version`) |
| GET | `/stats` | None | Free | Usage statistics and proof counts |
| POST | `/api/v1/feedback` | None | Free | Submit classification feedback/disputes |
| GET | `/openapi.json` | None | Free | OpenAPI 3.1 specification |
| GET | `/.well-known/ai-plugin.json` | None | Free | AI agent discovery manifest |
| GET | `/.well-known/llms.txt` | None | Free | LLM-readable API description |
| GET | `/` | None | Free | Web dashboard |

The `/api/v1/evaluate` endpoint accepts two request formats:
- **Name lookup:** `{"skill": "skill-slug"}` — fetches skill data from ClawHub, then classifies
- **Full skill data:** `{"skill": {"name": "...", "version": "...", ...}}` — classifies directly

Both formats return the same response with classification, confidence, scores, reasoning, and a mandatory zkML proof bundle. Every classification is cryptographically verified — there is no proof-optional mode. Every response includes a `model_version` field.

---

## How the Model Works (Plain English)

SkillGuard's brain is a small neural network — a program that learned to spot dangerous patterns by studying real-world examples of safe and malicious skills from the OpenClaw ecosystem.

### What it looks at

When a skill is submitted, SkillGuard doesn't try to "understand" the code the way a human would. Instead, it counts things. It reads through the skill's documentation, scripts, and metadata and produces 45 numbers — a kind of fingerprint. These numbers capture questions like:

- **How many times does this skill try to run shell commands?** Legitimate tools might run one or two; malware often runs many.
- **Does it download and execute anything from the internet?** A `curl | bash` pattern is a classic attack vector.
- **Are there reverse shell patterns?** Code that opens a connection back to an attacker's server is almost never legitimate.
- **Is the code obfuscated?** Base64-encoded eval() calls, character code assembly, and similar tricks are red flags.
- **How old is the author's account? How many stars does the skill have?** Brand-new accounts with no history publishing skills that request elevated permissions deserve extra scrutiny.
- **What's the entropy of the script bytes?** Encrypted or heavily encoded payloads have unusually high randomness.
- **How dense are the suspicious patterns?** One shell exec in a 500-line script is normal; ten in a 20-line script is suspicious.

Each of these 45 measurements is scaled to a number between 0 and 128, creating a fixed-size numeric fingerprint regardless of how big or complex the original skill is.

### How it decides

The fingerprint feeds into a **3-layer neural network** — three stacked layers of simple math operations (multiply, add, apply a threshold). The network has 4,979 tunable parameters (weights) that were learned during training.

- **Layer 1** (45 → 56 neurons): Takes the 45 features and mixes them through 56 neurons. Each neuron learns a different combination — one might activate when it sees "high obfuscation + shell exec + new account," while another fires on "network calls + credential access."
- **Layer 2** (56 → 40 neurons): Combines the first layer's patterns into higher-level concepts. This is where the network builds compound indicators like "this looks like a credential stealer" vs "this looks like a legitimate API client."
- **Layer 3** (40 → 3 outputs): Produces three scores — one for each safety class: SAFE, CAUTION, DANGEROUS. The highest score wins.

The raw output scores are converted to probabilities using a **softmax function** (with a calibrated temperature of T=0.95, ECE=0.045). This turns the scores into percentages that sum to 100%, giving a confidence level for each class.

### How it handles uncertainty

The model doesn't just pick the top class and move on. It checks how confident it is:

- If the top probability is high and the others are low (low **entropy**), the model is confident. SAFE/CAUTION skills get **ALLOW**. DANGEROUS skills get **DENY**.
- If the probabilities are spread out (high entropy, above 0.67 normalized), the model isn't sure. These predictions get **FLAG**ged for human review regardless of the top class.
- DANGEROUS classifications with less than 50% confidence also get **FLAG** instead of **DENY** — the model errs on the side of caution rather than blocking something it's unsure about.

### Why fixed-point arithmetic?

All the math inside the network uses integers instead of floating-point numbers (every weight is multiplied by 128 and stored as an `i32`). This is unusual for neural networks, but it's required because the entire forward pass runs inside a **zero-knowledge proof system** (Jolt Atlas). ZK circuits work with integers, not floats.

The training process uses **quantization-aware training (QAT)** — during training, every forward pass simulates the exact Rust i32 integer path: `matmul → (result + 64) // 128 → add bias → ReLU`. Gradients flow through these integer operations using straight-through estimators (STE). This means the model learns to produce correct classifications despite the rounding and truncation that integer arithmetic introduces. The result: zero gap between the Python training model and the Rust production inference.

### What it was trained on

The current model (v2.3) is trained on 619 LLM-labeled real OpenClaw skills plus augmented samples, across three safety classes. v2.3 restored DANGEROUS recall to 91.3% (from 80.2% in v2.2) through danger-sensitive loss (weight=20), DANGEROUS-priority checkpoint selection, improved hard negative mining, and aggressive SMOTE oversampling. A three-layer defense ensures safety: MLP classifier, 7 deterministic danger-floor rules for pattern-matched threats, and a safe-floor rule that prevents false positives on trivially benign skills. The v3.0 pipeline scales this to 1,400+ labeled skills through a multi-phase data collection process:

- **Safe skills:** Documentation tools, calculators, formatters, shell wrappers for standard tools (git, npm, docker) — typical utility skills with no concerning patterns.
- **Caution skills:** Legitimate tools that use shell commands, network calls, or file writes in normal ways, access multiple API keys, or have minimal metadata.
- **Dangerous skills:** Credential harvesters, reverse shells, obfuscated payloads, persistence installers, crypto miners, privilege escalation, data exfiltration, multi-vector attacks.

The training pipeline uses Claude API to label skills at scale with an improved prompt that includes explicit SAFE/CAUTION boundary rules and few-shot examples, reducing mislabeling at class boundaries. Data collection is phased: DANGEROUS-priority labeling first (fixing the worst class imbalance), then SAFE/CAUTION boundary enrichment (targeting skills where the model is most confused), then stratified diversity fill across categories and confidence bands. A human review queue prioritizes the highest-impact labels for manual verification. Training uses **adversarial examples** (FGSM perturbations on 30% of batches) and a **held-out test set** (15% stratified split) for unbiased final evaluation.

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | 3-layer MLP: 45→56→40→3 (ReLU). 4,979 parameters. Fixed-point i32 arithmetic (scale=7, rounding division). QAT with exact i32 simulation + FGSM adversarial training. v2.3: 619 LLM-labeled real skills + augmented, danger-sensitive loss. Three-layer defense: MLP + 7 danger-floor rules + safe-floor rule. v3.0 target: 1,400+ real skills. |
| Proving | [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) SNARK with Dory commitment (BN254 curve). ~53 KB proofs, ~4s proving time. |
| Payment | Free (no payment required) |
| Server | Axum async HTTP. LRU per-IP rate limiting (IPv6 /64 aggregation), constant-time API key auth, CORS, graceful shutdown, JSONL access logging. |
| MCP | stdio-based MCP server (`skillguard-mcp`) for zero-code agent integration via `skillguard_evaluate` tool. |
| Runtime | Docker on Render. Rust nightly. Pre-generated Dory SRS bundled in image. |

### Feature List

The classifier extracts 45 features from each skill:

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
| 28 | `string_obfuscation_score` | Hex escapes, join(), chr(), Unicode confusables, split-string evasion |
| 29 | `shell_exec_per_line` | Shell execution density (calls / script lines) |
| 30 | `network_per_script` | Network call density (calls / script count) |
| 31 | `credential_density` | Credential pattern density (patterns / doc lines) |
| 32 | `shell_and_network` | Shell + network co-occurrence (interaction term) |
| 33 | `obfuscation_and_exec` | Obfuscation + execution co-occurrence (interaction term) |
| 34 | `file_extension_diversity` | Count of unique file extensions in the skill package |
| 35 | `has_shebang` | Whether any script starts with `#!` |
| 36 | `documented_shell_ratio` | Shell commands in markdown prose vs. raw scripts (high = legitimate tutorial/tool) |
| 37 | `has_readme_or_docs` | Skill has README, docs, or substantial markdown (>50 lines) |
| 38 | `safe_tool_patterns` | Matches for known-safe tool patterns (git, npm, pip, cargo, docker, make, etc.) |
| 39 | `suspicious_url_ratio` | Ratio of non-HTTPS/non-standard domains to total domains (IP addresses, raw ports, ngrok) |
| 40 | `code_to_prose_ratio` | Code block lines / prose lines (high = mostly code, no explanation) |
| 41 | `credential_and_exfil` | Credential access + outbound data co-occurrence (credential harvesting signal) |
| 42 | `obfuscation_and_privilege` | Obfuscation + privilege escalation co-occurrence (hiding root actions) |
| 43 | `undocumented_risk` | Shell exec weighted by lack of documentation (undocumented commands = suspicious) |
| 44 | `risk_signal_count` | Count of active risk categories (download, priv_esc, llm_secret, etc.) |
| 45 | `stealth_composite` | Obfuscation weighted by lack of comments (obfuscated + uncommented = hiding) |

---

## Model Training

SkillGuard includes a full training pipeline in `training/` for reproducing or improving the classifier.

### Architecture

| Property | Value |
|----------|-------|
| Architecture | 45→56→40→3 MLP (ReLU activations, no output activation) |
| Parameters | 4,979 |
| Classes | 3 (SAFE, CAUTION, DANGEROUS) |
| Arithmetic | Fixed-point i32, scale=7 (×128), rounding division `(x+64)/128` |
| Training | QAT with exact i32 integer-division simulation + STE gradient flow |
| Adversarial | FGSM perturbations during training (ε=1.0, 30% of batches) |
| Augmentation | 35 DANGEROUS + SAFE sparse anchor samples + SMOTE oversampling (0.28 target ratio) |
| Loss | Danger-sensitive loss (weight=20) with focal-gamma=1.0 |
| Validation | 5-fold stratified cross-validation + 15% held-out test set |
| Dataset | 619 LLM-labeled real OpenClaw skills + augmented (v2.3), 1,400+ target (v3.0) |
| Calibration | Softmax temperature T=0.95 on integer logits, ECE=0.045 |
| Entropy threshold | 0.67 normalized (5% flag rate) |
| Model version | v2.3 (2026-02-25), v3.0 pipeline in progress |

### Training Pipeline

The training pipeline (`training/`) supports both synthetic and real data modes:

```bash
# Real data mode (3-class, LLM-labeled) — recommended
python train.py --dataset real --num-classes 3 --danger-fn-weight 20.0 --focal-gamma 1.0 --augment-dangerous 35 --oversample-dangerous 0.28 --adv-epsilon 1.0 --holdout-fraction 0.15 --export

# Synthetic data mode (4-class, legacy)
python train.py --dataset synthetic --export
```

**Real data pipeline:**
1. `training/fetch_and_label.py` — Fetches SKILL.md from ClawHub, labels via Claude API. Supports `--prioritize-dangerous` (DANGEROUS-class enrichment) and `--prioritize-boundary` (SAFE/CAUTION boundary enrichment)
2. `training/llm_label.py` — Claude-based labeling with boundary-aware prompt and few-shot examples
3. `training/stratified_sample.py` — Stratified sampling across categories and confidence bands for diversity fill
4. `training/dataset.py` — Loads labeled data, extracts features via `skillguard extract-features`, generates augmented samples
5. `training/train.py` — QAT training with exact i32 simulation, class weighting, FGSM adversarial examples, 5-fold CV, optional held-out test set (`--holdout-fraction`)
6. `training/generate_review_queue.py` — Generates prioritized human review queue (DANGEROUS labels, MLP/LLM disagreements, short reasoning)
7. `training/export_weights.py` — Exports fixed-point weights to Rust `model.rs`, validates against Rust i32 simulation
8. `training/calibrate.py` — Calibrates softmax temperature on QAT integer-scale logits

### Key Design Decisions

- **3 classes (not 4):** The original MALICIOUS class was merged into DANGEROUS. In practice, the distinction between "dangerous" and "malicious" was subjective and inconsistent. Three classes (safe / use-with-caution / block) map cleanly to agent policy actions.
- **LLM-labeled data:** Claude API labels real OpenClaw skills at scale, replacing synthetic data. The labeling prompt includes explicit boundary rules (SAFE vs CAUTION thresholds for env var access, shell wrapper patterns) and few-shot examples to reduce mislabeling at class boundaries.
- **Phased data collection:** Priority DANGEROUS labeling first (fixing class imbalance from 14 real examples), then SAFE/CAUTION boundary enrichment (targeting model-confused skills by `|SAFE - CAUTION|` score proximity), then stratified diversity fill across categories.
- **Held-out evaluation:** A 15% stratified holdout set provides unbiased accuracy measurement independent of cross-validation, catching overfitting that CV alone might miss.
- **Human review queue:** `generate_review_queue.py` prioritizes label verification by impact: DANGEROUS labels first (wrong DANGEROUS = false denial in production), then MLP/LLM disagreements, then short reasoning.
- **QAT with exact i32 simulation:** The forward pass during training simulates the exact Rust integer arithmetic path (`matmul → (result + 64) // 128 → add bias → relu`) using straight-through estimators. This eliminates the quantization gap between float training and integer inference — the Python model and Rust binary produce identical outputs.
- **Sparse anchor augmentation:** Reduced from 80 to 20 synthetic DANGEROUS samples as real labeled data replaces synthetic. SAFE anchor samples ensure the model handles high-metadata feature combinations.
- **Class weighting:** Inverse-frequency weighting handles the natural imbalance (most skills are safe).
- **Feature parity:** The `skillguard extract-features` CLI subcommand ensures training uses the exact same feature extraction as production.

### Reproducing

```bash
cd training
pip install -r requirements.txt

# Phase 1: Label DANGEROUS-priority skills (requires ANTHROPIC_API_KEY)
python fetch_and_label.py --scan-report scan-report.json --prioritize-dangerous --fetch-limit 200

# Phase 2: Label SAFE/CAUTION boundary skills
python fetch_and_label.py --scan-report scan-report.json --prioritize-boundary --fetch-limit 300

# Phase 3: Stratified diversity fill
python stratified_sample.py --scan-report scan-report.json --existing training/real-labels.json --target-count 500
python fetch_and_label.py --scan-report training/stratified-candidates.json --fetch-limit 500

# Phase 4: Generate human review queue
python generate_review_queue.py

# Phase 5: Train with holdout evaluation
python train.py --dataset real --num-classes 3 --danger-fn-weight 20.0 --focal-gamma 1.0 --augment-dangerous 35 --oversample-dangerous 0.28 --adv-epsilon 1.0 --holdout-fraction 0.15 --export

# Calibrate temperature on QAT integer-scale logits
cd .. && python training/calibrate.py --dataset real --num-classes 3

# Export & validate fixed-point weights
python training/export_weights.py --num-classes 3 --dataset real --validate --output data/weights.rs

# Run all tests (111 tests: 58 unit + 24 integration + 29 regression)
cargo test
```

Or use the automated retrain script:

```bash
bash training/retrain.sh
```

---

## Batch Scanning

SkillGuard includes CLI commands for crawling and batch-scanning OpenClaw skills. These are behind the `crawler` feature gate:

```bash
cargo build --release --features crawler
```

### Crawl

Fetch SKILL.md files from the [awesome-openclaw-skills](https://github.com/OpenClaw/awesome-openclaw-skills) list:

```bash
skillguard crawl --output-dir crawled-skills --concurrency 5 --delay-ms 200
```

| Flag | Description | Default |
|------|-------------|---------|
| `--awesome-url` | URL of the awesome list README (raw markdown) | built-in |
| `--limit` | Maximum skills to crawl (0 = all) | `0` |
| `--concurrency` | Maximum concurrent fetches | `5` |
| `--delay-ms` | Delay between fetches in milliseconds | `200` |
| `--output-dir` | Output directory for crawled SKILL.md files | `crawled-skills` |

### Scan

Batch-classify skills and produce a report. Supports two modes:

**Live mode** — fetch and classify directly from the awesome list:
```bash
skillguard scan --from-awesome --format json --output scan-report.json
```

**Directory mode** — classify previously crawled skills:
```bash
skillguard scan --input-dir crawled-skills --format csv --output scan-report.csv
```

| Flag | Description | Default |
|------|-------------|---------|
| `--from-awesome` | Fetch and scan from the awesome list (live mode) | `false` |
| `--input-dir` | Input directory of crawled SKILL.md files (directory mode) | `crawled-skills` |
| `--format` | Output format: `json`, `csv`, or `summary` | `summary` |
| `--output` | Output file (defaults to stdout) | stdout |
| `--filter` | Filter by classification (comma-separated, e.g. `DANGEROUS,CAUTION`) | all |
| `--concurrency` | Maximum concurrent classifications | `5` |
| `--limit` | Max skills to scan from awesome list (live mode, 0 = all) | `0` |

---

## MCP Server

SkillGuard includes an MCP (Model Context Protocol) server for zero-code integration with AI agents like Claude Code. The MCP server exposes a `skillguard_evaluate` tool over stdio transport with full ZK proof parity — every response includes the same cryptographic proof, raw logits, entropy, and model hash as the HTTP API.

### Setup

**Claude Code** (one command):

```bash
claude mcp add skillguard -- cargo run --release --bin skillguard-mcp
```

Or from a pre-built binary:

```bash
cargo build --release --bin skillguard-mcp
claude mcp add skillguard -- ./target/release/skillguard-mcp
```

**Other MCP hosts** (Cursor, Windsurf, etc.) — use the config at [`mcp.json`](mcp.json):

```bash
claude mcp add-from-file mcp.json   # or copy into your host's MCP config
```

**Manual config** — add to your MCP host's settings:

```json
{
  "mcpServers": {
    "skillguard": {
      "command": "/path/to/skillguard-mcp"
    }
  }
}
```

### Usage

The MCP server exposes one tool:

- **`skillguard_evaluate`** — Classify a skill by name or by raw SKILL.md content.

Parameters (provide one):
- `skill_name` (string) — Skill name for minimal evaluation
- `skill_md` (string) — Raw SKILL.md content to classify directly

Response fields (identical to the HTTP API):

| Field | Description |
|-------|-------------|
| `classification` | SAFE, CAUTION, or DANGEROUS |
| `decision` | Human-readable allow/warn/block decision |
| `confidence` | Model confidence (0–1) |
| `scores` | Per-class softmax probabilities |
| `reasoning` | Plain-language explanation |
| `raw_logits` | Raw model output logits |
| `entropy` | Shannon entropy of score distribution (0–1) |
| `model_hash` | SHA-256 of the model weights |
| `proof` | ZK proof bundle (`proof_b64`, `program_io`, `proof_size_bytes`, `proving_time_ms`) |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SKILLGUARD_API_KEY` | Bearer token for API authentication. If unset, all endpoints are open. | (none) |
| `SKILLGUARD_SKIP_PROVER` | Set to `1` to disable the ZKML prover. | `0` |
| `REDIS_URL` | Redis connection URL for durable metrics persistence. If set, counters are persisted to Redis in addition to disk and survive container redeployments. | (none) |
| `RUST_LOG` | Log level filter. | `info` |

See `.env.example` for a documented template of all variables.

---

## Links

- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) — ZKML proving stack
- [Jolt](https://github.com/a16z/jolt) — SNARK VM by a16z
- [OpenClaw](https://openclaw.ai) — Open framework for AI agent skills
- [ClawHub](https://clawhub.ai) — Registry for OpenClaw skills
- [Novanet](https://novanet.xyz) — Verifiable inference network

---

## License

MIT
