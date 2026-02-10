# Contributing to SkillGuard

Thank you for your interest in contributing to SkillGuard!

## Getting Started

```bash
git clone https://github.com/hshadab/skillguard.git
cd skillguard
cp .env.example .env
make setup-hooks
cargo build --release
```

Requires **Rust nightly** (for arkworks const generics).

## Code Style

- Format with `cargo fmt --all` (rustfmt `max_width=100`)
- Lint with `cargo clippy --all-targets -- -D warnings`
- All warnings are treated as errors in CI

## Testing

All tests must pass before submitting a PR:

```bash
cargo test --all
```

Regression tests in `tests/regression.rs` verify that known-safe and known-malicious skills classify correctly. If you change model weights or feature extraction, ensure these still pass.

### Crawler / Batch Scanner

The `crawl` and `scan` CLI commands are behind a feature gate. To build and test them:

```bash
cargo build --release --features crawler
cargo test --features crawler
```

See `src/crawler.rs` and `src/batch.rs` for implementation details.

## Pull Request Guidelines

1. Create a feature branch from `main`
2. Keep commits focused and atomic
3. Include tests for new functionality
4. Ensure `cargo fmt`, `cargo clippy`, and `cargo test --all` pass
5. Describe what changed and why in the PR description

## Lint and Audit

```bash
make fmt        # Format code
make lint       # Run clippy
make check-fmt  # Check formatting (CI equivalent)
make audit      # Check for known vulnerabilities
```

## Training Pipeline

To reproduce or improve the classifier:

```bash
cd training
pip install -r requirements.txt

# Label real skills (requires ANTHROPIC_API_KEY)
python fetch_and_label.py --existing-only

# Train with QAT and DANGEROUS augmentation
python train.py --dataset real --num-classes 3 --augment-dangerous 80 --export

# Calibrate softmax temperature
cd .. && python training/calibrate.py --dataset real --num-classes 3

# Export weights and validate against Rust i32 simulation
python training/export_weights.py --num-classes 3 --dataset real --validate

# Run all 111 tests
cargo test
```

See `training/` for details on dataset generation, QAT model architecture, and augmentation.

## Reporting Security Issues

See [SECURITY.md](SECURITY.md) for responsible disclosure guidelines.
