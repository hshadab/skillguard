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
python train.py --export
python calibrate.py
python export_weights.py --validate --output data/weights.rs
cd .. && cargo test
```

See `training/README.md` for details on dataset generation and model architecture.

## Reporting Security Issues

See [SECURITY.md](SECURITY.md) for responsible disclosure guidelines.
