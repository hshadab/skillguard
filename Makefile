.PHONY: build test lint fmt check-fmt audit docker clean doc bench setup-hooks help

build:
	cargo build --release

test:
	cargo test --all

lint:
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt --all

check-fmt:
	cargo fmt --all -- --check

audit:
	cargo audit

docker:
	docker build -t skillguard:latest .

clean:
	cargo clean

doc:
	cargo doc --no-deps --document-private-items

bench:
	cargo bench

setup-hooks:
	@mkdir -p .git/hooks
	@printf '#!/bin/sh\ncargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings\n' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "pre-commit hook installed (fmt + clippy)"

help:
	@echo "Available targets: build test lint fmt check-fmt audit docker clean doc bench setup-hooks help"
