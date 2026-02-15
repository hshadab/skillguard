.PHONY: build test lint fmt audit docker clean

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

setup-hooks:
	@mkdir -p .git/hooks
	@printf '#!/bin/sh\ncargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings\n' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "pre-commit hook installed (fmt + clippy)"
