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
