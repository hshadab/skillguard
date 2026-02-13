# SkillGuard ZKML Docker build
#
# Build:
#   docker build -t skillguard .
#
# Run:
#   docker run -p 8080:8080 skillguard

# --- Builder stage ---
# Nightly required: arkworks-algebra dev/twist-shout branch uses const generics
# features that need Rust >= 1.95 nightly.
FROM rustlang/rust:nightly-bookworm AS builder

WORKDIR /build

# Limit parallel jobs to avoid OOM with ZKML deps
ENV CARGO_BUILD_JOBS=1

# Override fat LTO from Cargo.toml — thin LTO uses far less memory
ENV CARGO_PROFILE_RELEASE_LTO=thin

# Copy manifest + lockfile first for layer caching
COPY Cargo.toml Cargo.lock ./

# Create dummy sources so cargo can fetch & compile dependencies first (layer cache)
RUN mkdir -p src \
    && echo 'fn main(){}' > src/main.rs \
    && echo '' > src/lib.rs \
    && cargo build --release --bin skillguard || true \
    && rm -rf src

# Copy real source and static assets
COPY src/ src/
COPY static/ static/

# Build in release mode (only re-compiles skillguard crate, deps are cached)
RUN cargo build --release --bin skillguard

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /build/target/release/skillguard /app/skillguard

# Copy data directory (registry)
COPY data/ /app/data/

EXPOSE 8080

CMD ["/app/skillguard", "serve", "--bind", "0.0.0.0:8080", "--rate-limit", "30"]
