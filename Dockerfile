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
# Pin to a known-good nightly to avoid breakage from compiler updates.
# We use a base Debian image + rustup instead of rustlang/rust:nightly-*
# because date-pinned bookworm tags don't exist on Docker Hub.
FROM debian:bookworm AS builder

# Install build dependencies and rustup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pinned nightly toolchain via rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain nightly-2026-01-29 && \
    rustc --version

WORKDIR /build

# Limit parallel jobs to avoid OOM with ZKML deps
ENV CARGO_BUILD_JOBS=1

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

# Touch main.rs to force cargo to re-link even if Docker layer is cached
RUN touch src/main.rs && cargo build --release --bin skillguard

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /build/target/release/skillguard /app/skillguard

# Copy data directory (registry)
COPY data/ /app/data/

# Pre-generated Dory SRS file (avoids runtime generation which can OOM)
COPY dory_srs_24_variables.srs /app/dory_srs_24_variables.srs

# Render sets PORT=10000 for web services
ENV PORT=10000
EXPOSE 10000

CMD /app/skillguard serve --bind "0.0.0.0:${PORT}" --rate-limit 30
