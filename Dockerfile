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
FROM debian:bookworm@sha256:34e7f0ae7c10a61bfbef6e1b2ed205d9b47bb12e90c50696f729a5c7a01cf1f2 AS builder

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

# Touch both lib.rs and main.rs to force cargo to recompile the crate
# (the dummy-source step caches an empty lib; we must invalidate it)
RUN touch src/lib.rs src/main.rs && cargo build --release --bin skillguard

# --- Runtime stage ---
FROM debian:bookworm-slim@sha256:98f4b71de414932439ac6ac690d7060df1f27161073c5036a7553723881bffbe

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /sbin/nologin skillguard

WORKDIR /app

# Copy binary
COPY --from=builder /build/target/release/skillguard /app/skillguard

# Strip release binary to reduce image size
RUN strip /app/skillguard

# Copy data directory (registry)
COPY data/ /app/data/

# Pre-generated Dory SRS file (avoids runtime generation which can OOM)
COPY dory_srs_24_variables.srs /app/dory_srs_24_variables.srs

# Ensure cache directory exists and is writable by skillguard user
RUN mkdir -p /var/data/skillguard-cache && chown -R skillguard:skillguard /app /var/data/skillguard-cache

# Render sets PORT=10000 for web services
ENV PORT=10000
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

USER skillguard

CMD /app/skillguard serve --bind "0.0.0.0:${PORT}" --rate-limit 30
