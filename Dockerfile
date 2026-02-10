# SkillGuard Docker build
#
# Build:
#   docker build -t skillguard .
#
# Run:
#   docker run -p 8080:8080 skillguard

# --- Builder stage ---
FROM rust:1.88-bookworm AS builder

WORKDIR /build

# Copy manifest first for layer caching
COPY Cargo.toml Cargo.lock* ./

# Copy source
COPY src/ src/

# Build in release mode
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
