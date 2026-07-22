# Stage 1: Build
FROM rust:latest AS builder

WORKDIR /app

# System dependencies (cached unless changed)
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy manifests first (cached unless Cargo.toml/Cargo.lock change)
COPY Cargo.toml Cargo.lock ./

# Download all dependencies into the image layer cache
RUN cargo fetch

# Copy the rest of the source code
COPY . .

# Build release binary (uses cached deps from cargo fetch)
RUN cargo build --release

# Stage 2: Runtime (Hardened, ultra-minimal Distroless + Busybox for healthcheck)
FROM gcr.io/distroless/cc-debian12

# Copy busybox for healthcheck tools (wget)
COPY --from=busybox:stable-musl /bin/busybox /bin/busybox

WORKDIR /app

# Copy the compiled release binary
COPY --from=builder /app/target/release/cascade-llm .

# Expose the gateway port
EXPOSE 3000

# Run the binary
CMD ["./cascade-llm"]