# Stage 1: Build
FROM rust:1.86-slim AS builder

WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release

# Stage 2: Runtime (Hardened, ultra-minimal Distroless)
FROM gcr.io/distroless/cc-debian12

WORKDIR /app

# Copy the compiled release binary
COPY --from=builder /app/target/release/cascade-llm .

# Expose the gateway port
EXPOSE 3000

# Run the binary
CMD ["./cascade-llm"]