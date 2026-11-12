# Stage 1: Build
FROM rust:latest AS builder

WORKDIR /app

# System dependencies (cached unless changed)
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy manifests first (cached unless Cargo.toml/Cargo.lock change)
COPY Cargo.toml Cargo.lock ./

# Download all dependencies into the image layer cache
RUN cargo fetch

# ★ Pre-compile the DEPENDENCY graph in its own layer.
#   `cargo fetch` only DOWNLOADS the crates — COMPILING them (axum, reqwest,
#   tokio, serde, …) is ~90 % of the build time. Without this layer the
#   single `cargo build --release` below also compiles every dependency, so
#   ANY change in src/ (which invalidates `COPY . .`) rebuilds axum, reqwest
#   and tokio from scratch: ~3 min per build, all of it wasted.
#   With the dummy source the dependency artifacts live in a layer that stays
#   cached as long as Cargo.toml/Cargo.lock do; a src/-only change then
#   recompiles just this crate (~30 s).
#   The dummy lib/bin are deliberately dependency-free stubs: the real
#   main.rs uses `cascade_llm::…`, which does not exist yet at this point.
RUN mkdir -p src \
 && echo 'fn main() {}' > src/main.rs \
 && echo '// dummy crate for dependency pre-compilation' > src/lib.rs \
 && cargo build --release \
 && rm -rf src

# Copy the rest of the source code
COPY . .

# Build release binary (deps are already compiled -> only this crate rebuilds)
#
# ★ `touch` is load-bearing. Cargo fingerprints a crate by the mtime of its
#   sources, and `COPY` preserves the mtime from the build context — which can
#   be OLDER than the artifact left behind by the dummy dependency layer. Cargo
#   then declares the crate fresh, compiles nothing and the image silently
#   ships the PREVIOUS binary: "the build succeeded" and "the binary is the one
#   you just wrote" are two different claims (gotcha 109). Touching the
#   sources after the copy makes the fingerprint newer than the artifact, so a
#   changed source ALWAYS rebuilds.
RUN find src -name '*.rs' -exec touch {} + \
 && cargo build --release

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