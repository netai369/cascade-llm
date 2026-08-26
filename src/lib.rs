//! cascade-llm — unified AI gateway with dynamic multi-node routing.
//!
//! Library target so integration tests can exercise the router, registry and
//! state machinery directly; the binary in `main.rs` is a thin bootstrap.

pub mod audio;
pub mod cascade_features;
pub mod config;
pub mod db;
pub mod handlers;
pub mod language;
pub mod media;
pub mod providers;
pub mod registry;
pub mod router;
pub mod state;
pub mod types;
