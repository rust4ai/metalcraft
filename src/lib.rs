//! metalcraft: a stateful graph orchestrator for AI agents in Rust.
//!
//! Inspired by LangGraph's orchestration model — typed state, reducer updates,
//! cyclic graphs, checkpointed execution, streaming, and interrupts — but built
//! idiomatically in Rust with compile-time safety guarantees.
//!
//! For LLM integration, use the `rig` feature flag which re-exports the
//! [Rig](https://docs.rig.rs) crate for provider-agnostic model access.

mod graph;
mod executor;
mod checkpoint;
mod error;
pub mod tools;

pub use error::{GraphError, Result};
pub use graph::{Graph, CompiledGraph, Edge, Node, NodeOutcome, Reducer, START, END};
pub use executor::{Executor, StepEvent, RunOutcome};
pub use checkpoint::{Checkpointer, MemoryCheckpointer};
pub use tools::{Tool, ToolRegistry};

/// Re-export Rig when the `rig` feature is enabled.
#[cfg(feature = "rig")]
pub use rig;
