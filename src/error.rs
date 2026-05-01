use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("node not found: {0}")]
    NodeNotFound(String),

    #[error("no edge from node: {0}")]
    NoEdge(String),

    #[error("no entry point set")]
    NoEntryPoint,

    #[error("execution step limit exceeded ({0} steps)")]
    StepLimitExceeded(usize),

    #[error("node error in '{node}': {message}")]
    Node { node: String, message: String },

    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    #[error("graph interrupted at '{node}': {reason}")]
    Interrupted { node: String, reason: String },

    #[error("tool call failed for '{tool}': {message}")]
    ToolCallFailed { tool: String, message: String },
}

pub type Result<T> = std::result::Result<T, GraphError>;
