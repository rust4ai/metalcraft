//! Streaming events emitted during graph execution.
//!
//! When an [`EventSender`] is provided (via [`AgentState::with_events`]),
//! nodes emit fine-grained events for tool calls, LLM invocations, and
//! state transitions — enabling real-time SSE streaming to clients.

use serde::Serialize;
use tokio::sync::mpsc;

/// A channel sender for streaming events. Clone-safe, Send + Sync.
pub type EventSender = mpsc::Sender<StreamEvent>;

/// A channel receiver for consuming streaming events.
pub type EventReceiver = mpsc::Receiver<StreamEvent>;

/// Create a bounded event channel. Returns (sender, receiver).
///
/// The sender can be attached to [`AgentState`] via [`AgentState::with_events`].
/// The receiver yields events as the agent executes.
pub fn event_channel(buffer: usize) -> (EventSender, EventReceiver) {
    mpsc::channel(buffer)
}

/// Events emitted during agent execution.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// LLM completion request started.
    LlmStart,

    /// LLM completion finished. Reports whether tool calls were requested.
    LlmEnd {
        tool_calls: usize,
        has_answer: bool,
    },

    /// A tool is about to be executed.
    ToolStart {
        name: String,
        call_id: String,
    },

    /// A tool finished executing.
    ToolEnd {
        name: String,
        call_id: String,
        is_error: bool,
        duration_ms: u64,
    },

    /// Agent produced a final answer.
    Done {
        answer: String,
    },

    /// Agent execution errored.
    Error {
        message: String,
    },
}

/// Helper to send an event without blocking or panicking if receiver dropped.
pub(crate) fn emit(tx: &EventSender, event: StreamEvent) {
    let _ = tx.try_send(event);
}
