//! Example: step guards for loop detection and error-spiral protection.
//!
//! Demonstrates `Executor::with_step_guard()` to halt runaway agents.
//! The guard runs after each step and can stop execution early when it
//! detects repeated identical actions or too many consecutive errors.

use async_trait::async_trait;
use metalcraft::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<String>,
    tool_calls: Vec<String>,
    errors: u32,
}

#[allow(dead_code)]
enum Update {
    AddMessage(String),
    ToolCall(String),
    Error,
}

impl Reducer for AgentState {
    type Update = Update;
    fn apply(&mut self, update: Update) {
        match update {
            Update::AddMessage(m) => self.messages.push(m),
            Update::ToolCall(name) => self.tool_calls.push(name),
            Update::Error => self.errors += 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Nodes — deliberately misbehaving to trigger guards
// ---------------------------------------------------------------------------

/// Agent that always wants to call a tool.
struct StubbyAgent;

#[async_trait]
impl Node<AgentState> for StubbyAgent {
    async fn run(&self, state: &AgentState) -> Result<NodeOutcome<Update>> {
        // Always calls the same tool with the same "args" — triggers loop detection
        let msg = format!("agent: calling search(\"rust\") (attempt #{})", state.tool_calls.len() + 1);
        println!("  {msg}");
        Ok(NodeOutcome::Update(Update::AddMessage(msg)))
    }
}

/// Tool that always fails — triggers error spiral detection.
struct FailingTool;

#[async_trait]
impl Node<AgentState> for FailingTool {
    async fn run(&self, _state: &AgentState) -> Result<NodeOutcome<Update>> {
        println!("  tool: search(\"rust\") -> ERROR: connection timeout");
        Ok(NodeOutcome::Update(Update::Error))
    }
}

fn route_after_agent(state: &AgentState) -> String {
    if state.tool_calls.len() >= 10 {
        // Safety net — never reached if guard works
        return END.to_string();
    }
    "tool".to_string()
}

fn route_after_tool(_state: &AgentState) -> String {
    "agent".to_string()
}

// ---------------------------------------------------------------------------
// Guard — the interesting part
// ---------------------------------------------------------------------------

/// Tracks recent call hashes and consecutive errors across steps.
struct GuardState {
    recent_hashes: Vec<u64>,
    consecutive_errors: u32,
}

impl GuardState {
    fn new() -> Self {
        Self {
            recent_hashes: Vec::new(),
            consecutive_errors: 0,
        }
    }
}

fn hash_action(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Build a step guard that detects loops and error spirals.
///
/// This is the pattern — adapt thresholds and detection logic to your needs.
fn make_guard(
    max_consecutive_errors: u32,
    max_repeated_calls: usize,
) -> StepGuard<AgentState> {
    let guard_state = Arc::new(Mutex::new(GuardState::new()));

    Arc::new(move |state: &AgentState, event: &StepEvent| {
        let mut gs = guard_state.lock().unwrap();

        // --- Error spiral detection ---
        // Check if the last step was an error
        if event.node == "tool" && state.errors > 0 {
            gs.consecutive_errors = state.errors;
        }
        if gs.consecutive_errors >= max_consecutive_errors {
            return GuardAction::Stop(format!(
                "Error spiral: {} consecutive tool failures. Stopping.",
                gs.consecutive_errors
            ));
        }

        // --- Loop detection ---
        // Hash the last message to detect repeated identical actions
        if let Some(last_msg) = state.messages.last() {
            let h = hash_action(last_msg);
            let repeated = gs.recent_hashes.iter().filter(|&&x| x == h).count();
            gs.recent_hashes.push(h);

            // Keep sliding window
            if gs.recent_hashes.len() > 20 {
                gs.recent_hashes.remove(0);
            }

            if repeated >= max_repeated_calls {
                return GuardAction::Stop(format!(
                    "Loop detected: same action repeated {} times. Stopping.",
                    repeated + 1
                ));
            }
        }

        GuardAction::Continue
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== metalcraft step guard example ===\n");

    let graph = Graph::<AgentState>::new()
        .add_node("agent", StubbyAgent)
        .add_node("tool", FailingTool)
        .add_conditional("agent", route_after_agent)
        .add_conditional("tool", route_after_tool)
        .set_entry("agent")
        .compile()?;

    println!("Graph:\n{}\n", graph.to_mermaid());

    // Guard: stop after 3 consecutive errors OR 2 repeated identical calls
    let guard = make_guard(3, 2);

    let executor = Executor::new(graph)
        .with_step_guard(guard)
        .max_steps(20);

    let initial = AgentState {
        messages: vec![],
        tool_calls: vec![],
        errors: 0,
    };

    println!("Running agent (will be stopped by guard)...\n");
    let outcome = executor.run(initial, "thread-1").await?;

    match outcome {
        RunOutcome::Completed(state) => {
            println!("\nCompleted (unexpected). Steps: {}", state.tool_calls.len());
        }
        RunOutcome::Interrupted {
            reason,
            state,
            resume_from,
        } => {
            println!("\n=== Guard stopped the agent ===");
            println!("Reason: {reason}");
            println!("Was about to run: {resume_from}");
            println!("Tool calls made: {}", state.tool_calls.len());
            println!("Errors: {}", state.errors);
        }
    }

    Ok(())
}
