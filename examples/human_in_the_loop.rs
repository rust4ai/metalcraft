//! Example: human-in-the-loop with interrupt/resume.
//!
//! An agent that drafts a response, then pauses for human approval
//! before sending. Demonstrates metalcraft's interrupt/resume pattern.

use async_trait::async_trait;
use metalcraft::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ReviewState {
    task: String,
    draft: Option<String>,
    approved: bool,
    final_output: Option<String>,
}

enum Update {
    SetDraft(String),
    Approve,
    SetFinalOutput(String),
}

impl Reducer for ReviewState {
    type Update = Update;
    fn apply(&mut self, update: Update) {
        match update {
            Update::SetDraft(d) => self.draft = Some(d),
            Update::Approve => self.approved = true,
            Update::SetFinalOutput(o) => self.final_output = Some(o),
        }
    }
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

/// Generates a draft response.
struct DraftNode;

#[async_trait]
impl Node<ReviewState> for DraftNode {
    async fn run(&self, state: &ReviewState) -> Result<NodeOutcome<Update>> {
        println!("  [draft] Generating draft for: {}", state.task);
        let draft = format!(
            "Dear customer, regarding '{}': we've investigated and resolved the issue. \
             Your account has been credited $50.",
            state.task
        );
        println!("  [draft] Draft: {draft}");
        Ok(NodeOutcome::Update(Update::SetDraft(draft)))
    }
}

/// Pauses for human review. This is where the interrupt happens.
struct ReviewNode;

#[async_trait]
impl Node<ReviewState> for ReviewNode {
    async fn run(&self, state: &ReviewState) -> Result<NodeOutcome<Update>> {
        if state.approved {
            // Already approved (set during resume), continue
            println!("  [review] Already approved, continuing...");
            Ok(NodeOutcome::Update(Update::Approve)) // no-op but keeps types happy
        } else {
            // Interrupt and wait for human
            println!("  [review] ⏸ Waiting for human approval...");
            Ok(NodeOutcome::interrupt(format!(
                "Please review this draft: {}",
                state.draft.as_deref().unwrap_or("(no draft)")
            )))
        }
    }
}

/// Sends the approved response.
struct SendNode;

#[async_trait]
impl Node<ReviewState> for SendNode {
    async fn run(&self, state: &ReviewState) -> Result<NodeOutcome<Update>> {
        let output = format!(
            "SENT: {}",
            state.draft.as_deref().unwrap_or("(no draft)")
        );
        println!("  [send] {output}");
        Ok(NodeOutcome::Update(Update::SetFinalOutput(output)))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== metalcraft human-in-the-loop example ===\n");

    let graph = Graph::<ReviewState>::new()
        .add_node("draft", DraftNode)
        .add_node("review", ReviewNode)
        .add_node("send", SendNode)
        .add_edge("draft", "review")
        .add_edge("review", "send")
        .add_edge("send", END)
        .set_entry("draft")
        .compile()?;

    let cp = Arc::new(MemoryCheckpointer::<ReviewState>::new());
    let executor = Executor::new(graph).with_checkpointer(cp);

    let initial = ReviewState {
        task: "Billing dispute #4521".to_string(),
        draft: None,
        approved: false,
        final_output: None,
    };

    // --- First run: will hit the interrupt ---
    println!("--- Run 1: Agent drafts, then pauses for review ---\n");
    let outcome = executor.run(initial, "thread-review").await?;

    match &outcome {
        RunOutcome::Interrupted {
            reason,
            resume_from,
            ..
        } => {
            println!("\n  ⏸ INTERRUPTED: {reason}");
            println!("  Resume from: {resume_from}");
        }
        RunOutcome::Completed(_) => {
            println!("\n  (completed unexpectedly)");
            return Ok(());
        }
    }

    // --- Simulate human approval ---
    println!("\n--- Human approves the draft ---\n");

    // Resume with the Approve update injected
    let outcome = executor
        .resume("thread-review", Some(Update::Approve))
        .await?;

    match outcome {
        RunOutcome::Completed(state) => {
            println!("\n=== Workflow completed ===");
            println!("Final output: {:?}", state.final_output);
        }
        RunOutcome::Interrupted { reason, .. } => {
            println!("\n  ⏸ Interrupted again: {reason}");
        }
    }

    Ok(())
}
