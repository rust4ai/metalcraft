//! Example: a simple agent loop that calls tools until done.
//!
//! This mirrors the classic LangGraph agent pattern:
//!   agent → (needs tool?) → tool → agent → (done?) → END

use async_trait::async_trait;
use metalcraft::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<String>,
    tool_calls: u32,
}

enum Update {
    AddMessage(String),
    IncToolCalls,
}

impl Reducer for AgentState {
    type Update = Update;
    fn apply(&mut self, update: Update) {
        match update {
            Update::AddMessage(m) => self.messages.push(m),
            Update::IncToolCalls => self.tool_calls += 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

struct AgentNode;

#[async_trait]
impl Node<AgentState> for AgentNode {
    async fn run(&self, state: &AgentState) -> Result<NodeOutcome<Update>> {
        if state.tool_calls < 2 {
            let msg = format!(
                "agent: I need to call a tool (call #{})",
                state.tool_calls + 1
            );
            println!("  → {msg}");
            Ok(NodeOutcome::Update(Update::AddMessage(msg)))
        } else {
            let msg = "agent: I have enough info. Generating final answer.".to_string();
            println!("  → {msg}");
            Ok(NodeOutcome::Update(Update::AddMessage(msg)))
        }
    }
}

struct ToolNode;

#[async_trait]
impl Node<AgentState> for ToolNode {
    async fn run(&self, state: &AgentState) -> Result<NodeOutcome<Update>> {
        let msg = format!("tool: executed tool call #{}", state.tool_calls + 1);
        println!("  → {msg}");
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        Ok(NodeOutcome::Update(Update::IncToolCalls))
    }
}

fn route_after_agent(state: &AgentState) -> String {
    if let Some(last) = state.messages.last() {
        if last.contains("final answer") {
            return END.to_string();
        }
    }
    "tool".to_string()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== metalcraft agent loop example ===\n");

    let graph = Graph::<AgentState>::new()
        .add_node("agent", AgentNode)
        .add_node("tool", ToolNode)
        .add_conditional("agent", route_after_agent)
        .add_edge("tool", "agent")
        .set_entry("agent")
        .compile()?;

    println!("Graph (Mermaid):\n{}\n", graph.to_mermaid());

    let cp = Arc::new(MemoryCheckpointer::<AgentState>::new());
    let executor = Executor::new(graph).with_checkpointer(cp);

    let initial = AgentState {
        messages: vec![],
        tool_calls: 0,
    };

    println!("Running agent...\n");
    let outcome = executor.run(initial, "thread-1").await?;

    match outcome {
        RunOutcome::Completed(state) => {
            println!("\n=== Agent completed ===");
            println!("Messages:");
            for msg in &state.messages {
                println!("  {msg}");
            }
            println!("Total tool calls: {}", state.tool_calls);
        }
        RunOutcome::Interrupted {
            reason,
            resume_from,
            ..
        } => {
            println!("\n=== Agent interrupted ===");
            println!("Reason: {reason}");
            println!("Would resume at: {resume_from}");
        }
    }

    Ok(())
}
