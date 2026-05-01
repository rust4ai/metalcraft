//! Example: graph composition with SubgraphNode.
//!
//! Demonstrates running a compiled graph as a node inside another graph.
//! The inner graph has its own state type, entry point, and edges.

use async_trait::async_trait;
use metalcraft::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Inner graph state — a simple counter
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct InnerState {
    count: u32,
    done: bool,
}

enum InnerUpdate {
    Increment,
    MarkDone,
}

impl Reducer for InnerState {
    type Update = InnerUpdate;
    fn apply(&mut self, update: InnerUpdate) {
        match update {
            InnerUpdate::Increment => self.count += 1,
            InnerUpdate::MarkDone => self.done = true,
        }
    }
}

struct IncrementNode;

#[async_trait]
impl Node<InnerState> for IncrementNode {
    async fn run(&self, state: &InnerState) -> Result<NodeOutcome<InnerUpdate>> {
        println!("    [inner] incrementing count to {}", state.count + 1);
        if state.count + 1 >= 3 {
            Ok(NodeOutcome::Update(InnerUpdate::MarkDone))
        } else {
            Ok(NodeOutcome::Update(InnerUpdate::Increment))
        }
    }
}

fn inner_route(state: &InnerState) -> String {
    if state.done {
        END.to_string()
    } else {
        "increment".to_string()
    }
}

// ---------------------------------------------------------------------------
// Outer graph state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct OuterState {
    label: String,
    inner_result: Option<u32>,
    finalized: bool,
}

enum OuterUpdate {
    SetInnerResult(u32),
    Finalize,
}

impl Reducer for OuterState {
    type Update = OuterUpdate;
    fn apply(&mut self, update: OuterUpdate) {
        match update {
            OuterUpdate::SetInnerResult(n) => self.inner_result = Some(n),
            OuterUpdate::Finalize => self.finalized = true,
        }
    }
}

struct FinalizeNode;

#[async_trait]
impl Node<OuterState> for FinalizeNode {
    async fn run(&self, state: &OuterState) -> Result<NodeOutcome<OuterUpdate>> {
        println!(
            "  [outer] finalizing — inner_result = {:?}",
            state.inner_result
        );
        Ok(NodeOutcome::Update(OuterUpdate::Finalize))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== metalcraft subgraph example ===\n");

    // Build the inner graph
    let inner_graph = Graph::<InnerState>::new()
        .add_node("increment", IncrementNode)
        .add_conditional("increment", inner_route)
        .set_entry("increment")
        .compile()?;

    let inner_graph = Arc::new(inner_graph);

    // Build the outer graph with a SubgraphNode
    let subgraph_node = SubgraphNode::new(
        inner_graph,
        // extract: outer → inner
        |outer: &OuterState| {
            println!("  [outer] extracting inner state for '{}'", outer.label);
            InnerState {
                count: outer.inner_result.unwrap_or(0),
                done: false,
            }
        },
        // merge: inner → outer update
        |inner: InnerState| {
            println!("  [outer] merging inner result: count={}", inner.count);
            OuterUpdate::SetInnerResult(inner.count)
        },
    );

    let graph = Graph::<OuterState>::new()
        .add_node("subgraph", subgraph_node)
        .add_node("finalize", FinalizeNode)
        .add_edge("subgraph", "finalize")
        .add_edge("finalize", END)
        .set_entry("subgraph")
        .compile()?;

    println!("Graph:\n{}\n", graph.to_mermaid());

    let initial = OuterState {
        label: "my-task".to_string(),
        inner_result: None,
        finalized: false,
    };

    let executor = Executor::new(graph);
    let outcome = executor.run(initial, "thread-1").await?;

    match outcome {
        RunOutcome::Completed(state) => {
            println!("\n=== Done ===");
            println!("Label: {}", state.label);
            println!("Inner result: {:?}", state.inner_result);
            println!("Finalized: {}", state.finalized);
        }
        RunOutcome::Interrupted { reason, .. } => {
            println!("\nInterrupted: {reason}");
        }
    }

    Ok(())
}
