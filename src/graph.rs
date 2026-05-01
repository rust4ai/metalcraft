use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{GraphError, Result};

pub const START: &str = "__start__";
pub const END: &str = "__end__";

// ---------------------------------------------------------------------------
// Reducer — your state defines how to merge partial updates
// ---------------------------------------------------------------------------

pub trait Reducer: Clone + Send + Sync + 'static {
    /// The enum of possible state mutations. Each node returns one of these.
    type Update: Send + Sync + 'static;

    /// Apply an update to the state. The compiler ensures every variant is handled.
    fn apply(&mut self, update: Self::Update);
}

// ---------------------------------------------------------------------------
// NodeOutcome — a node can update state, or request an interrupt
// ---------------------------------------------------------------------------

pub enum NodeOutcome<U> {
    /// Normal state update.
    Update(U),
    /// Request a pause for human-in-the-loop. Includes an optional update
    /// to apply before pausing, plus a reason string.
    Interrupt { update: Option<U>, reason: String },
}

impl<U> NodeOutcome<U> {
    /// Convenience: create an interrupt with no state change.
    pub fn interrupt(reason: impl Into<String>) -> Self {
        Self::Interrupt {
            update: None,
            reason: reason.into(),
        }
    }

    /// Convenience: create an interrupt that also applies an update first.
    pub fn interrupt_with(update: U, reason: impl Into<String>) -> Self {
        Self::Interrupt {
            update: Some(update),
            reason: reason.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Node — an async unit of work that reads state and returns an outcome
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Node<S: Reducer>: Send + Sync {
    async fn run(&self, state: &S) -> Result<NodeOutcome<S::Update>>;
}

/// Blanket impl: any async closure `Fn(S) -> Future<Result<NodeOutcome<S::Update>>>` is a Node.
#[async_trait]
impl<S, F, Fut> Node<S> for F
where
    S: Reducer,
    F: Fn(S) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<NodeOutcome<S::Update>>> + Send,
{
    async fn run(&self, state: &S) -> Result<NodeOutcome<S::Update>> {
        (self)(state.clone()).await
    }
}

// ---------------------------------------------------------------------------
// Edges — how nodes connect
// ---------------------------------------------------------------------------

pub type CondFn<S> = Arc<dyn Fn(&S) -> String + Send + Sync>;

pub enum Edge<S: Reducer> {
    /// Always go to this node.
    Static(String),
    /// Decide the next node based on current state.
    Conditional(CondFn<S>),
    /// Fan out to multiple nodes in parallel, then converge.
    Parallel(Vec<String>),
}

// ---------------------------------------------------------------------------
// Graph builder
// ---------------------------------------------------------------------------

pub struct Graph<S: Reducer> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: HashMap<String, Edge<S>>,
    entry: Option<String>,
}

impl<S: Reducer> Graph<S> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            entry: None,
        }
    }

    /// Add a named node to the graph.
    pub fn add_node<N: Node<S> + 'static>(mut self, name: &str, node: N) -> Self {
        self.nodes.insert(name.to_string(), Arc::new(node));
        self
    }

    /// Add a static edge: after `from` finishes, always go to `to`.
    pub fn add_edge(mut self, from: &str, to: &str) -> Self {
        self.edges
            .insert(from.to_string(), Edge::Static(to.to_string()));
        self
    }

    /// Add a conditional edge: after `from` finishes, call `f(&state)` to decide the next node.
    pub fn add_conditional<F>(mut self, from: &str, f: F) -> Self
    where
        F: Fn(&S) -> String + Send + Sync + 'static,
    {
        self.edges
            .insert(from.to_string(), Edge::Conditional(Arc::new(f)));
        self
    }

    /// Add a parallel fan-out edge: after `from`, run all `targets` concurrently.
    pub fn add_parallel(mut self, from: &str, targets: Vec<&str>) -> Self {
        self.edges.insert(
            from.to_string(),
            Edge::Parallel(targets.into_iter().map(String::from).collect()),
        );
        self
    }

    /// Set the entry point node (where execution begins).
    pub fn set_entry(mut self, name: &str) -> Self {
        self.entry = Some(name.to_string());
        self
    }

    /// Validate the graph and produce an immutable CompiledGraph.
    pub fn compile(self) -> Result<CompiledGraph<S>> {
        let entry = self
            .entry
            .ok_or(GraphError::NoEntryPoint)?;

        // Validate entry node exists
        if !self.nodes.contains_key(&entry) {
            return Err(GraphError::NodeNotFound(entry));
        }

        // Validate all edge targets reference existing nodes or END
        for (from, edge) in &self.edges {
            let targets: Vec<&str> = match edge {
                Edge::Static(to) => vec![to.as_str()],
                Edge::Conditional(_) => vec![], // can't validate at compile time
                Edge::Parallel(targets) => targets.iter().map(|s| s.as_str()).collect(),
            };
            for target in targets {
                if target != END && !self.nodes.contains_key(target) {
                    return Err(GraphError::NodeNotFound(format!(
                        "{target} (referenced by edge from '{from}')"
                    )));
                }
            }
        }

        // Validate all nodes (except END targets) have an outgoing edge
        for name in self.nodes.keys() {
            if !self.edges.contains_key(name) {
                return Err(GraphError::NoEdge(format!(
                    "node '{name}' has no outgoing edge"
                )));
            }
        }

        Ok(CompiledGraph {
            nodes: self.nodes,
            edges: self.edges,
            entry,
        })
    }
}

impl<S: Reducer> Default for Graph<S> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompiledGraph — immutable, validated, ready to execute
// ---------------------------------------------------------------------------

pub struct CompiledGraph<S: Reducer> {
    pub(crate) nodes: HashMap<String, Arc<dyn Node<S>>>,
    pub(crate) edges: HashMap<String, Edge<S>>,
    pub(crate) entry: String,
}

impl<S: Reducer> CompiledGraph<S> {
    /// Export the graph structure as a Mermaid flowchart string.
    pub fn to_mermaid(&self) -> String {
        let mut lines = vec!["flowchart TD".to_string()];
        lines.push(format!("    {START} --> {}", self.entry));

        for (from, edge) in &self.edges {
            match edge {
                Edge::Static(to) => {
                    lines.push(format!("    {from} --> {to}"));
                }
                Edge::Conditional(_) => {
                    lines.push(format!("    {from} -.->|conditional| ???"));
                }
                Edge::Parallel(targets) => {
                    for t in targets {
                        lines.push(format!("    {from} --> {t}"));
                    }
                }
            }
        }

        lines.join("\n")
    }
}
