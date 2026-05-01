use futures::stream::{FuturesUnordered, StreamExt};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info_span, Instrument};

use crate::checkpoint::Checkpointer;
use crate::error::{GraphError, Result};
use crate::graph::{CompiledGraph, Edge, NodeOutcome, Reducer, END};

// ---------------------------------------------------------------------------
// StepEvent — emitted for each node execution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StepEvent {
    /// The node that just ran.
    pub node: String,
    /// The next node to run (or END, or "__interrupted__").
    pub next: String,
}

// ---------------------------------------------------------------------------
// RunOutcome — execution can complete or be interrupted
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RunOutcome<S> {
    /// Graph reached END normally.
    Completed(S),
    /// A node requested an interrupt (human-in-the-loop).
    Interrupted {
        state: S,
        reason: String,
        /// The node that will re-run when resumed.
        resume_from: String,
    },
}

/// Internal result from executing a single step.
enum StepResult {
    /// Continue to this next node.
    Continue(String),
    /// Node requested an interrupt; resume from this node.
    Interrupt { reason: String, resume_from: String },
}

// ---------------------------------------------------------------------------
// Executor — runs a compiled graph to completion
// ---------------------------------------------------------------------------

pub struct Executor<S: Reducer> {
    graph: Arc<CompiledGraph<S>>,
    checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    max_steps: usize,
}

impl<S: Reducer> Executor<S> {
    pub fn new(graph: CompiledGraph<S>) -> Self {
        Self {
            graph: Arc::new(graph),
            checkpointer: None,
            max_steps: 100,
        }
    }

    /// Create an executor from a pre-shared Arc<CompiledGraph>.
    /// Useful when the same graph is reused across multiple test runs.
    pub fn new_from_arc(graph: Arc<CompiledGraph<S>>) -> Self {
        Self {
            graph,
            checkpointer: None,
            max_steps: 100,
        }
    }

    /// Attach a checkpointer for state persistence.
    pub fn with_checkpointer(mut self, cp: Arc<dyn Checkpointer<S>>) -> Self {
        self.checkpointer = Some(cp);
        self
    }

    /// Set the maximum number of execution steps before erroring.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    /// Run the graph to completion (or interruption).
    pub async fn run(&self, mut state: S, thread_id: &str) -> Result<RunOutcome<S>> {
        let mut current = self.graph.entry.clone();

        for step in 0..self.max_steps {
            if current == END {
                return Ok(RunOutcome::Completed(state));
            }

            match self.execute_step(&mut state, &current, step).await? {
                StepResult::Continue(next) => {
                    if let Some(cp) = &self.checkpointer {
                        cp.save(thread_id, &state, &next).await?;
                    }
                    current = next;
                }
                StepResult::Interrupt {
                    reason,
                    resume_from,
                } => {
                    // Save checkpoint so we can resume later
                    if let Some(cp) = &self.checkpointer {
                        cp.save(thread_id, &state, &resume_from).await?;
                    }
                    return Ok(RunOutcome::Interrupted {
                        state,
                        reason,
                        resume_from,
                    });
                }
            }
        }

        Err(GraphError::StepLimitExceeded(self.max_steps))
    }

    /// Resume execution from a checkpoint, optionally injecting an update first.
    pub async fn resume(
        &self,
        thread_id: &str,
        inject: Option<S::Update>,
    ) -> Result<RunOutcome<S>> {
        let cp = self
            .checkpointer
            .as_ref()
            .ok_or_else(|| GraphError::Checkpoint("no checkpointer configured".into()))?;

        let (mut state, next_node) = cp
            .load(thread_id)
            .await?
            .ok_or_else(|| {
                GraphError::Checkpoint(format!("no checkpoint found for thread '{thread_id}'"))
            })?;

        // Apply any injected update (e.g. human input) before continuing
        if let Some(update) = inject {
            state.apply(update);
        }

        let mut current = next_node;

        for step in 0..self.max_steps {
            if current == END {
                return Ok(RunOutcome::Completed(state));
            }

            match self.execute_step(&mut state, &current, step).await? {
                StepResult::Continue(next) => {
                    cp.save(thread_id, &state, &next).await?;
                    current = next;
                }
                StepResult::Interrupt {
                    reason,
                    resume_from,
                } => {
                    cp.save(thread_id, &state, &resume_from).await?;
                    return Ok(RunOutcome::Interrupted {
                        state,
                        reason,
                        resume_from,
                    });
                }
            }
        }

        Err(GraphError::StepLimitExceeded(self.max_steps))
    }

    /// Stream step events as the graph executes.
    pub fn stream(
        self: Arc<Self>,
        state: S,
        thread_id: String,
    ) -> Pin<Box<dyn Stream<Item = Result<(StepEvent, S)>> + Send>> {
        let (tx, rx) = mpsc::channel(16);

        tokio::spawn(async move {
            let mut state = state;
            let mut current = self.graph.entry.clone();

            for step in 0..self.max_steps {
                if current == END {
                    break;
                }

                match self.execute_step(&mut state, &current, step).await {
                    Ok(StepResult::Continue(next)) => {
                        if let Some(cp) = &self.checkpointer {
                            if let Err(e) = cp.save(&thread_id, &state, &next).await {
                                let _ = tx.send(Err(e)).await;
                                return;
                            }
                        }

                        let event = StepEvent {
                            node: current.clone(),
                            next: next.clone(),
                        };
                        if tx.send(Ok((event, state.clone()))).await.is_err() {
                            return;
                        }
                        current = next;
                    }
                    Ok(StepResult::Interrupt { resume_from, .. }) => {
                        let event = StepEvent {
                            node: current.clone(),
                            next: "__interrupted__".to_string(),
                        };
                        let _ = tx.send(Ok((event, state.clone()))).await;
                        if let Some(cp) = &self.checkpointer {
                            let _ = cp.save(&thread_id, &state, &resume_from).await;
                        }
                        return;
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    async fn execute_step(
        &self,
        state: &mut S,
        current: &str,
        step: usize,
    ) -> Result<StepResult> {
        let span = info_span!("node", name = current, step = step);

        async {
            match self.graph.edges.get(current) {
                Some(Edge::Parallel(targets)) => {
                    self.execute_parallel(state, targets).await
                }
                _ => {
                    let node = self
                        .graph
                        .nodes
                        .get(current)
                        .ok_or_else(|| GraphError::NodeNotFound(current.to_string()))?;

                    let outcome = node.run(state).await.map_err(|e| GraphError::Node {
                        node: current.to_string(),
                        message: e.to_string(),
                    })?;

                    match outcome {
                        NodeOutcome::Update(update) => {
                            state.apply(update);

                            let next = match self.graph.edges.get(current) {
                                Some(Edge::Static(next)) => next.clone(),
                                Some(Edge::Conditional(f)) => f(state),
                                None => return Err(GraphError::NoEdge(current.to_string())),
                                Some(Edge::Parallel(_)) => unreachable!(),
                            };

                            Ok(StepResult::Continue(next))
                        }
                        NodeOutcome::Interrupt { update, reason } => {
                            if let Some(u) = update {
                                state.apply(u);
                            }
                            // Resume from the CURRENT node so it re-runs with new input
                            Ok(StepResult::Interrupt {
                                reason,
                                resume_from: current.to_string(),
                            })
                        }
                    }
                }
            }
        }
        .instrument(span)
        .await
    }

    async fn execute_parallel(
        &self,
        state: &mut S,
        targets: &[String],
    ) -> Result<StepResult> {
        let mut tasks = FuturesUnordered::new();

        for name in targets {
            let node = self
                .graph
                .nodes
                .get(name)
                .ok_or_else(|| GraphError::NodeNotFound(name.clone()))?
                .clone();
            let s = state.clone();
            let name = name.clone();
            tasks.push(async move {
                let result = node.run(&s).await;
                (name, result)
            });
        }

        let mut results = Vec::new();
        while let Some((name, res)) = tasks.next().await {
            let outcome = res.map_err(|e| GraphError::Node {
                node: name.clone(),
                message: e.to_string(),
            })?;

            match outcome {
                NodeOutcome::Update(update) => {
                    results.push((name, update));
                }
                NodeOutcome::Interrupt { update, reason } => {
                    // Apply any partial updates collected so far
                    results.sort_by(|a, b| a.0.cmp(&b.0));
                    for (_, u) in results {
                        state.apply(u);
                    }
                    if let Some(u) = update {
                        state.apply(u);
                    }
                    return Ok(StepResult::Interrupt {
                        reason,
                        resume_from: name,
                    });
                }
            }
        }

        // Apply in deterministic order
        results.sort_by(|a, b| a.0.cmp(&b.0));
        for (_, update) in results {
            state.apply(update);
        }

        if let Some(first) = targets.first() {
            match self.graph.edges.get(first) {
                Some(Edge::Static(next)) => Ok(StepResult::Continue(next.clone())),
                Some(Edge::Conditional(f)) => Ok(StepResult::Continue(f(state))),
                _ => Err(GraphError::NoEdge(format!(
                    "parallel branch '{first}' has no outgoing edge"
                ))),
            }
        } else {
            Err(GraphError::NoEdge("empty parallel targets".into()))
        }
    }
}
