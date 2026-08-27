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
    /// Wall-clock duration of this node's execution.
    pub duration: std::time::Duration,
    /// Whether the node completed successfully, was interrupted, or errored.
    pub outcome: StepOutcome,
}

/// Outcome of a single node execution.
#[derive(Debug, Clone)]
pub enum StepOutcome {
    /// Node completed successfully and produced an update.
    Success,
    /// Node requested an interrupt.
    Interrupted { reason: String },
    /// Node failed with an error.
    Failed { error: String },
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
    /// A node failed. Carries the state accumulated up to the failure (every
    /// successful node's update was already applied) so callers can inspect or
    /// persist partial progress, alongside the error message. Previously a
    /// node failure surfaced as `Err(GraphError)` and the partial state was
    /// dropped; this variant preserves it.
    Failed {
        state: S,
        /// The node that failed.
        node: String,
        error: String,
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

/// Action returned by a step guard callback.
pub enum GuardAction {
    /// Continue execution normally.
    Continue,
    /// Stop execution with an interrupt reason.
    Stop(String),
}

/// A callback invoked after each step. Receives the current state, the node
/// that just ran, and the next node. Can halt execution early.
pub type StepGuard<S> = Arc<dyn Fn(&S, &StepEvent) -> GuardAction + Send + Sync>;

/// A source of updates from *outside* the graph, polled at every step boundary.
///
/// A running graph is otherwise closed: the only things that change its state
/// are its own nodes. That is the right default, but it makes one thing
/// impossible — reaching a run that is already going. A person who types "no,
/// stop, do the other thing" while an agent is eight tool calls deep can today
/// only be heard after the run ends, because there is no seam to hand them
/// through.
///
/// This is that seam. It is given the same `(&S, &StepEvent)` a [`StepGuard`]
/// sees, and whatever [`Reducer::Update`]s it returns are applied to the state
/// before the next node runs. Returning an empty vec — the common case — costs
/// one call and changes nothing.
///
/// **Choosing the boundary is the caller's job, and it matters.** The mailbox is
/// polled after *every* step, including between a node that produced tool calls
/// and the node that answers them. Injecting there can leave a message list the
/// downstream provider rejects. `event.next` says which node is about to run;
/// use it to inject only where the state is coherent, and return an empty vec
/// everywhere else.
///
/// [`Reducer::Update`]: crate::Reducer::Update
pub type Mailbox<S> =
    Arc<dyn Fn(&S, &StepEvent) -> Vec<<S as Reducer>::Update> + Send + Sync>;

/// An async observer called after each node execution with rich diagnostics.
/// Purely observational — cannot halt execution. Errors are logged but ignored.
#[async_trait::async_trait]
pub trait StepObserver<S: Reducer>: Send + Sync {
    async fn on_step(&self, state: &S, event: &StepEvent);
}

/// Blanket impl: async closures work as observers.
#[async_trait::async_trait]
impl<S, F, Fut> StepObserver<S> for F
where
    S: Reducer,
    F: Fn(StepEvent) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = ()> + Send,
{
    async fn on_step(&self, _state: &S, event: &StepEvent) {
        (self)(event.clone()).await;
    }
}

pub struct Executor<S: Reducer> {
    graph: Arc<CompiledGraph<S>>,
    checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    max_steps: usize,
    step_guard: Option<StepGuard<S>>,
    mailbox: Option<Mailbox<S>>,
    observer: Option<Arc<dyn StepObserver<S>>>,
}

impl<S: Reducer> Executor<S> {
    pub fn new(graph: CompiledGraph<S>) -> Self {
        Self {
            graph: Arc::new(graph),
            checkpointer: None,
            max_steps: 100,
            step_guard: None,
            mailbox: None,
            observer: None,
        }
    }

    /// Create an executor from a pre-shared Arc<CompiledGraph>.
    /// Useful when the same graph is reused across multiple test runs.
    pub fn new_from_arc(graph: Arc<CompiledGraph<S>>) -> Self {
        Self {
            graph,
            checkpointer: None,
            max_steps: 100,
            step_guard: None,
            mailbox: None,
            observer: None,
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

    /// Set a guard that runs after each step.
    ///
    /// The guard receives the current state and step event, and can
    /// halt execution by returning [`GuardAction::Stop`]. Useful for
    /// loop detection, error-spiral protection, or custom policies.
    pub fn with_step_guard(mut self, guard: StepGuard<S>) -> Self {
        self.step_guard = Some(guard);
        self
    }

    /// Attach a [`Mailbox`], letting the world outside the graph add to the
    /// state of a run already in progress.
    ///
    /// Polled at every step boundary, after the guard. Read the `Mailbox` docs
    /// before wiring one: the caller decides which boundaries are safe.
    pub fn with_mailbox(mut self, mailbox: Mailbox<S>) -> Self {
        self.mailbox = Some(mailbox);
        self
    }

    /// Attach a step observer for diagnostics. Called after each node
    /// execution with timing and outcome info. Does not affect control flow.
    pub fn with_observer<O: StepObserver<S> + 'static>(mut self, observer: O) -> Self {
        self.observer = Some(Arc::new(observer));
        self
    }

    /// Run the graph to completion (or interruption).
    pub async fn run(&self, mut state: S, thread_id: &str) -> Result<RunOutcome<S>> {
        let mut current = self.graph.entry.clone();

        for step in 0..self.max_steps {
            if current == END {
                return Ok(RunOutcome::Completed(state));
            }

            let started = std::time::Instant::now();
            let step_result = self.execute_step(&mut state, &current, step).await;
            let duration = started.elapsed();

            match step_result {
                Ok(StepResult::Continue(next)) => {
                    let event = StepEvent {
                        node: current.clone(),
                        next: next.clone(),
                        duration,
                        outcome: StepOutcome::Success,
                    };

                    // Notify observer
                    if let Some(obs) = &self.observer {
                        obs.on_step(&state, &event).await;
                    }

                    // Run step guard
                    if let Some(guard) = &self.step_guard {
                        if let GuardAction::Stop(reason) = guard(&state, &event) {
                            if let Some(cp) = &self.checkpointer {
                                cp.save(thread_id, &state, &next).await?;
                            }
                            return Ok(RunOutcome::Interrupted {
                                state,
                                reason,
                                resume_from: next,
                            });
                        }
                    }

                    // After the guard: a run being stopped should not first
                    // absorb messages it will never act on. Before the
                    // checkpoint, so what is saved is the state the next node
                    // will actually see.
                    if let Some(mailbox) = &self.mailbox {
                        for update in mailbox(&state, &event) {
                            state.apply(update);
                        }
                    }

                    if let Some(cp) = &self.checkpointer {
                        cp.save(thread_id, &state, &next).await?;
                    }
                    current = next;
                }
                Ok(StepResult::Interrupt {
                    reason,
                    resume_from,
                }) => {
                    let event = StepEvent {
                        node: current.clone(),
                        next: resume_from.clone(),
                        duration,
                        outcome: StepOutcome::Interrupted { reason: reason.clone() },
                    };
                    if let Some(obs) = &self.observer {
                        obs.on_step(&state, &event).await;
                    }

                    if let Some(cp) = &self.checkpointer {
                        cp.save(thread_id, &state, &resume_from).await?;
                    }
                    return Ok(RunOutcome::Interrupted {
                        state,
                        reason,
                        resume_from,
                    });
                }
                Err(e) => {
                    let event = StepEvent {
                        node: current.clone(),
                        next: String::new(),
                        duration,
                        outcome: StepOutcome::Failed { error: e.to_string() },
                    };
                    if let Some(obs) = &self.observer {
                        obs.on_step(&state, &event).await;
                    }
                    // Hand back the partial state instead of dropping it on the
                    // floor — the caller can persist or inspect what the graph
                    // accumulated before the failing node.
                    return Ok(RunOutcome::Failed {
                        state,
                        node: current,
                        error: e.to_string(),
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

            let started = std::time::Instant::now();
            let step_result = self.execute_step(&mut state, &current, step).await;
            let duration = started.elapsed();

            match step_result {
                Ok(StepResult::Continue(next)) => {
                    let event = StepEvent {
                        node: current.clone(),
                        next: next.clone(),
                        duration,
                        outcome: StepOutcome::Success,
                    };
                    if let Some(obs) = &self.observer {
                        obs.on_step(&state, &event).await;
                    }
                    if let Some(guard) = &self.step_guard {
                        if let GuardAction::Stop(reason) = guard(&state, &event) {
                            cp.save(thread_id, &state, &next).await?;
                            return Ok(RunOutcome::Interrupted {
                                state,
                                reason,
                                resume_from: next,
                            });
                        }
                    }
                    cp.save(thread_id, &state, &next).await?;
                    current = next;
                }
                Ok(StepResult::Interrupt {
                    reason,
                    resume_from,
                }) => {
                    let event = StepEvent {
                        node: current.clone(),
                        next: resume_from.clone(),
                        duration,
                        outcome: StepOutcome::Interrupted { reason: reason.clone() },
                    };
                    if let Some(obs) = &self.observer {
                        obs.on_step(&state, &event).await;
                    }
                    cp.save(thread_id, &state, &resume_from).await?;
                    return Ok(RunOutcome::Interrupted {
                        state,
                        reason,
                        resume_from,
                    });
                }
                Err(e) => {
                    let event = StepEvent {
                        node: current.clone(),
                        next: String::new(),
                        duration,
                        outcome: StepOutcome::Failed { error: e.to_string() },
                    };
                    if let Some(obs) = &self.observer {
                        obs.on_step(&state, &event).await;
                    }
                    return Ok(RunOutcome::Failed {
                        state,
                        node: current,
                        error: e.to_string(),
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

                let started = std::time::Instant::now();
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
                            duration: started.elapsed(),
                            outcome: StepOutcome::Success,
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
                            duration: started.elapsed(),
                            outcome: StepOutcome::Interrupted { reason: "interrupted".into() },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, NodeOutcome, END};

    /// A state that records what happened to it, in order — enough to tell
    /// *when* an injected update landed, not merely that it did.
    #[derive(Clone, Default, Debug)]
    struct Log {
        entries: Vec<String>,
    }

    impl Reducer for Log {
        type Update = String;
        fn apply(&mut self, update: String) {
            self.entries.push(update);
        }
    }

    /// A graph of `a → b → END`, each node appending its own name.
    fn two_step() -> crate::graph::CompiledGraph<Log> {
        Graph::<Log>::new()
            .add_node("a", |_s: Log| async move { Ok(NodeOutcome::Update("a".to_string())) })
            .add_node("b", |_s: Log| async move { Ok(NodeOutcome::Update("b".to_string())) })
            .add_edge("a", "b")
            .add_edge("b", END)
            .set_entry("a")
            .compile()
            .expect("graph compiles")
    }

    #[tokio::test]
    async fn a_mailbox_update_lands_before_the_next_node_runs() {
        // Delivered once, after the first step. The ordering is the assertion:
        // "a", then the injected message, then "b" — if it landed late the run
        // would read a, b, injected, and the next node would never have seen it.
        let delivered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mailbox: Mailbox<Log> = {
            let delivered = delivered.clone();
            Arc::new(move |_state, _event| {
                if delivered.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    Vec::new()
                } else {
                    vec!["injected".to_string()]
                }
            })
        };

        let outcome = Executor::new(two_step())
            .with_mailbox(mailbox)
            .run(Log::default(), "t")
            .await
            .expect("run succeeds");

        let RunOutcome::Completed(state) = outcome else {
            panic!("expected completion");
        };
        assert_eq!(state.entries, vec!["a", "injected", "b"]);
    }

    #[tokio::test]
    async fn an_empty_mailbox_changes_nothing() {
        let mailbox: Mailbox<Log> = Arc::new(|_state, _event| Vec::new());
        let outcome = Executor::new(two_step())
            .with_mailbox(mailbox)
            .run(Log::default(), "t")
            .await
            .expect("run succeeds");
        let RunOutcome::Completed(state) = outcome else {
            panic!("expected completion");
        };
        assert_eq!(state.entries, vec!["a", "b"]);
    }

    /// `event.next` is what lets a caller pick its boundary — the whole reason
    /// the mailbox is handed the event rather than just the state.
    #[tokio::test]
    async fn a_mailbox_can_choose_which_boundary_it_delivers_on() {
        let mailbox: Mailbox<Log> = Arc::new(|_state, event| {
            if event.next == "b" {
                Vec::new()
            } else {
                vec![format!("before:{}", event.next)]
            }
        });
        let outcome = Executor::new(two_step())
            .with_mailbox(mailbox)
            .run(Log::default(), "t")
            .await
            .expect("run succeeds");
        let RunOutcome::Completed(state) = outcome else {
            panic!("expected completion");
        };
        // Nothing injected before "b"; one injection before END.
        assert_eq!(state.entries, vec!["a", "b", "before:__end__"]);
    }

    /// A run being stopped must not first absorb messages it will never act on:
    /// the guard is asked before the mailbox is drained.
    #[tokio::test]
    async fn a_stopped_run_does_not_absorb_the_mailbox() {
        let guard: StepGuard<Log> = Arc::new(|_state, _event| GuardAction::Stop("stopped".into()));
        let mailbox: Mailbox<Log> = Arc::new(|_state, _event| vec!["injected".to_string()]);

        let outcome = Executor::new(two_step())
            .with_step_guard(guard)
            .with_mailbox(mailbox)
            .run(Log::default(), "t")
            .await
            .expect("run succeeds");

        let RunOutcome::Interrupted { state, reason, .. } = outcome else {
            panic!("expected an interrupt");
        };
        assert_eq!(reason, "stopped");
        assert_eq!(state.entries, vec!["a"], "nothing should have been injected");
    }
}
