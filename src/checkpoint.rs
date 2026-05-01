use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::Result;
use crate::graph::Reducer;

// ---------------------------------------------------------------------------
// Checkpointer trait — pluggable persistence for state snapshots
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Checkpointer<S: Reducer>: Send + Sync {
    /// Save a checkpoint: the current state and which node to run next.
    async fn save(&self, thread_id: &str, state: &S, next_node: &str) -> Result<()>;

    /// Load the latest checkpoint for a thread. Returns None if no checkpoint exists.
    async fn load(&self, thread_id: &str) -> Result<Option<(S, String)>>;
}

// ---------------------------------------------------------------------------
// MemoryCheckpointer — in-memory, great for tests and short-lived agents
// ---------------------------------------------------------------------------

pub struct MemoryCheckpointer<S: Reducer> {
    inner: Arc<Mutex<HashMap<String, (S, String)>>>,
}

impl<S: Reducer> MemoryCheckpointer<S> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<S: Reducer> Default for MemoryCheckpointer<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: Reducer> Checkpointer<S> for MemoryCheckpointer<S> {
    async fn save(&self, thread_id: &str, state: &S, next_node: &str) -> Result<()> {
        self.inner
            .lock()
            .await
            .insert(thread_id.to_string(), (state.clone(), next_node.to_string()));
        Ok(())
    }

    async fn load(&self, thread_id: &str) -> Result<Option<(S, String)>> {
        Ok(self.inner.lock().await.get(thread_id).cloned())
    }
}
