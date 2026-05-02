//! Tool trait and executor for AI agent tool-calling patterns.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{GraphError, Result};
use crate::graph::{Node, NodeOutcome, Reducer};

// ---------------------------------------------------------------------------
// Tool trait — define tools as typed Rust structs
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's unique name (used by the LLM to invoke it).
    fn name(&self) -> &str;

    /// A human-readable description for the LLM.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given arguments. Returns a JSON result.
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value>;
}

// ---------------------------------------------------------------------------
// ToolRegistry — collect tools and dispatch calls
// ---------------------------------------------------------------------------

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
        self
    }

    /// Look up and execute a tool by name.
    pub async fn call(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let tool = self.tools.get(name).ok_or_else(|| GraphError::Node {
            node: "tool_executor".into(),
            message: format!("unknown tool: {name}"),
        })?;
        tool.call(args).await
    }

    /// Generate the tools array for an LLM API request (Anthropic format).
    pub fn to_anthropic_tools(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "name": t.name(),
                    "description": t.description(),
                    "input_schema": t.parameters_schema(),
                })
            })
            .collect()
    }

    /// Generate the tools array for an LLM API request (OpenAI format).
    pub fn to_openai_tools(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.parameters_schema(),
                    }
                })
            })
            .collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ToolCallState — state trait for automatic tool execution
// ---------------------------------------------------------------------------

/// A pending tool call extracted from LLM output.
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    pub id: String,
    /// Secondary ID used by some providers (e.g. OpenAI's `call_id`) to link
    /// tool results back to tool calls. When present, tool results should
    /// reference this ID.
    pub call_id: Option<String>,
    pub name: String,
    pub args: serde_json::Value,
}

/// The result of executing a single tool call.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub id: String,
    pub call_id: Option<String>,
    pub name: String,
    pub result: std::result::Result<serde_json::Value, String>,
}

/// States that carry pending tool calls and can accept tool results.
///
/// Implement this trait on your state to use [`ToolNode`] for automatic
/// tool execution.
pub trait ToolCallState: Reducer {
    /// Return the list of pending tool calls that need to be executed.
    fn pending_tool_calls(&self) -> Vec<PendingToolCall>;

    /// Produce a state update that records the given tool results.
    fn tool_results_update(results: Vec<ToolResult>) -> Self::Update;
}

// ---------------------------------------------------------------------------
// BeforeToolCall hook
// ---------------------------------------------------------------------------

/// Result of a `before_tool_call` hook.
pub enum BeforeToolCallAction {
    /// Proceed with executing the tool.
    Proceed,
    /// Skip this tool call and return an error message instead.
    Deny(String),
}

/// A hook called before each tool execution.
///
/// Receives the tool name and arguments. Can approve or deny execution.
/// Useful for human-in-the-loop approval, logging, or policy enforcement.
pub type BeforeToolCallHook =
    Arc<dyn Fn(&str, &serde_json::Value) -> BeforeToolCallAction + Send + Sync>;

// ---------------------------------------------------------------------------
// ToolNode — executes pending tool calls from state via the registry
// ---------------------------------------------------------------------------

/// A graph node that reads pending tool calls from state, executes them
/// via a [`ToolRegistry`], and returns the results as a state update.
///
/// Supports an optional `before_tool_call` hook for approval/policy checks.
pub struct ToolNode {
    registry: Arc<ToolRegistry>,
    before_tool_call: Option<BeforeToolCallHook>,
}

impl ToolNode {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            before_tool_call: None,
        }
    }

    /// Set a hook that runs before each tool call.
    ///
    /// The hook receives `(tool_name, args)` and returns whether to proceed
    /// or deny the call. Denied calls produce an error tool result that the
    /// LLM can see and react to.
    pub fn with_before_hook(mut self, hook: BeforeToolCallHook) -> Self {
        self.before_tool_call = Some(hook);
        self
    }
}

#[async_trait]
impl<S: ToolCallState> Node<S> for ToolNode {
    async fn run(&self, state: &S) -> Result<NodeOutcome<S::Update>> {
        let pending = state.pending_tool_calls();
        let mut results = Vec::with_capacity(pending.len());

        for call in pending {
            // Run before-hook if configured
            if let Some(hook) = &self.before_tool_call {
                match hook(&call.name, &call.args) {
                    BeforeToolCallAction::Proceed => {}
                    BeforeToolCallAction::Deny(reason) => {
                        results.push(ToolResult {
                            id: call.id.clone(),
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            result: Err(reason),
                        });
                        continue;
                    }
                }
            }

            let result = match self.registry.call(&call.name, call.args.clone()).await {
                Ok(value) => ToolResult {
                    id: call.id.clone(),
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    result: Ok(value),
                },
                Err(e) => ToolResult {
                    id: call.id.clone(),
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    result: Err(e.to_string()),
                },
            };
            results.push(result);
        }

        Ok(NodeOutcome::Update(S::tool_results_update(results)))
    }
}
