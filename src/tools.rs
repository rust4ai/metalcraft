//! Tool trait and executor for AI agent tool-calling patterns.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{GraphError, Result};

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
