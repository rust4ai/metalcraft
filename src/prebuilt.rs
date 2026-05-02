//! Prebuilt agent patterns for common LLM workflows.
//!
//! Requires the `rig` feature flag.
//!
//! The star of this module is [`create_react_agent`], which builds a
//! ready-to-run ReAct (Reason + Act) agent graph from a Rig model and a
//! [`ToolRegistry`].

use async_trait::async_trait;
use rig::completion::{Chat, CompletionModel, Message as RigMessage};
use std::sync::Arc;

use crate::error::{GraphError, Result};
use crate::graph::{CompiledGraph, Graph, Node, NodeOutcome, Reducer, END};
use crate::tools::{
    BeforeToolCallHook, PendingToolCall, ToolCallState, ToolNode, ToolRegistry, ToolResult,
};

// ---------------------------------------------------------------------------
// AgentMessage — typed conversation history
// ---------------------------------------------------------------------------

/// A message in the agent's conversation history.
#[derive(Debug, Clone)]
pub enum AgentMessage {
    User(String),
    Assistant(String),
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    ToolResult {
        id: String,
        name: String,
        result: String,
    },
}

// ---------------------------------------------------------------------------
// AgentState — the ReAct agent's state
// ---------------------------------------------------------------------------

/// State for a ReAct agent built with [`create_react_agent`].
#[derive(Debug, Clone)]
pub struct AgentState {
    pub messages: Vec<AgentMessage>,
    pub pending_tool_calls: Vec<PendingToolCall>,
    pub is_done: bool,
}

impl AgentState {
    /// Create a new agent state with an initial user message.
    pub fn new(user_message: impl Into<String>) -> Self {
        Self {
            messages: vec![AgentMessage::User(user_message.into())],
            pending_tool_calls: vec![],
            is_done: false,
        }
    }

    /// Continue from a completed state with a new user message.
    ///
    /// Preserves the full conversation history and resets `is_done`
    /// so the graph can run another turn.
    pub fn continue_with(mut self, user_message: impl Into<String>) -> Self {
        self.messages.push(AgentMessage::User(user_message.into()));
        self.pending_tool_calls.clear();
        self.is_done = false;
        self
    }

    /// Get the final assistant response, if the agent is done.
    pub fn final_answer(&self) -> Option<&str> {
        if !self.is_done {
            return None;
        }
        self.messages.iter().rev().find_map(|m| match m {
            AgentMessage::Assistant(text) => Some(text.as_str()),
            _ => None,
        })
    }
}

// ---------------------------------------------------------------------------
// AgentUpdate — state mutations
// ---------------------------------------------------------------------------

/// Update variants for [`AgentState`].
pub enum AgentUpdate {
    /// The LLM wants to call tools.
    ToolCalls(Vec<PendingToolCall>),
    /// The LLM produced a final answer (no tool calls).
    FinalAnswer(String),
    /// Tool execution results.
    ToolResults(Vec<ToolResult>),
}

impl Reducer for AgentState {
    type Update = AgentUpdate;

    fn apply(&mut self, update: AgentUpdate) {
        match update {
            AgentUpdate::ToolCalls(calls) => {
                for call in &calls {
                    self.messages.push(AgentMessage::ToolCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        args: call.args.clone(),
                    });
                }
                self.pending_tool_calls = calls;
                self.is_done = false;
            }
            AgentUpdate::FinalAnswer(text) => {
                self.messages.push(AgentMessage::Assistant(text));
                self.pending_tool_calls.clear();
                self.is_done = true;
            }
            AgentUpdate::ToolResults(results) => {
                for r in &results {
                    let result_text = match &r.result {
                        Ok(v) => serde_json::to_string(v).unwrap_or_default(),
                        Err(e) => format!("ERROR: {e}"),
                    };
                    self.messages.push(AgentMessage::ToolResult {
                        id: r.id.clone(),
                        name: r.name.clone(),
                        result: result_text,
                    });
                }
                self.pending_tool_calls.clear();
            }
        }
    }
}

impl ToolCallState for AgentState {
    fn pending_tool_calls(&self) -> Vec<PendingToolCall> {
        self.pending_tool_calls.clone()
    }

    fn tool_results_update(results: Vec<ToolResult>) -> AgentUpdate {
        AgentUpdate::ToolResults(results)
    }
}

// ---------------------------------------------------------------------------
// ReactAgentNode — LLM node that parses tool calls
// ---------------------------------------------------------------------------

/// A graph node that calls a Rig [`CompletionModel`] and parses tool-call
/// directives from its response.
///
/// Produces [`AgentUpdate::ToolCalls`] when the LLM wants to use tools,
/// or [`AgentUpdate::FinalAnswer`] when it has a direct answer.
pub struct ReactAgentNode<M: CompletionModel> {
    agent: rig::agent::Agent<M>,
    registry: Arc<ToolRegistry>,
}

impl<M: CompletionModel> ReactAgentNode<M> {
    pub fn new(agent: rig::agent::Agent<M>, registry: Arc<ToolRegistry>) -> Self {
        Self { agent, registry }
    }
}

#[async_trait]
impl<M: CompletionModel + 'static> Node<AgentState> for ReactAgentNode<M> {
    async fn run(&self, state: &AgentState) -> Result<NodeOutcome<AgentUpdate>> {
        let tool_descriptions = self
            .registry
            .to_openai_tools()
            .iter()
            .map(|t| {
                let name = &t["function"]["name"];
                let desc = &t["function"]["description"];
                let params = &t["function"]["parameters"]["properties"];
                format!("- {name}: {desc}\n  Parameters: {params}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        let tool_instructions = format!(
            "You have these tools. To use one, respond EXACTLY with \
             `TOOL_CALL: tool_name({{\"param_name\": \"value\"}})` on its own line.\n\
             Use the EXACT parameter names shown below. \
             When done, respond normally with no TOOL_CALL.\n\nTools:\n{tool_descriptions}"
        );

        // Build full conversation history for the LLM
        let (prompt, history) = self.build_conversation(state, &tool_instructions);

        let response = self
            .agent
            .chat(&prompt, history)
            .await
            .map_err(|e| GraphError::Node {
                node: "agent".into(),
                message: e.to_string(),
            })?;

        // Parse tool calls from the response
        let mut tool_calls = Vec::new();
        let mut call_counter = 0u32;

        for line in response.lines() {
            if line.starts_with("TOOL_CALL:") {
                let rest = line.trim_start_matches("TOOL_CALL:").trim();
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim().to_string();
                    let args_str = rest[paren..].trim_start_matches('(').trim_end_matches(')');
                    let args: serde_json::Value =
                        serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));

                    tool_calls.push(PendingToolCall {
                        id: format!("call_{call_counter}"),
                        name,
                        args,
                    });
                    call_counter += 1;
                }
            }
        }

        if tool_calls.is_empty() {
            Ok(NodeOutcome::Update(AgentUpdate::FinalAnswer(response)))
        } else {
            Ok(NodeOutcome::Update(AgentUpdate::ToolCalls(tool_calls)))
        }
    }
}

impl<M: CompletionModel> ReactAgentNode<M> {
    /// Build the full conversation for the LLM from agent state.
    ///
    /// Returns (current_prompt, chat_history) where history contains
    /// the tool instructions and all prior messages.
    fn build_conversation(
        &self,
        state: &AgentState,
        tool_instructions: &str,
    ) -> (String, Vec<RigMessage>) {
        let mut history = vec![RigMessage::system(tool_instructions)];

        // Convert all messages except the last into chat history.
        // The last message becomes the prompt.
        if state.messages.is_empty() {
            return (String::new(), history);
        }

        let (earlier, last) = state.messages.split_at(state.messages.len() - 1);

        for msg in earlier {
            match msg {
                AgentMessage::User(text) => {
                    history.push(RigMessage::user(text));
                }
                AgentMessage::Assistant(text) => {
                    history.push(RigMessage::assistant(text));
                }
                AgentMessage::ToolCall { name, args, .. } => {
                    let args_str = serde_json::to_string(args).unwrap_or_default();
                    history.push(RigMessage::assistant(format!(
                        "TOOL_CALL: {name}({args_str})"
                    )));
                }
                AgentMessage::ToolResult { name, result, .. } => {
                    history.push(RigMessage::user(format!(
                        "[Tool result for '{name}']: {result}"
                    )));
                }
            }
        }

        // The last message becomes the prompt
        let prompt = match &last[0] {
            AgentMessage::User(text) => text.clone(),
            AgentMessage::Assistant(text) => text.clone(),
            AgentMessage::ToolResult { name, result, .. } => {
                format!(
                    "[Tool result for '{name}']: {result}\n\n\
                     Provide your final answer or call another tool."
                )
            }
            AgentMessage::ToolCall { name, args, .. } => {
                let args_str = serde_json::to_string(args).unwrap_or_default();
                format!("TOOL_CALL: {name}({args_str})")
            }
        };

        (prompt, history)
    }
}

// ---------------------------------------------------------------------------
// create_react_agent — convenience builder
// ---------------------------------------------------------------------------

/// Build a ready-to-run ReAct agent graph.
///
/// The returned [`CompiledGraph<AgentState>`] has the following topology:
///
/// ```text
/// agent → (conditional) → tools → agent
///                       → END
/// ```
///
/// Use with [`Executor`] to run it:
///
/// ```ignore
/// let graph = create_react_agent(model, tools, "You are helpful.")?;
/// let executor = Executor::new(graph).max_steps(20);
/// let outcome = executor.run(AgentState::new("Hello"), "thread-1").await?;
/// ```
///
/// For multi-turn conversations, use [`AgentState::continue_with`]:
///
/// ```ignore
/// let state = AgentState::new("What files are here?");
/// let outcome = executor.run(state, "thread-1").await?;
/// if let RunOutcome::Completed(state) = outcome {
///     let state = state.continue_with("Now read the README");
///     let outcome = executor.run(state, "thread-1").await?;
/// }
/// ```
pub fn create_react_agent<M: CompletionModel + 'static>(
    model: M,
    tools: ToolRegistry,
    system_prompt: impl Into<String>,
) -> Result<CompiledGraph<AgentState>> {
    create_react_agent_with_hooks(model, tools, system_prompt, None)
}

/// Build a ReAct agent graph with an optional before-tool-call hook.
///
/// The hook runs before each tool execution and can approve or deny calls.
/// See [`BeforeToolCallHook`] for details.
///
/// ```ignore
/// use metalcraft::{create_react_agent_with_hooks, BeforeToolCallAction};
/// use std::sync::Arc;
///
/// let hook = Arc::new(|name: &str, args: &serde_json::Value| {
///     if name == "bash" {
///         // prompt user...
///         BeforeToolCallAction::Proceed
///     } else {
///         BeforeToolCallAction::Proceed
///     }
/// });
///
/// let graph = create_react_agent_with_hooks(model, tools, "prompt", Some(hook))?;
/// ```
pub fn create_react_agent_with_hooks<M: CompletionModel + 'static>(
    model: M,
    tools: ToolRegistry,
    system_prompt: impl Into<String>,
    before_tool_call: Option<BeforeToolCallHook>,
) -> Result<CompiledGraph<AgentState>> {
    let registry = Arc::new(tools);

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(&system_prompt.into())
        .build();

    let agent_node = ReactAgentNode::new(agent, registry.clone());
    let mut tool_node = ToolNode::new(registry);
    if let Some(hook) = before_tool_call {
        tool_node = tool_node.with_before_hook(hook);
    }

    Graph::<AgentState>::new()
        .add_node("agent", agent_node)
        .add_node("tools", tool_node)
        .add_conditional("agent", |state: &AgentState| {
            if state.is_done {
                END.to_string()
            } else if !state.pending_tool_calls.is_empty() {
                "tools".to_string()
            } else {
                END.to_string()
            }
        })
        .add_edge("tools", "agent")
        .set_entry("agent")
        .compile()
}
