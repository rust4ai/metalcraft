//! Prebuilt agent patterns for common LLM workflows.
//!
//! Requires the `rig` feature flag.
//!
//! The star of this module is [`create_react_agent`], which builds a
//! ready-to-run ReAct (Reason + Act) agent graph from a Rig model and a
//! [`ToolRegistry`].

use async_trait::async_trait;
use rig::completion::{AssistantContent, CompletionModel, Message as RigMessage, ToolDefinition};
use std::sync::Arc;

use crate::error::{GraphError, Result};
use crate::graph::{CompiledGraph, Graph, Node, NodeOutcome, Reducer, END};
use crate::tools::{
    BeforeToolCallHook, PendingToolCall, ToolCallState, ToolNode, ToolRegistry, ToolResult,
};

// ---------------------------------------------------------------------------
// AgentTurn — structured turn data extracted from message history
// ---------------------------------------------------------------------------

/// A tool call within an agent turn.
#[derive(Debug, Clone)]
pub struct AgentToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// A tool result within an agent turn.
#[derive(Debug, Clone)]
pub struct AgentToolResult {
    pub id: String,
    pub name: String,
    pub result: String,
}

/// A single turn in the agent's execution, extracted from the message history.
///
/// Each turn represents one LLM response cycle: the agent either calls tools
/// (with subsequent results) or produces a final text answer.
#[derive(Debug, Clone)]
pub struct AgentTurn {
    pub index: usize,
    pub tool_calls: Vec<AgentToolCall>,
    pub tool_results: Vec<AgentToolResult>,
    pub assistant_text: Option<String>,
}

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
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
    },
    ToolResult {
        id: String,
        call_id: Option<String>,
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

    /// Extract structured turn data from the message history.
    ///
    /// Each turn corresponds to one LLM response: either a set of tool calls
    /// (followed by their results) or a final assistant text.
    pub fn turns(&self) -> Vec<AgentTurn> {
        let mut turns = Vec::new();
        let mut current_tool_calls: Vec<AgentToolCall> = Vec::new();
        let mut current_tool_results: Vec<AgentToolResult> = Vec::new();

        for msg in &self.messages {
            match msg {
                AgentMessage::User(_) => {
                    // Flush any pending turn
                    if !current_tool_calls.is_empty() {
                        turns.push(AgentTurn {
                            index: turns.len(),
                            tool_calls: std::mem::take(&mut current_tool_calls),
                            tool_results: std::mem::take(&mut current_tool_results),
                            assistant_text: None,
                        });
                    }
                }
                AgentMessage::ToolCall { id, name, args, .. } => {
                    // If we had tool results from a previous batch, flush that turn
                    if !current_tool_results.is_empty() {
                        turns.push(AgentTurn {
                            index: turns.len(),
                            tool_calls: std::mem::take(&mut current_tool_calls),
                            tool_results: std::mem::take(&mut current_tool_results),
                            assistant_text: None,
                        });
                    }
                    current_tool_calls.push(AgentToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        args: args.clone(),
                    });
                }
                AgentMessage::ToolResult { id, name, result, .. } => {
                    current_tool_results.push(AgentToolResult {
                        id: id.clone(),
                        name: name.clone(),
                        result: result.clone(),
                    });
                }
                AgentMessage::Assistant(text) => {
                    // Flush any pending tool turn first
                    if !current_tool_calls.is_empty() {
                        turns.push(AgentTurn {
                            index: turns.len(),
                            tool_calls: std::mem::take(&mut current_tool_calls),
                            tool_results: std::mem::take(&mut current_tool_results),
                            assistant_text: None,
                        });
                    }
                    // The assistant text is its own turn
                    turns.push(AgentTurn {
                        index: turns.len(),
                        tool_calls: vec![],
                        tool_results: vec![],
                        assistant_text: Some(text.clone()),
                    });
                }
            }
        }

        // Flush any remaining
        if !current_tool_calls.is_empty() {
            turns.push(AgentTurn {
                index: turns.len(),
                tool_calls: current_tool_calls,
                tool_results: current_tool_results,
                assistant_text: None,
            });
        }

        turns
    }

    /// Get all tool names called across the entire conversation.
    pub fn tools_called(&self) -> Vec<String> {
        self.messages
            .iter()
            .filter_map(|m| match m {
                AgentMessage::ToolCall { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
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
                        call_id: call.call_id.clone(),
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
                        call_id: r.call_id.clone(),
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
// ReactAgentNode — LLM node using native tool calling
// ---------------------------------------------------------------------------

/// A graph node that calls a Rig [`CompletionModel`] with native tool
/// definitions and parses structured tool calls from the response.
///
/// Produces [`AgentUpdate::ToolCalls`] when the LLM wants to use tools,
/// or [`AgentUpdate::FinalAnswer`] when it has a direct answer.
pub struct ReactAgentNode<M: CompletionModel> {
    model: M,
    system_prompt: String,
    registry: Arc<ToolRegistry>,
}

impl<M: CompletionModel> ReactAgentNode<M> {
    pub fn new(model: M, system_prompt: String, registry: Arc<ToolRegistry>) -> Self {
        Self {
            model,
            system_prompt,
            registry,
        }
    }
}

#[async_trait]
impl<M: CompletionModel + 'static> Node<AgentState> for ReactAgentNode<M> {
    async fn run(&self, state: &AgentState) -> Result<NodeOutcome<AgentUpdate>> {
        // Convert our ToolRegistry into rig ToolDefinitions
        let tool_defs: Vec<ToolDefinition> = self
            .registry
            .to_openai_tools()
            .iter()
            .map(|t| ToolDefinition {
                name: t["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                description: t["function"]["description"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                parameters: t["function"]["parameters"].clone(),
            })
            .collect();

        // Build conversation history
        let (prompt, history) = self.build_conversation(state);

        let response = self
            .model
            .completion_request(prompt)
            .preamble(self.system_prompt.clone())
            .messages(history)
            .tools(tool_defs)
            .send()
            .await
            .map_err(|e| GraphError::Node {
                node: "agent".into(),
                message: e.to_string(),
            })?;

        // Parse response: extract tool calls and text from AssistantContent
        let mut tool_calls = Vec::new();
        let mut text_parts = Vec::new();

        for content in response.choice {
            match content {
                AssistantContent::ToolCall(tc) => {
                    tool_calls.push(PendingToolCall {
                        id: tc.id.clone(),
                        call_id: tc.call_id.clone(),
                        name: tc.function.name.clone(),
                        args: tc.function.arguments.clone(),
                    });
                }
                AssistantContent::Text(t) => {
                    text_parts.push(t.text);
                }
                _ => {} // Reasoning, Image — ignore
            }
        }

        if tool_calls.is_empty() {
            let final_text = text_parts.join("\n");
            Ok(NodeOutcome::Update(AgentUpdate::FinalAnswer(final_text)))
        } else {
            Ok(NodeOutcome::Update(AgentUpdate::ToolCalls(tool_calls)))
        }
    }
}

impl<M: CompletionModel> ReactAgentNode<M> {
    /// Build the conversation history from agent state.
    ///
    /// Returns (prompt_message, chat_history).
    fn build_conversation(&self, state: &AgentState) -> (RigMessage, Vec<RigMessage>) {
        let mut history: Vec<RigMessage> = Vec::new();

        if state.messages.is_empty() {
            return (RigMessage::user(""), history);
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
                AgentMessage::ToolCall {
                    id, call_id, name, args,
                } => {
                    let mut tc = rig::completion::message::ToolCall::new(
                        id.clone(),
                        rig::completion::message::ToolFunction {
                            name: name.clone(),
                            arguments: args.clone(),
                        },
                    );
                    if let Some(cid) = call_id {
                        tc = tc.with_call_id(cid.clone());
                    }
                    history.push(RigMessage::from(tc));
                }
                AgentMessage::ToolResult {
                    id, call_id, result, ..
                } => {
                    let cid = call_id.clone().or_else(|| Some(id.clone()));
                    history.push(RigMessage::tool_result_with_call_id(id, cid, result));
                }
            }
        }

        // The last message becomes the prompt
        let prompt = match &last[0] {
            AgentMessage::User(text) => RigMessage::user(text),
            AgentMessage::Assistant(text) => RigMessage::user(text),
            AgentMessage::ToolResult {
                id, call_id, result, ..
            } => {
                let cid = call_id.clone().or_else(|| Some(id.clone()));
                RigMessage::tool_result_with_call_id(id, cid, result)
            }
            AgentMessage::ToolCall {
                id, call_id, name, args,
            } => {
                let mut tc = rig::completion::message::ToolCall::new(
                    id.clone(),
                    rig::completion::message::ToolFunction {
                        name: name.clone(),
                        arguments: args.clone(),
                    },
                );
                if let Some(cid) = call_id {
                    tc = tc.with_call_id(cid.clone());
                }
                RigMessage::from(tc)
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
pub fn create_react_agent_with_hooks<M: CompletionModel + 'static>(
    model: M,
    tools: ToolRegistry,
    system_prompt: impl Into<String>,
    before_tool_call: Option<BeforeToolCallHook>,
) -> Result<CompiledGraph<AgentState>> {
    let registry = Arc::new(tools);

    let agent_node = ReactAgentNode::new(model, system_prompt.into(), registry.clone());
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
