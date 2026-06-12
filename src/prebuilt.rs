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

/// Render an error together with its full `source()` chain.
///
/// Many lower-level errors carry the useful detail in their source, not their
/// top-level `Display`. The classic offender is `reqwest::Error` for a failed
/// decode: `to_string()` yields only `"error decoding response body"`, while
/// the source chain holds the actual cause (e.g. the serde line/column, or the
/// provider's raw error payload). Flattening with `to_string()` throws that
/// away; this walks the chain so callers see *why* a request failed.
fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        let s = cause.to_string();
        // Skip causes whose text the parent already embeds, to avoid
        // "decode error: decode error: ..." style duplication.
        if !out.contains(&s) {
            out.push_str(": ");
            out.push_str(&s);
        }
        source = cause.source();
    }
    out
}

// ---------------------------------------------------------------------------
// LlmCallHook — observe the raw context sent to the LLM each turn
// ---------------------------------------------------------------------------

/// Snapshot of the full context sent to the LLM in a single call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LlmCallSnapshot {
    /// The system prompt (preamble).
    pub system_prompt: String,
    /// The prompt message (last message, sent as the main user turn).
    pub prompt: serde_json::Value,
    /// Chat history preceding the prompt.
    pub history: Vec<serde_json::Value>,
    /// Tool definitions available to the model.
    pub tools: Vec<serde_json::Value>,
}

/// Hook called with the raw LLM request context before each `.send()`.
pub type LlmCallHook = Arc<dyn Fn(&LlmCallSnapshot) + Send + Sync>;

/// Token usage reported by the provider for a single LLM call.
///
/// A provider-neutral projection of `rig`'s usage so consumers don't depend on
/// rig types. Counts a provider doesn't report come back as `0`.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlmUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    /// Input tokens served from a provider-managed prompt cache.
    pub cached_input_tokens: u64,
    /// Tokens spent on internal reasoning by reasoning-capable models.
    pub reasoning_tokens: u64,
}

/// Snapshot of an LLM call's *result*, delivered after `.send()` returns.
///
/// Complements [`LlmCallSnapshot`] (which is the pre-call request). This is the
/// only place per-call token usage is available, since the request hook fires
/// before the provider responds.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LlmResponseSnapshot {
    /// Assistant text produced. Empty when the model only requested tool calls.
    pub output_text: String,
    /// Names of the tools the model requested in this response, in order.
    pub tool_calls: Vec<String>,
    /// Token usage reported by the provider.
    pub usage: LlmUsage,
}

/// Hook called with the LLM response (output + token usage) after each
/// `.send()`. Pairs with [`LlmCallHook`] to bracket one model call.
pub type LlmResponseHook = Arc<dyn Fn(&LlmResponseSnapshot) + Send + Sync>;

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
    /// so the graph can run another turn. Any orphaned tool calls
    /// (calls without matching results, e.g. from a guard interruption)
    /// get synthetic "interrupted" results so the API sees a valid
    /// conversation sequence.
    pub fn continue_with(mut self, user_message: impl Into<String>) -> Self {
        // Patch orphaned tool calls: collect IDs that have a ToolCall but no ToolResult.
        let mut unmatched: Vec<(String, Option<String>, String)> = Vec::new();
        let mut matched_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for msg in &self.messages {
            match msg {
                AgentMessage::ToolCall { id, call_id, name, .. } => {
                    unmatched.push((id.clone(), call_id.clone(), name.clone()));
                }
                AgentMessage::ToolResult { id, .. } => {
                    matched_ids.insert(id.clone());
                }
                _ => {}
            }
        }
        for (id, call_id, name) in unmatched {
            if !matched_ids.contains(&id) {
                self.messages.push(AgentMessage::ToolResult {
                    id,
                    call_id,
                    name,
                    result: "ERROR: interrupted by user".to_string(),
                });
            }
        }

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
    llm_call_hook: Option<LlmCallHook>,
    llm_response_hook: Option<LlmResponseHook>,
    tool_choice: ToolChoice,
}

impl<M: CompletionModel> ReactAgentNode<M> {
    pub fn new(model: M, system_prompt: String, registry: Arc<ToolRegistry>) -> Self {
        Self {
            model,
            system_prompt,
            registry,
            llm_call_hook: None,
            llm_response_hook: None,
            tool_choice: ToolChoice::Auto,
        }
    }

    /// Attach a hook that observes the raw context sent to the LLM each turn.
    pub fn with_llm_call_hook(mut self, hook: LlmCallHook) -> Self {
        self.llm_call_hook = Some(hook);
        self
    }

    /// Attach a hook that observes the LLM response (output + token usage)
    /// after each `.send()` returns.
    pub fn with_llm_response_hook(mut self, hook: LlmResponseHook) -> Self {
        self.llm_response_hook = Some(hook);
        self
    }

    /// Set the tool-choice policy. [`ToolChoice::Required`] forces the model to
    /// emit at least one tool call every step (never free text); pair it with a
    /// terminal tool (see [`AgentOptions::terminal_tools`]) so the loop can end.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
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
        let (prompt, history) = build_conversation(state);

        // Fire the LLM call hook with the full raw context
        if let Some(ref hook) = self.llm_call_hook {
            let snapshot = LlmCallSnapshot {
                system_prompt: self.system_prompt.clone(),
                prompt: serde_json::to_value(&prompt).unwrap_or_default(),
                history: history
                    .iter()
                    .map(|m| serde_json::to_value(m).unwrap_or_default())
                    .collect(),
                tools: tool_defs
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect(),
            };
            hook(&snapshot);
        }

        let mut builder = self
            .model
            .completion_request(prompt)
            .preamble(self.system_prompt.clone())
            .messages(history)
            .tools(tool_defs);
        // Force tool-only output when configured. With `Required` the model
        // never returns free text, so the turn must be ended by a terminal tool
        // (wired in the graph by `create_react_agent_with_options`).
        if matches!(self.tool_choice, ToolChoice::Required) {
            builder = builder.tool_choice(rig::completion::message::ToolChoice::Required);
        }
        let response = builder.send().await.map_err(|e| GraphError::Node {
            node: "agent".into(),
            message: error_chain(&e),
        })?;

        // Capture token usage before consuming `response.choice` below (both are
        // independent fields, so this is just a partial move).
        let usage = response.usage;

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

        // Fire the response hook with output + token usage. This is the only
        // place per-call usage is observable (the request hook fires pre-send).
        if let Some(ref hook) = self.llm_response_hook {
            let snapshot = LlmResponseSnapshot {
                output_text: text_parts.join("\n"),
                tool_calls: tool_calls.iter().map(|t| t.name.clone()).collect(),
                usage: LlmUsage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    total_tokens: usage.total_tokens,
                    cached_input_tokens: usage.cached_input_tokens,
                    reasoning_tokens: usage.reasoning_tokens,
                },
            };
            hook(&snapshot);
        }

        if tool_calls.is_empty() {
            let final_text = text_parts.join("\n");
            Ok(NodeOutcome::Update(AgentUpdate::FinalAnswer(final_text)))
        } else {
            Ok(NodeOutcome::Update(AgentUpdate::ToolCalls(tool_calls)))
        }
    }
}

/// Build the conversation history from agent state.
///
/// Returns (prompt_message, chat_history).
///
/// Assistant turns are replayed with an (empty) id via [`RigMessage::assistant_with_id`]
/// rather than [`RigMessage::assistant`]. rig-core 0.38.2's OpenAI Responses API
/// serializer emits `input_text` for assistant messages whose `id` is `None` while
/// still tagging them `role: assistant` — a combination OpenAI rejects with HTTP 400
/// ("Invalid value: 'input_text'. Supported values are: 'output_text' and 'refusal'").
/// Supplying an id forces the valid `output_text` form; the empty id is dropped on the
/// wire (`skip_serializing_if = "String::is_empty"`).
fn build_conversation(state: &AgentState) -> (RigMessage, Vec<RigMessage>) {
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
                history.push(RigMessage::assistant_with_id(String::new(), text));
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
    create_react_agent_with_hooks(model, tools, system_prompt, None, None, None)
}

/// Tool-choice policy for the ReAct agent node.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether to call tools or answer with free text. When it
    /// answers with text, the turn ends. This is the historical default.
    #[default]
    Auto,
    /// The model must emit at least one tool call on every step and may never
    /// answer with free text. Because the loop's natural exit (a free-text final
    /// answer) can no longer occur, a terminal tool MUST be configured (see
    /// [`AgentOptions::terminal_tools`]) so a turn can complete.
    Required,
}

/// Options for [`create_react_agent_with_options`]. All fields default to the
/// historical behavior, so `AgentOptions::default()` reproduces
/// [`create_react_agent`].
#[derive(Default)]
pub struct AgentOptions {
    /// Runs before each tool execution; can approve or deny calls.
    pub before_tool_call: Option<BeforeToolCallHook>,
    /// Observes the raw context sent to the LLM each turn.
    pub llm_call_hook: Option<LlmCallHook>,
    /// Observes the LLM response (output + token usage) after each call returns.
    pub llm_response_hook: Option<LlmResponseHook>,
    /// Whether the model may answer with free text or must call a tool.
    pub tool_choice: ToolChoice,
    /// Names of tools that, when called, end the turn after they execute (the
    /// graph routes to `END` instead of looping back to the agent). Empty means
    /// the turn only ends on a free-text answer (historical behavior). This is
    /// what makes [`ToolChoice::Required`] usable: e.g. a `say_to_user` tool that
    /// both delivers the reply and terminates the turn.
    pub terminal_tools: Vec<String>,
}

/// Build a ReAct agent graph with optional hooks.
///
/// - `before_tool_call`: runs before each tool execution; can approve or deny calls.
/// - `llm_call_hook`: observes the raw context sent to the LLM each turn.
/// - `llm_response_hook`: observes the LLM response (output + token usage) after
///   each call returns.
///
/// See [`BeforeToolCallHook`], [`LlmCallHook`] and [`LlmResponseHook`] for details.
pub fn create_react_agent_with_hooks<M: CompletionModel + 'static>(
    model: M,
    tools: ToolRegistry,
    system_prompt: impl Into<String>,
    before_tool_call: Option<BeforeToolCallHook>,
    llm_call_hook: Option<LlmCallHook>,
    llm_response_hook: Option<LlmResponseHook>,
) -> Result<CompiledGraph<AgentState>> {
    create_react_agent_with_options(
        model,
        tools,
        system_prompt,
        AgentOptions {
            before_tool_call,
            llm_call_hook,
            llm_response_hook,
            ..Default::default()
        },
    )
}

/// Build a ReAct agent graph with full options, including tool-choice forcing
/// and terminal tools.
///
/// When `options.tool_choice` is [`ToolChoice::Required`] the model is forced to
/// emit tool calls and never free text. In that mode you should also set
/// `options.terminal_tools` to the tool(s) that end a turn — otherwise the loop
/// only stops at `max_steps`. When `terminal_tools` is empty the graph behaves
/// exactly as [`create_react_agent_with_hooks`]: `tools` always loops back to
/// `agent`, and a turn ends on a free-text final answer.
pub fn create_react_agent_with_options<M: CompletionModel + 'static>(
    model: M,
    tools: ToolRegistry,
    system_prompt: impl Into<String>,
    options: AgentOptions,
) -> Result<CompiledGraph<AgentState>> {
    let AgentOptions {
        before_tool_call,
        llm_call_hook,
        llm_response_hook,
        tool_choice,
        terminal_tools,
    } = options;

    let registry = Arc::new(tools);

    let mut agent_node = ReactAgentNode::new(model, system_prompt.into(), registry.clone())
        .with_tool_choice(tool_choice);
    if let Some(hook) = llm_call_hook {
        agent_node = agent_node.with_llm_call_hook(hook);
    }
    if let Some(hook) = llm_response_hook {
        agent_node = agent_node.with_llm_response_hook(hook);
    }
    let mut tool_node = ToolNode::new(registry);
    if let Some(hook) = before_tool_call {
        tool_node = tool_node.with_before_hook(hook);
    }

    let graph = Graph::<AgentState>::new()
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
        });

    let graph = if terminal_tools.is_empty() {
        // Historical behavior: after tools run, always return to the agent.
        graph.add_edge("tools", "agent")
    } else {
        // After tools run, end the turn if the just-executed batch invoked a
        // terminal tool; otherwise loop back to the agent. The freshly-appended
        // ToolResult messages sit at the tail of `messages`.
        graph.add_conditional("tools", move |state: &AgentState| {
            if invoked_terminal_tool(state, &terminal_tools) {
                END.to_string()
            } else {
                "agent".to_string()
            }
        })
    };

    graph.set_entry("agent").compile()
}

/// True if the most recently executed tool batch (the trailing run of
/// `ToolResult` messages on `state`) invoked any tool named in `terminal_tools`.
/// Used by [`create_react_agent_with_options`] to decide whether a turn ends
/// after the `tools` node runs.
fn invoked_terminal_tool(state: &AgentState, terminal_tools: &[String]) -> bool {
    state
        .messages
        .iter()
        .rev()
        .take_while(|m| matches!(m, AgentMessage::ToolResult { .. }))
        .any(|m| match m {
            AgentMessage::ToolResult { name, .. } => terminal_tools.iter().any(|t| t == name),
            _ => false,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_result(name: &str) -> AgentMessage {
        AgentMessage::ToolResult {
            id: format!("call_{name}"),
            call_id: None,
            name: name.to_string(),
            result: "{}".to_string(),
        }
    }

    #[test]
    fn detects_terminal_tool_in_trailing_results() {
        let terminal = vec!["say_to_user".to_string()];
        let mut state = AgentState::new("hi");
        // A non-terminal tool just ran: keep looping.
        state.messages.push(tool_result("web_fetch"));
        assert!(!invoked_terminal_tool(&state, &terminal));
        // A batch that includes the terminal tool: end the turn.
        state.messages.push(tool_result("say_to_user"));
        assert!(invoked_terminal_tool(&state, &terminal));
    }

    #[test]
    fn terminal_only_counts_the_trailing_tool_batch() {
        let terminal = vec!["say_to_user".to_string()];
        let mut state = AgentState::new("hi");
        // say_to_user ran in an *earlier* batch...
        state.messages.push(tool_result("say_to_user"));
        // ...but the latest assistant turn called another tool, whose result is
        // the new tail. `take_while` stops at the Assistant message, so the old
        // say_to_user is not counted and the loop continues.
        state.messages.push(AgentMessage::Assistant(String::new()));
        state.messages.push(tool_result("grep"));
        assert!(!invoked_terminal_tool(&state, &terminal));
    }

    #[test]
    fn empty_terminal_list_never_terminates() {
        let mut state = AgentState::new("hi");
        state.messages.push(tool_result("say_to_user"));
        assert!(!invoked_terminal_tool(&state, &[]));
    }

    /// Regression for the multi-turn HTTP 400 from the OpenAI Responses API:
    ///
    ///   "Invalid value: 'input_text'. Supported values are: 'output_text' and
    ///    'refusal'." (param: input[N].content[0])
    ///
    /// A prior assistant turn replayed as conversation history must serialize as
    /// `output_text`. rig-core 0.38.2 emits `input_text` for assistant messages
    /// with `id: None` while keeping `role: assistant`, which OpenAI rejects.
    /// `build_conversation` must produce assistant items that avoid this.
    #[test]
    fn assistant_history_serializes_as_output_text() {
        use rig::providers::openai::responses_api::InputItem;

        // Mirror a real multi-turn chat: [user, assistant, user(question)].
        let mut state = AgentState::new("what is teller?");
        state
            .messages
            .push(AgentMessage::Assistant("Teller is a lending protocol.".into()));
        state
            .messages
            .push(AgentMessage::User("how do teller loans work".into()));

        let (prompt, history) = build_conversation(&state);

        // Assemble the request `input` array the way rig's Responses API does:
        // chat_history followed by the prompt.
        let mut messages = history;
        messages.push(prompt);

        let mut input_items: Vec<InputItem> = Vec::new();
        for m in messages {
            let items: Vec<InputItem> = m.try_into().expect("message converts to input items");
            input_items.extend(items);
        }

        let json = serde_json::to_value(&input_items).expect("serialize input items");
        let arr = json.as_array().expect("input serializes to an array");

        // Every assistant-role text item must use `output_text`, never `input_text`.
        let mut saw_assistant_text = false;
        for item in arr {
            if item.get("role").and_then(|r| r.as_str()) != Some("assistant") {
                continue;
            }
            let Some(content) = item.get("content").and_then(|c| c.as_array()) else {
                continue;
            };
            for part in content {
                if let Some(ty @ ("input_text" | "output_text")) =
                    part.get("type").and_then(|t| t.as_str())
                {
                    saw_assistant_text = true;
                    assert_eq!(
                        ty, "output_text",
                        "assistant history must serialize as output_text, not input_text \
                         (OpenAI Responses API rejects input_text on assistant role); \
                         offending item: {item}"
                    );
                }
            }
        }
        assert!(
            saw_assistant_text,
            "expected an assistant text item in the serialized input; got: {json}"
        );
    }
}
