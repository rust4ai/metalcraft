//! Prebuilt agent patterns for common LLM workflows.
//!
//! Requires the `rig` feature flag.
//!
//! The star of this module is [`create_react_agent`], which builds a
//! ready-to-run ReAct (Reason + Act) agent graph from a Rig model and a
//! [`ToolRegistry`].

use async_trait::async_trait;
use rig::OneOrMany;
use rig::completion::message::{Reasoning, ReasoningContent};
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

/// A reasoning item emitted by a reasoning-capable model (e.g. OpenAI's
/// `gpt-5.x` family on the Responses API), preserved so it can be replayed on a
/// later turn.
///
/// The Responses API pairs every `function_call` item with the `reasoning` item
/// (`rs_…`) that produced it, and **rejects** a turn that replays the
/// `function_call` without its reasoning item ("Item 'fc_…' of type
/// 'function_call' was provided without its required 'reasoning' item"). So once
/// a model reasons before calling a tool, we must carry the reasoning item
/// through history and send it back alongside the tool call. See
/// [`AgentMessage::Reasoning`] and [`build_conversation`].
#[derive(Debug, Clone)]
pub struct ReasoningItem {
    /// Provider-assigned id (OpenAI `rs_…`). Links the item to its tool call.
    pub id: String,
    /// The `reasoning.encrypted_content` payload. This is what the Responses API
    /// validates a replayed reasoning item against, so we only keep items that
    /// carry it — an id alone cannot be replayed. The provider returns it only
    /// when a `reasoning` request parameter is set (see
    /// [`ReactAgentNode::with_reasoning_effort`]).
    pub encrypted: String,
}

/// A message in the agent's conversation history.
#[derive(Debug, Clone)]
pub enum AgentMessage {
    User(String),
    Assistant(String),
    /// A reasoning item from a reasoning-capable model, kept so it can be
    /// replayed before the tool call(s) it produced. Only stored when a turn
    /// emits tool calls (a final text answer ends the turn, so its reasoning
    /// never needs replaying). See [`ReasoningItem`].
    Reasoning {
        id: String,
        encrypted: String,
    },
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
                // Reasoning items are an internal replay artifact, not a
                // user-facing turn — skip them in the structured view.
                AgentMessage::Reasoning { .. } => {}
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
    /// The LLM wants to call tools, optionally preceded by the reasoning
    /// item(s) that produced them (kept so they can be replayed — see
    /// [`ReasoningItem`]).
    ToolCalls {
        reasoning: Vec<ReasoningItem>,
        calls: Vec<PendingToolCall>,
    },
    /// The LLM produced a final answer (no tool calls).
    FinalAnswer(String),
    /// Tool execution results.
    ToolResults(Vec<ToolResult>),
    /// A message from the user, arriving mid-run through a [`Mailbox`].
    ///
    /// Ordinary turns do not use this: a turn starts *from* a user message, and
    /// the state already carries it. This is for the person who says something
    /// while the agent is still working — the message joins the conversation the
    /// run is already having, rather than waiting for a run that may be minutes
    /// from finishing.
    ///
    /// Where it lands is the caller's problem, not this variant's: appended to
    /// `messages` like any other, so injecting it between a tool call and its
    /// result produces exactly the orphaned-call history the Responses API
    /// rejects. See [`Mailbox`] for the rule.
    ///
    /// [`Mailbox`]: crate::Mailbox
    UserMessage(String),
}

impl Reducer for AgentState {
    type Update = AgentUpdate;

    fn apply(&mut self, update: AgentUpdate) {
        match update {
            AgentUpdate::UserMessage(text) => {
                self.messages.push(AgentMessage::User(text));
            }
            AgentUpdate::ToolCalls { reasoning, calls } => {
                // Reasoning items must precede the tool call(s) they produced so
                // the Responses API accepts the replayed `function_call`.
                for item in reasoning {
                    self.messages.push(AgentMessage::Reasoning {
                        id: item.id,
                        encrypted: item.encrypted,
                    });
                }
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
    reasoning_effort: Option<String>,
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
            reasoning_effort: None,
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

    /// Set the reasoning effort (`none` | `minimal` | `low` | `medium` | `high`
    /// | `xhigh` | `max`) for reasoning-capable models on OpenAI's Responses
    /// API. When set, the request carries a `reasoning` parameter, which also
    /// makes the provider return the `reasoning.encrypted_content` needed to
    /// replay reasoning items across turns (see [`ReasoningItem`]).
    ///
    /// Leave unset (the default) for non-reasoning models — passing a
    /// `reasoning` parameter to a model that doesn't support it is an error.
    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort.filter(|s| !s.trim().is_empty());
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
        // Enable reasoning for reasoning-capable models. Setting the `reasoning`
        // parameter also makes rig request `reasoning.encrypted_content`, which
        // is what lets reasoning items round-trip across turns so a replayed
        // tool call keeps its paired reasoning item (see [`ReasoningItem`]).
        if let Some(effort) = &self.reasoning_effort {
            builder = builder.additional_params(serde_json::json!({
                "reasoning": { "effort": effort }
            }));
        }
        let response = builder.send().await.map_err(|e| GraphError::Node {
            node: "agent".into(),
            message: error_chain(&e),
        })?;

        // Capture token usage before consuming `response.choice` below (both are
        // independent fields, so this is just a partial move).
        let usage = response.usage;

        // Parse response: extract tool calls, text, and reasoning items from
        // AssistantContent.
        let mut tool_calls = Vec::new();
        let mut text_parts = Vec::new();
        let mut reasoning_items = Vec::new();

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
                AssistantContent::Reasoning(r) => {
                    // Keep only reasoning items that carry both an id and the
                    // encrypted payload — those are the replayable ones the
                    // Responses API requires alongside their tool call. Items
                    // without an encrypted payload (e.g. when reasoning wasn't
                    // requested) can't be replayed, so we don't store them.
                    let encrypted = r.content.iter().find_map(|c| match c {
                        ReasoningContent::Encrypted(data) => Some(data.clone()),
                        _ => None,
                    });
                    if let (Some(id), Some(encrypted)) = (r.id, encrypted) {
                        reasoning_items.push(ReasoningItem { id, encrypted });
                    }
                }
                AssistantContent::Image(_) => {} // not surfaced by this agent
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
            // A final text answer ends the turn; its reasoning never needs to be
            // replayed, so we drop `reasoning_items` here.
            let final_text = text_parts.join("\n");
            Ok(NodeOutcome::Update(AgentUpdate::FinalAnswer(final_text)))
        } else {
            Ok(NodeOutcome::Update(AgentUpdate::ToolCalls {
                reasoning: reasoning_items,
                calls: tool_calls,
            }))
        }
    }
}

/// Ensure every `ToolCall` in the history is followed by a matching
/// `ToolResult`. Any call whose result was dropped (an unknown tool the
/// ToolNode skipped, a guard interruption, a provider hiccup) gets a synthetic
/// error result inserted at the end of its contiguous tool block — i.e. right
/// before the next user/assistant turn, or the end of the history. Without this
/// an orphaned assistant `tool_calls` makes OpenAI reject the whole request with
/// a 400. Mirrors the between-turn patch in [`AgentState::continue_with`], but
/// runs before *every* model call, not only when a new user message arrives.
fn backfill_orphaned_tool_results(messages: &[AgentMessage]) -> Vec<AgentMessage> {
    use std::collections::HashSet;

    let resolved: HashSet<&str> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::ToolResult { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();

    // Common case: nothing orphaned — skip the rebuild entirely.
    let has_orphan = messages.iter().any(|m| match m {
        AgentMessage::ToolCall { id, .. } => !resolved.contains(id.as_str()),
        _ => false,
    });
    if !has_orphan {
        return messages.to_vec();
    }

    fn flush(out: &mut Vec<AgentMessage>, pending: &mut Vec<(String, Option<String>, String)>) {
        for (id, call_id, name) in pending.drain(..) {
            out.push(AgentMessage::ToolResult {
                id,
                call_id,
                name,
                result: "ERROR: tool call produced no result (backfilled to keep the \
                         tool-call/result sequence valid)"
                    .to_string(),
            });
        }
    }

    let mut out: Vec<AgentMessage> = Vec::with_capacity(messages.len() + 1);
    let mut pending: Vec<(String, Option<String>, String)> = Vec::new();
    for msg in messages {
        match msg {
            AgentMessage::ToolCall { id, call_id, name, .. } => {
                out.push(msg.clone());
                if !resolved.contains(id.as_str()) {
                    pending.push((id.clone(), call_id.clone(), name.clone()));
                }
            }
            AgentMessage::ToolResult { .. } => out.push(msg.clone()),
            // Turn boundary: any orphan must be answered before the next turn.
            // A reasoning item leads a fresh assistant turn, so it is a boundary
            // too (its own turn's tool calls follow it and are resolved normally).
            AgentMessage::User(_)
            | AgentMessage::Assistant(_)
            | AgentMessage::Reasoning { .. } => {
                flush(&mut out, &mut pending);
                out.push(msg.clone());
            }
        }
    }
    flush(&mut out, &mut pending);
    out
}

/// Build a rig assistant message carrying a single encrypted reasoning item, so
/// it can be replayed ahead of the tool call it produced. The id links it to the
/// tool call and the encrypted payload is what the Responses API validates it
/// against (see [`ReasoningItem`]).
fn reasoning_message(id: &str, encrypted: &str) -> RigMessage {
    RigMessage::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::Reasoning(
            Reasoning::encrypted(encrypted).with_id(id.to_string()),
        )),
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

    // Guarantee every assistant tool_call is answered by a tool result before
    // the history reaches the provider. A dropped/missing result (an unknown
    // tool the ToolNode skipped, a guard interruption, a provider hiccup) would
    // otherwise leave an orphaned tool_call that OpenAI rejects with a 400
    // ("tool_call_ids did not have response messages"). This is the single choke
    // point every model call passes through, so patching here covers them all.
    let messages = backfill_orphaned_tool_results(&state.messages);
    let (earlier, last) = messages.split_at(messages.len() - 1);

    for msg in earlier {
        match msg {
            AgentMessage::User(text) => {
                history.push(RigMessage::user(text));
            }
            AgentMessage::Assistant(text) => {
                history.push(RigMessage::assistant_with_id(String::new(), text));
            }
            AgentMessage::Reasoning { id, encrypted } => {
                history.push(reasoning_message(id, encrypted));
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
        // A reasoning item is always followed by its tool call(s) (and their
        // results), so it is never the tail of the history in practice. Handle
        // it defensively: replay it as history and use an empty prompt.
        AgentMessage::Reasoning { id, encrypted } => {
            history.push(reasoning_message(id, encrypted));
            RigMessage::user("")
        }
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
    /// Reasoning effort for reasoning-capable models on OpenAI's Responses API
    /// (`none` | `minimal` | `low` | `medium` | `high` | `xhigh` | `max`). When
    /// set, the agent sends a `reasoning` parameter and preserves/replays
    /// reasoning items across turns — required for tool calling with reasoning
    /// models like `gpt-5.x`. Leave `None` for non-reasoning models. See
    /// [`ReactAgentNode::with_reasoning_effort`].
    pub reasoning_effort: Option<String>,
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
        reasoning_effort,
    } = options;

    let registry = Arc::new(tools);

    let mut agent_node = ReactAgentNode::new(model, system_prompt.into(), registry.clone())
        .with_tool_choice(tool_choice)
        .with_reasoning_effort(reasoning_effort);
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
/// `ToolResult` messages on `state`) invoked any tool named in `terminal_tools`
/// **and that call succeeded**. Used by [`create_react_agent_with_options`] to
/// decide whether a turn ends after the `tools` node runs.
///
/// The success requirement is the whole point of reading the result and not just
/// the name. A terminal tool is terminal because calling it *delivers* something
/// — `say_to_user` hands the reply to a channel. When that call fails, nothing
/// was delivered, so ending the turn on it ends it in silence: the user waits on
/// a reply the agent believes it already sent. Looping back to the agent instead
/// gives the model the error and a chance to say something that lands.
///
/// It also makes a terminal tool able to *decline*. Returning an error from
/// `say_to_user` now means "not yet" and returns control to the model, which is
/// how an agent can be held to an unfinished plan rather than being free to
/// close the turn the moment one delegation returns. Callers that use this must
/// bound their own refusals — a tool that always declines never lets a turn end.
fn invoked_terminal_tool(state: &AgentState, terminal_tools: &[String]) -> bool {
    state
        .messages
        .iter()
        .rev()
        .take_while(|m| matches!(m, AgentMessage::ToolResult { .. }))
        .any(|m| match m {
            AgentMessage::ToolResult { name, result, .. } => {
                terminal_tools.iter().any(|t| t == name) && !is_tool_error(result)
            }
            _ => false,
        })
}

/// The error marker the tool node writes for a call that returned `Err`
/// (`format!("ERROR: {e}")`), plus the interrupt/backfill results that use the
/// same prefix. Kept as one predicate so the terminal-tool rule and any future
/// reader agree on what "the call failed" looks like on the wire.
fn is_tool_error(result: &str) -> bool {
    result.starts_with("ERROR:")
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

    fn tool_call(name: &str) -> AgentMessage {
        AgentMessage::ToolCall {
            id: format!("call_{name}"),
            call_id: None,
            name: name.to_string(),
            args: serde_json::json!({}),
        }
    }

    fn ids_missing_results(msgs: &[AgentMessage]) -> Vec<String> {
        let resolved: std::collections::HashSet<&str> = msgs
            .iter()
            .filter_map(|m| match m {
                AgentMessage::ToolResult { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        msgs.iter()
            .filter_map(|m| match m {
                AgentMessage::ToolCall { id, .. } if !resolved.contains(id.as_str()) => {
                    Some(id.clone())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn backfill_leaves_fully_paired_history_untouched() {
        let msgs = vec![
            AgentMessage::User("hi".into()),
            tool_call("grep"),
            tool_result("grep"),
        ];
        let out = backfill_orphaned_tool_results(&msgs);
        assert_eq!(out.len(), msgs.len());
        assert!(ids_missing_results(&out).is_empty());
    }

    #[test]
    fn backfill_answers_orphan_within_a_parallel_batch() {
        // Two calls in one assistant turn, but only one came back — the classic
        // dropped-result case that OpenAI rejects with a 400.
        let msgs = vec![
            AgentMessage::User("hi".into()),
            tool_call("grep"),
            tool_call("read_file"),
            tool_result("grep"),
            AgentMessage::Assistant("done".into()),
        ];
        assert_eq!(ids_missing_results(&msgs), vec!["call_read_file".to_string()]);

        let out = backfill_orphaned_tool_results(&msgs);
        // Orphan is now answered, and the synthetic result sits before the next
        // assistant turn so the tool-call/result block stays contiguous.
        assert!(ids_missing_results(&out).is_empty());
        let assistant_pos = out
            .iter()
            .position(|m| matches!(m, AgentMessage::Assistant(_)))
            .unwrap();
        let backfilled_pos = out
            .iter()
            .position(|m| matches!(m, AgentMessage::ToolResult { id, .. } if id == "call_read_file"))
            .unwrap();
        assert!(backfilled_pos < assistant_pos);
    }

    #[test]
    fn backfill_answers_trailing_orphan() {
        let msgs = vec![
            AgentMessage::User("hi".into()),
            tool_call("grep"),
        ];
        let out = backfill_orphaned_tool_results(&msgs);
        assert!(ids_missing_results(&out).is_empty());
        assert!(matches!(out.last().unwrap(), AgentMessage::ToolResult { id, .. } if id == "call_grep"));
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

    fn failed_tool_result(name: &str, message: &str) -> AgentMessage {
        AgentMessage::ToolResult {
            id: format!("call_{name}"),
            call_id: None,
            name: name.to_string(),
            result: format!("ERROR: {message}"),
        }
    }

    /// A terminal tool that *failed* delivered nothing, so the turn must not end
    /// on it — otherwise the user waits forever on a reply the agent thinks it
    /// sent. The model gets the error back and can try again.
    #[test]
    fn failed_terminal_tool_does_not_end_the_turn() {
        let terminal = vec!["say_to_user".to_string()];
        let mut state = AgentState::new("hi");
        state
            .messages
            .push(failed_tool_result("say_to_user", "failed to deliver reply"));
        assert!(!invoked_terminal_tool(&state, &terminal));
    }

    /// The same rule is what lets a terminal tool decline: an error means "not
    /// yet", and control returns to the model instead of closing the turn.
    #[test]
    fn a_declining_terminal_tool_returns_control_to_the_agent() {
        let terminal = vec!["say_to_user".to_string()];
        let mut state = AgentState::new("hi");
        state.messages.push(failed_tool_result(
            "say_to_user",
            "plan has 2 open steps; finish them or ask the user",
        ));
        assert!(!invoked_terminal_tool(&state, &terminal));
        // ...and once it succeeds, the turn ends as before.
        state.messages.push(AgentMessage::Assistant(String::new()));
        state.messages.push(tool_result("say_to_user"));
        assert!(invoked_terminal_tool(&state, &terminal));
    }

    /// One failed terminal call in a batch does not veto a successful one.
    #[test]
    fn a_successful_terminal_call_still_ends_a_mixed_batch() {
        let terminal = vec!["say_to_user".to_string(), "ask_user".to_string()];
        let mut state = AgentState::new("hi");
        state
            .messages
            .push(failed_tool_result("say_to_user", "plan has open steps"));
        state.messages.push(tool_result("ask_user"));
        assert!(invoked_terminal_tool(&state, &terminal));
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
    fn reasoning_item_is_replayed_before_its_tool_call() {
        use rig::providers::openai::responses_api::InputItem;

        // A turn that reasoned before calling a tool: [user, reasoning, call,
        // result, user]. On the next request the reasoning item must be sent
        // back ahead of the function_call, carrying its encrypted payload — or
        // the Responses API rejects the call.
        let mut state = AgentState::new("hi");
        state.messages.push(AgentMessage::Reasoning {
            id: "rs_1".into(),
            encrypted: "ENC_PAYLOAD".into(),
        });
        state.messages.push(AgentMessage::ToolCall {
            id: "fc_read".into(),
            call_id: Some("call_read".into()),
            name: "read".into(),
            args: serde_json::json!({}),
        });
        state.messages.push(tool_result("read"));
        state.messages.push(AgentMessage::User("next".into()));

        let (prompt, history) = build_conversation(&state);
        let mut messages = history;
        messages.push(prompt);

        let mut input_items: Vec<InputItem> = Vec::new();
        for m in messages {
            let items: Vec<InputItem> = m.try_into().expect("message converts to input items");
            input_items.extend(items);
        }
        let s = serde_json::to_string(&input_items).expect("serialize input items");

        // The reasoning id and encrypted payload must reach the wire...
        assert!(s.contains("rs_1"), "reasoning id must be serialized: {s}");
        assert!(
            s.contains("ENC_PAYLOAD"),
            "encrypted reasoning payload must be serialized: {s}"
        );
        // ...and the reasoning item must precede the function_call it produced.
        let reasoning_at = s.find("ENC_PAYLOAD").unwrap();
        let call_at = s.find("function_call").expect("a function_call item is present");
        assert!(
            reasoning_at < call_at,
            "reasoning item must precede its function_call: {s}"
        );
    }

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
