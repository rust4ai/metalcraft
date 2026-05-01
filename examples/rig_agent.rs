//! A real LLM-powered agent using Rig for inference.
//!
//! Supports OpenAI or Anthropic via env vars:
//!
//!   OPENAI_API_KEY=sk-...  cargo run --example rig_agent
//!   ANTHROPIC_API_KEY=sk-... cargo run --example rig_agent -- --anthropic
//!
//! The agent can call tools (weather, calculator) and loops until done.

use async_trait::async_trait;
use metalcraft::*;
use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message as RigMessage};
use rig::providers::{anthropic, openai};
use std::sync::Arc;

// ===========================================================================
// State
// ===========================================================================

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<RigMessage>,
    tool_calls: Vec<(String, serde_json::Value, String)>, // (name, args, result)
    final_answer: Option<String>,
}

enum Update {
    AddToolCall {
        name: String,
        args: serde_json::Value,
        result: String,
    },
    SetFinalAnswer(String),
}

impl Reducer for AgentState {
    type Update = Update;
    fn apply(&mut self, update: Update) {
        match update {
            Update::AddToolCall { name, args, result } => {
                self.tool_calls.push((name, args, result.clone()));
                self.messages
                    .push(RigMessage::user(format!("[Tool result]: {result}")));
            }
            Update::SetFinalAnswer(answer) => {
                self.final_answer = Some(answer);
            }
        }
    }
}

// ===========================================================================
// Tools
// ===========================================================================

struct GetWeatherTool;

#[async_trait]
impl Tool for GetWeatherTool {
    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> &str {
        "Get the current weather for a city. Returns temperature and conditions."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "The city name" }
            },
            "required": ["city"]
        })
    }
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let city = args["city"].as_str().unwrap_or("Unknown");
        Ok(serde_json::json!({
            "city": city, "temperature_f": 72, "condition": "Partly cloudy"
        }))
    }
}

struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str {
        "Evaluate a simple math expression like '42 * 17'."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string", "description": "Math expression" }
            },
            "required": ["expression"]
        })
    }
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let expr = args["expression"].as_str().unwrap_or("0").trim().to_string();
        for op in [" + ", " - ", " * ", " / "] {
            if let Some(pos) = expr.find(op) {
                let l: f64 = expr[..pos].trim().parse().unwrap_or(0.0);
                let r: f64 = expr[pos + op.len()..].trim().parse().unwrap_or(0.0);
                let res = match op.trim() {
                    "+" => l + r, "-" => l - r, "*" => l * r,
                    "/" if r != 0.0 => l / r,
                    _ => return Ok(serde_json::json!({"error": "bad expression"})),
                };
                return Ok(serde_json::json!({"result": res}));
            }
        }
        Ok(serde_json::json!({"error": format!("cannot eval: {expr}")}))
    }
}

// ===========================================================================
// LLM Node — wraps any Rig model via the Chat trait
// ===========================================================================

struct LlmNode<M: CompletionModel> {
    agent: rig::agent::Agent<M>,
    registry: Arc<ToolRegistry>,
}

#[async_trait]
impl<M: CompletionModel + 'static> Node<AgentState> for LlmNode<M> {
    async fn run(&self, state: &AgentState) -> Result<NodeOutcome<Update>> {
        let tool_descriptions = self.registry
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

        let prompt = if let Some((name, _, result)) = state.tool_calls.last() {
            format!(
                "Tool '{name}' returned: {result}\n\n\
                 Provide your final answer or call another tool."
            )
        } else {
            state
                .messages
                .iter()
                .rev()
                .find_map(|m| match m {
                    RigMessage::User { content } => Some(format!("{content:?}")),
                    _ => None,
                })
                .unwrap_or_default()
        };

        let history = vec![RigMessage::system(&tool_instructions)];
        let response = self
            .agent
            .chat(&prompt, history)
            .await
            .map_err(|e| GraphError::Node {
                node: "llm".into(),
                message: e.to_string(),
            })?;

        println!("    [llm] {}", &response[..response.len().min(120)]);

        // Parse tool calls
        if let Some(line) = response.lines().find(|l| l.starts_with("TOOL_CALL:")) {
            let rest = line.trim_start_matches("TOOL_CALL:").trim();
            if let Some(paren) = rest.find('(') {
                let name = rest[..paren].trim().to_string();
                let args_str = rest[paren..].trim_start_matches('(').trim_end_matches(')');
                let args: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));

                println!("    [llm] → tool call: {name}({args})");
                let result = self.registry.call(&name, args.clone()).await?;
                let result_str = serde_json::to_string(&result).unwrap_or_default();
                println!("    [tool] → {result_str}");

                return Ok(NodeOutcome::Update(Update::AddToolCall {
                    name,
                    args,
                    result: result_str,
                }));
            }
        }

        Ok(NodeOutcome::Update(Update::SetFinalAnswer(response)))
    }
}

fn route(state: &AgentState) -> String {
    if state.final_answer.is_some() {
        END.to_string()
    } else {
        "llm".to_string()
    }
}

fn build_graph<M: CompletionModel + 'static>(
    agent: rig::agent::Agent<M>,
    registry: Arc<ToolRegistry>,
) -> std::result::Result<CompiledGraph<AgentState>, GraphError> {
    Graph::<AgentState>::new()
        .add_node("llm", LlmNode { agent, registry })
        .add_conditional("llm", route)
        .set_entry("llm")
        .compile()
}

// ===========================================================================
// Main
// ===========================================================================

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== metalcraft + rig agent ===\n");

    let args: Vec<String> = std::env::args().collect();
    let use_anthropic = args.iter().any(|a| a == "--anthropic");

    let registry = Arc::new(
        ToolRegistry::new()
            .register(GetWeatherTool)
            .register(CalculatorTool),
    );

    let graph = if use_anthropic {
        println!("Provider: Anthropic\n");
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .expect("Set ANTHROPIC_API_KEY env var");
        let client = anthropic::Client::new(&api_key)?;
        let agent = client
            .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
            .preamble("You are a concise assistant. Use tools when needed.")
            .build();
        build_graph(agent, registry.clone())?
    } else {
        println!("Provider: OpenAI\n");
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("Set OPENAI_API_KEY env var");
        let client = openai::Client::new(&api_key)?;
        let agent = client
            .agent(openai::completion::GPT_4O_MINI)
            .preamble("You are a concise assistant. Use tools when needed.")
            .build();
        build_graph(agent, registry.clone())?
    };

    println!("Graph:\n{}\n", graph.to_mermaid());

    let task = "What's the weather in Chicago and what is 42 * 17?";
    println!("Task: {task}\n");

    let initial = AgentState {
        messages: vec![RigMessage::user(task)],
        tool_calls: vec![],
        final_answer: None,
    };

    let executor = Executor::new(graph).max_steps(20);
    let outcome = executor.run(initial, "thread-1").await?;

    match outcome {
        RunOutcome::Completed(state) => {
            println!("\n=== Done ===");
            println!("Answer: {}", state.final_answer.unwrap_or("(none)".into()));
            println!("Tools called: {}", state.tool_calls.len());
            for (name, args, result) in &state.tool_calls {
                println!("  {name}({args}) → {result}");
            }
        }
        RunOutcome::Interrupted { reason, .. } => {
            println!("\nInterrupted: {reason}");
        }
    }

    Ok(())
}
