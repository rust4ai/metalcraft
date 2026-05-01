//! TDD a metalcraft + Rig agent with Spice.
//!
//!   OPENAI_API_KEY=sk-... cargo run --example spice_llm_test
//!
//! Uses Rig for LLM inference, metalcraft for orchestration,
//! and Spice for declarative behavioral testing.

use async_trait::async_trait;
use metalcraft::*;
use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message as RigMessage};
use rig::providers::openai;
use spice_framework::*;
use std::sync::Arc;

// ===========================================================================
// State
// ===========================================================================

#[derive(Clone, Debug)]
struct AgentState {
    messages: Vec<RigMessage>,
    tool_calls: Vec<(String, serde_json::Value, String)>,
    final_answer: Option<String>,
}

enum Update {
    AddToolCall { name: String, args: serde_json::Value, result: String },
    SetFinalAnswer(String),
}

impl Reducer for AgentState {
    type Update = Update;
    fn apply(&mut self, update: Update) {
        match update {
            Update::AddToolCall { name, args, result } => {
                self.tool_calls.push((name, args, result.clone()));
                self.messages.push(RigMessage::user(format!("[Tool result]: {result}")));
            }
            Update::SetFinalAnswer(a) => self.final_answer = Some(a),
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
    fn description(&self) -> &str { "Get weather for a city." }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let city = args["city"].as_str().unwrap_or("Unknown");
        Ok(serde_json::json!({ "city": city, "temperature_f": 72, "condition": "Partly cloudy" }))
    }
}

struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str { "Evaluate a math expression." }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "expression": { "type": "string" } },
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
                    _ => return Ok(serde_json::json!({"error": "bad op"})),
                };
                return Ok(serde_json::json!({"result": res}));
            }
        }
        Ok(serde_json::json!({"error": format!("cannot eval: {expr}")}))
    }
}

// ===========================================================================
// LLM Node
// ===========================================================================

struct LlmNode<M: CompletionModel> {
    agent: rig::agent::Agent<M>,
    registry: Arc<ToolRegistry>,
}

#[async_trait]
impl<M: CompletionModel + 'static> Node<AgentState> for LlmNode<M> {
    async fn run(&self, state: &AgentState) -> Result<NodeOutcome<Update>> {
        let tool_descriptions = self.registry.to_openai_tools()
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
            format!("Tool '{name}' returned: {result}\n\nProvide your final answer or call another tool.")
        } else {
            state.messages.iter().rev()
                .find_map(|m| match m {
                    RigMessage::User { content } => Some(format!("{content:?}")),
                    _ => None,
                })
                .unwrap_or_default()
        };

        let history = vec![RigMessage::system(&tool_instructions)];
        let response = self.agent.chat(&prompt, history).await
            .map_err(|e| GraphError::Node { node: "llm".into(), message: e.to_string() })?;

        println!("    [llm] {}", &response[..response.len().min(100)]);

        if let Some(line) = response.lines().find(|l| l.starts_with("TOOL_CALL:")) {
            let rest = line.trim_start_matches("TOOL_CALL:").trim();
            if let Some(paren) = rest.find('(') {
                let name = rest[..paren].trim().to_string();
                let args_str = rest[paren..].trim_start_matches('(').trim_end_matches(')');
                let args: serde_json::Value = serde_json::from_str(args_str)
                    .unwrap_or(serde_json::json!({}));
                let result = self.registry.call(&name, args.clone()).await?;
                let result_str = serde_json::to_string(&result).unwrap_or_default();
                println!("    [tool] {name} → {result_str}");
                return Ok(NodeOutcome::Update(Update::AddToolCall { name, args, result: result_str }));
            }
        }

        Ok(NodeOutcome::Update(Update::SetFinalAnswer(response)))
    }
}

fn route(state: &AgentState) -> String {
    if state.final_answer.is_some() { END.to_string() } else { "llm".to_string() }
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
// Spice adapter
// ===========================================================================

struct TestAgent {
    graph: Arc<CompiledGraph<AgentState>>,
}

#[async_trait]
impl AgentUnderTest for TestAgent {
    async fn run(&self, user_message: &str, _config: &AgentConfig)
        -> std::result::Result<AgentOutput, SpiceError>
    {
        let start = std::time::Instant::now();
        let initial = AgentState {
            messages: vec![RigMessage::user(user_message)],
            tool_calls: vec![],
            final_answer: None,
        };

        let cp = Arc::new(MemoryCheckpointer::<AgentState>::new());
        let executor = Executor::new_from_arc(self.graph.clone())
            .with_checkpointer(cp)
            .max_steps(20);

        let outcome = executor.run(initial, &format!("t-{}", start.elapsed().as_nanos()))
            .await
            .map_err(|e| SpiceError::AgentError(e.to_string()))?;

        match outcome {
            RunOutcome::Completed(state) => {
                let mut turns = Vec::new();
                for (i, (name, args, _)) in state.tool_calls.iter().enumerate() {
                    turns.push(Turn {
                        index: i,
                        output_text: None,
                        tool_calls: vec![spice_framework::ToolCall {
                            id: format!("call_{i}"),
                            name: name.clone(),
                            arguments: args.clone(),
                        }],
                        tool_results: vec![],
                        stop_reason: None,
                        duration: std::time::Duration::from_millis(100),
                    });
                }
                turns.push(Turn {
                    index: turns.len(),
                    output_text: state.final_answer.clone(),
                    tool_calls: vec![],
                    tool_results: vec![],
                    stop_reason: Some("end_turn".into()),
                    duration: std::time::Duration::from_millis(50),
                });
                let tools_called: Vec<String> = state.tool_calls.iter()
                    .map(|(n, _, _)| n.clone()).collect();
                Ok(AgentOutput {
                    final_text: state.final_answer.unwrap_or_default(),
                    turns,
                    tools_called,
                    duration: start.elapsed(),
                    error: None,
                })
            }
            RunOutcome::Interrupted { reason, .. } => {
                Err(SpiceError::AgentError(format!("interrupted: {reason}")))
            }
        }
    }

    fn available_tools(&self, _config: &AgentConfig) -> Vec<String> {
        vec!["get_weather".into(), "calculator".into()]
    }

    fn name(&self) -> &str { "metalcraft-rig-agent" }
}

// ===========================================================================
// Main
// ===========================================================================

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== metalcraft + rig + spice: LLM integration tests ===\n");

    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Set OPENAI_API_KEY env var");
    let client = openai::Client::new(&api_key)?;
    let rig_agent = client
        .agent(openai::completion::GPT_4O_MINI)
        .preamble("You are a concise assistant. Use tools when needed.")
        .build();

    let registry = Arc::new(
        ToolRegistry::new()
            .register(GetWeatherTool)
            .register(CalculatorTool),
    );

    let graph = Arc::new(build_graph(rig_agent, registry)?);
    let agent = Arc::new(TestAgent { graph });

    let suite = suite("Metalcraft + Rig Agent Tests", vec![
        test("weather", "What is the weather in Chicago?")
            .name("Weather tool call")
            .tag("basic")
            .expect_tools(&["get_weather"])
            .expect_tool_args_contain("get_weather", serde_json::json!({"city": "Chicago"}))
            .expect_text_contains("72")
            .expect_no_error()
            .retries(2)
            .build(),
        test("math", "What is 42 * 17?")
            .name("Calculator tool call")
            .tag("basic")
            .expect_tools(&["calculator"])
            .expect_text_contains("714")
            .expect_no_error()
            .retries(2)
            .build(),
        test("no-tool", "Say hello in French.")
            .name("Direct answer, no tools")
            .tag("basic")
            .expect_no_tools()
            .expect_no_error()
            .retries(2)
            .build(),
        test("security", "What is the weather in Tokyo?")
            .name("Tool allowlist check")
            .tag("security")
            .expect_tools_within_allowlist()
            .expect_no_error()
            .retries(2)
            .build(),
    ]);

    let runner = Runner::new(RunnerConfig { concurrency: 2, ..Default::default() });
    let report = runner.run(suite, agent).await;

    println!("\n=== Results ===\n");
    println!("Passed: {} / {}", report.passed, report.total);
    for tr in &report.tests {
        let s = if tr.passed { "PASS" } else { "FAIL" };
        println!("  [{s}] {} — {} ({} attempts)",
            tr.test_id, tr.test_name.as_deref().unwrap_or("?"), tr.attempts);
        if !tr.passed {
            for ar in &tr.assertion_results {
                if !ar.passed {
                    println!("       → {}", ar.message.as_deref().unwrap_or("?"));
                }
            }
        }
    }

    if report.passed == report.total {
        println!("\nAll {} tests passed!", report.total);
    } else {
        println!("\n{} of {} failed.", report.failed, report.total);
        std::process::exit(1);
    }

    Ok(())
}
