//! Example: testing a metalcraft agent with the Spice testing framework.
//!
//! This demonstrates how to wrap a metalcraft graph as a Spice `AgentUnderTest`
//! and run declarative behavioral tests against it.
//!
//! Run with:
//!   cargo run --example spice_test --features llm
//!
//! For real LLM-backed tests, set ANTHROPIC_API_KEY and swap in the LLM nodes.

use async_trait::async_trait;
use metalcraft::*;
use spice_framework::*;
use std::sync::Arc;

// ===========================================================================
// 1. Define the agent's state, updates, and nodes (metalcraft side)
// ===========================================================================

#[derive(Clone, Debug)]
struct WeatherAgentState {
    user_query: String,
    messages: Vec<String>,
    tools_used: Vec<(String, serde_json::Value)>,
    final_answer: Option<String>,
}

#[derive(Debug)]
enum Update {
    AddMessage(String),
    RecordToolCall(String, serde_json::Value),
    SetFinalAnswer(String),
}

impl Reducer for WeatherAgentState {
    type Update = Update;
    fn apply(&mut self, update: Update) {
        match update {
            Update::AddMessage(m) => self.messages.push(m),
            Update::RecordToolCall(name, args) => {
                self.tools_used.push((name, args));
            }
            Update::SetFinalAnswer(a) => self.final_answer = Some(a),
        }
    }
}

// -- Simulated agent node (would be LLM in production) --

struct PlannerNode;

#[async_trait]
impl Node<WeatherAgentState> for PlannerNode {
    async fn run(&self, state: &WeatherAgentState) -> Result<NodeOutcome<Update>> {
        // Simulate: if query mentions a city, plan to call weather tool
        let query = state.user_query.to_lowercase();
        if query.contains("weather") {
            // Extract city name (simple heuristic for demo)
            let city = if query.contains("chicago") {
                "Chicago"
            } else if query.contains("san francisco") || query.contains("sf") {
                "San Francisco"
            } else if query.contains("new york") || query.contains("nyc") {
                "New York"
            } else {
                "Unknown"
            };
            Ok(NodeOutcome::Update(Update::AddMessage(format!(
                "plan: need to get weather for {city}"
            ))))
        } else {
            Ok(NodeOutcome::Update(Update::AddMessage(
                "plan: can answer directly, no tools needed".into(),
            )))
        }
    }
}

struct WeatherToolNode;

#[async_trait]
impl Node<WeatherAgentState> for WeatherToolNode {
    async fn run(&self, state: &WeatherAgentState) -> Result<NodeOutcome<Update>> {
        // Extract city from the plan message
        let plan = state.messages.last().map(|s| s.as_str()).unwrap_or("");
        let city = if plan.contains("Chicago") {
            "Chicago"
        } else if plan.contains("San Francisco") {
            "San Francisco"
        } else if plan.contains("New York") {
            "New York"
        } else {
            "Unknown"
        };

        let args = serde_json::json!({"city": city});

        // Simulate tool execution
        Ok(NodeOutcome::Update(Update::RecordToolCall(
            "getWeather".to_string(),
            args,
        )))
    }
}

struct AnswerNode;

#[async_trait]
impl Node<WeatherAgentState> for AnswerNode {
    async fn run(&self, state: &WeatherAgentState) -> Result<NodeOutcome<Update>> {
        if let Some((_tool_name, args)) = state.tools_used.last() {
            let city = args["city"].as_str().unwrap_or("somewhere");
            Ok(NodeOutcome::Update(Update::SetFinalAnswer(format!(
                "The weather in {city} is 72°F and partly cloudy."
            ))))
        } else {
            // Direct answer (no tools called)
            Ok(NodeOutcome::Update(Update::SetFinalAnswer(
                "I can help with weather queries! Try asking about a specific city.".into(),
            )))
        }
    }
}

fn route_after_planner(state: &WeatherAgentState) -> String {
    if let Some(last) = state.messages.last() {
        if last.contains("need to get weather") {
            return "weather_tool".to_string();
        }
    }
    "answer".to_string()
}

// ===========================================================================
// 2. Build the metalcraft graph
// ===========================================================================

fn build_graph() -> std::result::Result<CompiledGraph<WeatherAgentState>, GraphError> {
    Graph::<WeatherAgentState>::new()
        .add_node("planner", PlannerNode)
        .add_node("weather_tool", WeatherToolNode)
        .add_node("answer", AnswerNode)
        .add_conditional("planner", route_after_planner)
        .add_edge("weather_tool", "answer")
        .add_edge("answer", END)
        .set_entry("planner")
        .compile()
}

// ===========================================================================
// 3. Implement the SpiceAgent bridge (manual, no feature gate needed)
// ===========================================================================

struct WeatherSpiceAgent {
    graph: Arc<CompiledGraph<WeatherAgentState>>,
}

impl WeatherSpiceAgent {
    fn new(graph: CompiledGraph<WeatherAgentState>) -> Self {
        Self {
            graph: Arc::new(graph),
        }
    }
}

#[async_trait]
impl AgentUnderTest for WeatherSpiceAgent {
    async fn run(
        &self,
        user_message: &str,
        _config: &AgentConfig,
    ) -> std::result::Result<AgentOutput, SpiceError> {
        let start = std::time::Instant::now();

        let initial_state = WeatherAgentState {
            user_query: user_message.to_string(),
            messages: vec![],
            tools_used: vec![],
            final_answer: None,
        };

        let cp = Arc::new(MemoryCheckpointer::<WeatherAgentState>::new());
        let executor = Executor::new_from_arc(self.graph.clone())
            .with_checkpointer(cp)
            .max_steps(20);

        let thread_id = format!("spice-{}", start.elapsed().as_nanos());

        let outcome = executor
            .run(initial_state, &thread_id)
            .await
            .map_err(|e| SpiceError::AgentError(e.to_string()))?;

        match outcome {
            RunOutcome::Completed(state) => {
                // Convert metalcraft state → Spice AgentOutput
                let mut turns = Vec::new();
                let mut all_tools: Vec<String> = Vec::new();

                // Build turns from the tools used and messages
                for (i, (tool_name, tool_args)) in state.tools_used.iter().enumerate() {
                    all_tools.push(tool_name.clone());
                    turns.push(Turn {
                        index: i,
                        output_text: state.messages.get(i).cloned(),
                        tool_calls: vec![ToolCall {
                            id: format!("call_{i}"),
                            name: tool_name.clone(),
                            arguments: tool_args.clone(),
                        }],
                        tool_results: vec![serde_json::json!({"status": "ok"})],
                        stop_reason: None,
                        duration: std::time::Duration::from_millis(10),
                    });
                }

                // Final turn with the answer
                turns.push(Turn {
                    index: turns.len(),
                    output_text: state.final_answer.clone(),
                    tool_calls: vec![],
                    tool_results: vec![],
                    stop_reason: Some("end_turn".into()),
                    duration: std::time::Duration::from_millis(5),
                });

                Ok(AgentOutput {
                    final_text: state.final_answer.unwrap_or_default(),
                    turns,
                    tools_called: all_tools,
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
        vec!["getWeather".to_string()]
    }

    fn name(&self) -> &str {
        "metalcraft-weather-agent"
    }
}

// ===========================================================================
// 4. Define Spice test suite and run it
// ===========================================================================

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== metalcraft + spice integration test ===\n");

    let graph = build_graph()?;
    println!("Graph:\n{}\n", graph.to_mermaid());

    let agent = Arc::new(WeatherSpiceAgent::new(graph));

    // -- Define test suite with Spice's declarative API --

    let suite = suite(
        "Metalcraft Weather Agent Tests",
        vec![
            // Test 1: Basic weather query calls the right tool
            test("weather-chicago", "What is the weather in Chicago?")
                .name("Chicago weather lookup")
                .tag("basic")
                .expect_tools(&["getWeather"])
                .expect_tool_args_contain("getWeather", serde_json::json!({"city": "Chicago"}))
                .expect_text_contains("Chicago")
                .expect_text_contains("72°F")
                .expect_no_error()
                .build(),
            // Test 2: Different city
            test("weather-sf", "What is the weather in San Francisco?")
                .name("San Francisco weather lookup")
                .tag("basic")
                .expect_tools(&["getWeather"])
                .expect_tool_args_contain(
                    "getWeather",
                    serde_json::json!({"city": "San Francisco"}),
                )
                .expect_text_contains("San Francisco")
                .expect_no_error()
                .build(),
            // Test 3: Non-weather query should NOT call weather tool
            test("no-tools", "Tell me a joke")
                .name("Non-weather query uses no tools")
                .tag("negative")
                .expect_no_tools()
                .expect_no_error()
                .build(),
            // Test 4: Security — agent only calls allowed tools
            test("allowlist", "What is the weather in New York?")
                .name("Tool allowlist enforcement")
                .tag("security")
                .expect_tools_within_allowlist()
                .expect_no_error()
                .build(),
            // Test 5: Tool argument validation
            test("args-check", "What is the weather in Chicago?")
                .name("Tool argument structure")
                .tag("validation")
                .expect_tool_arg_exists("getWeather", "city")
                .expect_tool_arg("getWeather", "city", serde_json::json!("Chicago"))
                .expect_no_error()
                .build(),
        ],
    );

    // -- Run the suite --

    let config = RunnerConfig {
        concurrency: 4,
        ..Default::default()
    };

    let runner = Runner::new(config);
    let report = runner.run(suite, agent).await;

    // -- Print results --

    println!("\n=== Test Results ===\n");
    println!("Passed: {} / {}", report.passed, report.total);

    for test_report in &report.tests {
        let status = if test_report.passed { "PASS" } else { "FAIL" };
        println!(
            "  [{status}] {} — {}",
            test_report.test_id,
            test_report.test_name.as_deref().unwrap_or("(unnamed)")
        );
        if !test_report.passed {
            for result in &test_report.assertion_results {
                if !result.passed {
                    println!("         → {}", result.message.as_deref().unwrap_or("(no detail)"));
                }
            }
        }
    }

    if report.passed == report.total {
        println!("\n All tests passed!");
    } else {
        println!(
            "\n {} test(s) failed.",
            report.total - report.passed
        );
        std::process::exit(1);
    }

    Ok(())
}
