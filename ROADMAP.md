# Metalcraft Roadmap & Gap Analysis

A comparison of Metalcraft against the LangGraph ecosystem, identifying gaps and a prioritized plan to close them.

## Current State

Metalcraft's core graph execution engine is solid and on par with LangGraph's fundamentals:

- Typed state management via `Reducer` trait with compile-time safety
- Static, conditional, and parallel edge types
- Cyclic graph support (agent loops)
- Checkpointed execution with interrupt/resume (human-in-the-loop)
- Async streaming of step events
- Tool registry with Anthropic and OpenAI format export
- LLM integration via Rig (optional)
- Mermaid diagram export
- Spice framework integration for declarative behavioral testing

The Rust type system gives Metalcraft an inherent advantage in correctness — the opportunity is to layer convenience, composition, and ecosystem on top of that foundation.

---

## Gap Analysis

### Tier 1 — Critical Gaps (core library)

#### 1. Subgraphs / Graph Composition

LangGraph lets you nest a graph inside another graph as a node. This is how multi-agent systems work — each agent is its own subgraph, composed into a supervisor graph. Metalcraft has no concept of this today. This is the single biggest feature gap.

**What's needed:** A `SubgraphNode` adapter that wraps a `CompiledGraph` and implements `Node<S>`, with state mapping functions to translate between parent and child state types.

#### 2. Prebuilt Agent Patterns

LangGraph ships `create_react_agent()` — a ready-to-use ReAct agent. Metalcraft requires wiring up every node, edge, and routing function from scratch each time.

**What's needed:** A set of prebuilt graph constructors:
- `create_react_agent()` — ReAct loop with tool calling
- `create_plan_and_execute()` — planning step followed by execution steps
- `create_supervisor()` — supervisor that delegates to worker agents

#### 3. Multi-Agent Orchestration

LangGraph supports supervisor/worker topologies, hierarchical teams, and agent handoffs. Metalcraft can technically achieve this with conditional routing, but there's no first-class support, no agent-as-node abstraction, and no message-passing protocol between agents.

**What's needed:** First-class multi-agent primitives built on top of subgraph composition — supervisor pattern, handoff protocol, shared message bus.

#### 4. State Channels / Field-Level Reducers

LangGraph defines reducers per field: `messages: Annotated[list, operator.add]`. Metalcraft requires a single monolithic `Reducer` impl for the entire state. Field-level reducers would reduce boilerplate and make state composition easier, especially when combining subgraphs.

**What's needed:** A derive macro (`#[derive(Reducer)]`) that generates the `Reducer` impl from per-field annotations:
```rust
#[derive(Reducer)]
struct AgentState {
    #[reducer(append)]
    messages: Vec<Message>,
    #[reducer(replace)]
    final_answer: Option<String>,
}
```

---

### Tier 2 — Important Gaps (production readiness)

#### 5. Persistent Checkpointers

Metalcraft only has `MemoryCheckpointer`. LangGraph ships SQLite, PostgreSQL, and Redis checkpointers. Production use requires durable persistence.

**What's needed:** Separate crates or feature-gated modules:
- `metalcraft-checkpoint-sqlite` (via rusqlite or sqlx)
- `metalcraft-checkpoint-postgres` (via sqlx)
- `metalcraft-checkpoint-redis` (via redis-rs)

#### 6. Observability / Tracing

Metalcraft has `tracing` as a dependency but doesn't instrument anything with spans. LangGraph integrates deeply with LangSmith for trace visualization, cost tracking, latency monitoring, and debugging.

**What's needed:** Structured `tracing` spans for:
- Each node execution (with node name, duration, outcome type)
- Edge traversal decisions (with source, target, edge type)
- Checkpoint save/load operations
- Overall graph run lifecycle

This enables integration with any tracing backend (Jaeger, Datadog, OpenTelemetry) for free.

#### 7. Message Types / Chat History

LangGraph has built-in message types (`HumanMessage`, `AIMessage`, `ToolMessage`) that form a standard conversation protocol. Metalcraft leaves message representation entirely to the user.

**What's needed:** A standard `Message` enum and `MessageState` struct with a built-in append reducer:
```rust
enum Message {
    Human { content: String },
    Ai { content: String, tool_calls: Vec<ToolCall> },
    Tool { tool_call_id: String, content: String },
    System { content: String },
}
```

This would cover 80% of agentic use cases out of the box.

#### 8. Node/Task Caching

LangGraph 2.0 added node result caching to skip redundant computation. Metalcraft re-runs every node every time.

**What's needed:** An optional content-addressable cache keyed on `(node_name, state_hash)`. Requires `Hash` bound on state or a configurable cache key function.

#### 9. Error Recovery / Retry

LangGraph has built-in retry policies on nodes. Metalcraft propagates errors up immediately with no retry mechanism.

**What's needed:** Per-node retry configuration:
```rust
graph.add_node_with_config("llm_call", LlmNode, NodeConfig {
    retries: 3,
    backoff: ExponentialBackoff::default(),
})
```

---

### Tier 3 — Ecosystem Gaps (developer experience)

#### 10. Deployment Story

LangGraph has LangGraph Platform (now LangSmith Deployment) — one-click cloud deploy, managed infrastructure, BYOC. Metalcraft has no deployment guidance, no server wrapper, no REST API exposure.

**What's needed:** An `axum`-based server wrapper that exposes a compiled graph as a REST/WebSocket API with endpoints for run, stream, resume, and thread management.

#### 11. Studio / Visual Editor

LangGraph Studio lets you visually build, debug, and step through graphs. Metalcraft has `to_mermaid()` (read-only).

**What's needed (incremental):**
1. Interactive CLI debugger with step-through execution
2. Web-based visualizer that renders live Mermaid + state inspector
3. Full visual editor (long-term)

#### 12. Pre/Post Hooks & Guardrails

LangGraph 2.0 added guardrail nodes, pre/post model hooks, content filtering, and rate limiting as built-in configuration. Metalcraft has none of these.

**What's needed:** Hook points in the executor:
```rust
executor
    .before_node(|node_name, state| { /* validate, filter, log */ })
    .after_node(|node_name, state, outcome| { /* audit, transform */ })
```

#### 13. Documentation & Examples

LangGraph has extensive docs, tutorials, cookbooks, and a large community. Metalcraft has no README, no docs, and 5 examples.

**What's needed:**
- README.md with quickstart, installation, and feature overview
- API reference (rustdoc with examples on all public types)
- Cookbook with common patterns (ReAct agent, multi-agent, RAG pipeline, human-in-the-loop)
- Architecture guide explaining the Reducer/Node/Edge/Executor model

#### 14. Long-Term Memory

LangGraph distinguishes short-term (thread) memory from long-term (cross-thread) memory. Metalcraft checkpointing is thread-scoped only.

**What's needed:** A `MemoryStore` trait separate from `Checkpointer`:
```rust
#[async_trait]
trait MemoryStore: Send + Sync {
    async fn put(&self, namespace: &str, key: &str, value: serde_json::Value) -> Result<()>;
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<serde_json::Value>>;
    async fn search(&self, namespace: &str, query: &str) -> Result<Vec<serde_json::Value>>;
}
```

---

## Prioritized Roadmap

### P0 — Foundation (do first)

| Feature | Effort | Impact | Notes |
|---------|--------|--------|-------|
| README + docs + more examples | Low | High | Table stakes for adoption |
| Structured tracing spans on nodes/edges | Low | High | Enables all observability tooling |
| Built-in `Message` types + `MessageState` | Medium | High | Covers 80% of agentic use cases |

### P1 — Core Capabilities

| Feature | Effort | Impact | Notes |
|---------|--------|--------|-------|
| Subgraph composition (graph-as-node) | Medium | Very High | Unlocks multi-agent |
| Prebuilt ReAct agent pattern | Medium | High | Dramatically lowers barrier to entry |
| SQLite + Postgres checkpointers | Medium | High | Required for production |
| Node retry policies | Low | Medium | Simple but important for reliability |

### P2 — Power Features

| Feature | Effort | Impact | Notes |
|---------|--------|--------|-------|
| Field-level reducer derive macro | High | High | Major ergonomic win |
| Multi-agent supervisor pattern | Medium | High | Builds on subgraph composition |
| Node result caching | Low | Medium | Performance optimization |
| Pre/post execution hooks | Low | Medium | Guardrails and auditing |

### P3 — Ecosystem

| Feature | Effort | Impact | Notes |
|---------|--------|--------|-------|
| REST API server wrapper (axum) | Medium | Medium | Deployment story |
| Long-term cross-thread memory | Medium | Medium | Stateful agents across sessions |
| Visual debugger / step-through | High | High | Best-in-class DX |

---

## Metalcraft's Unique Advantages

While closing gaps with LangGraph, Metalcraft should lean into strengths that LangGraph cannot match:

1. **Compile-time safety** — invalid graphs don't compile. LangGraph catches errors at runtime.
2. **Performance** — Rust's zero-cost abstractions and async model are orders of magnitude faster than Python.
3. **Memory efficiency** — no GC, no interpreter overhead. Critical for edge deployment and high-throughput servers.
4. **Type-safe state mutations** — the `Update` enum ensures every state transition is explicitly handled. No silent key overwrites.
5. **Embeddability** — Metalcraft can be embedded in any Rust application, CLI tool, or WASM target. LangGraph requires a Python runtime.

The goal is not to clone LangGraph — it's to bring the same level of **capability and convenience** while preserving the **correctness and performance** that Rust uniquely enables.
