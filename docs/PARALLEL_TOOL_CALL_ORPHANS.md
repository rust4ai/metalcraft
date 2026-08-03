# Parallel tool calls produce an invalid chat/completions history

**Status:** known limitation in `0.8.x`. Worked around downstream by using the
OpenAI **Responses API** instead of chat/completions. This note explains the bug
and sketches the proper in-crate fix (a candidate for a `0.9` upgrade).

## Symptom

When a pod routes inference through the chat/completions surface (OpenAI's
`/v1/chat/completions`, or an OpenAI-compatible gateway), a turn that emits **two
or more tool calls at once** (parallel / multi tool-calling) fails with:

```
openai 400 Bad Request: An assistant message with 'tool_calls' must be followed
by tool messages responding to each 'tool_call_id'. The following tool_call_ids
did not have response messages: call_XXX
  ... "param": "messages.[N].role"
```

This is extremely easy to hit with integration packs (e.g. a notes pack that the
model drives with `list_pages` + `read_page` in a single turn). It surfaced most
visibly in the `sub_agent` / persona path, because those agents lean on packs and
run without compaction, so nothing else masks it.

## Root cause

The failure is a **message-ordering** bug, not a missing-result bug. Every tool
call *does* get a result within the run — but the results are laid out in an order
chat/completions rejects.

1. **The reducer stores one message per call.** `Reducer::apply` in
   [`src/prebuilt.rs`](../src/prebuilt.rs) pushes a separate `AgentMessage::ToolCall`
   for each parallel call (`AgentUpdate::ToolCalls`, ~L334), then a separate
   `AgentMessage::ToolResult` for each result (`AgentUpdate::ToolResults`, ~L351).
   For parallel calls `A` and `B` the history becomes:

   ```
   ToolCall(A), ToolCall(B), ToolResult(A), ToolResult(B)
   ```

2. **`build_conversation` converts each `ToolCall` message to its own rig
   message**, with no coalescing (`src/prebuilt.rs`, ~L615; the `ToolCall` arm at
   ~L639/L673 does `history.push(RigMessage::from(tc))` per call).

3. **rig wraps exactly one tool call per assistant message**
   (`impl From<ToolCall> for Message` → `Message::Assistant { content:
   OneOrMany::one(ToolCall(..)) }`), and the OpenAI provider flattens messages
   without merging consecutive assistant messages.

The bytes that reach OpenAI are therefore:

```
assistant { tool_calls: [A] }
assistant { tool_calls: [B] }   <-- A never got a tool response before this
tool (A)
tool (B)
```

chat/completions requires the message *immediately after*
`assistant{tool_calls:[A]}` to be the tool response(s) for `A`. It is another
assistant message instead → 400.

The **Responses API tolerates** this layout (it matches tool outputs to calls by
`call_id` rather than by strict adjacency), which is why the downstream workaround
is to keep pods on the Responses client and expose a `POST /responses` passthrough
on the inference gateway.

## Why we might want to fix it in-crate (the `0.9` upgrade)

Relying on the Responses API works but pins us to one OpenAI surface. Any
OpenAI-*compatible* backend that only implements chat/completions (most gateways,
vLLM, many router products, Azure's older surface) will 400 on parallel tool
calls. Fixing the serialization makes the agent portable across both surfaces and
removes a latent foot-gun.

### Proposed fix

Coalesce adjacent `AgentMessage::ToolCall` entries from the same turn into a
**single** rig assistant message that carries all of the turn's tool calls, emitted
*before* the corresponding tool-result messages:

```
assistant { tool_calls: [A, B] }   // one message, OneOrMany::many([A, B])
tool (A)
tool (B)
```

Concretely, in `build_conversation` (`src/prebuilt.rs`): when walking the history,
group a maximal run of consecutive `ToolCall` messages and build one
`Message::Assistant { content: OneOrMany::many([ToolCall(A), ToolCall(B), ...]) }`
instead of one assistant message per call, then append each `ToolResult` as its own
`tool` message. This is valid for **both** chat/completions and the Responses API,
so it is a safe unconditional change.

### Also harden orphan handling

Two related orphan sources are worth folding into the same upgrade, because
chat/completions is strict about them where the Responses API was lenient:

- **Mid-run orphans are only patched between user turns.** `continue_with`
  (`src/prebuilt.rs` ~L181) synthesizes `"ERROR: interrupted by user"` results for
  any `ToolCall` lacking a `ToolResult` — but it only runs when a *new user
  message* is added. A one-shot `sub_agent` run never calls it, and a guard/step
  interruption that stops right after tool calls are emitted (before the tool node
  runs) leaves an orphan in the history that is sent as-is on the next completion
  call. Consider patching orphans **inside `build_conversation`** (right before
  handing history to the model) so every request is always well-formed, regardless
  of how the run was interrupted.

- **History compaction can split a parallel group.** Downstream compaction that
  keeps the last *N* messages can land its boundary inside a
  `ToolCall(A), ToolCall(B), ToolResult(A), ToolResult(B)` group and summarize
  `ToolCall(A)` away while keeping `ToolResult(A)` — producing both an orphaned
  result and the same unpaired-assistant-message ordering. Compaction boundaries
  must treat a parallel tool-call/result group as one atomic unit. (This lives in
  the consumer today, but a helper in this crate that returns "safe split points"
  would let every consumer get it right.)

## Test to add alongside the fix

A turn with two tool calls, serialized via `build_conversation`, should produce a
single assistant message whose `tool_calls` contains both ids, followed by exactly
two `tool` messages — and the resulting array should pass OpenAI chat/completions
validation (assert the first message after the assistant is a `tool` message whose
`tool_call_id` matches one of the calls).
