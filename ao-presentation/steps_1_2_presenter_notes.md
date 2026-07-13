# Presenter notes — Steps 1 & 2: collect activations, build the oracle prompt

## What's on the slide

Above the two step boxes, a dashed "target model" container shows the ordinary side of the pipeline: a **user prompt** ("Capital of France?") flows through a normal **forward pass** (layers 0–35, unchanged) to a **normal response** — the target model has no idea an oracle exists. A hook fires at **layer 18 of 36 (50% depth)**, capturing an **activation tensor of shape [seq_len × 4096]**, one 4096-dim vector per token.

Below that:

- **Step 1** — `collect_activations_multiple_layers(model, submodules, inputs_BL)`: registers a forward hook on `model.model.layers[18]`, which stores `outputs[0]` into a dict, then raises `EarlyStopException` to stop early.
- **Step 2** — `get_introspection_prefix(layer=18, num_positions=5)` returns `"Layer: 18\n ? ? ? ? ? \n"` prepended to the user's question; `find_pattern_in_tokens(token_ids, " ?", n, tokenizer)` scans the tokenised prompt for all `" ?"` IDs, verifies the count equals `n` and that they're consecutive, and returns their positions.

## What to say

Frame this as two separate models, two separate prompts — that distinction runs through the whole paper.

**The target model runs completely normally first.** This is worth dwelling on: nothing about running the target model differs from a standard forward pass. The model isn't fine-tuned for this, isn't told an oracle exists, and produces exactly the response it would produce anyway. All the paper's machinery attaches *around* this pass, not inside it.

**Step 1 — why a hook, and why raise an exception.** The hook on `model.model.layers[18]` is a standard PyTorch forward hook: it intercepts the layer's output before the model finishes generating. Storing `outputs[0]` captures the residual stream activations at that layer for every token position. Raising `EarlyStopException` right after is a performance detail worth calling out — once the target layer's output is captured, there's no need to run layers 19–35 or decode a token, so the hook aborts the forward pass early rather than wastefully completing it. **Layer 18 of 36 is 50% depth** — that's not arbitrary: the paper's appendix (C.5) found layer-50%-depth activations outperform 75%-depth on secret-keeping evals, with performance robust across layers 25–75% but poor in early layers (0–10%). This slide's choice of layer 18 is the paper's evaluation default.

**The activation tensor itself.** `[seq_len × 4096]` — one 4096-dimensional vector per token in the prompt. This is the *entire* signal the oracle will ever see about the target model's internal state; there's no text, no logits, just these vectors. It's worth noting these are raw residual-stream floats, not sparse or compressed in any way at this stage — unlike PCD, which routes captured activations through an encoder bottleneck before anything downstream sees them.

**Step 2 — building the question the oracle will answer.** This step has nothing to do with the target model — it's entirely about constructing the **oracle's** input prompt. `get_introspection_prefix` builds the literal string `"Layer: 18\n ? ? ? ? ? \n"`, with one `?` placeholder for each activation vector to be injected (five in this example), then prepends that to whatever natural-language question the user wants answered about those activations (e.g. "Is this a positive sentiment?"). This matches the paper's Section 3.1 input structure directly: layer number, K placeholder tokens, free text question.

**Why `find_pattern_in_tokens` matters.** Once the oracle prompt is tokenised, the `?` characters don't necessarily map to a single easy-to-find token — tokenisation can be messy around punctuation. This function scans the token IDs for the `" ?"` pattern, confirms it found exactly `n` of them (matching the number of activation vectors) and that they're consecutive, then returns their exact positions. Those positions are what Step 3's `OracleInput.positions` field carries forward, and ultimately what the Layer 1 steering hook uses to know precisely which residual-stream positions to modify.

## Connecting to the paper

This pair of steps operationalizes the paper's core terminology split: the **target prompt** (what the target model sees, Steps in the dashed container) versus the **oracle prompt** (what the oracle sees, built in Step 2). The activations captured here are exactly the **v** vectors that later appear in the norm-matching injection formula from Section 3.1:

```
h'_i = h_i + ‖h_i‖ · (v_i / ‖v_i‖)
```

Steps 1 and 2 don't touch that formula at all — they're purely about *gathering the ingredients*: capturing **v** from the target model, and building an oracle prompt with placeholder slots ready to receive it. The actual injection doesn't happen until the oracle's own forward pass reaches layer 1, several steps later in the pipeline.
