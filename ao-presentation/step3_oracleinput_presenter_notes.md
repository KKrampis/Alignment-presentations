# Presenter notes — Step 3: package into OracleInput

## What's on the slide

The Step 3 box shows `create_oracle_input(prompt, layer, n, tokenizer, acts_BD)`, which returns an `OracleInput` dataclass with four fields: `input_ids` (the tokenised oracle prompt) and `steering_vectors` (activations cloned to CPU, detached from the graph), alongside `layer` and `positions`.

## What to say

This is the handoff step — everything captured from the *target* model in Steps 1 and 2 gets packaged into a single object the *oracle* can actually consume.

Walk through it in order:

- **Where the inputs come from.** By this point we already have two things: the oracle prompt text, built in Step 2 with literal `?` placeholder tokens marking where injection should happen, and the raw activation vectors captured from the target model's forward pass in Step 1, at a chosen layer and set of token positions.
- **What `create_oracle_input` does.** It's a packaging function, not a transformation. It tokenises the oracle prompt into `input_ids`, and takes the captured activations and stores them as `steering_vectors`. Crucially, these are **cloned to CPU and detached from the autograd graph** — they're just saved numbers at this point, with no gradient connection back to the target model. That detachment is what makes injection safe: nothing that happens to the oracle can leak gradients back into the model we captured activations from.
- **Why a dataclass, not a dict.** `OracleInput` gives four typed fields — `input_ids`, `layer`, `steering_vectors`, `positions` — so every downstream function (the steering hook, `run_oracle`) has one unambiguous contract for what it needs, rather than passing four loose arguments around.
- **What `positions` is really doing.** These are the exact token indices of the `?` placeholders inside `input_ids`, found earlier via `find_pattern_in_tokens`. Step 3 doesn't discover them — it carries them forward so the Layer 1 hook (next step in the diagram) knows precisely which positions to modify and which to leave untouched.

## Connecting to the paper

This step is the practical bridge to Section 3.1 of the paper: the `steering_vectors` here are exactly the **v** vectors in the norm-matching injection formula,

```
h'_i = h_i + ‖h_i‖ · (v_i / ‖v_i‖)
```

`OracleInput` is what carries **v** — plus the layer number and placeholder positions — into the oracle's forward pass, where the Layer 1 hook applies that formula. Nothing in Step 3 touches the oracle's own activations (**h**) yet; it's purely assembling the *inputs* the hook will need once generation starts.

It's also worth flagging for the audience: at this point the pipeline still has **no bottleneck and no interpretability step of its own** — unlike PCD's encoder, which compresses activations into a sparse concept list before handing anything to a decoder, the Activation Oracle passes the raw captured vectors straight through. `OracleInput` is honest about that: it's a container, not a compressor.
