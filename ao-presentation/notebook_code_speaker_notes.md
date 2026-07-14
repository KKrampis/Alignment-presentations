# Speaker notes — notebook snippets shown in the two diagrams

## Background — target models vs. suspect models, and how the oracle is trained

### Is the "target model" the same as the suspect models from the auditing table?

Yes, with one nuance. **"Target model" is the general term** for whatever model the oracle is reading activations from — in the injection-mechanism diagrams, the toy example uses a plain "Capital of France?" prompt on an unmodified target model. **The suspect models from the auditing table are specific instances of a target model** — the same base architecture (Gemma-2-9B-IT, Llama-3.3-70B), but with a secret or misaligned behavior fine-tuned into them.

So: "target model" = the role in the pipeline (whose activations get captured and fed to the oracle); "suspect model" = a target model that's been deliberately fine-tuned to hide something, used specifically for the auditing evals.

### How are the oracles themselves trained?

Key asymmetry worth flagging: **the oracle is trained entirely on activations from the clean base model, never on activations from any fine-tuned/suspect model.** This is the headline surprising result — it generalizes to suspect models it's never seen activations from.

**Setup**

- The oracle is a **LoRA adapter** (r=64, on attention + MLP projections) on top of the *same base architecture* as the target it will later audit — so a Qwen3-8B oracle audits Qwen3-8B-family targets, a Llama-3.3-70B oracle audits Llama-3.3-70B-family targets, etc.
- Four models were trained this way: **Llama-3.3-70B-Instruct, Gemma-2-9B-IT, Qwen3-8B, Claude Haiku 3.5** (the last one differs — full fine-tuning instead of LoRA).

**Training data — three task types, mixed together**

1. **System Prompt QA (SPQA)** — oracle answers questions about a model's system prompt from activations of a user-assistant exchange (e.g. "What is the assistant's tone?" → "Formal and professional").
2. **Classification** — 7 binary yes/no tasks (sentiment, truth value, language ID, NER, gender, tense, entailment), activations taken near the end of the sequence.
3. **Self-supervised context prediction** — predict K contiguous tokens before/after an activation sequence, no overlap between input and target tokens, so the oracle can't just "reconstruct" — it has to actually use the activation content. Scales to any text corpus with no labeling needed.

**Scale and mechanics**

- ~1M training examples total, LoRA throughout.
- Activations collected at **3 depths** — 25%, 50%, 75% — always captured from the **base model, before LoRA fine-tuning**.
- Compute: 10 H100-hrs for Qwen3-8B, 90 H200-hrs for Llama-3.3-70B.
- Evaluation later uses **50% depth** specifically (ablations in Appendix C.5 found 50% beats 75%, and early layers 0–10% perform poorly).

**Ablation findings on the mixture (Section 6)**

The full 3-task mixture beats SPQA-only or SPQA+Classification on 10/12 model×eval combinations; task *diversity* helps even holding total example count fixed, and raw *quantity* helps independently on top of that (full 1M mixture beats a diversity-matched 400k subset).

---

## Notebook code snippets shown in the two diagrams

Source: `1_3_4_Activation_Oracles_exercises.ipynb` (ARENA 3.0), solution cells. Each snippet keeps the function's signature and docstring (`"""..."""`) verbatim for context, followed by only the body lines that correspond to what's drawn in the matching SVG box, with their original inline comments.

---

## Diagram 1 — `steps_1_2.svg`

### Step 1 box — "registers forward hook... stores outputs[0]... raises EarlyStopException"

> **Slide:** "Steps 1 & 2 — Collect activations and build the oracle prompt"

```python
def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, Int[Tensor, "batch seq"]],
    start_offset: int | None,
    end_offset: int | None,
) -> dict[int, Float[Tensor, "batch seq d_model"]]:
    """
    Collect activations from multiple layers using forward hooks.

    Args:
        model: The target model
        submodules: Dict mapping layer number to submodule to hook
        inputs_BL: Tokenized inputs (input_ids, attention_mask)
        start_offset: Start of the token slice (negative index from end). Only used when `end_offset`
            is also non-None; if `end_offset` is None, this must also be None (no slicing is applied).
        end_offset: End of the token slice (negative index from end, exclusive). Set both `start_offset`
            and `end_offset` to non-None values to enable token-position slicing; if both are None,
            the full sequence activations are returned.

    Returns:
        Dict mapping layer → activations tensor [batch, length, d_model]
    """
    ...

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]
        # Handle different output formats
        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs

        # Early stop after max layer
        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    # Register hooks
    handles.append(submodule.register_forward_hook(gather_target_act_hook))
```

### Step 2 box — `get_introspection_prefix(layer=18, num_positions=5)`

> **Slide:** "Steps 1 & 2 — Collect activations and build the oracle prompt"

```python
def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """Create the prefix for oracle prompts with ? tokens."""
    prefix = f"Layer: {layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix
```

### Step 2 box — `find_pattern_in_tokens(...)` — "scans... verifies count = n and consecutive, returns positions"

> **Slide:** "Steps 1 & 2 — Collect activations and build the oracle prompt"

```python
def find_pattern_in_tokens(
    token_ids: list[int],
    special_token_str: str,
    num_positions: int,
    tokenizer: AutoTokenizer,
) -> list[int]:
    """
    Find positions of special token in tokenized sequence.

    Args:
        token_ids: List of token IDs
        special_token_str: The special token string (e.g., " ?")
        num_positions: Expected number of occurrences
        tokenizer: Tokenizer to encode special token

    Returns:
        List of positions where special token appears
    """
    ...

    # Find ALL positions where this token appears (don't stop early)
    positions = [i for i, tid in enumerate(token_ids) if tid == special_token_id]

    if len(positions) != num_positions:
        raise ValueError(
            f"Expected {num_positions} occurrences of special token, but found {len(positions)}. "
            f"This can happen if your oracle prompt contains a literal '?' character."
        )
    assert positions[-1] - positions[0] == num_positions - 1, f"Positions are not consecutive: {positions}"

    return positions
```

---

## Diagram 2 — `steps_3_with_formula.svg`

### Step 3 box — "returns an `OracleInput`... `input_ids`... `steering_vectors` (cloned to CPU, detached)"

> **Slide:** "Steps 3–5 — Package, inject and generate" *(note: this slide's title still says "3–5" even though the diagram now only shows Step 3 — worth updating the slide title to match)*

```python
def create_oracle_input(
    prompt: str,
    layer: int,
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: Float[Tensor, "num_pos d_model"],
) -> OracleInput:
    """
    Create an oracle input for inference.

    Args:
        prompt: Question to ask the oracle
        layer: Layer the activations came from
        num_positions: Number of ? tokens (equals length of acts_BD)
        tokenizer: Tokenizer
        acts_BD: Activation vectors [num_positions, d_model]

    Returns:
        OracleInput ready for generation
    """
    ...

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )

    # Find ? token positions in the prompt
    positions = find_pattern_in_tokens(input_prompt_ids, SPECIAL_TOKEN, num_positions, tokenizer)

    # Ensure activations are on CPU and detached
    acts_BD = acts_BD.cpu().clone().detach()

    return OracleInput(
        input_ids=input_prompt_ids,
        layer=layer,
        steering_vectors=acts_BD,
        positions=positions,
    )
```

### Layer 1 box — `orig = resid[0, positions, :]` / `norms = orig.norm(dim=-1)` / `steered = normed_v * norms` / `resid[positions] += steered`

> **Slides:** "Steps 3–5 — Package, inject and generate" (the Layer 1 box itself) **and** "Norm-matching injection formula (Eq. 1)" (this is the code behind that formula)

```python
def get_hf_activation_steering_hook(
    vectors: Float[Tensor, "num_pos d_model"],
    positions: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Create hook that injects activations at specified positions (assumes batch_size=1).

    Args:
        vectors: Steering vectors [K, d_model] where K is number of positions
        positions: List of positions to inject at
        steering_coefficient: Multiplier for steering strength
        device: Device for tensors
        dtype: Data type for steering

    Returns:
        Hook function that modifies activations during forward pass
    """
    # Normalize vectors to unit norm
    normed_vectors = torch.nn.functional.normalize(vectors, dim=-1).detach()

    ...

    def hook_fn(module, _input, output):
        ...

        # Get original activations at steering positions
        orig_KD = resid_BLD[0, positions_tensor, :]  # [K, d_model]
        norms_K1 = orig_KD.norm(dim=-1, keepdim=True)  # [K, 1]

        # Scale normalized steering vectors by original magnitudes
        steered_KD = (normed_vectors * norms_K1 * steering_coefficient).to(dtype)

        # Inject (add to original)
        resid_BLD[0, positions_tensor, :] = steered_KD.detach() + orig_KD
```

**Mapping the diagram's pseudocode to this code:** `orig` = `orig_KD`, `norms` = `norms_K1`, `normed_v` = `normed_vectors`, `steered` = `steered_KD`. The diagram compresses `h'_i = h_i + ‖h_i‖ · (v_i / ‖v_i‖)` into these four lines — `normed_vectors` is the unit vector `v_i/‖v_i‖`, `norms_K1` is `‖h_i‖`, and the final line is the addition `h_i + steered`.
