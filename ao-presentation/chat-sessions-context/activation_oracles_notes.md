# Activation Oracles — Research Notes

Sources:
- Paper: *Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers* — arXiv:2512.15674 (Karvonen, Chua, Dumas, Fraser-Taliente, Kantamneni, Minder, Ong, Sen Sharma, Wen, Evans, Marks)
- Related follow-ups: *Confidence and Calibration of Activation Oracles* (arXiv:2605.26045), *Building Better Activation Oracles* (arXiv:2606.02609)
- Companion post: https://alignment.anthropic.com/2025/activation-oracles/
- Code: github.com/adamkarvonen/activation_oracles
- Notebook: `KKrampis/ARENA_3.0`, chapter1_transformer_interp, `part34_activation_oracles/1.3.4_Activation_Oracles_exercises.ipynb`

---

## 1. Paper overview

**Core idea.** Builds on LatentQA (Pan et al. 2024): train an LLM to directly accept another model's activations as input and answer free-form natural-language questions about them, instead of relying on narrow specialized tools (probes, SAEs). This paper generalizes LatentQA into **Activation Oracles (AOs)** — trained to answer *arbitrary* questions about activations, evaluated on far out-of-distribution tasks.

**Headline result.** AOs can recover information fine-tuned into a model (biographical facts, malign propensities) that never appears in the input text — despite never being trained on activations from a fine-tuned model. Best AOs match or exceed prior white-box interpretability baselines on all 4 downstream auditing tasks studied, and are best-overall on 3 of 4.

**Flagship demo (Figure 1).** A model secretly fine-tuned to play Taboo (guide the user to guess "tree" without ever saying it). An AO is fed the suspect model's activations at the assistant token and exposes both the goal and the secret word — purely from activations, not outputs.

### Background / relation to prior work
- **LatentQA** (Pan et al. 2024): introduced the QA-about-activations task and Latent Interpretation Tuning method. Prior work was narrow — SAE feature interpretation, system-prompt QA, user-belief modeling.
- AOs are compared against: (1) **PatchScopes** — untrained AOs (same weights, no fine-tuning); (2) narrowly-trained AOs (Pan et al.'s original data); (3) task-specific baselines where applicable.

---

## 2. Methods (Section 3)

### 3.1 Activation steering for latent verbalization

**Terminology.**
- *Target prompt* — input to the target model; activations are collected from here.
- *Oracle prompt* — input to the Activation Oracle; contains placeholder tokens + a natural-language question.

**Input structure.** Given K activation vectors {v₁, ..., v_K} from layer ℓ of the target model, the oracle prompt is built from: (1) the layer number ℓ, (2) K placeholder tokens (`?`) where the vectors get injected, (3) free natural-language text. Example:

```
Layer 18: ? ? ? Is this a positive sentiment?
```

**Injection mechanism.** At layer 1 of the *oracle's* forward pass (not layer 0 — see Appendix A.5), a hook modifies the residual stream at each placeholder position by **adding** a norm-matched steering vector:

$$\mathbf{h}_i' = \mathbf{h}_i + \|\mathbf{h}_i\| \cdot \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \tag{1}$$

where **h**ᵢ is the oracle's own original activation at placeholder position i, and ‖·‖ is the L2 norm.

They add rather than replace (unlike Pan et al. 2024) because direct replacement caused catastrophic norm growth during training: **20× at layer 0, 100,000× at layer 1** (Appendix A.5). Addition avoids this.

#### Step-by-step mechanics (walkthrough, not in the paper's text but reconstructed from the method + notebook)

1. **Run the target model**, collect activations **v**ᵢ at chosen layer/positions during a normal forward pass. These are just saved vectors, detached from the target model.
2. **Build the oracle prompt** with literal `?` tokens — one per activation vector — plus the natural-language question. Tokenize normally.
3. **Start the oracle's forward pass.** A hook is attached at layer 1.
4. **The hook locates the `?` token positions** (e.g. via a pattern search over token IDs — `find_pattern_in_tokens` in the notebook).
5. **Overwrite only those positions** using formula (1); every other position (real text tokens) passes through unmodified.
6. **Continue the forward pass** through remaining layers. Attention lets later tokens (the question) read the modified `?` positions.
7. **Generate the answer** via normal decoding.

The `?` tokens are empty vessels whose only job is to mark exact positions for the hook to overwrite — their original embedding content is irrelevant.

#### Understanding the norm-matching formula (bullets 2 & 3 explained)

- **Norm** ‖**v**‖ = vector length, e.g. for **v** = [3,4], ‖**v**‖ = √(3²+4²) = 5.
- **Unit vector** **v**/‖**v**‖: divide by its own norm → length becomes exactly 1, direction preserved. E.g. [3,4]/5 = [0.6, 0.8]. This strips out magnitude, keeping only *direction*.
- **h**ᵢ is a *separate* vector: the oracle's own activation at that position (nothing to do with **v**ᵢ). It has its own norm, e.g. ‖**h**ᵢ‖ = 12.
- **‖h**ᵢ‖ · (unit vector)**: rescale the unit vector (length 1, **v**ᵢ's direction) up to length 12 — i.e. 12 · [0.6, 0.8] = [7.2, 9.6], which has norm exactly 12 (√(7.2²+9.6²) = 12) but still points in **v**ᵢ's direction.
- **Why "norm-matching":** activation vectors from different sources (raw residual stream, SAE features, difference vectors) have wildly different typical magnitudes. Feeding them in raw could either blow out the position (huge norm) or barely register (tiny norm) — inconsistent behavior the oracle wasn't trained to expect. By stripping the target vector to pure direction, then rescaling to match ‖**h**ᵢ‖ (the magnitude the oracle already expects at that position), every injection lands at a magnitude the oracle is "used to," regardless of source. Direction carries the actual information (what the oracle has to interpret); scale is standardized automatically.
- Final step: **h**ᵢ' = **h**ᵢ + (norm-matched vector) — additive, not a replacement, which is what avoids the runaway norm explosion mentioned above.

### 3.2 Latent verbalization datasets

Three training task types:

1. **System Prompt QA (SPQA)** — from Pan et al. (2024). Oracle answers questions about a model's system prompt given activations from user-assistant interactions (e.g. "What is the assistant's tone?" → "Formal and professional"). Activations collected from different segments (system prompt only / user prompt only / user+assistant).
2. **Classification datasets** — 7 binary yes/no tasks adapted to natural language (sentiment, truth value, language ID, NER, gender, tense, entailment). Activations from near end of sequence.
3. **Self-supervised context prediction** — predict K contiguous tokens before/after an activation sequence (K, J ~ Uniform(1,20)), scalable to any text corpus. No overlap between input and target tokens, forcing reliance on activation content rather than reconstruction.

### 3.3 Implementation details

- **Models trained:** Llama-3.3-70B-Instruct, Gemma-2-9B-IT, Qwen3-8B, Claude Haiku 3.5.
- **Activation collection (training):** 3 layers — 25%, 50%, 75% depth — always from base model before LoRA fine-tuning.
- **Evaluation:** activations from 50% depth (see Appendix C.5 for ablations).
- **Scale:** ~1M training examples. LoRA adapters. 10 H100-hrs for Qwen3-8B, 90 H200-hrs for Llama-3.3-70B.

---

## 3. Evaluation on downstream auditing tasks (Section 4)

Based on Marks et al. (2025) "auditing games" — controlled benchmarks testing whether tools can uncover fine-tuned knowledge/behaviors that a model won't reveal through ordinary prompting.

### 4.1 Secret-keeping evaluations
Models from Cywiński et al. (2025), three secret types:
- **Taboo** (Gemma-2-9B-IT): 20 models each hiding a secret word (e.g. "gold," "moon").
- **User Gender** (Gemma-2-9B-IT): models trained to assume/deny knowledge of user's gender.
- **Secret Side Constraint (SSC)** (Llama-3.3-70B): base64-encoded hidden instructions the model follows but won't reveal.

**Result:** AOs beat all white-box interpretability baselines (logit lens, SAE-based) on all three tasks, and beat black-box baselines on two of three. Notable gain: 21% vs 4% accuracy on Taboo.

### 4.2 Emergent misalignment audit
Three narrowly misaligned models (risky finance, extreme sports, bad medical advice) × 2 base models. AO given the *difference* between base/fine-tuned activations, asked an open-ended question ("Can you describe what is happening in this text?"). Fed into an auditing agent alongside model-diffing baseline (ADL from Minder et al. 2025).

**Result:** AO matches ADL baseline (2/5 average score — near-max given most models show minimal emergent misalignment due to 50% chat data mixed in).

---

## 4. Additional OOD generalization experiments (Section 5)

### 5.1 PersonaQA
Tests whether AOs do genuine interpretation vs. "text inversion" (recovering input text rather than deeper content). 100 fictional personas, 6 attributes each, fine-tuned into models; evaluated on prompts containing only the persona's name.

**Result:** Accuracy scales with training mixture diversity — base model ~5% → full AO ~14% (open-source models). Claude Haiku 3.5 AO reaches ~35%. All trained AOs beat PatchScopes baseline. Caveat: fine-tuned models' own knowledge is brittle (>80% on open-ended Qs, near-chance on yes/no reformulations).

### 5.2 Classification (held-out)
7 OOD binary classification datasets (AG News, headline topic ID, language ID, etc.). AOs generalize well; untrained base models near chance.

---

## 5. Training dataset ablations (Section 6)

- **Scaling the training mixture:** comparing untrained base → SPQA-only → SPQA+Classification → full mixture. 10/12 model×eval combinations peak with the full mixture. SPQA-only is already competitive on secret-keeping but more hyperparameter-sensitive.
- **Diversity vs. quantity:** controlled ablation on Qwen3-8B holding total examples fixed (400k) while varying diversity. Diverse mixture (with context-prediction task) beats SPQA+Classification baseline even at equal sample count — diversity helps independent of raw data volume. Full 1M mixture still outperforms the diversity-matched 400k mixture — quantity also helps.

---

## 6. Discussion & limitations (Section 8)

- AOs trade mechanistic understanding for a simpler black-box QA interface — no drill-down into mechanisms or targeted interventions.
- **Reliability/calibration is a real limitation** — the model isn't trained to express uncertainty and frequently answers confidently even when wrong.
- Promising future directions: better/more diverse training data, self-supervised scaling, multi-turn data, on-policy generations, RL training.
- AOs may not beat simple prompting on all practical tasks (e.g. hallucination detection — anecdotal experiment in Appendix F found a simple black-box follow-up-question baseline performs similarly).
- Benchmarks are narrow, single-behavior fine-tunes; real post-training pipelines are richer.

---

## 7. Key appendix details

- **Steering layer selection (A.5):** layer 1 outperforms layer 0 for Llama-3.3-70B (1–11% better across evals); no significant difference for Qwen3/Gemma-2 in training loss.
- **Claude Haiku 3.5 differs (A.6):** replacement-based (not additive) injection at layer 0, full fine-tuning instead of LoRA, single middle layer only, different placeholder/prompt format.
- **Evaluation token selection (C.4):** full-sequence activations outperform single-token on most evals, *except* Taboo, where single-token (assistant start-of-turn) is notably better (SPQA-only: 4.4% → 21.6%).
- **Layer choice at inference (C.5):** layer 50% beats 75% on secret-keeping evals; classification performance is robust across layers 25–75%; early layers (0–10%) perform poorly.
- **SAE experiments (Appendix G):** tried using SAE feature vectors as additional training signal — mixed results, not worth the added complexity/cost (~$1000 in API costs), excluded from final mixture.

---

## 8. ARENA Notebook — Section 1: Introduction & Using Activation Oracles

Source: `1.3.4_Activation_Oracles_exercises.ipynb`, cells 9–33.

**Purpose:** build intuition using a *pre-trained* oracle before Section 2 has you implement the injection mechanics from scratch.

### Core concept
AO = LLM trained to accept another model's activation vectors as part of its input, answering questions in plain English. Oracle is loaded as a **LoRA adapter on the same base architecture** as the target model. Mechanically: feed the oracle `Layer N: ? ? ? Is this a positive sentiment?`, then after an early layer, swap in the target's real layer-N activations at the `?` positions.

Framed vs. alternatives: linear probes answer only their one trained question; SAEs decompose activations but still need human interpretation; oracles generalize to unseen questions but sacrifice mechanistic grounding (no error bars, no easy verification of what signal is actually being read).

### Loading models (cells 11–16)
- Base model: `Qwen/Qwen3-8B`
- Oracle adapter: `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`, loaded via PEFT (`model.load_adapter(..., adapter_name="oracle")`)
- LoRA config: rank `r=64`, target modules `down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj`

### Oracle queries (cells 17–25)
Uses provided `utils.run_oracle()` helper (built from scratch in Section 2). Key params: `target_prompt`, `oracle_prompt`, `oracle_lora_path`, `oracle_input_type` (`"full_seq"`, `"segment"`, or `"tokens"`).

**Demo 1:** Target prompt "What is the capital of France?" → oracle prompt "What answer will the model give?" → correctly answers "Paris" using only activations, not raw output.

**Skepticism exercises:**
1. *"Answering question" hypothesis* — is the oracle just latching onto the "France" token embedding? Test: feed only the segment *after* the France token. Still answers correctly → info has moved forward in the residual stream, not just echoed.
2. *"Logit lens" hypothesis* — is the oracle just reading the top next-token prediction? Test: check top-10 logits at that position directly — "Paris" is indeed there, so logit lens is a plausible partial explanation in this case. (Bonus: try multi-step reasoning prompts where the direct next-token wouldn't give it away.)

### Token-by-token analysis (cells 26–28)
Philosopher chain prompt ("hemlock" → Socrates → "student who founded" → Plato → "most famous pupil" → Aristotle). Querying every token position independently (`oracle_input_type="tokens"`) shows representations accumulate incrementally through the sequence.

### Final exercise (cells 29–32)
Given `def foo(x,y): return x+y \n result = foo(3,4)`, feed oracle activations only from `result = foo(3, 4)` onward (`oracle_input_type="segment"`), excluding the function definition. Oracle still answers "7" — activations encode the *computed result*, not just surface tokens.

### Transition (cell 33)
Sets up Section 2 (build `run_oracle()` internals: hooks, `?` token mechanism, steering injection, training-data formatting) before Section 3 (advanced applications: secret extraction, misalignment detection).

---

## References (selected, from paper bibliography)

- Pan, Chen, Steinhardt (2024) — *LatentQA: Teaching LLMs to Decode Activations into Natural Language* — arXiv:2412.08686
- Ghandeharioun et al. (2024) — *PatchScopes* — arXiv:2401.06102
- Cywiński et al. (2025) — *Eliciting Secret Knowledge from Language Models* — arXiv:2510.01070
- Minder et al. (2025) — *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences* — arXiv:2510.13900
- Marks et al. (2025) — *Auditing Language Models for Hidden Objectives* — arXiv:2503.10965
- Li et al. (2025a) — *Training Language Models to Explain Their Own Computations* — arXiv:2511.08579
- Li et al. (2025b) — *Do Natural Language Descriptions of Model Activations Convey Privileged Information?* — arXiv:2509.13316
- Betley et al. (2025) — *Emergent Misalignment* — arXiv:2502.17424
- Turner et al. (2025) — *Model Organisms for Emergent Misalignment* — arXiv:2506.11613
